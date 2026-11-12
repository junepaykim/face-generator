"""Finding the save to read, copying it safely, and syncing it.

The tool never parses a save in the game's folder: it copies it to data/snapshots/ and
reads the copy (AGENTS.md hard rule 1). A save that changes while it is copied is mid-write;
the copy is retried after a pause. The watch loop itself arrives in Phase 4.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from facegen.datasource.base import DataSource, SaveReadError, UnsupportedSaveError
from facegen.settings import Settings
from facegen.sync import SyncReport, record_failed_sync, run_sync

# FM27 keeps the previous save under this name when one is overwritten.
_BACKUP_PREFIX = "last save overwrite backup"


def list_saves(games_dir: Path) -> list[Path]:
    """Career saves in the game's saves folder, newest first."""
    if not games_dir.is_dir():
        return []
    saves = [
        p
        for p in games_dir.glob("*.fm")
        if p.is_file() and not p.name.lower().startswith(_BACKUP_PREFIX)
    ]
    return sorted(saves, key=lambda p: p.stat().st_mtime, reverse=True)


def find_latest_save(games_dir: Path) -> Path | None:
    saves = list_saves(games_dir)
    return saves[0] if saves else None


def _signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def copy_snapshot(
    source: Path,
    snapshots_dir: Path,
    *,
    attempts: int = 3,
    settle_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Copy a save into snapshots_dir and return the copy's path.

    The source is only read. The copy keeps the source's modification time. If the source's
    size or modification time changes during the copy, wait and try again.
    """
    source = source.resolve()
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshots_dir.resolve() / source.name
    if dest == source:
        raise ValueError("the snapshot would overwrite its source")
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        before = _signature(source)
        shutil.copyfile(source, partial)
        after = _signature(source)
        if before == after and partial.stat().st_size == after[0]:
            st = source.stat()
            os.utime(partial, ns=(st.st_atime_ns, st.st_mtime_ns))
            os.replace(partial, dest)
            return dest
        if attempt + 1 < attempts:
            sleep(settle_seconds)
    partial.unlink(missing_ok=True)
    raise SaveReadError(f"{source.name} kept changing while it was copied; is FM27 saving?")


class NoSaveError(LookupError):
    """There is no save to read."""


class SaveUnreadableError(RuntimeError):
    """The save could not be copied or parsed; existing pictures and the queue are unchanged."""

    def __init__(self, save: Path, cause: Exception) -> None:
        super().__init__(f"Cannot read {save.name}: {cause}")
        self.save = save


def sync_save(
    conn: sqlite3.Connection,
    settings: Settings,
    source: DataSource,
    save: Path | None = None,
    *,
    dry_run: bool = False,
    preview: int = 20,
    with_table: bool = False,
) -> SyncReport:
    """Copy a save (default: the newest in FM27's games folder), parse the copy and sync it.

    A save already inside data/snapshots/ is read in place. A save that cannot be read is
    recorded as a failed sync (except in a dry run) and raises SaveUnreadableError.
    """
    snapshots = settings.paths.snapshots_dir
    if save is None:
        save = find_latest_save(settings.paths.games_dir)
        if save is None:
            raise NoSaveError(f"No save found in {settings.paths.games_dir}")
    if not save.is_file():
        raise NoSaveError(f"No such save: {save}")
    try:
        if save.resolve().is_relative_to(snapshots.resolve()):
            snapshot_path = save
        else:
            snapshot_path = copy_snapshot(save, snapshots)
        snapshot = source.load(snapshot_path)
    except (UnsupportedSaveError, SaveReadError) as exc:
        if not dry_run:
            record_failed_sync(conn, save.name, str(exc))
        raise SaveUnreadableError(save, exc) from exc
    return run_sync(
        conn,
        snapshot,
        settings,
        save_file=save.name,
        save_mtime=save.stat().st_mtime,
        dry_run=dry_run,
        preview=preview,
        with_table=with_table,
    )
