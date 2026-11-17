"""Save discovery, snapshot copying, sync execution and watcher policy.

Parse only copies in data/snapshots/. Retry copies if the source changes during copying.
`SaveWatcher` waits for an unsynced version to stay unchanged for `settle_seconds`;
database sync records prevent duplicate reads across restarts and manual syncs.
The watcher polls because Windows file-change events do not reach WSL.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from facegen.core.settings_values import Settings
from facegen.datasource import photos
from facegen.datasource.base import (
    DataSource,
    SaveBusyError,
    SaveReadError,
    UnsupportedSaveError,
)
from facegen.persistence.settings import folder_not_set_message
from facegen.pipeline.sync import SyncReport, record_failed_sync, run_sync

SETTLE_SECONDS = 10.0
"""A changed save must stay unchanged this long before it is copied."""

UNSUPPORTED_ADVICE = (
    "The save reader (fmsave, pinned in pyproject.toml) cannot read this game build. Existing "
    "pictures and the queue are unchanged, and nothing new is queued from this save. When an "
    "fmsave release supports the build: bump the pin, re-run the Phase 0.1 checks and note the "
    "result in docs/research.md (AGENTS.md rule 5)."
)

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
    raise SaveBusyError(f"{source.name} kept changing while it was copied; is FM27 saving?")


class NoSaveError(LookupError):
    """There is no save to read."""


class FolderNotSetError(NoSaveError):
    """FM27's user folder is neither set nor found, so there is no saves folder to look in."""

    def __init__(self) -> None:
        super().__init__(folder_not_set_message())


class SaveUnreadableError(RuntimeError):
    """The save could not be copied or parsed; existing pictures and the queue are unchanged."""

    def __init__(self, save: Path, cause: Exception) -> None:
        super().__init__(f"Cannot read {save.name}: {cause}")
        self.save = save
        self.unsupported = isinstance(cause, UnsupportedSaveError)
        """The game or build is one the save reader cannot read (see UNSUPPORTED_ADVICE)."""
        self.busy = isinstance(cause, SaveBusyError)
        """The game was still writing the save; a later try should work."""


def sync_save(
    conn: sqlite3.Connection,
    settings: Settings,
    source: DataSource,
    save: Path | None = None,
    *,
    dry_run: bool = False,
    preview: int = 20,
    with_table: bool = False,
    trigger: str | None = None,
) -> SyncReport:
    """Copy a save (default: the newest in FM27's games folder), parse the copy and sync it.

    A save already inside data/snapshots/ is read in place. A save that cannot be read is
    recorded as a failed sync (except in a dry run) and raises SaveUnreadableError. The
    recorded save time is the copy's, which keeps the source's time from when it was copied.
    Each sync also looks again at who FM27 shows a photo for (photos.py): its own photo list
    and the user's facepacks, read again only if they changed.
    """
    snapshots = settings.paths.snapshots_dir
    if save is None:
        games = settings.paths.games_dir
        if games is None:
            raise FolderNotSetError()
        save = find_latest_save(games)
        if save is None:
            raise NoSaveError(f"No save found in {games}")
    if not save.is_file():
        raise NoSaveError(f"No such save: {save}")
    save_mtime: float | None = None
    try:
        if save.resolve().is_relative_to(snapshots.resolve()):
            snapshot_path = save
        else:
            snapshot_path = copy_snapshot(save, snapshots)
        save_mtime = snapshot_path.stat().st_mtime
        snapshot = source.load(snapshot_path)
    except (UnsupportedSaveError, SaveReadError) as exc:
        if not dry_run:
            record_failed_sync(
                conn,
                save.name,
                str(exc),
                getattr(exc, "build", None),
                # A save still being written is tried again; any other failure waits for the
                # save to change.
                save_mtime=None if isinstance(exc, SaveBusyError) else save_mtime,
                unsupported=isinstance(exc, UnsupportedSaveError),
                trigger=trigger,
            )
        raise SaveUnreadableError(save, exc) from exc
    return run_sync(
        conn,
        snapshot,
        settings,
        save_file=save.name,
        save_mtime=save_mtime,
        dry_run=dry_run,
        preview=preview,
        with_table=with_table,
        trigger=trigger,
        photos=photos.scan(settings.paths, photos.stored_state(conn)),
    )


@dataclass(frozen=True, slots=True)
class SaveVersion:
    """One version of a save file: its path, size and modification time."""

    path: Path
    size: int
    mtime_ns: int
    mtime: float

    @classmethod
    def of(cls, path: Path) -> SaveVersion:
        st = path.stat()
        return cls(path, st.st_size, st.st_mtime_ns, st.st_mtime)

    def saved_at(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True, slots=True)
class WatchDecision:
    due: SaveVersion | None
    """Sync this save now; None: nothing to do yet."""
    wait: float
    """Seconds until the watcher should look again."""
    state: str
    """'no save', 'synced', 'settling' or 'due'."""
    detail: str


class SaveWatcher:
    """Decides when the newest save in FM27's games folder is due for a sync.

    Pure apart from the clock time passed in; the caller finds the newest save and asks the
    database whether that version is already synced.
    """

    def __init__(self, settle_seconds: float = SETTLE_SECONDS) -> None:
        self.settle_seconds = settle_seconds
        self._pending: tuple[SaveVersion, float] | None = None

    def look(
        self, newest: SaveVersion | None, already_synced: bool, now: float, interval: float
    ) -> WatchDecision:
        if newest is None:
            self._pending = None
            return WatchDecision(None, interval, "no save", "no save in FM27's games folder yet")
        name = newest.path.name
        if already_synced:
            self._pending = None
            detail = f"newest save {name} (saved {newest.saved_at()}) is synced"
            return WatchDecision(None, interval, "synced", detail)
        if self._pending is None or self._pending[0] != newest:
            self._pending = (newest, now)
            return WatchDecision(
                None,
                self.settle_seconds,
                "settling",
                f"{name} changed; it is synced once it stops changing",
            )
        waited = now - self._pending[1]
        if waited < self.settle_seconds:
            return WatchDecision(
                None,
                self.settle_seconds - waited,
                "settling",
                f"{name} changed; it is synced once it stops changing",
            )
        return WatchDecision(newest, 0.0, "due", f"reading {name}")

    def done(self) -> None:
        """The due save was synced, or failed in a way the database records."""
        self._pending = None
