"""Discover FM saves and copy stable snapshots without writing to the source folder."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kitshot.datasource.base import SaveBusyError

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
        copied = partial.stat()
        if before == after and copied.st_size == after[0]:
            # The time checked with the copy, not one read later: a save written since then
            # is newer than this copy, and the watcher must still see it as new.
            os.utime(partial, ns=(copied.st_atime_ns, after[1]))
            os.replace(partial, dest)
            return dest
        if attempt + 1 < attempts:
            sleep(settle_seconds)
    partial.unlink(missing_ok=True)
    raise SaveBusyError(f"{source.name} kept changing while it was copied; is FM27 saving?")


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
