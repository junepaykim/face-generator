"""Finding the save to read and copying it safely.

The tool never parses a save in the game's folder: it copies it to data/snapshots/ and
reads the copy (AGENTS.md hard rule 1). A save that changes while it is copied is mid-write;
the copy is retried after a pause. The watch loop itself arrives in Phase 4.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from facegen.datasource.base import SaveReadError

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
