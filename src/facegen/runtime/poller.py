"""Policy for watching the newest save until it stops changing."""

from __future__ import annotations

from dataclasses import dataclass

from facegen.datasource.snapshots import SaveVersion, copy_snapshot, find_latest_save, list_saves
from facegen.pipeline.sync_operation import (
    UNSUPPORTED_ADVICE,
    FolderNotSetError,
    NoSaveError,
    SaveUnreadableError,
    sync_save,
)

__all__ = [
    "SETTLE_SECONDS",
    "UNSUPPORTED_ADVICE",
    "FolderNotSetError",
    "NoSaveError",
    "SaveUnreadableError",
    "SaveVersion",
    "SaveWatcher",
    "WatchDecision",
    "copy_snapshot",
    "find_latest_save",
    "list_saves",
    "sync_save",
]

SETTLE_SECONDS = 10.0
"""A changed save must stay unchanged this long before it is copied."""


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
