"""The thread adapter that claims and reports one sync at a time."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from facegen.datasource.base import DataSource
from facegen.pipeline.sync_operation import execute_sync
from facegen.runtime.contracts import SyncOutcome, SyncRequest, SyncResult, SyncStatus
from facegen.runtime.logging import stamp


class SyncService:
    """Runs one sync at a time: "Sync now" in a thread of its own, or the save watcher's."""

    def __init__(
        self,
        db_path: Path,
        source_factory: Callable[[], DataSource],
        *,
        defaults_path: Path | None = None,
        on_done: Callable[[], Any] = lambda: None,
        log: Callable[[str], None] = lambda line: None,
    ) -> None:
        self.db_path = db_path
        self.defaults_path = defaults_path
        self._source_factory = source_factory
        self._on_done = on_done
        self._log = log
        self._lock = threading.Lock()
        self._status = SyncStatus()
        self._thread: threading.Thread | None = None

    def status(self) -> SyncStatus:
        with self._lock:
            return SyncStatus(**vars(self._status))

    def _claim(self, trigger: str) -> bool:
        with self._lock:
            if self._status.running:
                return False
            self._status = SyncStatus(running=True, started=stamp(), trigger=trigger)
            return True

    def start(self) -> bool:
        """Start a sync of the newest save in the background; False if one is running."""
        if not self._claim("sync now"):
            return False
        self._thread = threading.Thread(
            target=self._run, args=(None, "sync now"), name="facegen-sync", daemon=True
        )
        self._thread.start()
        return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self, save: Path | None = None, trigger: str = "sync now") -> SyncOutcome | None:
        """Sync `save` (default: the newest) in this thread. Returns SYNCED, UNREADABLE or
        FAILED, or None if another sync is running."""
        if not self._claim(trigger):
            return None
        return self._run(save, trigger)

    def _run(self, save: Path | None, trigger: str) -> SyncOutcome:
        result = execute_sync(
            SyncRequest(
                db_path=str(self.db_path),
                defaults_path=str(self.defaults_path) if self.defaults_path is not None else None,
                save_path=str(save) if save is not None else None,
                trigger=trigger,
            ),
            self._source_factory,
            log=self._log,
            on_result=self._finished,
        )
        self._on_done()
        return result.outcome

    def _finished(self, result: SyncResult) -> None:
        with self._lock:
            self._status = SyncStatus(
                running=False,
                started=self._status.started,
                trigger=self._status.trigger,
                message=result.message,
                error=result.error,
                unsupported=result.unsupported,
                warnings=list(result.warnings),
                notice=result.notice,
            )
