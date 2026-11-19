"""The adapter that claims and reports one sync at a time.

With a `command`, each sync runs in a child process (runtime.sync_process): `facegen serve`
and the desktop app with the real save reader. With a `source_factory` instead, it runs in
the calling thread: the tests' fake sources, as WorkerService stays beside WorkerProxy.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from facegen.datasource.base import DataSource
from facegen.persistence import db, publication
from facegen.pipeline.sync_operation import execute_sync
from facegen.runtime.contracts import FAILED, SyncOutcome, SyncRequest, SyncResult, SyncStatus
from facegen.runtime.logging import stamp
from facegen.runtime.policy import BUSY_TIMEOUT_MS
from facegen.runtime.sync_process import SyncProcess

STOP_JOIN_SECONDS = 10.0
"""How long `stop` waits for "Sync now"'s thread once its sync process has ended."""

logger = logging.getLogger("facegen")


class SyncService:
    """Runs one sync at a time: "Sync now" in a thread of its own, or the save watcher's."""

    def __init__(
        self,
        db_path: Path,
        source_factory: Callable[[], DataSource] | None = None,
        *,
        command: Sequence[str] | None = None,
        defaults_path: Path | None = None,
        on_done: Callable[[], Any] = lambda: None,
        log: Callable[[str], None] = lambda line: None,
        echo_stderr: bool = True,
    ) -> None:
        if (source_factory is None) == (command is None):
            raise ValueError("SyncService takes either a source factory or a sync process command")
        self.db_path = db_path
        self.defaults_path = defaults_path
        self._source_factory = source_factory
        self._process = (
            SyncProcess(command, log=log, echo_stderr=echo_stderr) if command is not None else None
        )
        self._on_done = on_done
        self._log = log
        self._lock = threading.Condition()
        self._status = SyncStatus()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._recovery_thread: threading.Thread | None = None

    @property
    def process(self) -> SyncProcess | None:
        """The child-process runner, or None when syncs run in the calling thread."""
        return self._process

    def status(self) -> SyncStatus:
        with self._lock:
            return SyncStatus(**vars(self._status))

    def _claim(self, trigger: str, *, wait: bool = False) -> bool:
        with self._lock:
            if wait:
                self._lock.wait_for(lambda: self._stopped or not self._status.running)
            if self._status.running or self._stopped:
                return False
            self._status = SyncStatus(running=True, started=stamp(), trigger=trigger)
            return True

    def recover_if_pending(self) -> None:
        """For starts without a watcher, check for an unfinished pack in the background."""

        def recover() -> None:
            try:
                with closing(db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)) as conn:
                    pending = publication.pending(conn)
                if pending and self._claim("publish recovery", wait=True):
                    self._run(None, "publish recovery")
            except (OSError, sqlite3.Error):
                logger.exception("could not check for an interrupted publication at startup")

        with self._lock:
            if self._stopped or self._recovery_thread is not None:
                return
            self._recovery_thread = threading.Thread(
                target=recover, name="facegen-publish-recovery", daemon=True
            )
            self._recovery_thread.start()

    def start(self) -> bool:
        """Start a sync of the newest save in the background; False if one is running, or
        facegen is quitting."""
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
        if self._recovery_thread is not None:
            self._recovery_thread.join(timeout)

    def run(self, save: Path | None = None, trigger: str = "sync now") -> SyncOutcome | None:
        """Sync `save` (default: the newest) in this thread. Returns SYNCED, UNREADABLE or
        FAILED, or None if another sync is running or facegen is quitting."""
        if not self._claim(trigger):
            return None
        return self._run(save, trigger)

    def stop(self, timeout: float | None = STOP_JOIN_SECONDS) -> None:
        """For Quit: start no further sync and end a running sync process (its database
        transaction is rolled back; a committed one publishes first, see sync_process), then
        wait up to `timeout` seconds for "Sync now"'s thread. A sync in the calling thread
        (the tests' fakes) cannot be ended."""
        with self._lock:
            self._stopped = True
            self._lock.notify_all()
        if self._process is not None:
            self._process.stop()
        self.join(timeout)

    def _run(self, save: Path | None, trigger: str) -> SyncOutcome:
        request = SyncRequest(
            db_path=str(self.db_path),
            defaults_path=str(self.defaults_path) if self.defaults_path is not None else None,
            save_path=str(save) if save is not None else None,
            trigger=trigger,
        )
        if self._process is None:
            assert self._source_factory is not None
            result = execute_sync(
                request, self._source_factory, log=self._log, on_result=self._finished
            )
        else:
            try:
                result = self._process.run(request)
            except Exception as exc:
                logger.error("the sync process could not be run", exc_info=exc)
                result = SyncResult(FAILED, error=f"sync failed: {type(exc).__name__}: {exc}")
            self._finished(result)
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
            self._lock.notify_all()
