"""The save-watcher thread; sync execution is supplied through SyncControl."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from facegen.datasource.snapshots import SaveVersion, find_latest_save
from facegen.persistence import db, publication
from facegen.persistence.settings import folder_not_set_message, load_settings
from facegen.pipeline.sync import sync_of_save
from facegen.runtime.contracts import FAILED, SYNCED, PollerStatus, SyncControl
from facegen.runtime.logging import stamp
from facegen.runtime.policy import BUSY_TIMEOUT_MS, FM_POLL_SECONDS
from facegen.runtime.poller import SaveWatcher


class PollerService:
    """Watches FM27's saves folder and syncs the newest save once it stops changing."""

    def __init__(
        self,
        db_path: Path,
        syncer: SyncControl,
        *,
        defaults_path: Path | None = None,
        settle_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self.db_path = db_path
        self.defaults_path = defaults_path
        self._syncer = syncer
        self._watcher = SaveWatcher() if settle_seconds is None else SaveWatcher(settle_seconds)
        self._clock = clock
        self._conn = conn
        self._owns_conn = conn is None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = PollerStatus("starting", "", None, "starting")

    def status(self) -> PollerStatus:
        with self._lock:
            return self._status

    def _set(self, state: str, detail: str, reason: str, **values: Any) -> None:
        """A new state, with the English line for the report (`detail`) and the code and
        values the Log page words (contracts.WATCHER_REASONS)."""
        with self._lock:
            self._status = PollerStatus(state, detail, stamp(), reason, values)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        return self._conn

    def step(self) -> float:
        """Look at the folder once and sync if a save is due. Returns seconds to wait."""
        conn = self._connection()
        settings = load_settings(conn, self.defaults_path)
        interval = float(settings.schedule.poll_seconds)
        recovery_error = None
        if publication.pending(conn):
            self._set("syncing", "finishing publication of the active career's pack", "publishing")
            recovered = self._syncer.run(trigger="publish recovery")
            if recovered is None:
                return self._watcher.settle_seconds
            if recovered == FAILED:
                recovery_error = self._syncer.status().error or "could not publish the pack"
                self._set("error", recovery_error, "publish_failed", error=recovery_error)
        games = settings.paths.games_dir
        if games is None:
            if recovery_error is None:
                self._set("not set", folder_not_set_message(), "not_set")
            return interval
        try:
            newest_path = find_latest_save(games)
            newest = SaveVersion.of(newest_path) if newest_path is not None else None
        except OSError as exc:  # the save vanished between listing and stat, or /mnt/c is gone
            self._set("error", f"cannot look at {games}: {exc}", "cannot_look", error=str(exc))
            return interval
        done = sync_of_save(conn, newest.path.name, newest.mtime) if newest else None
        decision = self._watcher.look(newest, done is not None, self._clock(), interval)
        if decision.due is None:
            if recovery_error is not None:
                return decision.wait
            if newest is not None and done is not None and done["error"]:
                name = newest.path.name
                why = "its game build is not supported" if done["unsupported"] else done["error"]
                detail = f"newest save {name} could not be read ({why}); waiting for the next save"
                if done["unsupported"]:
                    self._set("error", detail, "unsupported", name=name)
                else:
                    self._set("error", detail, "unreadable", name=name, error=done["error"])
            else:
                self._set(decision.state, decision.detail, decision.reason, **decision.values)
            return decision.wait
        self._set("syncing", decision.detail, decision.reason, **decision.values)
        outcome = self._syncer.run(decision.due.path, trigger="poller")
        if outcome is None:
            self._set("settling", "waiting for a sync started by hand", "by_hand")
            return self._watcher.settle_seconds
        self._watcher.done()
        name = decision.due.path.name
        status = self._syncer.status()
        if outcome == FAILED:
            error = status.error or "the sync failed"
            self._set("error", f"{error}; trying again shortly", "sync_failed", error=error)
        elif outcome == SYNCED:
            saved = decision.due.saved_at()
            self._set("synced", status.message or "synced", "synced", name=name, saved=saved)
        else:
            detail = f"{status.error}; waiting for the next save"
            if status.unsupported:
                self._set("error", detail, "unsupported", name=name)
            else:
                self._set("error", detail, "unreadable", name=name, error=status.error or "")
        return interval

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="facegen-poller", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    def _main(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    delay = self.step()
                except Exception as exc:  # noqa: BLE001 - keep watching; show it in the UI
                    error = f"{type(exc).__name__}: {exc}"
                    self._set("error", f"{error}; trying again shortly", "failed", error=error)
                    delay = FM_POLL_SECONDS
                self._wake.wait(delay)
                self._wake.clear()
        finally:
            self._set("stopped", "the save watcher has stopped", "stopped")
            if self._owns_conn and self._conn is not None:
                self._conn.close()
                self._conn = None
