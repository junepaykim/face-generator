"""The save-watcher thread; sync execution is supplied through SyncControl."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from facegen import db
from facegen.poller import SaveVersion, SaveWatcher, find_latest_save
from facegen.runtime_policy import BUSY_TIMEOUT_MS, FM_POLL_SECONDS
from facegen.service_contracts import FAILED, SYNCED, PollerStatus, SyncControl, SyncOutcome
from facegen.service_logging import stamp
from facegen.settings import folder_not_set_message, load_settings
from facegen.sync import sync_of_save


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
        self._status = PollerStatus("starting", "", None)

    def status(self) -> PollerStatus:
        with self._lock:
            return self._status

    def _set(self, state: str, detail: str) -> None:
        with self._lock:
            self._status = PollerStatus(state, detail, stamp())

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        return self._conn

    def step(self) -> float:
        """Look at the folder once and sync if a save is due. Returns seconds to wait."""
        conn = self._connection()
        settings = load_settings(conn, self.defaults_path)
        interval = float(settings.schedule.poll_seconds)
        games = settings.paths.games_dir
        if games is None:
            self._set("not set", folder_not_set_message())
            return interval
        try:
            newest_path = find_latest_save(games)
            newest = SaveVersion.of(newest_path) if newest_path is not None else None
        except OSError as exc:  # the save vanished between listing and stat, or /mnt/c is gone
            self._set("error", f"cannot look at {games}: {exc}")
            return interval
        done = sync_of_save(conn, newest.path.name, newest.mtime) if newest else None
        decision = self._watcher.look(newest, done is not None, self._clock(), interval)
        if decision.due is None:
            if newest is not None and done is not None and done["error"]:
                why = "its game build is not supported" if done["unsupported"] else done["error"]
                self._set(
                    "error",
                    f"newest save {newest.path.name} could not be read ({why});"
                    " waiting for the next save",
                )
            else:
                self._set(decision.state, decision.detail)
            return decision.wait
        self._set("syncing", decision.detail)
        outcome = self._syncer.run(decision.due.path, trigger="poller")
        if outcome is None:  # "Sync now" is running; look again shortly
            self._set("settling", "waiting for a sync started by hand")
            return self._watcher.settle_seconds
        self._watcher.done()
        if outcome == FAILED:
            status = self._syncer.status()
            self._set("error", f"{status.error or 'the sync failed'}; trying again shortly")
            return interval
        self._set("synced" if outcome == SYNCED else "error", self._after(outcome))
        return interval

    def _after(self, outcome: SyncOutcome) -> str:
        status = self._syncer.status()
        if outcome == SYNCED:
            return status.message or "synced"
        return f"{status.error}; waiting for the next save"

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
                    self._set("error", f"{type(exc).__name__}: {exc}; trying again shortly")
                    delay = FM_POLL_SECONDS
                self._wake.wait(delay)
                self._wake.clear()
        finally:
            self._set("stopped", "the save watcher has stopped")
            if self._owns_conn and self._conn is not None:
                self._conn.close()
                self._conn = None
