"""The background work behind `facegen serve`: the worker loop and "sync now".

**Worker.** One thread owns the GPU. Each pass re-reads the settings and the queue, so a
threshold change or a manual request applied in the browser is picked up before the next
job (spec section 7). It draws manual requests first, then the automatic jobs of the groups
in `schedule.auto_groups`, in queue order, and publishes after each picture.

- While FM27 runs it waits and frees the GPU, unless "generate now" is set or
  `schedule.generate_while_fm_running` is true (spec section 6).
- "Generate now" draws the whole queue as it stands, every group, FM27 running or not, and
  lasts until nothing is left to draw. Only re-draws waiting for a reference picture
  (Phase 5) stay behind.
- When the queue has been empty for a few minutes, or the worker is paused, the model is
  unloaded so the card is free for the game.
- An error (model failed to load, out of memory) pauses the worker and shows the error;
  the job goes back to the queue. Resume tries again.
- `step` runs one pass and says how long to wait, so tests drive the loop without a thread.

**Sync now** reads the newest save in a thread of its own, as `facegen sync` does.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from facegen import db
from facegen.datasource.base import DataSource
from facegen.kits import KitTable, load_kits
from facegen.poller import NoSaveError, SaveUnreadableError, sync_save
from facegen.publish import PublishError
from facegen.settings import load_settings
from facegen.worker import (
    Job,
    JobFilter,
    JobGone,
    Renderer,
    Tagger,
    Worker,
    WorkerBusy,
    fm_running,
    load_career,
    worker_lock,
)

BUSY_TIMEOUT_MS = 60_000
IDLE_POLL_SECONDS = 15.0
"""How often an idle worker looks at the queue when nothing wakes it."""
FM_POLL_SECONDS = 30.0
"""How often a worker waiting for FM27 to close checks again."""
IDLE_UNLOAD_SECONDS = 300.0
"""An empty queue for this long frees the GPU."""
LOG_LINES = 200


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """What the dashboard shows about the worker."""

    state: str
    """starting, idle, paused, waiting, running, error or stopped."""
    detail: str
    current: Job | None
    paused: bool
    generate_now: bool
    models_loaded: bool
    done: int
    failed: int
    waiting_for_reference: int
    last_error: str | None
    log: tuple[str, ...]


class WorkerService:
    def __init__(
        self,
        db_path: Path,
        *,
        renderer_factory: Callable[[], Renderer],
        tagger_factory: Callable[[], Tagger],
        defaults_path: Path | None = None,
        kits: KitTable | None = None,
        fm_check: Callable[[], bool | None] = fm_running,
        start_paused: bool = False,
        clock: Callable[[], float] = time.monotonic,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self.db_path = db_path
        self.defaults_path = defaults_path
        self._kits = kits
        self._renderer_factory = renderer_factory
        self._tagger_factory = tagger_factory
        self._fm_check = fm_check
        self._clock = clock
        self._conn = conn
        self._owns_conn = conn is None
        self._worker: Worker | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log: deque[str] = deque(maxlen=LOG_LINES)
        self._state = "starting"
        self._detail = ""
        self._current: Job | None = None
        self._paused = start_paused
        self._generate_now = False
        self._done = 0
        self._failed = 0
        self._waiting_for_reference = 0
        self._last_error: str | None = None
        self._idle_since: float | None = None
        self._unpublished = False

    # -- control, from the web threads --

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        self.log("Paused.")
        self.wake()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._last_error = None
        self.log("Resumed.")
        self.wake()

    def generate_now(self) -> None:
        """Draw the whole current queue, every group, even while FM27 runs (spec section 6)."""
        with self._lock:
            self._generate_now = True
            self._paused = False
        self.log("Generate now: drawing the whole queue, every group, even if FM27 is running.")
        self.wake()

    def wake(self) -> None:
        """Look at the queue now rather than at the next poll."""
        self._wake.set()

    def status(self) -> WorkerStatus:
        with self._lock:
            return WorkerStatus(
                state=self._state,
                detail=self._detail,
                current=self._current,
                paused=self._paused,
                generate_now=self._generate_now,
                models_loaded=self._worker is not None and self._worker.models_loaded,
                done=self._done,
                failed=self._failed,
                waiting_for_reference=self._waiting_for_reference,
                last_error=self._last_error,
                log=tuple(self._log),
            )

    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(f"{_stamp()}  {line}")

    def _set(self, state: str, detail: str = "", current: Job | None = None) -> None:
        with self._lock:
            self._state, self._detail, self._current = state, detail, current

    # -- the thread --

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="facegen-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Ask the loop to stop after the current job; wait up to `timeout` seconds."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _main(self) -> None:
        conn = self._connection()
        try:
            while not self._stop.is_set():
                settings = load_settings(conn, self.defaults_path)
                try:
                    with worker_lock(settings.paths.data_dir):
                        self.log("Worker started.")
                        self._requeue_stale(conn)
                        self._loop()
                    break
                except WorkerBusy:
                    self._set("waiting", "another facegen worker is running (facegen run?)")
                    self._wait(FM_POLL_SECONDS)
        except Exception as exc:  # noqa: BLE001 - never die silently: show it in the UI
            self._set("error", f"worker crashed: {exc}")
            self.log("".join(traceback.format_exception_only(exc)).strip())
        finally:
            if self._worker is not None:
                self._worker.unload()
            self._set("stopped", "the worker has stopped")
            if self._owns_conn:
                conn.close()
                self._conn = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                delay = self.step()
            except Exception as exc:  # noqa: BLE001 - e.g. the database stayed locked: retry
                message = f"{type(exc).__name__}: {exc}"
                self._set("error", f"{message}; trying again shortly")
                self.log(f"Worker error: {message}")
                delay = FM_POLL_SECONDS
            if delay > 0:
                self._wait(delay)

    def _wait(self, seconds: float) -> None:
        self._wake.wait(seconds)
        self._wake.clear()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        return self._conn

    def _requeue_stale(self, conn: sqlite3.Connection) -> None:
        career = load_career(conn)
        if career is None:
            return
        worker = self._worker_for(conn)
        if worker is not None and (n := worker.requeue_stale()):
            self.log(f"Put {n} interrupted job(s) back in the queue.")

    def _worker_for(self, conn: sqlite3.Connection) -> Worker | None:
        """The worker for the latest career, with fresh settings; models are kept."""
        career = load_career(conn)
        if career is None:
            return None
        settings = load_settings(conn, self.defaults_path)
        if self._worker is None:
            self._worker = Worker(
                conn,
                settings,
                self._kits or load_kits(),
                career,
                renderer_factory=self._renderer_factory,
                tagger_factory=self._tagger_factory,
                fm_check=self._fm_check,
                log=self.log,
            )
        else:
            if self._worker.career.id != career.id:
                self.log(f"Now working on career {career.display_name}.")
            self._worker.career = career
            self._worker.settings = settings
        return self._worker

    # -- one pass --

    def step(self) -> float:
        """One pass of the loop: run at most one job. Returns seconds to wait (0: go on)."""
        conn = self._connection()
        worker = self._worker_for(conn)
        if worker is None:
            self._set("idle", "no career yet: sync a save")
            return IDLE_POLL_SECONDS
        settings = worker.settings
        with self._lock:
            paused, generate_now = self._paused, self._generate_now

        if paused:
            self._publish_pending(worker)
            if worker.unload():
                self.log("Paused: the GPU is free.")
            self._set("paused", self._last_error or "")
            return IDLE_POLL_SECONDS

        if generate_now:
            f = JobFilter(manual_always=True)
        else:
            f = JobFilter(groups=settings.schedule.auto_groups, manual_always=True)
        ready, waiting = worker.runnable(f)
        with self._lock:
            self._waiting_for_reference = waiting
        if not ready:
            with self._lock:
                self._generate_now = False
            self._publish_pending(worker)
            now = self._clock()
            if self._idle_since is None:
                self._idle_since = now
            elif now - self._idle_since >= IDLE_UNLOAD_SECONDS and worker.unload():
                self.log("Queue empty: the GPU is free.")
            self._set("idle", self._idle_detail(worker, waiting))
            return IDLE_POLL_SECONDS
        self._idle_since = None

        if not (generate_now or settings.schedule.generate_while_fm_running):
            running = self._fm_check()
            if running is None or running:
                self._publish_pending(worker)
                if worker.unload():
                    self.log("FM27 is running: the GPU is free.")
                why = (
                    "FM27 is running"
                    if running
                    else "cannot tell whether FM27 is running (tasklist.exe failed)"
                )
                self._set("waiting", f"{why}; {len(ready)} job(s) wait. Generate now overrides.")
                return FM_POLL_SECONDS

        job = ready[0]
        self._set("running", f"{job.name}: {job.kind} {job.group.upper()}", job)
        self.log(f"{job.name} ({job.unique_id}), {job.kind} {job.group.upper()}")
        try:
            outcome = worker.run_job(job)
        except JobGone:
            return 0.0  # cancelled or changed meanwhile (a settings change); look again
        except Exception as exc:  # noqa: BLE001 - model load, out of memory: pause and report
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._paused = True
                self._last_error = f"{job.name}: {message}"
            worker.unload()
            self.log(f"  error, worker paused: {message}")
            self._set("error", self._last_error or message)
            return IDLE_POLL_SECONDS
        with self._lock:
            if outcome.status == "done":
                self._done += 1
            else:
                self._failed += 1
        self.log(f"  {outcome.message} ({outcome.seconds:.1f} s)")
        self._unpublished = True
        self._publish_pending(worker)
        self._set("running", "next job")
        return 0.0

    @staticmethod
    def _idle_detail(worker: Worker, waiting_for_reference: int) -> str:
        """Why nothing is drawn: the queue is empty, or what is held back."""
        held = worker.runnable(JobFilter())[0]  # every group; none of them drawn now
        parts = []
        if held:
            groups = ", ".join(sorted({j.group.upper() for j in held}))
            parts.append(
                f"{len(held)} job(s) in group {groups} are not drawn automatically yet;"
                " Generate now draws them"
            )
        if waiting_for_reference:
            parts.append(f"{waiting_for_reference} wait for a reference picture (Phase 5)")
        return "; ".join(parts) or "the queue is empty"

    def _publish_pending(self, worker: Worker) -> None:
        if not self._unpublished:
            return
        try:
            report = worker.publish()
        except PublishError as exc:
            self.log(f"Publishing failed: {exc}")
            with self._lock:
                self._last_error = f"publishing failed: {exc}"
            return
        self._unpublished = False
        if report.written:
            self.log(
                f"Published {report.written} picture(s); FM27: Settings -> Reload Skin to see them."
            )


@dataclass
class SyncStatus:
    running: bool = False
    started: str | None = None
    message: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


class SyncService:
    """Runs one sync at a time in a background thread ("Sync now")."""

    def __init__(
        self,
        db_path: Path,
        source_factory: Callable[[], DataSource],
        *,
        defaults_path: Path | None = None,
        on_done: Callable[[], Any] = lambda: None,
    ) -> None:
        self.db_path = db_path
        self.defaults_path = defaults_path
        self._source_factory = source_factory
        self._on_done = on_done
        self._lock = threading.Lock()
        self._status = SyncStatus()
        self._thread: threading.Thread | None = None

    def status(self) -> SyncStatus:
        with self._lock:
            return SyncStatus(**vars(self._status))

    def start(self) -> bool:
        """Start a sync of the newest save; False if one is already running."""
        with self._lock:
            if self._status.running:
                return False
            self._status = SyncStatus(running=True, started=_stamp())
        self._thread = threading.Thread(target=self.run, name="facegen-sync", daemon=True)
        self._thread.start()
        return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self) -> None:
        message = error = None
        warnings: list[str] = []
        conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        try:
            settings = load_settings(conn, self.defaults_path)
            report = sync_save(conn, settings, self._source_factory(), preview=0)
            queued = sum(report.enqueued.values())
            message = (
                f"Synced {report.save_file} (in-game {report.game_date or 'unknown'}): "
                f"{queued} queued, {report.updated} updated, {report.cancelled} cancelled."
            )
            warnings = report.warnings
        except (NoSaveError, SaveUnreadableError) as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - a background sync reports, never raises
            error = f"sync failed: {type(exc).__name__}: {exc}"
        finally:
            conn.close()
            with self._lock:
                self._status = SyncStatus(
                    running=False,
                    started=self._status.started,
                    message=message,
                    error=error,
                    warnings=warnings,
                )
        self._on_done()
