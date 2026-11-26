"""GPU worker loop; settings and queue changes apply before the next job."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kitshot.core.job_records import Job, JobFilter
from kitshot.core.kit_values import KitTable
from kitshot.datasource.kits import load_kits
from kitshot.persistence import db, jobs
from kitshot.persistence.jobs import JobGone, load_career, queued_counts
from kitshot.persistence.settings import load_settings
from kitshot.pipeline.publish import RELOAD_HINT, PublishError
from kitshot.pipeline.worker import PRE_ENCODE_JOBS, Worker, WorkerBusy, fm_running, worker_lock
from kitshot.render.gpu import describe as describe_gpu
from kitshot.render.gpu import is_fatal
from kitshot.render.renderer import Renderer, Tagger
from kitshot.runtime.contracts import Exit, ExitReason, Link, LogLevel, LogLine, WorkerStatus
from kitshot.runtime.logging import LOG_LINES, PYTHON_LEVELS, logger, stamp, utc_now
from kitshot.runtime.policy import BUSY_TIMEOUT_MS, FM_POLL_SECONDS

STOPPED_EXIT = 4
"""A worker process's exit code when Quit ended it while it was drawing (`stop_now`)."""
SKIPPED_EXIT = 5
"""A worker process's exit code when Skip ended it while it was drawing (`skip`)."""

IDLE_POLL_SECONDS = 15.0
"""How often an idle worker looks at the queue when nothing wakes it."""
IDLE_UNLOAD_SECONDS = 300.0
"""An empty queue for this long frees the GPU."""
MODELS_MISSING = (
    "the models are not downloaded yet: download them once from the dashboard (Set up), or"
    " with `kitshot models --download`; {n} job(s) wait"
)
GPU_FAULT_ADVICE = (
    "The graphics driver reported an error, and this run of Kitshot cannot draw any more. "
    "Close Kitshot and start it again to continue; the queue is kept."
)
CHILD_FAULT_DETAIL = "the graphics driver reported an error; this worker process ends"


class WorkerService:
    def __init__(
        self,
        db_path: Path,
        *,
        renderer_factory: Callable[[str], Renderer],
        tagger_factory: Callable[[], Tagger],
        defaults_path: Path | None = None,
        kits: KitTable | None = None,
        fm_check: Callable[[], bool | None] = fm_running,
        start_paused: bool = False,
        generate_now: bool = False,
        clock: Callable[[], float] = time.monotonic,
        conn: sqlite3.Connection | None = None,
        link: Link | None = None,
        models_ready: Callable[[str], bool] | None = None,
    ) -> None:
        self.db_path = db_path
        self._models_ready = models_ready
        """Whether the selected model's files are on this PC (models.ready_now), asked with
        `style.model` before each job; None: not asked, as with the tests' fake renderers."""
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
        self._log: deque[LogLine] = deque(maxlen=LOG_LINES)
        self._state = "starting"
        self._detail = ""
        self._reason = "starting"
        self._values: dict[str, Any] = {}
        self._since: str | None = utc_now()
        self._current: Job | None = None
        self._paused = start_paused
        self._generate_now = generate_now
        self._done = 0
        self._failed = 0
        self._last_error: str | None = None
        self._gpu_fault = False
        self._idle_since: float | None = None
        self._unpublished = True
        """Publish at the next chance: after a picture, and once at start."""
        self._reload_hint = False
        """Pictures reached FM27's active pack since the queue last ran out: the Reload Skin
        hint is logged once when it does (Phase 17i)."""
        self._link = link
        self.exit: Exit | None = None
        """In a worker process, why the loop ended; None until it does."""
        self._drawing = False
        """A job is being drawn (models loading, rendering, storing it): Quit ends the worker
        process at once then (`stop_now`)."""

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        self.log("Paused.")
        self.wake()

    def resume(self) -> bool:
        """Go on drawing. False, with nothing changed, after a GPU fault."""
        with self._lock:
            if self._gpu_fault:
                return False
            self._paused = False
            self._last_error = None
        self.log("Resumed.")
        self.wake()
        return True

    def generate_now(self) -> bool:
        """Draw the whole current queue, every group, even while FM27 runs (spec section 6).

        False, with nothing changed, after a GPU fault.
        """
        with self._lock:
            if self._gpu_fault:
                return False
            self._generate_now = True
            self._paused = False
        self.log("Generate now: drawing the whole queue, every group, even if FM27 is running.")
        self.wake()
        return True

    def skip(self, job_id: int, exit_now: Callable[[int], Any] | None = None) -> bool:
        """Skip current (Phase 17m, D8): the job being drawn goes back to the queue behind the
        rest (`jobs.skip`), keeping its sheet and seeds. In a worker process, `exit_now`
        (`os._exit`) then ends the process at once, as Quit does, after telling the parent why;
        the parent starts another. In a thread the picture under way cannot be stopped: it is
        dropped when it ends, as its job is no longer running. False, with nothing changed, if
        that job is not being drawn (it finished, or another one is)."""
        with self._lock:
            job = self._current if self._drawing else None
        if job is None or job.id != job_id:
            return False
        conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        try:
            moved = jobs.skip(conn, job_id)
        finally:
            conn.close()
        if not moved:
            return False
        self.log(f"Skipped {job.name}: drawn again after the rest of the queue.")
        if exit_now is not None and self._leave("skipped", None, job):
            assert self._link is not None and self.exit is not None
            self._link.exit(self.exit)  # what serve_child would say, had it the chance
            exit_now(SKIPPED_EXIT)
        return True

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
                gpu_fault=self._gpu_fault,
                generate_now=self._generate_now,
                models_loaded=self._worker is not None and self._worker.models_loaded,
                done=self._done,
                failed=self._failed,
                last_error=self._last_error,
                log=tuple(self._log),
                reason=self._reason,
                values=dict(self._values),
                since=self._since,
            )

    def log(self, line: str, level: LogLevel = "info") -> None:
        """A line for the Log page and the log file; warnings and errors pass their level."""
        entry = LogLine(stamp(), line, level)
        with self._lock:
            self._log.append(entry)
        logger.log(PYTHON_LEVELS.get(level, logging.INFO), "%s", line)
        if self._link is not None:
            self._link.line(entry)
            self._link.status(self.status)

    def _set(
        self,
        state: str,
        detail: str = "",
        current: Job | None = None,
        *,
        reason: str = "",
        **values: Any,
    ) -> None:
        """A new state, with the English line for the log and the tray (`detail`) and the
        reason code and values the pages word (contracts.WORKER_REASONS)."""
        with self._lock:
            job_now = current.id if current else None
            job_before = self._current.id if self._current else None
            if (state, reason, job_now) != (self._state, self._reason, job_before):
                self._since = utc_now()
            self._state, self._detail, self._current = state, detail, current
            self._reason, self._values = reason, values
        if self._link is not None:
            self._link.status(self.status)

    def _leave_if_paused(self) -> bool:
        """In a worker process, a pause ends it wherever the loop is waiting."""
        with self._lock:
            paused = self._paused
        return paused and self._leave("paused")

    def _leave(
        self, reason: ExitReason, message: str | None = None, job: Job | None = None
    ) -> bool:
        """In a worker process, end the loop so the process exits; else do nothing.

        Returns whether the loop ends. The first reason given is the one kept.
        """
        if self._link is None:
            return False
        if self.exit is None:
            self.exit = Exit(reason, message, job)
        self._stop.set()
        self._wake.set()
        return True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="kitshot-worker", daemon=True)
        self._thread.start()

    def run(self) -> None:
        """The loop in this thread, until stopped: a worker process's main thread."""
        self._main()

    def stop(self, timeout: float | None = None) -> None:
        """Ask the loop to stop after the current job; wait up to `timeout` seconds."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def stop_now(self, exit_now: Callable[[int], Any]) -> None:
        """Quit, in a worker process: draw nothing more, and while a job is being drawn call
        `exit_now` (`os._exit`), abandoning it. Nothing is lost: pictures and pack files are
        written whole or not at all, the database rolls an open transaction back, and the job,
        left running, goes back to the queue when a worker starts (`_requeue_stale`) and is drawn
        again with the same sheet and seeds. Between jobs the loop ends by itself, finishing a
        publication under way."""
        with self._lock:
            self._stop.set()
            drawing = self._drawing
        self._wake.set()
        if drawing:
            exit_now(STOPPED_EXIT)

    def _main(self) -> None:
        conn = self._connection()
        try:
            while not self._stop.is_set():
                settings = load_settings(conn, self.defaults_path)
                try:
                    with worker_lock(settings.paths.data_dir):
                        if self._link is None:
                            self.log("Worker started.")
                        else:  # one per process: the log file is enough
                            logger.info("worker process %d started its loop", os.getpid())
                        self._requeue_stale(conn)
                        self._loop()
                    break
                except WorkerBusy:
                    self._set(
                        "waiting",
                        "another Kitshot worker is running (kitshot run?)",
                        reason="other_worker",
                    )
                    if not self._leave_if_paused():
                        self._wait(FM_POLL_SECONDS)
        except Exception as exc:  # noqa: BLE001 - never die silently: show it in the UI
            self._set("error", f"worker crashed: {exc}", reason="crashed", error=str(exc))
            self.log("".join(traceback.format_exception_only(exc)).strip(), "error")
            logger.error("worker crashed", exc_info=exc)
            self._leave("crashed", f"worker crashed: {type(exc).__name__}: {exc}")
        finally:
            if self._worker is not None:
                self._worker.unload()
            self._set("stopped", "the worker has stopped", reason="stopped")
            if self._owns_conn:
                conn.close()
                self._conn = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                delay = self.step()
            except Exception as exc:  # noqa: BLE001 - e.g. the database stayed locked: retry
                message = f"{type(exc).__name__}: {exc}"
                logger.error("worker pass failed", exc_info=exc)
                if is_fatal(exc):  # outside a job, e.g. while publishing frees the card
                    self._fault(message)
                    delay = IDLE_POLL_SECONDS
                else:
                    self._set(
                        "error",
                        f"{message}; trying again shortly",
                        reason="retrying",
                        error=message,
                    )
                    self.log(f"Worker error: {message}", "error")
                    delay = FM_POLL_SECONDS
            if delay > 0:
                self._wait(delay)

    def _fault(self, message: str, job: Job | None = None) -> None:
        """A GPU fault: stop drawing for good, let go of the models, say what to do.

        In a worker process the process ends instead and its parent decides what follows;
        the requests (generate now) are left as they were for the next process.
        """
        leaving = self._leave("fault", message, job)
        with self._lock:
            self._gpu_fault = True
            self._last_error = f"{job.name}: {message}" if job else message
            if not leaving:
                self._paused = True
                self._generate_now = False
        if self._worker is not None:
            self._worker.unload()
        self.log(
            f"  GPU fault: {message}" if leaving else f"  GPU fault, drawing stopped: {message}",
            "error",
        )
        logger.error("GPU fault (%s)", describe_gpu())
        if leaving:
            self._set("gpu-fault", CHILD_FAULT_DETAIL, reason="gpu_fault_child", error=message)
        else:
            self._set("gpu-fault", GPU_FAULT_ADVICE, reason="gpu_fault", error=message)

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
                self._kits
                or load_kits(
                    data_dir=settings.paths.data_dir,
                    install_dir=settings.paths.fm_install_dir,
                ),
                career,
                renderer_factory=self._renderer_factory,
                tagger_factory=self._tagger_factory,
                fm_check=self._fm_check,
                read_settings=lambda: load_settings(conn, self.defaults_path),
                log=self.log,
            )
        else:
            if self._worker.career.id != career.id:
                self.log(f"Now working on career {career.display_name} (its save is the newest).")
            self._worker.career = career
            self._worker.settings = settings
        return self._worker

    def step(self) -> float:
        """One pass of the loop: run at most one job. Returns seconds to wait (0: go on)."""
        conn = self._connection()
        worker = self._worker_for(conn)
        if worker is None:
            self._set("idle", "no career yet: sync a save", reason="no_career")
            self._leave_if_paused()
            return IDLE_POLL_SECONDS
        settings = worker.settings
        with self._lock:
            paused, generate_now = self._paused, self._generate_now
            gpu_fault = self._gpu_fault

        if gpu_fault:
            self._publish_pending(worker)  # needs no GPU: finished pictures still reach FM27
            with self._lock:
                error = self._last_error or ""
            self._set("gpu-fault", GPU_FAULT_ADVICE, reason="gpu_fault", error=error)
            return IDLE_POLL_SECONDS
        if paused:
            self._publish_pending(worker)
            if worker.unload():
                self.log("Paused: the GPU is free.")
            with self._lock:
                error = self._last_error
            if error:
                self._set("paused", error, reason="paused_error", error=error)
            else:
                self._set("paused", "", reason="paused")
            self._leave("paused")
            return IDLE_POLL_SECONDS

        if generate_now:
            f = JobFilter(manual_always=True)
        else:
            f = JobFilter(groups=settings.schedule.auto_groups, manual_always=True)
        ready = worker.runnable(f, limit=PRE_ENCODE_JOBS)
        if not ready:
            with self._lock:
                self._generate_now = False
            self._publish_pending(worker)
            if self._reload_hint:
                self._reload_hint = False
                self.log(f"Queue done. {RELOAD_HINT}")
            now = self._clock()
            if self._idle_since is None:
                self._idle_since = now
            elif now - self._idle_since >= IDLE_UNLOAD_SECONDS and worker.unload():
                self.log("Queue empty: the GPU is free.")
                self._leave("freed")
            self._idle(worker)
            return IDLE_POLL_SECONDS
        self._idle_since = None

        if self._models_ready is not None and not self._models_ready(settings.style.model):
            self._publish_pending(worker)
            # The model chosen before may still be loaded: free the GPU while we wait.
            if worker.unload():
                self.log("The chosen model is not downloaded: the GPU is free.")
                self._leave("freed")
            count = sum(queued_counts(conn, worker.career.id, f).values())
            self._set(
                "setup",
                MODELS_MISSING.format(n=count),
                reason="models_missing",
                count=count,
            )
            return FM_POLL_SECONDS

        if not (generate_now or settings.schedule.generate_while_fm_running):
            running = self._fm_check()
            if running is None or running:
                self._publish_pending(worker)
                if worker.unload():
                    self.log("FM27 is running: the GPU is free.")
                    self._leave("freed")
                why = (
                    "FM27 is running"
                    if running
                    else "cannot tell whether FM27 is running (tasklist.exe failed)"
                )
                count = sum(queued_counts(conn, worker.career.id, f).values())
                self._set(
                    "waiting",
                    f"{why}; {count} job(s) wait. Generate now overrides.",
                    reason="fm_running" if running else "fm_unknown",
                    count=count,
                )
                return FM_POLL_SECONDS

        job = ready[0]
        with self._lock:
            if self._stop.is_set():  # Quit came while the queue was read: draw nothing
                return 0.0
            self._drawing = True
        self._set(
            "running",
            f"{job.name}: {job.kind} {job.group.upper()}",
            job,
            reason="drawing",
            name=job.name,
            kind=job.kind,
            group=job.group.upper(),
        )
        self.log(f"{job.name} ({job.unique_id}), {job.kind} {job.group.upper()}")
        try:
            try:
                outcome = worker.run_job(job, ready[1:])
            finally:
                with self._lock:
                    self._drawing = False
        except JobGone:
            return 0.0  # cancelled or changed meanwhile (a settings change); look again
        except Exception as exc:  # noqa: BLE001 - model load, out of memory: pause and report
            message = f"{type(exc).__name__}: {exc}"
            logger.error("job failed: %s (%s)", job.name, job.unique_id, exc_info=exc)
            if is_fatal(exc):
                self._fault(message, job)
                return IDLE_POLL_SECONDS
            with self._lock:
                self._paused = True
                self._last_error = f"{job.name}: {message}"
            worker.unload()
            self.log(f"  error, worker paused: {message}", "error")
            self._set(
                "error", f"{job.name}: {message}", reason="job_error", name=job.name, error=message
            )
            self._leave("error", self._last_error, job)
            return IDLE_POLL_SECONDS
        with self._lock:
            if outcome.status == "done":
                self._done += 1
            else:
                self._failed += 1
        self.log(
            f"  {outcome.message} ({outcome.seconds:.1f} s)",
            "info" if outcome.status == "done" else "warning",
        )
        self._unpublished = True
        self._publish_pending(worker)
        self._set("running", "next job", reason="next_job")
        return 0.0

    def _idle(self, worker: Worker) -> None:
        """Why nothing is drawn: the queue is empty, or what is held back."""
        held = queued_counts(worker.conn, worker.career.id, JobFilter())
        if not held:
            self._set("idle", "the queue is empty", reason="queue_empty")
            return
        groups = ", ".join(g.upper() for g in sorted(held))
        count = sum(held.values())
        self._set(
            "idle",
            f"{count} job(s) in group {groups} are not drawn automatically"
            " (Settings, Worker); Generate now draws them",
            reason="held",
            count=count,
            groups=groups,
        )

    def _publish_pending(self, worker: Worker) -> None:
        if not self._unpublished or worker.settings.paths.fm_user_dir is None:
            return  # without FM27's folder the pictures wait; the next sync publishes them
        try:
            report = worker.publish()
        except PublishError as exc:
            self.log(f"Publishing failed: {exc}", "error")
            with self._lock:
                self._last_error = f"publishing failed: {exc}"
            return
        self._unpublished = False
        if not report.written:
            return
        if report.active:  # the Reload Skin hint once, when the queue runs out (Phase 17i)
            self._reload_hint = True
            self.log(f"Published {report.written} picture(s).")
        else:
            self.log(
                f"Published {report.written} picture(s); not shown in FM27: another career's"
                " save is newer."
            )
