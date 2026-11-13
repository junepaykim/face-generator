"""The background work behind `facegen serve`: the worker loop and "sync now".

**Worker.** One thread owns the GPU. Each pass re-reads the settings and the queue, so a
threshold change or a manual request applied in the browser is picked up before the next
job (spec section 7). It draws manual requests first, then the automatic jobs of the groups
in `schedule.auto_groups`, in queue order, and publishes after each picture.

- While FM27 runs it waits and frees the GPU, unless "generate now" is set or
  `schedule.generate_while_fm_running` is true (spec section 6).
- "Generate now" draws the whole queue as it stands, every group, FM27 running or not, and
  lasts until nothing is left to draw.
- When the queue has been empty for a few minutes, or the worker is paused, the model is
  unloaded so the card is free for the game.
- An error (model failed to load, out of memory) pauses the worker and shows the error;
  the job goes back to the queue. Resume tries again.
- At start it publishes the active career's pictures once, so a pack left behind (pictures
  drawn by `facegen run --no-publish`, or older square pictures now cropped to 3:4) is put
  right without waiting for the next picture.
- `step` runs one pass and says how long to wait, so tests drive the loop without a thread.

**Syncs.** `SyncService` runs one sync at a time: "Sync now" from the dashboard, or the
save watcher. After a sync it shows FM27 the active career's pack (publish.publish_active)
and wakes the worker.

**Save watcher.** `PollerService` looks at FM27's games folder every `schedule.poll_seconds`
and syncs the newest save once it has changed and stopped changing (poller.SaveWatcher).
Every save or autosave is therefore a sync (spec section 6). It runs whether or not FM27
does; only the GPU waits for the game.
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
from facegen.poller import (
    UNSUPPORTED_ADVICE,
    NoSaveError,
    SaveUnreadableError,
    SaveVersion,
    SaveWatcher,
    find_latest_save,
    sync_save,
)
from facegen.publish import PublishError, PublishReport, publish_active
from facegen.settings import load_settings
from facegen.sync import SyncReport, sync_of_save
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
        self._last_error: str | None = None
        self._idle_since: float | None = None
        self._unpublished = True
        """Publish at the next chance: after a picture, and once at start."""

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
                self.log(f"Now working on career {career.display_name} (its save is the newest).")
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
        ready = worker.runnable(f)
        if not ready:
            with self._lock:
                self._generate_now = False
            self._publish_pending(worker)
            now = self._clock()
            if self._idle_since is None:
                self._idle_since = now
            elif now - self._idle_since >= IDLE_UNLOAD_SECONDS and worker.unload():
                self.log("Queue empty: the GPU is free.")
            self._set("idle", self._idle_detail(worker))
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
            outcome = worker.run_job(job, ready[1:])
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
    def _idle_detail(worker: Worker) -> str:
        """Why nothing is drawn: the queue is empty, or what is held back."""
        held = worker.runnable(JobFilter())  # every group; none of them drawn now
        if not held:
            return "the queue is empty"
        groups = ", ".join(sorted({j.group.upper() for j in held}))
        return (
            f"{len(held)} job(s) in group {groups} are not drawn automatically"
            " (Settings, Worker); Generate now draws them"
        )

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
            where = (
                "FM27: Settings -> Reload Skin to see them"
                if report.active
                else "not shown in FM27: another career's save is newer"
            )
            self.log(f"Published {report.written} picture(s); {where}.")


@dataclass
class SyncStatus:
    running: bool = False
    started: str | None = None
    trigger: str | None = None
    message: str | None = None
    error: str | None = None
    unsupported: bool = False
    """The last sync failed because the save reader cannot read the save's build."""
    warnings: list[str] = field(default_factory=list)


SYNCED, UNREADABLE, FAILED = "synced", "unreadable", "failed"
"""How a sync ended. UNREADABLE is recorded in the database and not retried until the save
changes; FAILED (no save, a save still being written, an error) is worth trying again."""


def describe_sync(report: SyncReport) -> str:
    queued = sum(report.enqueued.values())
    text = (
        f"Synced {report.save_file} (in-game {report.game_date or 'unknown'}): "
        f"{queued} queued, {report.updated} updated, {report.cancelled} cancelled."
    )
    if report.new_at_my_club:
        text += f" {report.new_at_my_club} new at my club, drawn first."
    return text


def describe_activation(report: PublishReport | None) -> str | None:
    """A line for the log when a sync changed what FM27 is shown, else None."""
    if report is None or not (report.deactivated or report.config_changed or report.written):
        return None
    parts = [f"FM27 pack {report.folder.name}: {report.mapped} picture(s) mapped"]
    if report.deactivated:
        parts.append(f"other careers' packs unmapped ({', '.join(report.deactivated)})")
    return "; ".join(parts) + ". FM27 loads it at its next start or on Reload Skin."


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
            self._status = SyncStatus(running=True, started=_stamp(), trigger=trigger)
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

    def run(self, save: Path | None = None, trigger: str = "sync now") -> str | None:
        """Sync `save` (default: the newest) in this thread. Returns SYNCED, UNREADABLE or
        FAILED, or None if another sync is running."""
        if not self._claim(trigger):
            return None
        return self._run(save, trigger)

    def _run(self, save: Path | None, trigger: str) -> str:
        message = error = None
        unsupported = False
        outcome = FAILED
        warnings: list[str] = []
        conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        try:
            settings = load_settings(conn, self.defaults_path)
            report = sync_save(
                conn, settings, self._source_factory(), save, preview=0, trigger=trigger
            )
            outcome = SYNCED
            message = describe_sync(report)
            warnings = list(report.warnings)
            self._log(message)
            try:
                if line := describe_activation(publish_active(conn, settings)):
                    self._log(line)
            except PublishError as exc:
                warnings.append(f"publishing the active career's pack failed: {exc}")
                self._log(f"Publishing failed: {exc}")
        except SaveUnreadableError as exc:
            error = str(exc)
            outcome = FAILED if exc.busy else UNREADABLE
            unsupported = exc.unsupported
            self._log(error + (f" {UNSUPPORTED_ADVICE}" if unsupported else ""))
        except NoSaveError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - a background sync reports, never raises
            error = f"sync failed: {type(exc).__name__}: {exc}"
            self._log(error)
        finally:
            conn.close()
            with self._lock:
                self._status = SyncStatus(
                    running=False,
                    started=self._status.started,
                    trigger=trigger,
                    message=message,
                    error=error,
                    unsupported=unsupported,
                    warnings=warnings,
                )
        self._on_done()
        return outcome


@dataclass(frozen=True, slots=True)
class PollerStatus:
    state: str
    """starting, no save, synced, settling, due, syncing, error or stopped."""
    detail: str
    checked: str | None
    """When the folder was last looked at (local time)."""


class PollerService:
    """Watches FM27's saves folder and syncs the newest save once it stops changing."""

    def __init__(
        self,
        db_path: Path,
        syncer: SyncService,
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
            self._status = PollerStatus(state, detail, _stamp())

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        return self._conn

    def step(self) -> float:
        """Look at the folder once and sync if a save is due. Returns seconds to wait."""
        conn = self._connection()
        settings = load_settings(conn, self.defaults_path)
        interval = float(settings.schedule.poll_seconds)
        try:
            newest_path = find_latest_save(settings.paths.games_dir)
            newest = SaveVersion.of(newest_path) if newest_path is not None else None
        except OSError as exc:  # the save vanished between listing and stat, or /mnt/c is gone
            self._set("error", f"cannot look at {settings.paths.games_dir}: {exc}")
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

    def _after(self, outcome: str) -> str:
        status = self._syncer.status()
        if outcome == SYNCED:
            return status.message or "synced"
        return f"{status.error}; waiting for the next save"

    # -- the thread --

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
