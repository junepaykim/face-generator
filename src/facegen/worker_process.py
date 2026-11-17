"""The GPU worker in a process of its own (Phase 7b).

`facegen serve` starts `facegen worker` as a child process and supervises it. Only a new
process gets a new CUDA context, so after a GPU fault (gpu.is_fatal) the child exits and a
new one carries on. A process that exits also gives back all of its GPU memory, so Pause,
Quit, FM27 starting and a queue left empty for a while free the card for certain.

The pipe carries one JSON object per line, in ASCII, so no console encoding can break it.

child to parent, on the child's stdout:

    {"type": "status", "status": {...}}       WorkerStatus without the log, after a change
    {"type": "line", "text": "..."}            a line for the dashboard's log, time-stamped
    {"type": "log", "level": 40, "text": "..."}   a "facegen" logger record, with its traceback
    {"type": "exit", "reason": "...", "message": "..." | null, "job": {...} | null}

    "exit" comes last; its reasons are service.EXIT_REASONS. Anything else the child's
    libraries print goes to its stderr, which the parent writes to the log file.

parent to child, on the child's stdin:

    {"cmd": "pause" | "resume" | "generate-now" | "wake" | "stop"}

    "stop", and the end of stdin (the parent went away), end the child after the job it is
    drawing. Pause does the same once the pictures are published.

The queue itself is in SQLite: the parent and the web app change it, the child reads it.

`WorkerProxy` is the parent's side. It has WorkerService's public methods
(service.WorkerControl), so the web app does not know which one it has. It starts a child
unless paused, keeps the dashboard's log and counts across children, and after a failure
follows supervision.FailureHistory: start again after a wait, give up, send a job to review.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

from facegen import db, jobs, procs
from facegen.datasource.records import Personality
from facegen.jobs import Job
from facegen.kits import KitError, KitTable, load_kits
from facegen.pipes import (  # noqa: F401 - also used from here by the tests and cli
    PIPE_POLL_SECONDS,
    decode,
    encode,
    fd_lines,
    pipe_reader,
    protocol_streams,
)
from facegen.renderer import Renderer, Tagger
from facegen.service import (
    BUSY_TIMEOUT_MS,
    LOG_LINES,
    Exit,
    WorkerService,
    WorkerStatus,
    _stamp,
)
from facegen.settings import Settings
from facegen.supervision import DEFAULT_POLICY, FailureHistory, RestartPolicy

STOP_TIMEOUT = 60.0
"""Quit waits this long for the job being drawn, then ends the worker process."""
RESPAWN_SECONDS = 1.0
"""The wait before a new worker process when the last one ended with nothing wrong (it freed
the GPU): a floor, so a child that keeps ending at once cannot spin."""
REAP_SECONDS = 10.0
"""How long a child whose output has closed (it is exiting) may take to end before it is
killed."""
IDLE_SECONDS = 3600.0
"""How long the supervisor sleeps while paused, unless woken."""

CHILD_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "TQDM_DISABLE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}
"""The worker process's stderr goes to the log file: no progress bars there (a model load
drew about 30 lines of them, at each start of a worker process)."""

logger = logging.getLogger("facegen")
child_logger = logging.getLogger("facegen.child")
"""The parent writes the child's log records here, so they reach the same log file."""


# -- the messages --------------------------------------------------------------------------


def job_to_json(job: Job | None) -> dict[str, Any] | None:
    return None if job is None else dataclasses.asdict(job)


def job_from_json(data: dict[str, Any] | None) -> Job | None:
    if data is None:
        return None
    fields = dict(data)
    personality = fields.get("personality")
    fields["personality"] = Personality(**personality) if personality else None
    return Job(**fields)


_STATUS_FIELDS = (
    "state",
    "detail",
    "paused",
    "gpu_fault",
    "generate_now",
    "models_loaded",
    "done",
    "failed",
    "last_error",
)


def status_to_json(status: WorkerStatus) -> dict[str, Any]:
    """The status without its log: log lines travel one by one."""
    data = {name: getattr(status, name) for name in _STATUS_FIELDS}
    data["current"] = job_to_json(status.current)
    return data


def status_from_json(data: dict[str, Any]) -> WorkerStatus:
    return WorkerStatus(
        current=job_from_json(data.get("current")),
        log=(),
        **{name: data[name] for name in _STATUS_FIELDS},
    )


# -- the child's side ----------------------------------------------------------------------


class ChildLink:
    """The child's end of the pipe (service.Link): messages to the parent."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._lock = threading.RLock()
        self._last_status: dict[str, Any] | None = None
        self.broken = False
        """The parent stopped reading; the end of stdin will stop the loop."""

    def send(self, message: dict[str, Any]) -> None:
        text = encode(message)
        with self._lock:
            if self.broken:
                return
            try:
                self._out.write(text)
                self._out.flush()
            except (OSError, ValueError):
                self.broken = True

    def line(self, text: str) -> None:
        self.send({"type": "line", "text": text})

    def status(self, read: Callable[[], WorkerStatus]) -> None:
        # Read and send under one lock, so two threads cannot send in the wrong order.
        with self._lock:
            data = status_to_json(read())
            if data != self._last_status:
                self._last_status = data
                self.send({"type": "status", "status": data})

    def exit(self, ex: Exit) -> None:
        self.send(
            {"type": "exit", "reason": ex.reason, "message": ex.message, "job": job_to_json(ex.job)}
        )


class _ForwardHandler(logging.Handler):
    """Sends the child's "facegen" log records to the parent, tracebacks included."""

    def __init__(self, link: ChildLink) -> None:
        super().__init__(logging.INFO)
        self._link = link
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._link.send({"type": "log", "level": record.levelno, "text": self.format(record)})
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


def _read_commands(commands: Iterable[str], svc: WorkerService) -> None:
    try:
        for raw in commands:
            message = decode(raw)
            match message.get("cmd") if message else None:
                case "pause":
                    svc.pause()
                case "resume":
                    svc.resume()
                case "generate-now":
                    svc.generate_now()
                case "wake":
                    svc.wake()
                case "stop":
                    break
                case _:
                    logger.warning("worker process: not a command: %r", raw[:200])
    except (OSError, ValueError):
        pass  # the pipe broke: the parent is gone
    svc.stop()  # asked to, or the parent went away


def serve_child(
    settings: Settings,
    *,
    defaults_path: Path | None,
    renderer_factory: Callable[[], Renderer],
    tagger_factory: Callable[[], Tagger],
    fm_check: Callable[[], bool | None],
    commands: Iterable[str],
    out: TextIO,
    generate_now: bool = False,
    kits: KitTable | None = None,
    models_ready: Callable[[], bool] | None = None,
) -> Exit:
    """`facegen worker`: run the worker loop, talking to the parent, until it ends. Returns
    why it ended, as told to the parent."""
    link = ChildLink(out)
    handler = _ForwardHandler(link)
    saved = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        if kits is None:
            try:
                kits = load_kits(data_dir=settings.paths.data_dir)
            except KitError as exc:
                ex = Exit("crashed", f"kits: {exc}")
                link.exit(ex)
                return ex
        svc = WorkerService(
            settings.paths.db_path,
            renderer_factory=renderer_factory,
            tagger_factory=tagger_factory,
            defaults_path=defaults_path,
            kits=kits,
            fm_check=fm_check,
            generate_now=generate_now,
            link=link,
            models_ready=models_ready,
        )
        reader = threading.Thread(
            target=_read_commands, args=(commands, svc), name="facegen-commands", daemon=True
        )
        reader.start()
        svc.run()
        ex = svc.exit or Exit("stopped")
        link.exit(ex)
        return ex
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]


# -- the parent's side ---------------------------------------------------------------------


def worker_command(program: Sequence[str] | None = None, config: Path | None = None) -> list[str]:
    """The command line that starts `facegen worker`: this Python, unless `program` says."""
    base = list(program) if program else [sys.executable, "-m", "facegen.cli"]
    return [*base, *(["--config", str(config)] if config else []), "worker"]


def describe_exit(code: int | None) -> str:
    if code is None:
        return "it could not be started"
    if code < 0:
        try:
            return f"ended by {signal.Signals(-code).name}"
        except ValueError:
            return f"ended by signal {-code}"
    if code >= 0x80000000:  # a Windows status, e.g. 0xC0000005 for an access violation
        return f"exit code {code:#010x}"
    return f"exit code {code}"


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


class WorkerProxy:
    """The GPU worker in a child process, with WorkerService's public methods."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        db_path: Path,
        start_paused: bool = False,
        policy: RestartPolicy | None = None,
        stop_timeout: float = STOP_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
        echo_stderr: bool = True,
    ) -> None:
        self._command = list(command)
        self.db_path = db_path
        self._failures = FailureHistory(policy if policy is not None else DEFAULT_POLICY)
        self.stop_timeout = stop_timeout
        self._clock = clock
        self._echo_stderr = echo_stderr
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._child: subprocess.Popen[str] | None = None
        self._child_status: WorkerStatus | None = None
        self._log: deque[str] = deque(maxlen=LOG_LINES)
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._paused = start_paused
        self._state = "paused" if start_paused else "starting"
        self._detail = ""
        self._restart_at: float | None = None
        self._generate_now = False
        self._generate_now_seen = False
        """A child has reported generate now on since it was asked for: its going off then
        means the queue was drawn."""
        self._done = 0
        self._failed = 0
        self._last_error: str | None = None
        self.starts = 0
        """Worker processes started so far."""

    # -- control, from the web threads --

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._restart_at = None
            if self._child is None and self._state != "gpu-fault":
                self._state, self._detail = "paused", ""
        if not self._send("pause"):
            self.log("Paused.")
        self._wake.set()

    def resume(self) -> bool:
        """Go on drawing: the child goes on, or a new one starts."""
        with self._lock:
            self._paused = False
            self._last_error = None
            self._restart_at = None
            self._starting_again()
        if not self._send("resume"):
            self.log("Resumed.")
        self._wake.set()
        return True

    def generate_now(self) -> bool:
        """Draw the whole current queue, every group, even while FM27 runs (spec section 6)."""
        with self._lock:
            self._generate_now = True
            self._generate_now_seen = False
            self._paused = False
            self._restart_at = None
            self._starting_again()
        if not self._send("generate-now"):
            self.log("Generate now: drawing the whole queue, every group, even if FM27 is running.")
        self._wake.set()
        return True

    def wake(self) -> None:
        """Look at the queue now rather than at the next poll."""
        self._send("wake")

    def _starting_again(self) -> None:
        """Caller holds the lock: the user asked to draw. After the restarts were given up,
        the count of failures starts afresh."""
        if self._state == "gpu-fault":
            self._failures.clear()
        if self._child is None:
            self._state, self._detail = "starting", "starting the worker process"

    def status(self) -> WorkerStatus:
        with self._lock:
            child = self._child_status if self._child is not None else None
            if child is not None:
                state, detail, current = child.state, child.detail, child.current
                models, done, failed = child.models_loaded, child.done, child.failed
                last_error = self._last_error or child.last_error
            else:
                state, detail, current = self._state, self._detail_now(), None
                models, done, failed, last_error = False, 0, 0, self._last_error
            return WorkerStatus(
                state=state,
                detail=detail,
                current=current,
                paused=self._paused,
                gpu_fault=False,  # a new process can always be started
                generate_now=self._generate_now,
                models_loaded=models,
                done=self._done + done,
                failed=self._failed + failed,
                last_error=last_error,
                log=tuple(self._log),
            )

    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(f"{_stamp()}  {line}")
        logger.info("%s", line)

    @property
    def child_pid(self) -> int | None:
        with self._lock:
            return self._child.pid if self._child is not None else None

    def _detail_now(self) -> str:
        if self._state == "restarting" and self._restart_at is not None:
            left = max(0.0, self._restart_at - self._clock())
            return f"{self._detail}; a new worker process starts in {left:.0f} s"
        return self._detail

    def _send(self, cmd: str) -> bool:
        """Send a command to the child; False if there is none to send it to."""
        with self._lock:
            child = self._child
        if child is None or child.stdin is None:
            return False
        with self._send_lock:
            try:
                child.stdin.write(encode({"cmd": cmd}))
                child.stdin.flush()
            except (OSError, ValueError):
                return False
        return True

    # -- the supervisor thread --

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="facegen-supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop the child after its current job, or end it after `stop_timeout` seconds;
        then wait up to `timeout` seconds for the supervisor."""
        with self._lock:
            self._stop.set()
            child = self._child
        self._wake.set()
        if child is not None:
            self._send("stop")
            try:
                child.wait(timeout=self.stop_timeout)
            except subprocess.TimeoutExpired:
                self.log(
                    f"The worker process did not stop within {self.stop_timeout:g} s; ended it."
                )
                child.kill()
        if self._thread is not None:
            self._thread.join(timeout)

    def _main(self) -> None:
        try:
            while not self._stop.is_set():
                wait = self._until_start()
                if wait > 0:
                    self._wake.wait(wait)
                    self._wake.clear()
                    continue
                self._run_child()
        except Exception as exc:  # never die silently: show it in the UI
            logger.error("worker supervisor crashed", exc_info=exc)
            with self._lock:
                self._paused = True
                self._last_error = f"the worker supervisor crashed: {type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._state, self._detail = "stopped", "the worker has stopped"

    def _until_start(self) -> float:
        """Seconds until a child should start; 0: now."""
        with self._lock:
            if self._paused:
                return IDLE_SECONDS
            if self._restart_at is not None:
                return max(0.0, self._restart_at - self._clock())
            return 0.0

    def _spawn(self, generate_now: bool) -> subprocess.Popen[str]:
        command = [*self._command, *(["--generate-now"] if generate_now else [])]
        if os.name == "nt":  # Ctrl+C in the console is the parent's; it sends "stop"
            extra: dict[str, Any] = {
                "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            }
        else:
            extra = {"start_new_session": True}
        # No console window of its own on Windows (procs.py), even when the parent has none.
        return procs.popen_quiet(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, **CHILD_ENV},
            **extra,
        )

    def _run_child(self) -> None:
        with self._lock:
            if self._stop.is_set() or self._paused:
                return
            try:
                child = self._spawn(self._generate_now)
            except OSError as exc:
                failed: OSError | None = exc
            else:
                failed = None
                self._child, self._child_status = child, None
                self._restart_at = None
                self._generate_now_seen = False
                self._state, self._detail = "starting", "starting the worker process"
                self.starts += 1
        if failed is not None:
            logger.error("could not start the worker process", exc_info=failed)
            text = f"could not start the worker process: {failed}"
            self._failed_child(text, None, text)
            return
        logger.info("worker process %d started: %s", child.pid, " ".join(child.args))
        reader = threading.Thread(
            target=self._drain_stderr, args=(child,), name="facegen-child-stderr", daemon=True
        )
        reader.start()
        ended: dict[str, Any] | None = None
        assert child.stdout is not None
        for raw in child.stdout:
            message = decode(raw)
            if message is None:
                logger.warning("worker process: not a message: %r", raw[:200])
                continue
            try:
                if message.get("type") == "exit":
                    ended = message
                else:
                    self._receive(message)
            except Exception as exc:  # noqa: BLE001 - a bad message must not end the supervisor
                logger.warning("worker process: bad message %r (%s)", raw[:200], exc)
        code = self._reap(child)
        reader.join(5)
        for stream in (child.stdin, child.stdout, child.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        with self._lock:
            last = self._child_status
            if last is not None:
                self._done += last.done
                self._failed += last.failed
            self._child, self._child_status = None, None
        logger.info("worker process %d ended (%s)", child.pid, describe_exit(code))
        self._after(ended, code, last.current if last else None)

    def _receive(self, message: dict[str, Any]) -> None:
        match message.get("type"):
            case "status":
                status = status_from_json(message["status"])
                with self._lock:
                    self._child_status = status
                    if status.generate_now:
                        self._generate_now_seen = True
                    elif self._generate_now_seen:  # the queue was drawn
                        self._generate_now = self._generate_now_seen = False
            case "line":
                with self._lock:
                    self._log.append(str(message["text"]))
            case "log":
                child_logger.log(int(message["level"]), "%s", message["text"])
            case _:
                logger.warning("worker process: unknown message %r", message)

    def _drain_stderr(self, child: subprocess.Popen[str]) -> None:
        assert child.stderr is not None
        for raw in child.stderr:
            line = raw.rstrip("\r\n")
            self._stderr_tail.append(line)
            # Library notices, mostly; a crash's last words are logged again as an error.
            child_logger.info("stderr: %s", line)
            if self._echo_stderr and sys.stderr is not None:
                print(line, file=sys.stderr, flush=True)

    def _reap(self, child: subprocess.Popen[str]) -> int:
        """The exit code; a child that keeps running after closing its output is ended."""
        try:
            return child.wait(timeout=REAP_SECONDS)
        except subprocess.TimeoutExpired:
            child.kill()
            return child.wait()

    # -- after a child ended --

    def _after(self, ended: dict[str, Any] | None, code: int | None, current: Job | None) -> None:
        if self._stop.is_set():
            return
        reason = ended.get("reason") if ended else None
        message = ended.get("message") if ended else None
        job = current
        if ended and ended.get("job"):
            try:
                job = job_from_json(ended["job"])
            except (TypeError, ValueError) as exc:  # a child of another version
                logger.warning("worker process: unreadable job in %r (%s)", ended, exc)
        match reason:
            case "fault":
                self._failed_child(
                    message or "GPU fault", job, "the graphics driver reported an error"
                )
            case "error" | "crashed":
                text = message or reason
                with self._lock:
                    self._paused = True
                    self._last_error = text
                    self._state, self._detail = "error", text
            case "paused":
                with self._lock:
                    if self._paused:
                        self._state, self._detail = "paused", ""
                    else:  # resumed while the child was ending: start a new one
                        self._respawn_soon()
            case "freed" | "stopped":
                with self._lock:
                    self._respawn_soon()
            case _:  # ended without saying why: a crash outside Python, or killed
                text = f"the worker process ended unexpectedly ({describe_exit(code)})"
                self.log(text[0].upper() + text[1:] + ".")
                if self._stderr_tail:
                    logger.error("worker process's last output:\n%s", "\n".join(self._stderr_tail))
                self._failed_child(text, job, "the worker process ended unexpectedly")

    def _respawn_soon(self) -> None:
        """Caller holds the lock."""
        self._state, self._detail = "starting", "starting the worker process"
        self._restart_at = self._clock() + RESPAWN_SECONDS

    def _failed_child(self, message: str, job: Job | None, what: str) -> None:
        """A GPU fault or a crash: wait and start again, send the job to review, or give up.

        `message` is the error, `what` the plain words for the dashboard."""
        decision = self._failures.record(self._clock(), job.id if job else None)
        logger.error(
            "worker process failed (%d within %.0f min): %s%s",
            decision.recent,
            self._failures.policy.window / 60,
            message,
            f" while drawing {job.name} ({job.unique_id}, job {job.id})" if job else "",
        )
        reviewed = decision.review and job is not None and self._send_to_review(job, message)
        if decision.restart_in is None:
            minutes = self._failures.policy.window / 60
            advice = (
                f"Drawing has stopped: the graphics card or its driver failed {decision.recent}"
                f" times within {minutes:.0f} minutes. The queue is kept. Resume tries again; if"
                " it keeps happening, close other programs that use the graphics card, or"
                " restart the PC."
            )
            with self._lock:
                self._paused = True
                self._restart_at = None
                self._last_error = message
                self._state, self._detail = "gpu-fault", advice
            self.log(f"Drawing stopped after {decision.recent} failures; Resume tries again.")
            return
        delay = decision.restart_in
        with self._lock:
            self._restart_at = self._clock() + delay
            self._state = "restarting"
            if job is None:
                self._detail = what
            elif reviewed:
                self._detail = f"{what}; {job.name} went to review"
            else:
                self._detail = f"{what}; {job.name} is back in the queue"
        self.log(f"Starting a new worker process in {delay:.0f} s.")

    def _send_to_review(self, job: Job, message: str) -> bool:
        n = self._failures.policy.review_after
        error = f"the graphics driver failed {_times(n)} while drawing this picture: {message}"
        try:
            conn = db.connect(self.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
            try:
                sent = jobs.send_to_review(conn, job.id, error[:500])
            finally:
                conn.close()
        except Exception as exc:  # say so; the job is drawn again instead
            logger.error("could not send job %d to review", job.id, exc_info=exc)
            self.log(f"Could not send {job.name} to review: {type(exc).__name__}: {exc}")
            return False
        if sent:
            self.log(
                f"{job.name} goes to review: the graphics driver failed {_times(n)}"
                " while drawing it."
            )
        return sent
