"""Run each sync in a child process, so reading a save leaves the app's process free.

A sync reads the save with fmsave for 45 to 60 s of pure-Python work; inside the app's process
it made the pages 3 to 6 times slower and the tray late, and kept the parsed save's memory
(docs/research.md section 13). `facegen sync-process` runs pipeline.sync_operation.execute_sync,
the operation the in-thread sync runs, with its own settings, connection and source; the
parsed save goes when the child ends. The pipe carries one ASCII JSON object per line, as for
the worker (platform.pipes); library output goes to stderr, which the parent logs.

Parent to child stdin:

    {"cmd": "sync", "request": {...}}   first: runtime.contracts.SyncRequest
    {"cmd": "stop"}                     facegen quits: end now, unless the sync has committed

Child stdout:

    {"type": "line", "text": "..."}     a line for the dashboard's log (the parent stamps it)
    {"type": "committed"}               the sync's transaction committed; publication follows
    {"type": "result", "result": {...}} runtime.contracts.SyncResult, last

Stopping. Everything before the commit can be abandoned: the copy into data/snapshots/ goes
through a `.part` file, parsing only reads, and SQLite rolls an unfinished transaction back,
so the database stays as it was and the save watcher reads that save again at the next start.
After the commit only publication remains (each file is written whole, then renamed), and it
is let finish. So a stop command, or the end of stdin (the parent is gone), ends the child at
once while reading/planning. The child protects its final commit from a simultaneous stop.
The parent kills a child that has not ended within `STOP_SECONDS`, extending the deadline
when it receives the commit notification. SQLite retains pending publication across a kill.

`SyncProcess` is the parent's side; runtime.sync_service.SyncService claims and reports.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TextIO

from facegen.platform import procs
from facegen.platform.pipes import JsonWriter, decode, drain_lines, encode
from facegen.runtime.contracts import FAILED, SYNCED, UNREADABLE, SyncRequest, SyncResult

if TYPE_CHECKING:
    from facegen.datasource.base import DataSource

STOP_SECONDS = 5.0
"""Quit waits this long for a sync that has not committed (it ends at once), then kills it."""
PUBLISH_SECONDS = 30.0
"""Quit waits this long for a committed sync to finish publishing, then kills it."""
REAP_SECONDS = 10.0
"""How long a child whose output has closed may take to end before it is killed."""
STOPPED_EXIT = 4
"""The child's exit code when a stop ended it before the commit."""
STDERR_LINES = 20
"""A crashed child's last stderr lines, logged again as an error."""

CHILD_ENV = {"PYTHONIOENCODING": "utf-8"}

STOPPED_ERROR = (
    "the sync was stopped because facegen quit; the save is read again at the next start"
)
STOPPED_PUBLISHING_ERROR = (
    "facegen quit while the sync was publishing: the sync is stored, and FM27's pack is"
    " published again at the next start"
)

OUTCOMES = (SYNCED, UNREADABLE, FAILED)

logger = logging.getLogger("facegen")
child_logger = logging.getLogger("facegen.child")
"""The child's stderr goes here, as the worker process's does, so it reaches the log file."""


def sync_command(program: Sequence[str] | None = None) -> list[str]:
    """The command line that starts `facegen sync-process`: this Python, unless `program`
    says. The request names the database and the defaults file, so no `--config`."""
    base = list(program) if program else [sys.executable, "-m", "facegen.cli"]
    return [*base, "sync-process"]


# -- the messages ------------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def request_to_json(request: SyncRequest) -> dict[str, Any]:
    return dataclasses.asdict(request)


def request_from_json(data: dict[str, Any]) -> SyncRequest:
    return SyncRequest(
        db_path=str(data["db_path"]),
        defaults_path=_text(data.get("defaults_path")),
        save_path=_text(data.get("save_path")),
        trigger=str(data.get("trigger") or "sync now"),
    )


def result_to_json(result: SyncResult) -> dict[str, Any]:
    data = dataclasses.asdict(result)
    data["warnings"] = list(result.warnings)
    return data


def result_from_json(data: dict[str, Any]) -> SyncResult:
    outcome = data.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError(f"not a sync outcome: {outcome!r}")
    return SyncResult(
        outcome=outcome,
        message=_text(data.get("message")),
        error=_text(data.get("error")),
        unsupported=bool(data.get("unsupported")),
        warnings=tuple(str(w) for w in data.get("warnings") or ()),
        notice=_text(data.get("notice")),
    )


# -- the child ---------------------------------------------------------------------------------


class _Link:
    """The child's messages to the parent; a parent that stopped reading is not an error."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._writer = JsonWriter()
        self.broken = False

    def send(self, message: dict[str, Any]) -> None:
        text = encode(message)
        with self._writer.lock:
            if self.broken:
                return
            try:
                self._writer.send_encoded(self._out, text)
            except (OSError, ValueError):
                self.broken = True

    def line(self, text: str) -> None:
        self.send({"type": "line", "text": text})


class _Progress:
    """How far the child's sync is: before its commit, a stop may end it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage = "reading"

    def enter(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    def stop_if_reading(self, exit_now: Callable[[int], Any]) -> None:
        with self._lock:
            if self._stage == "reading":
                exit_now(STOPPED_EXIT)


def _watch_parent(
    commands: Iterator[str], progress: _Progress, exit_now: Callable[[int], Any]
) -> None:
    """Wait for a stop or the end of stdin, then end the process if nothing is committed."""
    try:
        for raw in commands:
            message = decode(raw)
            if message is not None and message.get("cmd") == "stop":
                break
    except (OSError, ValueError):
        pass  # the pipe broke: the parent is gone
    progress.stop_if_reading(exit_now)


def serve_child(
    commands: Iterable[str],
    out: TextIO,
    source_factory: Callable[[], DataSource],
    *,
    exit_now: Callable[[int], Any] = os._exit,
) -> SyncResult | None:
    """`facegen sync-process`: read the request, run the sync, send its lines and result.
    None when the first line is not a request. A stop before the commit calls `exit_now`
    (`os._exit`: no clean-up, as if killed; the database rolls the transaction back)."""
    from facegen.pipeline.sync_operation import execute_sync

    lines = iter(commands)
    first = next(lines, None)
    message = decode(first) if first is not None else None
    try:
        if message is None or message.get("cmd") != "sync":
            raise ValueError("the first line is not a sync request")
        request = request_from_json(message["request"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"sync process: {exc}: {(first or '')[:200]!r}", file=sys.stderr, flush=True)
        return None
    link = _Link(out)
    progress = _Progress()
    threading.Thread(
        target=_watch_parent,
        args=(lines, progress, exit_now),
        name="facegen-sync-commands",
        daemon=True,
    ).start()

    def committed() -> None:
        progress.enter("publishing")
        link.send({"type": "committed"})

    def finished(result: SyncResult) -> None:
        progress.enter("done")
        link.send({"type": "result", "result": result_to_json(result)})

    return execute_sync(
        request,
        source_factory,
        log=link.line,
        before_commit=lambda: progress.enter("committing"),
        on_committed=committed,
        on_result=finished,
    )


# -- the parent --------------------------------------------------------------------------------


class SyncProcess:
    """Runs one sync at a time in a child process, for SyncService; `stop` for Quit."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        log: Callable[[str], None] = lambda line: None,
        echo_stderr: bool = True,
        stop_seconds: float = STOP_SECONDS,
        publish_seconds: float = PUBLISH_SECONDS,
    ) -> None:
        self.command = list(command)
        self._log = log
        self._echo_stderr = echo_stderr
        self.stop_seconds = stop_seconds
        self.publish_seconds = publish_seconds
        self._lock = threading.Lock()
        self._writer = JsonWriter()
        self._child: subprocess.Popen[str] | None = None
        self._committed_at: float | None = None
        self._stopped = False
        self.starts = 0
        """Sync processes started so far."""

    @property
    def child_pid(self) -> int | None:
        with self._lock:
            return self._child.pid if self._child is not None else None

    def run(self, request: SyncRequest) -> SyncResult:
        """Run `request` in a new child process and say how it ended. Never raises: a child
        that cannot start, crashes or is stopped is a failed sync."""
        with self._lock:
            if self._stopped:
                return SyncResult(FAILED, error=STOPPED_ERROR)
            try:
                child = procs.popen_json(
                    self.command,
                    detached=True,  # Ctrl+C belongs to the parent, which stops the child
                    env={**os.environ, **CHILD_ENV},
                )
            except OSError as exc:
                failed: OSError | None = exc
            else:
                failed = None
                self._child, self._committed_at = child, None
                self.starts += 1
        if failed is not None:
            logger.error("could not start the sync process", exc_info=failed)
            text = f"could not start the sync process: {failed}"
            self._log(f"Sync failed: {text}.")
            return SyncResult(FAILED, error=text)
        logger.info("sync process %d started: %s", child.pid, " ".join(child.args))
        tail: deque[str] = deque(maxlen=STDERR_LINES)
        reader = threading.Thread(
            target=self._drain_stderr, args=(child, tail), name="facegen-sync-stderr", daemon=True
        )
        reader.start()
        result: SyncResult | None = None
        try:
            self._send(child, {"cmd": "sync", "request": request_to_json(request)})
            assert child.stdout is not None
            for raw in child.stdout:
                message = decode(raw)
                if message is None:
                    logger.warning("sync process: not a message: %r", raw[:200])
                    continue
                try:
                    match message.get("type"):
                        case "line":
                            self._log(str(message["text"]))
                        case "committed":
                            with self._lock:
                                self._committed_at = time.monotonic()
                        case "result":
                            result = result_from_json(message["result"])
                        case _:
                            logger.warning("sync process: unknown message %r", message)
                except Exception as exc:  # noqa: BLE001 - a bad message must not end the sync
                    logger.warning("sync process: bad message %r (%s)", raw[:200], exc)
        finally:
            code = procs.wait_or_kill(child, timeout=REAP_SECONDS)
            reader.join(5)
            with self._writer.lock:
                procs.close_streams(child.stdin, child.stdout, child.stderr)
            with self._lock:
                self._child = None
                stopped, committed = self._stopped, self._committed_at is not None
        logger.info("sync process %d ended (%s)", child.pid, procs.describe_exit(code))
        if result is not None:
            return result
        if stopped:
            error = STOPPED_PUBLISHING_ERROR if committed else STOPPED_ERROR
            self._log(f"Sync stopped: {error}.")
            return SyncResult(FAILED, error=error)
        text = f"the sync process ended unexpectedly ({procs.describe_exit(code)})"
        if committed:
            text += "; the sync is stored and unfinished publication will be retried"
        self._log(f"Sync failed: {text}.")
        if tail:
            logger.error("sync process's last output:\n%s", "\n".join(tail))
        return SyncResult(FAILED, error=text)

    def stop(self) -> None:
        """Start no other sync, and end the running one: it ends at once before its commit
        and publishes first after it. A child still running after `stop_seconds` (or
        `publish_seconds` once committed) is killed."""
        with self._lock:
            self._stopped = True
            child = self._child
        if child is None:
            return
        started = time.monotonic()
        self._send(child, {"cmd": "stop"})
        with self._writer.lock:
            procs.close_streams(child.stdin)
        with self._lock:
            committed_at = self._committed_at
        grace = self.stop_seconds
        if committed_at is None:
            try:
                child.wait(timeout=self.stop_seconds)
                return
            except subprocess.TimeoutExpired:
                # The child can commit after Quit starts waiting.
                with self._lock:
                    committed_at = self._committed_at
        remaining = 0.0
        if committed_at is not None:
            grace = self.publish_seconds
            remaining = max(0.0, max(started, committed_at) + grace - time.monotonic())
            logger.info("the sync is stored; waiting up to %.0f s for it to publish", grace)

        def killing() -> None:
            logger.warning("the sync process did not stop within %.0f s; ended it", grace)

        procs.wait_or_kill(child, timeout=remaining, on_timeout=killing)

    def _send(self, child: subprocess.Popen[str], message: dict[str, Any]) -> bool:
        if child.stdin is None:
            return False
        try:
            self._writer.send(child.stdin, message)
        except (OSError, ValueError):  # the child has gone, or its stdin is closed
            return False
        return True

    def _drain_stderr(self, child: subprocess.Popen[str], tail: deque[str]) -> None:
        assert child.stderr is not None

        def receive(line: str) -> None:
            tail.append(line)
            child_logger.info("sync stderr: %s", line)
            if self._echo_stderr and sys.stderr is not None:
                print(line, file=sys.stderr, flush=True)

        drain_lines(child.stderr, receive)
