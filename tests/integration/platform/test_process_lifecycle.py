"""Closing a protocol pipe and exiting a child process are different lifecycle events."""

from __future__ import annotations

import io
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from facegen.desktop.app_window import WindowProcess
from facegen.platform import pipes, procs
from facegen.runtime import worker_process
from facegen.runtime.worker_process import WorkerProxy
from facegen.setup.models import ModelDownload


class ChildAfterEOF:
    """Empty output pipes, but wait blocks until exit or a requested timeout."""

    def __init__(self, *, exit_on_kill: bool = True) -> None:
        self.args = ["fake-child"]
        self.pid = 123
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.returncode: int | None = None
        self.waiting = threading.Event()
        self.exited = threading.Event()
        self.killed = threading.Event()
        self.timeouts: list[float | None] = []
        self.exit_on_kill = exit_on_kill

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.timeouts.append(timeout)
        self.waiting.set()
        # An incorrect controller must fail the test instead of leaving a blocked thread.
        if not self.exited.wait(timeout if timeout is not None else 5):
            raise subprocess.TimeoutExpired(self.args, timeout)
        assert self.returncode is not None
        return self.returncode

    def finish(self, code: int = 0) -> None:
        self.returncode = code
        self.exited.set()

    def kill(self) -> None:
        self.killed.set()
        if self.exit_on_kill:
            self.finish(-9)

    @property
    def process(self) -> subprocess.Popen[str]:
        return cast(subprocess.Popen[str], self)


class Background:
    def __init__(self, action: Callable[[], None]) -> None:
        self.errors: list[BaseException] = []
        self.done = threading.Event()

        def run() -> None:
            try:
                action()
            except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
                self.errors.append(exc)
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def join(self) -> None:
        self.thread.join(3)
        assert not self.thread.is_alive()
        if self.errors:
            raise self.errors[0]


def test_worker_reaps_a_child_still_alive_after_stdout_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = ChildAfterEOF()
    worker = WorkerProxy(child.args, db_path=tmp_path / "unused.db", echo_stderr=False)
    monkeypatch.setattr(worker_process, "REAP_SECONDS", 0.05)
    monkeypatch.setattr(worker, "_spawn", lambda generate_now: child.process)
    ended: list[int | None] = []
    monkeypatch.setattr(worker, "_after", lambda message, code, current: ended.append(code))
    reader = Background(worker._run_child)
    try:
        assert child.waiting.wait(3)
        reader.join()
        assert child.timeouts[0] == 0.05
        assert child.killed.is_set() and child.poll() == -9
        assert worker.child_pid is None and ended == [-9]
        assert child.stdin.closed and child.stdout.closed and child.stderr.closed
        assert not (tmp_path / "unused.db").exists()
    finally:
        child.finish(-9)
        reader.join()


def _fixed_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The worker proxy's monotonic clock, set by the test: [now]."""
    now = [100.0]
    monkeypatch.setattr(worker_process, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def test_worker_stop_allows_the_child_to_finish_before_its_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixed_clock(monkeypatch)
    child = ChildAfterEOF()
    worker = WorkerProxy(
        child.args, db_path=tmp_path / "unused.db", stop_timeout=5, echo_stderr=False
    )
    worker._child = child.process
    stopping = Background(worker.stop)
    try:
        assert child.waiting.wait(3)
        assert child.timeouts == [5]  # Quit began just now: the whole budget is left
        assert pipes.decode(child.stdin.getvalue()) == {"cmd": "stop"}
        assert not stopping.done.is_set() and not child.killed.is_set()
        child.finish()
        stopping.join()
        assert not child.killed.is_set() and child.poll() == 0
    finally:
        child.finish()
        stopping.join()


def test_worker_stop_waits_only_for_what_is_left_of_the_quit_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop budget counts from the Quit request, not from the wait that follows it."""
    now = _fixed_clock(monkeypatch)
    child = ChildAfterEOF()
    worker = WorkerProxy(
        child.args, db_path=tmp_path / "unused.db", stop_timeout=5, echo_stderr=False
    )
    worker._child = child.process
    worker.request_stop()
    now[0] += 3  # other services stopped first
    stopping = Background(worker.stop)
    try:
        assert child.waiting.wait(3)
        assert child.timeouts == [2]
        assert pipes.decode(child.stdin.getvalue()) == {"cmd": "stop"}  # sent once
        child.finish()
        stopping.join()
        assert not child.killed.is_set() and child.poll() == 0
    finally:
        child.finish()
        stopping.join()


@pytest.mark.parametrize("close_window", [False, True])
def test_window_stdout_eof_waits_for_exit_or_explicit_close(close_window: bool) -> None:
    child = ChildAfterEOF()
    window = WindowProcess("http://127.0.0.1:8765/")
    window._child = child.process
    window._started_at = window._clock()
    answered = threading.Event()
    window._answers[1] = (answered, [None])
    reader = Background(lambda: window._read(child.process))
    try:
        assert child.waiting.wait(3)
        assert child.timeouts == [None]
        assert not reader.done.is_set() and not child.killed.is_set()
        assert not answered.is_set() and window.available
        if close_window:
            window.destroy(timeout=0.05)
            assert child.killed.is_set() and child.stdin.closed
            assert 0.05 in child.timeouts
        else:
            child.finish()
        reader.join()
        assert answered.is_set() and window.available
        assert child.killed.is_set() is close_window
    finally:
        child.finish()
        reader.join()


@pytest.mark.parametrize("cancel", [False, True])
def test_download_stdout_eof_waits_for_exit_and_cancellation_does_not_wait_for_reaping(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    child = ChildAfterEOF(exit_on_kill=False)
    monkeypatch.setattr(procs, "popen_quiet", lambda *args, **kwargs: child.process)
    completed = threading.Event()
    download = ModelDownload(child.args, on_done=completed.set)
    assert download.start("animagine")
    try:
        assert child.waiting.wait(3)
        assert child.timeouts == [None]
        assert download.status().running and not completed.is_set()
        assert not child.killed.is_set()
        if cancel:
            download.stop()
            assert child.killed.is_set()
            assert download.status().running and not completed.is_set()
        child.finish(-9 if cancel else 0)
        download.join(3)
        assert not download.status().running and completed.is_set()
        assert child.killed.is_set() is cancel
        assert (download.status().error is not None) is cancel
    finally:
        child.finish()
        download.join(3)
        assert download._thread is not None and not download._thread.is_alive()
