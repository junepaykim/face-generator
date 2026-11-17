"""Concurrent pipe writes must reach the child as complete JSON lines."""

from __future__ import annotations

import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from facegen import models
from facegen.app_window import WindowProcess
from facegen.pipes import JsonWriter
from facegen.service import WorkerStatus
from facegen.worker_process import ChildLink, WorkerProxy


class FragmentedOutput:
    """Expose overlapping writes even when the real pipe usually accepts a whole line."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.events: list[str] = []

    def write(self, text: str) -> int:
        self.events.append("write")
        middle = len(text) // 2
        self.parts.append(text[:middle])
        time.sleep(0.001)
        self.parts.append(text[middle:])
        return len(text)

    def flush(self) -> None:
        self.events.append("flush")


@pytest.mark.parametrize("sender", ["window", "worker", "child", "writer"])
def test_commands_are_serialized_through_flush(sender: str, tmp_path: Path) -> None:
    out = FragmentedOutput()
    child = SimpleNamespace(stdin=out, poll=lambda: None)
    if sender == "window":
        window = WindowProcess("http://127.0.0.1:8765")
        window._child = child
        send_message = lambda message: window._send({"cmd": message})
    elif sender == "worker":
        worker = WorkerProxy([], db_path=tmp_path / "unused.db")
        worker._child = child
        send_message = worker._send
    elif sender == "child":
        link = ChildLink(out)
        send_message = lambda message: link.send({"cmd": message})
    else:
        writer = JsonWriter()
        send_message = lambda message: writer.send(out, {"cmd": message})
    start = threading.Barrier(8)

    def send(ident: int) -> None:
        start.wait(timeout=5)
        for index in range(10):
            assert send_message(f"한글-{ident * 10 + index}") is not False

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(send, range(8)))
    messages = [json.loads(line) for line in "".join(out.parts).splitlines()]
    assert {message["cmd"] for message in messages} == {f"한글-{i}" for i in range(80)}
    assert "".join(out.parts).isascii()
    assert out.events == ["write", "flush"] * 80


def test_worker_status_read_and_send_share_the_writer_lock() -> None:
    writing, release, second_started, second_read = (threading.Event() for _ in range(4))

    class BlockingOutput(io.StringIO):
        def write(self, text: str) -> int:
            writing.set()
            assert release.wait(5)
            return super().write(text)

    out = BlockingOutput()
    link = ChildLink(out)

    def status(done: int) -> WorkerStatus:
        return WorkerStatus("idle", "", None, False, False, False, False, done, 0, None, ())

    def read_second() -> WorkerStatus:
        second_read.set()
        return status(2)

    def send_second() -> None:
        second_started.set()
        link.status(read_second)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(link.status, lambda: status(1))
        try:
            assert writing.wait(5)
            second = pool.submit(send_second)
            assert second_started.wait(5)
            assert not second_read.wait(0.05)
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert [json.loads(line)["status"]["done"] for line in out.getvalue().splitlines()] == [1, 2]


@pytest.mark.parametrize("error", [BrokenPipeError, ValueError])
@pytest.mark.parametrize("operation", ["write", "flush"])
def test_pipe_errors_remain_caller_policy(
    error: type[Exception], operation: str, tmp_path: Path
) -> None:
    class FailsOnce(io.StringIO):
        failed = False

        def fail(self, at: str) -> None:
            if at == operation and not self.failed:
                self.failed = True
                raise error("closed pipe")

        def write(self, text: str) -> int:
            self.fail("write")
            return super().write(text)

        def flush(self) -> None:
            self.fail("flush")
            super().flush()

    out = FailsOnce()
    writer = JsonWriter()
    with pytest.raises(error, match="closed pipe"):
        writer.send(out, {"type": "ready"})
    writer.send(out, {"type": "ready"})  # the primitive does not latch a broken state

    out = FailsOnce()
    link = ChildLink(out)
    link.line("first")
    assert link.broken
    after_error = out.getvalue()
    link.line("second")
    assert out.getvalue() == after_error  # the worker child does latch it

    for parent in (
        WindowProcess("http://127.0.0.1:8765"),
        WorkerProxy([], db_path=tmp_path / "db"),
    ):
        parent._child = SimpleNamespace(stdin=FailsOnce(), poll=lambda: None)
        command = {"cmd": "show"} if isinstance(parent, WindowProcess) else "wake"
        assert not parent._send(command)
        assert parent._send(command)  # parents report the error and allow a later send

    with pytest.raises(error, match="closed pipe"):
        models.run_child(FailsOnce(), cache=models.HubCache(tmp_path), files=())


def test_model_progress_and_file_messages_share_a_writer() -> None:
    files = tuple(models.ModelFile("made/up", f"{i}.json", 10) for i in range(4))
    have: set[models.ModelFile] = set()
    cache = SimpleNamespace(has=lambda f: f in have, partial_bytes=lambda repo: 3)
    reported = threading.Event()

    class ProgressOutput(FragmentedOutput):
        def write(self, text: str) -> int:
            count = super().write(text)
            if json.loads(text).get("done") == 3:
                reported.set()
            return count

    def fetch(f: models.ModelFile) -> None:
        assert reported.wait(5)
        have.add(f)

    out = ProgressOutput()
    assert models.run_child(out, cache=cache, files=files, fetch=fetch, interval=0.001) == 0
    messages = [json.loads(line) for line in "".join(out.parts).splitlines()]
    assert messages[0]["type"] == "plan"
    assert messages[-1]["type"] == "done"
    assert sum(message["type"] == "file" for message in messages) == len(files)
    assert any(message.get("done") == 3 for message in messages)
    assert out.events == ["write", "flush"] * len(messages)


def test_worker_does_not_mistake_encoding_errors_for_broken_pipes() -> None:
    link = ChildLink(io.StringIO())
    circular: dict = {}
    circular["cycle"] = circular
    with pytest.raises(ValueError, match="Circular reference"):
        link.send(circular)
    assert not link.broken
