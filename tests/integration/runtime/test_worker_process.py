"""Phase 7b: the GPU worker in a child process. The pipe protocol, WorkerProxy with a fake
child that speaks the protocol by a plan (fake_child.py), and the exit test: `facegen serve`
with a worker process whose renderer fails on its third job (gpu_child.py)."""

from __future__ import annotations

import dataclasses
import io
import logging
import os
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from factories import ABROAD, NEWGEN, RIVAL, player, snapshot
from support.paths import TESTS

from facegen.core.records import Personality
from facegen.persistence import db
from facegen.persistence.jobs import Job
from facegen.persistence.settings import load_settings
from facegen.pipeline.sync import run_sync
from facegen.runtime import worker_process
from facegen.runtime.supervision import RestartPolicy
from facegen.runtime.worker_process import (
    ChildLink,
    WorkerProxy,
    decode,
    describe_exit,
    encode,
    status_from_json,
    status_to_json,
    worker_command,
)
from facegen.service import Exit, WorkerStatus

FAST = RestartPolicy(backoff=(0.05,))


def wait_for(check: Callable[[], bool], timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


# -- the messages -----------------------------------------------------------------------------


def job(**kw: object) -> Job:
    fields: dict[str, object] = {
        "id": 7,
        "unique_id": 1002,
        "kind": "initial",
        "group": "a",
        "target_club_id": None,
        "target_band": None,
        "name": "James Rodríguez",
        "at_my_club": True,
        "goalkeeper": False,
        "height_cm": 180,
        "strength": 12,
        "age": 22,
    }
    fields.update(kw)
    return Job(**fields)  # type: ignore[arg-type]


def test_a_status_survives_the_pipe() -> None:
    personality = Personality(10, 11, 12, 13, 14, 15, 16, 17)
    status = WorkerStatus(
        "running", "drawing", job(personality=personality, staff=True, staff_job="Coach"),
        False, False, True, True, 3, 1, None, ("a line",),
        reason="drawing", values={"name": "James Rodríguez", "kind": "initial", "group": "A"},
        since="2026-11-17T10:00:00+00:00",
    )  # fmt: skip
    line = encode({"type": "status", "status": status_to_json(status)})
    assert line.isascii() and line.endswith("\n") and line.count("\n") == 1
    message = decode(line)
    assert message is not None
    back = status_from_json(message["status"])
    assert back == dataclasses.replace(status, log=())  # the log travels line by line


def test_a_status_from_a_child_without_reasons_still_reads() -> None:
    """A status line written before Phase 12c has no reason, values or since."""
    data = status_to_json(
        WorkerStatus("idle", "the queue is empty", None, *[False] * 4, 0, 0, None, ())
    )
    for name in ("reason", "values", "since"):
        del data[name]
    back = status_from_json(data)
    assert (back.reason, back.values, back.since, back.detail) == (
        "",
        {},
        None,
        "the queue is empty",
    )


def test_commands_are_read_from_the_descriptor_line_by_line() -> None:

    read, write = os.pipe()
    os.write(write, b'{"cmd": "wake"}\n{"cmd": "st')
    os.write(write, b'op"}\npartial')
    os.close(write)
    assert list(worker_process.fd_lines(read)) == ['{"cmd": "wake"}', '{"cmd": "stop"}', "partial"]
    os.close(read)


def test_lines_that_are_not_messages() -> None:
    assert decode("Loading pipeline components...\n") is None
    assert decode("[1, 2]\n") is None
    assert decode('{"type": "line", "text": "x"}') == {"type": "line", "text": "x"}


def test_the_child_link_sends_each_change_once_and_survives_a_broken_pipe() -> None:
    out = io.StringIO()
    link = ChildLink(out)
    status = WorkerStatus("idle", "", None, False, False, False, False, 0, 0, None, ())
    link.status(lambda: status)
    link.status(lambda: status)
    link.line("12:00:00  hello")
    link.exit(Exit("fault", "CUDA error", job()))
    messages = [decode(line) for line in out.getvalue().splitlines()]
    assert [m["type"] for m in messages if m] == ["status", "line", "exit"]
    assert messages[-1] and messages[-1]["job"]["name"] == "James Rodríguez"

    class Broken(io.StringIO):
        def write(self, s: str) -> int:
            raise BrokenPipeError

    broken = ChildLink(Broken())
    broken.line("x")  # does not raise
    assert broken.broken


def test_the_command_line_and_exit_codes() -> None:
    assert worker_command(None, None) == [sys.executable, "-m", "facegen.cli", "worker"]
    assert worker_command(("py", "x.py"), Path("c.toml")) == [
        "py", "x.py", "--config", "c.toml", "worker",
    ]  # fmt: skip
    assert describe_exit(3) == "exit code 3"
    assert describe_exit(-11) == "ended by SIGSEGV"
    assert describe_exit(0xC0000005) == "exit code 0xc0000005"
    assert describe_exit(None) == "it could not be started"


# -- WorkerProxy with a fake child ------------------------------------------------------------


class Fake:
    """A proxy over fake_child.py with a plan, and what the fake children noted."""

    def __init__(self, tmp_path: Path, db_path: Path) -> None:
        self.state = tmp_path / "child"
        self.db_path = db_path
        self.proxies: list[WorkerProxy] = []

    def proxy(self, plan: str, **kw: object) -> WorkerProxy:
        command = [sys.executable, str(TESTS / "fake_child.py"), str(self.state), plan]
        kw.setdefault("policy", FAST)
        p = WorkerProxy(command, db_path=self.db_path, echo_stderr=False, **kw)  # type: ignore[arg-type]
        self.proxies.append(p)
        return p

    def starts(self) -> list[str]:
        path = self.state / "starts"
        return path.read_text().splitlines() if path.exists() else []

    def commands(self) -> list[str]:
        path = self.state / "commands"
        return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def fake(tmp_path: Path, config_file: Path) -> Iterator[Fake]:
    settings = load_settings(defaults_path=config_file)
    conn = db.connect(settings.paths.db_path)
    run_sync(conn, snapshot(world()), settings, save_file="career.fm")
    conn.close()
    f = Fake(tmp_path, settings.paths.db_path)
    yield f
    for p in f.proxies:
        p.stop(timeout=10)


def world() -> list:
    return [
        player(1001, name="Mine", age=29),
        player(NEWGEN + 2, name="B Newgen", ca=90, pa=152, club_id=RIVAL),
        player(NEWGEN + 4, name="A Newgen", ca=160, pa=170, club_id=ABROAD),
    ]


def a_job_id(db_path: Path) -> int:
    conn = db.connect(db_path)
    try:
        return conn.execute("SELECT MIN(id) FROM jobs WHERE status = 'queued'").fetchone()[0]
    finally:
        conn.close()


def job_row(db_path: Path, job_id: int) -> sqlite3.Row:
    conn = db.connect(db_path)
    try:
        return conn.execute("SELECT status, error FROM jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        conn.close()


def log_text(p: WorkerProxy) -> str:
    return "\n".join(p.status().log)


def test_a_clean_start_and_stop(fake: Fake) -> None:
    p = fake.proxy("idle")
    p.start()
    wait_for(lambda: p.status().state == "idle")
    assert p.status().detail == "run 1" and p.child_pid is not None
    p.stop(timeout=10)
    status = p.status()
    assert status.state == "stopped" and p.child_pid is None
    assert fake.commands() == ["stop"] and p.starts == 1
    assert status.done == 1  # the child's count is kept after it ended


def test_a_gpu_fault_starts_a_new_worker_process(fake: Fake) -> None:
    job_id = a_job_id(fake.db_path)
    p = fake.proxy(f"fault:{job_id},idle")
    p.start()
    wait_for(lambda: p.starts == 2 and p.status().state == "idle")
    assert log_text(p).count("Starting a new worker process") == 1
    assert job_row(fake.db_path, job_id)["status"] == "queued"  # one fault: drawn again
    assert not p.status().paused and p.status().last_error is None


def test_a_job_drawn_during_two_faults_goes_to_review(fake: Fake) -> None:
    job_id = a_job_id(fake.db_path)
    p = fake.proxy(f"fault:{job_id},fault:{job_id},idle")
    p.start()
    wait_for(lambda: p.starts == 3 and p.status().state == "idle")
    row = job_row(fake.db_path, job_id)
    assert row["status"] == "failed"
    assert row["error"].startswith("the graphics driver failed twice while drawing this picture")
    assert "Player " in log_text(p) and "goes to review" in log_text(p)


def test_three_failures_give_up_and_resume_tries_again(fake: Fake) -> None:
    p = fake.proxy("fault,die,fault,idle")
    p.start()
    wait_for(lambda: p.status().state == "gpu-fault")
    status = p.status()
    assert status.paused and not status.gpu_fault  # Resume is offered: a new process can start
    assert "failed 3 times within 30 minutes" in status.detail and "Resume" in status.detail
    assert status.reason == "gave_up" and (status.values["count"], status.values["minutes"]) == (
        3,
        30,
    )
    assert "illegal memory access" in (status.last_error or "")
    assert "ended unexpectedly (exit code 3)" in log_text(p)
    time.sleep(0.3)
    assert p.starts == 3  # no more
    assert p.resume()
    wait_for(lambda: p.status().state == "idle")
    assert p.starts == 4 and p.status().last_error is None


def test_a_child_that_dies_is_restarted_and_its_last_words_are_logged(fake: Fake) -> None:
    records: list[str] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    log = logging.getLogger("facegen")
    handler, level = Keep(), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        p = fake.proxy("die:5,idle")
        p.start()
        wait_for(lambda: p.starts == 2 and p.status().state == "idle")
    finally:
        log.removeHandler(handler)
        log.setLevel(level)
    text = "\n".join(records)
    assert "stderr: Fatal Python error: Segmentation fault" in text
    assert "while drawing Player 5" in text
    assert "Player 5 is back in the queue" not in p.status().detail  # it is drawing again


def test_a_child_that_hangs_on_stop_is_killed_after_a_timeout(fake: Fake) -> None:
    p = fake.proxy("hang", stop_timeout=0.5)
    p.start()
    wait_for(lambda: p.status().state == "idle")
    pid = p.child_pid
    t0 = time.monotonic()
    p.stop(timeout=10)
    assert time.monotonic() - t0 < 5
    assert "The worker process did not stop within 0.5 s; ended it." in log_text(p)
    assert p.status().state == "stopped" and pid is not None and p.child_pid is None
    assert fake.commands() == ["stop"]


def test_pause_ends_the_child_and_resume_starts_another(fake: Fake) -> None:
    p = fake.proxy("idle")
    p.start()
    wait_for(lambda: p.status().state == "idle")
    p.pause()
    wait_for(lambda: p.status().state == "paused" and p.child_pid is None)
    assert p.status().paused and fake.commands() == ["pause"]
    assert "Paused." in log_text(p)
    time.sleep(0.2)
    assert p.starts == 1
    p.resume()
    wait_for(lambda: p.starts == 2 and p.status().state == "idle")
    assert p.status().detail == "run 2" and p.status().done == 2


def test_generate_now_reaches_the_next_child(fake: Fake) -> None:
    p = fake.proxy("idle", start_paused=True)
    p.start()
    time.sleep(0.2)
    assert p.starts == 0 and p.status().state == "paused"
    assert p.generate_now()
    wait_for(lambda: p.status().state == "idle")
    assert fake.starts() == ["--generate-now"] and p.status().generate_now
    assert "Generate now" in log_text(p)


def test_a_child_that_freed_the_gpu_is_replaced_without_counting_a_failure(fake: Fake) -> None:
    p = fake.proxy("freed,idle")
    p.start()
    wait_for(lambda: p.starts == 2 and p.status().state == "idle")
    assert "Starting a new worker process" not in log_text(p)
    assert p.status().last_error is None


def test_an_error_pauses_and_waits_for_resume(fake: Fake) -> None:
    p = fake.proxy("error,idle")
    p.start()
    wait_for(lambda: p.status().state == "error")
    status = p.status()
    assert status.paused and "CUDA out of memory" in (status.last_error or "")
    time.sleep(0.2)
    assert p.starts == 1
    p.resume()
    wait_for(lambda: p.status().state == "idle")


def test_a_stray_line_on_the_pipe_is_skipped(fake: Fake) -> None:
    p = fake.proxy("junk")
    p.start()
    wait_for(lambda: p.status().state == "idle")


def test_a_worker_that_cannot_be_started_gives_up(fake: Fake, tmp_path: Path) -> None:
    p = WorkerProxy(
        [str(tmp_path / "no-such-python")], db_path=fake.db_path, policy=FAST, echo_stderr=False
    )
    fake.proxies.append(p)
    p.start()
    wait_for(lambda: p.status().state == "gpu-fault")
    assert "could not start the worker process" in (p.status().last_error or "")
    assert p.starts == 0


def test_wake_and_resume_reach_a_running_child(fake: Fake) -> None:
    p = fake.proxy("idle")
    p.start()
    wait_for(lambda: p.status().state == "idle")
    p.wake()
    p.resume()
    wait_for(lambda: fake.commands() == ["wake", "resume"])


# -- the exit test ----------------------------------------------------------------------------


def test_exit_test_a_fault_on_the_third_job_is_shown_once_and_drawn_after_a_restart(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 7's exit test: with a renderer that raises a fake fatal error on its third job,
    `facegen serve` shows the fault once, keeps the job queued, and draws it after an
    automatic restart; no AttributeError appears anywhere."""
    from fastapi.testclient import TestClient

    from facegen.cli import Gpu, main

    monkeypatch.setattr(worker_process, "DEFAULT_POLICY", RestartPolicy(backoff=(0.2,)))
    settings = load_settings(defaults_path=config_file)
    conn = db.connect(settings.paths.db_path)
    run_sync(conn, snapshot(world()), settings, save_file="career.fm")
    conn.close()
    marker = tmp_path / "faulted"
    gpu = Gpu(program=(sys.executable, str(TESTS / "gpu_child.py"), str(marker)))
    seen: dict[str, str] = {}

    def fake_server(app, host: str, port: int) -> None:  # type: ignore[no-untyped-def]
        with TestClient(app, base_url=f"http://{host}:{port}") as client:

            def all_drawn() -> bool:
                seen["status"] = client.get("/status").text
                about = client.get("/settings").text  # the run's counts are here (Phase 12)
                return "3 made, 0 sent to review" in about and "state-idle" in seen["status"]

            wait_for(all_drawn, timeout=60)
            seen["page"] = client.get("/log").text  # the worker's log has a page (Phase 12)

    log = logging.getLogger("facegen")
    out = io.StringIO()
    try:
        code = main(
            ["--config", str(config_file), "serve", "--no-watch"],
            gpu=gpu,
            out=out,
            run_server=fake_server,
        )
    finally:
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()
    assert code == 0 and marker.exists()  # the fault happened
    page = seen["page"]
    assert page.count("GPU fault: RuntimeError: CUDA error: an illegal memory access") == 1
    assert page.count("Starting a new worker process") == 1
    conn = db.connect(settings.paths.db_path)
    try:
        rows = conn.execute("SELECT unique_id, status FROM jobs ORDER BY id").fetchall()
        drawn = conn.execute("SELECT unique_id FROM images WHERE is_current = 1").fetchall()
        errors = [r[0] or "" for r in conn.execute("SELECT error FROM jobs")]
    finally:
        conn.close()
    assert {r["status"] for r in rows} == {"done"}  # the third job too, after the restart
    assert len(drawn) == 3
    text = (settings.paths.data_dir / "logs" / "facegen.log").read_text(encoding="utf-8")
    assert "Traceback" in text and "illegal memory access" in text  # the real stack is kept
    # Both worker processes ended by themselves: the one that faulted, and the one stopped.
    assert text.count("ended (exit code 0)") == 2 and "Fatal Python error" not in text
    for where in (page, seen["status"], text, *errors, out.getvalue()):
        assert "AttributeError" not in where
