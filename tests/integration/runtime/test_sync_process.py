"""Phase 14: each sync in a process of its own. The messages and the child's side in this
process, then SyncService over the real `facegen sync-process` with a fake reader by a plan
(sync_child.py): synced, an unsupported save, a crash, Quit while reading, writing and
publishing, and the save watcher and Sync now over the child."""

from __future__ import annotations

import io
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from factories import ABROAD, NEWGEN, RIVAL, player, snapshot
from support.paths import ROOT, TESTS
from support.processes import has_exited

from facegen.core.records import Snapshot
from facegen.persistence import db
from facegen.persistence.settings import Settings, load_settings
from facegen.pipeline.sync import run_sync
from facegen.platform import procs
from facegen.platform.pipes import decode, encode
from facegen.runtime import sync_process
from facegen.runtime.contracts import FAILED, SYNCED, UNREADABLE, SyncRequest, SyncResult
from facegen.runtime.poller_service import PollerService
from facegen.runtime.sync_process import (
    STOPPED_ERROR,
    STOPPED_EXIT,
    STOPPED_PUBLISHING_ERROR,
    request_from_json,
    request_to_json,
    result_from_json,
    result_to_json,
    serve_child,
    sync_command,
)
from facegen.runtime.sync_service import SyncService


def wait_for(check: Callable[[], bool], timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def world() -> Snapshot:
    """As sync_child.py's world."""
    return snapshot(
        [
            player(1001, name="Mine", age=29),
            player(NEWGEN + 2, name="B Newgen", ca=90, pa=152, club_id=RIVAL),
            player(NEWGEN + 4, name="A Newgen", ca=160, pa=170, club_id=ABROAD),
        ]
    )


class WorldSource:
    def load(self, path: Path) -> Snapshot:
        return world()


def write_save(settings: Settings, name: str = "career.fm", mtime: int = 1_800_000_000) -> Path:
    """The game writes a save."""
    games = settings.paths.games_dir
    assert games is not None
    games.mkdir(parents=True, exist_ok=True)
    path = games / name
    path.write_bytes(b"save " + str(mtime).encode())
    os.utime(path, (mtime, mtime))
    return path


def dump(db_path: Path) -> list[str]:
    """Everything the database holds, as SQL."""
    with closing(sqlite3.connect(db_path)) as conn:
        return list(conn.iterdump())


# -- the messages -------------------------------------------------------------------------------


def test_requests_and_results_survive_the_pipe() -> None:
    request = SyncRequest("state.db", None, "C:/games/Career ü.fm", "poller")
    line = encode({"cmd": "sync", "request": request_to_json(request)})
    assert line.isascii() and line.count("\n") == 1
    message = decode(line)
    assert message is not None and request_from_json(message["request"]) == request
    result = SyncResult(UNREADABLE, None, "Cannot read x.fm: build 27.9", True, ("a warning",))
    line = encode({"type": "result", "result": result_to_json(result)})
    message = decode(line)
    assert message is not None and result_from_json(message["result"]) == result
    with pytest.raises(ValueError, match="not a sync outcome"):
        result_from_json({"outcome": "finished"})


def test_the_command_line() -> None:
    assert sync_command() == [sys.executable, "-m", "facegen.cli", "sync-process"]
    assert sync_command(("py", "x.py")) == ["py", "x.py", "sync-process"]


# -- the child, in this process -----------------------------------------------------------------


class Commands:
    """The child's stdin: lines put in by the test, open until closed."""

    def __init__(self, *lines: str) -> None:
        self._lines: queue.Queue[str | None] = queue.Queue()
        for line in lines:
            self.put(line)

    def put(self, line: str) -> None:
        self._lines.put(line)

    def close(self) -> None:
        self._lines.put(None)

    def __iter__(self) -> Iterator[str]:
        while (line := self._lines.get()) is not None:
            yield line


def messages(out: io.StringIO) -> list[dict]:
    found = [decode(line) for line in out.getvalue().splitlines()]
    assert all(m is not None for m in found)
    return found  # type: ignore[return-value]


def test_the_child_commits_then_logs_publishes_and_sends_its_result(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    save = write_save(settings)
    request = SyncRequest(str(settings.paths.db_path), str(config_file), str(save), "poller")
    commands = Commands(encode({"cmd": "sync", "request": request_to_json(request)}))
    out = io.StringIO()
    exits: list[int] = []
    result = serve_child(commands, out, WorldSource, exit_now=exits.append)
    commands.close()
    assert result is not None and result.outcome == SYNCED
    sent = messages(out)
    assert sent[0] == {"type": "committed"}  # before anything is said or published
    assert sent[1] == {"type": "line", "text": result.message}
    assert result.message is not None and result.message.startswith("Synced career.fm")
    assert sent[-1] == {"type": "result", "result": result_to_json(result)}
    assert [m["type"] for m in sent].count("result") == 1
    with closing(db.connect(settings.paths.db_path)) as conn:
        assert conn.execute("SELECT trigger FROM sync_runs").fetchone()[0] == "poller"
        assert conn.execute("SELECT name FROM players WHERE unique_id = 1001").fetchone()[0] == (
            "Mine"
        )
    time.sleep(0.1)  # the end of stdin after the result ends nothing
    assert exits == []


def test_a_stop_ends_the_child_before_its_commit_only() -> None:
    progress = sync_process._Progress()
    exits: list[int] = []
    sync_process._watch_parent(iter(['{"cmd": "wake"}', '{"cmd": "stop"}']), progress, exits.append)
    assert exits == [STOPPED_EXIT]
    sync_process._watch_parent(iter([]), progress, exits.append)  # the parent has gone
    assert exits == [STOPPED_EXIT, STOPPED_EXIT]
    for stage in ("publishing", "done"):
        progress.enter(stage)
        sync_process._watch_parent(iter(['{"cmd": "stop"}']), progress, exits.append)
        sync_process._watch_parent(iter([]), progress, exits.append)
    assert exits == [STOPPED_EXIT, STOPPED_EXIT]


def test_a_child_without_a_request_says_so_and_ends_with_2(tmp_path: Path) -> None:
    out = io.StringIO()
    assert serve_child(iter(["not json"]), out, WorldSource) is None
    assert out.getvalue() == ""
    done = procs.run_quiet(
        [sys.executable, "-m", "facegen.cli", "sync-process"],
        input='{"cmd": "wake"}\n',
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=60,
    )
    assert done.returncode == 2 and done.stdout == ""
    assert "not a sync request" in done.stderr


# -- SyncService over the real child with a fake reader -------------------------------------------


@dataclass
class Child:
    settings: Settings
    config: Path
    state: Path
    log: list[str] = field(default_factory=list)
    services: list[SyncService] = field(default_factory=list)

    def syncer(self, plan: str, **kw: object) -> SyncService:
        command = [sys.executable, str(TESTS / "sync_child.py"), str(self.state), plan]
        service = SyncService(
            self.settings.paths.db_path,
            command=sync_command(command),
            defaults_path=self.config,
            log=self.log.append,
            echo_stderr=False,
            **kw,  # type: ignore[arg-type]
        )
        self.services.append(service)
        return service

    def poller(self, syncer: SyncService, clock: Callable[[], float]) -> PollerService:
        return PollerService(
            self.settings.paths.db_path,
            syncer,
            defaults_path=self.config,
            settle_seconds=10.0,
            clock=clock,
        )

    def starts(self) -> list[str]:
        path = self.state / "starts"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def marker(self, name: str) -> int | None:
        path = self.state / name
        return int(path.read_text(encoding="utf-8")) if path.exists() else None

    def runs(self) -> list[sqlite3.Row]:
        with closing(db.connect(self.settings.paths.db_path)) as conn:
            return conn.execute("SELECT * FROM sync_runs ORDER BY id").fetchall()


@pytest.fixture
def child(tmp_path: Path, config_file: Path) -> Iterator[Child]:
    settings = load_settings(defaults_path=config_file)
    with closing(db.connect(settings.paths.db_path)) as conn:
        run_sync(conn, world(), settings, save_file="earlier.fm")  # what is there already
    c = Child(settings, config_file, tmp_path / "child")
    yield c
    for service in c.services:
        service.stop(timeout=10)


@dataclass
class Clock:
    now: float = 1000.0

    def __call__(self) -> float:
        return self.now


def poll_until_synced(poller: PollerService, clock: Clock) -> None:
    """Look, wait out the settle time, look again."""
    poller.step()
    assert poller.status().state == "settling", poller.status()
    clock.now += 10
    poller.step()


def test_sync_now_runs_in_a_child_and_reports_as_before(child: Child) -> None:
    write_save(child.settings)
    done: list[int] = []
    syncer = child.syncer("world", on_done=lambda: done.append(1))
    assert syncer.start()
    syncer.join(60)
    status = syncer.status()
    assert not status.running and status.error is None and status.trigger == "sync now"
    assert status.message is not None and status.message.startswith("Synced career.fm")
    assert child.log[0] == status.message  # the child's lines reach the log, in order
    assert done == [1] and child.starts() == ["world"]
    assert syncer.process is not None and syncer.process.child_pid is None
    assert child.runs()[-1]["trigger"] == "sync now"
    assert (child.settings.paths.snapshots_dir / "career.fm").is_file()  # read from a copy
    # FM27 has not saved since: the notice comes back from the child too.
    assert syncer.start()
    syncer.join(60)
    assert (syncer.status().notice or "").startswith("FM27 has not saved since career.fm")


def test_the_save_watcher_syncs_once_over_the_child(child: Child) -> None:
    clock = Clock()
    syncer = child.syncer("world")
    poller = child.poller(syncer, clock)
    before = len(child.runs())
    write_save(child.settings)
    poll_until_synced(poller, clock)
    assert poller.status().state == "synced"
    assert (poller.status().detail or "").startswith("Synced career.fm")
    runs = child.runs()
    assert len(runs) == before + 1 and runs[-1]["trigger"] == "poller"
    for _ in range(3):  # nothing new: not read again
        poller.step()
    assert child.starts() == ["world"]


def test_an_unsupported_save_is_unreadable_and_waits_for_the_next_save(child: Child) -> None:
    clock = Clock()
    syncer = child.syncer("unsupported,world")
    poller = child.poller(syncer, clock)
    write_save(child.settings)
    poll_until_synced(poller, clock)
    status = syncer.status()
    assert status.unsupported and "27.9.0+1" in (status.error or "")
    assert any("bump the pin" in line for line in child.log)
    run = child.runs()[-1]
    assert (run["unsupported"], run["build"], run["save_mtime"]) == (1, "27.9.0+1", 1.8e9)
    poller.step()
    assert child.starts() == ["unsupported"]  # not tried again until the save changes
    assert "its game build is not supported" in poller.status().detail
    write_save(child.settings, mtime=1_800_000_600)
    poll_until_synced(poller, clock)
    assert poller.status().state == "synced" and child.starts() == ["unsupported", "world"]


def test_a_crash_is_a_failed_sync_with_its_last_words_in_the_log_and_is_tried_again(
    child: Child, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Clock()
    syncer = child.syncer("die,world")
    poller = child.poller(syncer, clock)
    before = len(child.runs())
    write_save(child.settings)
    with caplog.at_level(logging.INFO, logger="facegen"):
        poll_until_synced(poller, clock)
    error = "the sync process ended unexpectedly (exit code 3)"
    assert syncer.status().error == error and not syncer.status().running
    assert child.log == [f"Sync failed: {error}."]
    assert any(
        r.levelno == logging.ERROR and "fmsave: the reader broke down" in r.getMessage()
        for r in caplog.records
    )
    assert poller.status().state == "error" and "trying again" in poller.status().detail
    assert len(child.runs()) == before  # nothing recorded: the save is read again
    poll_until_synced(poller, clock)
    assert poller.status().state == "synced" and child.starts() == ["die", "world"]


@pytest.mark.parametrize(
    ("plan", "marker"),
    [("slow", "reading"), ("slow-write", "writing")],
    ids=["while reading the save", "inside the transaction"],
)
def test_quit_during_a_sync_ends_it_at_once_with_the_database_as_it_was(
    child: Child, caplog: pytest.LogCaptureFixture, plan: str, marker: str
) -> None:
    """Phase 14's exit test, without the real save: Quit ends the sync within a few seconds,
    nothing it read or wrote is kept, and the save is read again at the next start."""
    write_save(child.settings)
    before = dump(child.settings.paths.db_path)
    syncer = child.syncer(plan)
    caplog.set_level(logging.INFO, logger="facegen")
    assert syncer.start()
    wait_for(lambda: child.marker(marker) is not None)
    pid = child.marker(marker)
    assert pid is not None
    started = time.monotonic()
    syncer.stop()
    took = time.monotonic() - started
    assert took < 3.0, took  # it ended itself: no kill after STOP_SECONDS
    assert has_exited(pid)
    assert f"ended (exit code {STOPPED_EXIT})" in caplog.text
    status = syncer.status()
    assert not status.running and status.error == STOPPED_ERROR
    assert child.log[-1] == f"Sync stopped: {STOPPED_ERROR}."
    assert dump(child.settings.paths.db_path) == before
    with closing(db.connect(child.settings.paths.db_path)) as conn, db.transaction(conn):
        conn.execute("UPDATE settings SET value = value WHERE 0")  # nothing holds it
    # No other sync starts while facegen quits.
    assert not syncer.start() and syncer.run() is None and child.starts() == [plan]


def test_quit_after_the_commit_lets_publication_finish_for_a_while(
    child: Child, caplog: pytest.LogCaptureFixture
) -> None:
    write_save(child.settings)
    before = len(child.runs())
    syncer = child.syncer("slow-publish")
    assert syncer.process is not None
    syncer.process.publish_seconds = 1.0
    caplog.set_level(logging.INFO, logger="facegen")
    assert syncer.start()
    wait_for(lambda: child.marker("publishing") is not None and bool(child.log))
    assert child.log[0].startswith("Synced career.fm")  # sent after "committed"
    started = time.monotonic()
    syncer.stop()
    took = time.monotonic() - started
    assert 1.0 <= took < 5.0, took  # it was given its time, then ended
    assert "did not stop within 1 s; ended it" in caplog.text
    pid = child.marker("publishing")
    assert pid is not None and has_exited(pid)
    assert syncer.status().error == STOPPED_PUBLISHING_ERROR
    assert len(child.runs()) == before + 1  # the stored sync stays


def test_a_sync_process_that_cannot_start_is_a_failed_sync(child: Child, tmp_path: Path) -> None:
    syncer = SyncService(
        child.settings.paths.db_path,
        command=[str(tmp_path / "no-such-program")],
        defaults_path=child.config,
        log=child.log.append,
    )
    assert syncer.run() == FAILED
    assert (syncer.status().error or "").startswith("could not start the sync process")


def test_a_service_takes_a_source_or_a_command(child: Child) -> None:
    with pytest.raises(ValueError, match="either"):
        SyncService(child.settings.paths.db_path)
    with pytest.raises(ValueError, match="either"):
        SyncService(child.settings.paths.db_path, WorldSource, command=["x"])
    in_thread = SyncService(child.settings.paths.db_path, WorldSource)
    assert in_thread.process is None
    in_thread.stop()  # nothing to end; no further sync
    assert not in_thread.start()


def test_sync_now_from_another_thread_while_the_watcher_waits(child: Child) -> None:
    """The watcher and Sync now share one claim over the child, as they did in a thread."""
    write_save(child.settings)
    syncer = child.syncer("slow,world")
    assert syncer.start()
    wait_for(lambda: child.marker("reading") is not None)
    assert syncer.run(trigger="poller") is None  # the watcher waits for it
    stopper = threading.Thread(target=syncer.stop)
    stopper.start()
    stopper.join(30)
    assert not stopper.is_alive() and child.starts() == ["slow"]
