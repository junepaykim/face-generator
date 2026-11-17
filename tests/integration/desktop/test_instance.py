"""One desktop app at a time (Phase 11): the lock in data/, and a second start asking the first
to open its window."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from support.paths import ROOT

from facegen.desktop import instance
from facegen.platform import procs

HOLDER = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from facegen.desktop.instance import InstanceLock
lock = InstanceLock(Path(sys.argv[1]))
print("held" if lock.acquire() else "busy", flush=True)
sys.stdin.readline()
"""


def test_the_lock_is_taken_once_and_given_back(tmp_path: Path) -> None:
    first, second = instance.InstanceLock(tmp_path), instance.InstanceLock(tmp_path)
    assert first.acquire() and first.held
    assert first.acquire()  # already held: still ours
    assert not second.acquire() and not second.held
    assert instance.held_elsewhere(tmp_path)
    first.release()
    assert not instance.held_elsewhere(tmp_path)
    with second:
        assert second.acquire()
    assert not second.held


def test_another_process_holding_the_lock_is_seen_and_its_end_frees_it(tmp_path: Path) -> None:
    # A Windows venv's python.exe can be a launcher with the actual interpreter as its
    # child. Kill the lock owner itself, not that launcher (which can leave the owner alive).
    python = getattr(sys, "_base_executable", sys.executable) if os.name == "nt" else sys.executable
    source = str(ROOT / "src")
    holder = procs.popen_quiet(
        [python, "-c", HOLDER, str(tmp_path), source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        assert instance.held_elsewhere(tmp_path)
        assert not instance.InstanceLock(tmp_path).acquire()
    finally:
        holder.kill()  # as Task Manager would: no stale lock is left behind
        holder.wait()
    assert not instance.held_elsewhere(tmp_path)


def test_a_request_to_open_is_answered_once(tmp_path: Path) -> None:
    assert not instance.take_open_request(tmp_path)
    path = instance.request_open(tmp_path)
    assert path == tmp_path / instance.OPEN_REQUEST_NAME and path.is_file()
    assert not list(tmp_path.glob(".*.part"))  # written whole
    assert instance.take_open_request(tmp_path)
    assert not instance.take_open_request(tmp_path)


def test_a_second_start_waits_for_the_first_to_take_its_request(tmp_path: Path) -> None:
    instance.request_open(tmp_path)

    def take_soon() -> None:
        time.sleep(0.2)
        instance.take_open_request(tmp_path)

    threading.Thread(target=take_soon).start()
    assert instance.wait_taken(tmp_path, timeout=5)


def test_a_second_start_gives_up_when_nobody_answers(tmp_path: Path) -> None:
    instance.request_open(tmp_path)
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    assert not instance.wait_taken(tmp_path, timeout=2, sleep=sleep, clock=lambda: now[0])
    assert now[0] >= 2
