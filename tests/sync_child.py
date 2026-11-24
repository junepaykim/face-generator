"""`kitshot sync-process` with a fake save reader, by a plan: the real child (the request on
stdin, execute_sync, JSON lines out) for SyncService's process mode (Phase 14).

    python sync_child.py STATE_DIR PLAN sync-process

PLAN has one step per process start, comma-separated; the last step repeats:

    world         reads three made-up players (as the tests' world)
    unsupported   the reader cannot read the save's build (27.9.0+1)
    die           writes its last words to stderr and exits with code 3, saying nothing
    slow          says it is reading (STATE_DIR/reading holds its pid), then reads for up to
                  60 s: a stop before the commit ends it
    slow-write    reads the world, then waits up to 60 s inside the sync's open transaction,
                  its players written (STATE_DIR/writing holds its pid): a stop rolls it back
    slow-publish  reads the world, commits, then publishes for up to 60 s (STATE_DIR/publishing
                  holds its pid): a stop after the commit is not obeyed

Each start appends its plan step to STATE_DIR/starts.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factories import ABROAD, NEWGEN, RIVAL, player, snapshot

from kitshot.cli import main
from kitshot.core.records import Snapshot
from kitshot.datasource.base import UnsupportedSaveError
from kitshot.pipeline import reclassification, sync_operation
from kitshot.runtime import sync_process

WAIT = 60.0


def world() -> Snapshot:
    return snapshot(
        [
            player(1001, name="Mine", age=29),
            player(NEWGEN + 2, name="B Newgen", ca=90, pa=152, club_id=RIVAL),
            player(NEWGEN + 4, name="A Newgen", ca=160, pa=170, club_id=ABROAD),
        ]
    )


class PlannedSource:
    def __init__(self, step: str, state: Path) -> None:
        self.step = step
        self.state = state

    def load(self, path: Path) -> Snapshot:
        if self.step == "unsupported":
            raise UnsupportedSaveError("FM27 build 27.9.0+1 cannot be read", build="27.9.0+1")
        if self.step == "die":
            print("fmsave: the reader broke down", file=sys.stderr, flush=True)
            os._exit(3)
        if self.step == "slow":
            (self.state / "reading").write_text(str(os.getpid()), encoding="utf-8")
            time.sleep(WAIT)
        return world()


def slow_publish(state: Path):  # type: ignore[no-untyped-def]
    def publish(conn, settings, **kwargs):  # type: ignore[no-untyped-def]
        (state / "publishing").write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(WAIT)

    return publish


def slow_write(state: Path):  # type: ignore[no-untyped-def]
    plan = reclassification.classify_and_plan

    def classify_and_plan(conn, career_id, settings):  # type: ignore[no-untyped-def]
        assert conn.in_transaction
        (state / "writing").write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(WAIT)
        return plan(conn, career_id, settings)

    return classify_and_plan


def delayed_commit_notice(state: Path) -> None:
    """Let the test deliver Quit between the SQLite commit and its wire notification."""
    sync_save = sync_operation.sync_save
    publish_active = sync_operation.publish_active
    stop_if_reading = sync_process._Progress.stop_if_reading

    def after_sync(*args, **kwargs):  # type: ignore[no-untyped-def]
        report = sync_save(*args, **kwargs)
        (state / "stored").write_text(str(os.getpid()), encoding="utf-8")
        wait_for_file(state / "quit-received")
        return report

    def receive_stop(progress, exit_now):  # type: ignore[no-untyped-def]
        stop_if_reading(progress, exit_now)
        (state / "quit-received").touch()

    def publish(conn, settings, **kwargs):  # type: ignore[no-untyped-def]
        (state / "publishing").write_text(str(os.getpid()), encoding="utf-8")
        wait_for_file(state / "finish-publication")
        return publish_active(conn, settings, **kwargs)

    sync_operation.sync_save = after_sync  # type: ignore[assignment]
    sync_process._Progress.stop_if_reading = receive_stop
    sync_operation.publish_active = publish


def wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + WAIT
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.01)


if __name__ == "__main__":
    state = Path(sys.argv[1])
    state.mkdir(parents=True, exist_ok=True)
    plan = sys.argv[2].split(",")
    starts = state / "starts"
    done = starts.read_text(encoding="utf-8").splitlines() if starts.exists() else []
    step = plan[min(len(done), len(plan) - 1)]
    with starts.open("a", encoding="utf-8") as f:
        f.write(step + "\n")
    if step == "slow-write":
        reclassification.classify_and_plan = slow_write(state)  # type: ignore[assignment]
    if step == "slow-publish":
        sync_operation.publish_active = slow_publish(state)  # type: ignore[assignment]
    if step == "delayed-notice":
        delayed_commit_notice(state)
    raise SystemExit(main(sys.argv[3:], source_factory=lambda: PlannedSource(step, state)))
