"""`facegen serve`'s background work: the worker loop (driven step by step) and "sync now"."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from factories import ABROAD, MANUAL, NEWGEN, RIVAL, player, snapshot
from fakes import FakeRenderer, FakeTagger
from support.paths import ROOT

from facegen.core.records import Snapshot
from facegen.datasource.kits import load_kits
from facegen.persistence import db
from facegen.persistence.db import transaction
from facegen.persistence.settings import Settings, load_settings, set_overrides
from facegen.pipeline import manual
from facegen.pipeline.reclassification import classify_and_plan
from facegen.pipeline.sync import run_sync
from facegen.pipeline.worker import load_career, worker_lock
from facegen.runtime.contracts import WORKER_REASONS
from facegen.service import (
    FM_POLL_SECONDS,
    IDLE_UNLOAD_SECONDS,
    PollerService,
    SyncService,
    WorkerService,
    WorkerStatus,
)

MINE = 1001  # my club: A
MINE_2 = 1002  # my club: A, arrived later, so drawn first
STAR = 1003  # pre-existing at another club: hand-picked only
B_NEWGEN = NEWGEN + 2  # rival, CA 90, PA 152: B
A_NEWGEN = NEWGEN + 4  # abroad, CA 160: A elsewhere


def world() -> list:
    from datetime import date

    return [
        player(MINE, name="Mine", age=29, join=date(2024, 7, 1)),
        player(MINE_2, name="Mine Two", age=22, join=date(2025, 7, 1)),
        player(STAR, name="Star", ca=180, pa=185, club_id=ABROAD),
        player(B_NEWGEN, name="B Newgen", ca=90, pa=152, club_id=RIVAL),
        player(A_NEWGEN, name="A Newgen", ca=160, pa=170, club_id=ABROAD),
    ]


@dataclass
class Clock:
    now: float = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Gpu:
    """Counts model loads; every load gets a fresh fake."""

    renderers: list[FakeRenderer] = field(default_factory=list)
    taggers: list[FakeTagger] = field(default_factory=list)
    fail_first_render: BaseException | None = None

    def renderer(self, model: str = "animagine") -> FakeRenderer:
        fail = self.fail_first_render if not self.renderers else None
        r = FakeRenderer(model, fail_on_call=1, exc=fail) if fail else FakeRenderer(model)
        self.renderers.append(r)
        return r

    def tagger(self) -> FakeTagger:
        t = FakeTagger()
        self.taggers.append(t)
        return t

    @property
    def renders(self) -> int:
        return sum(r.renders for r in self.renderers)


@dataclass
class Env:
    conn: sqlite3.Connection
    settings: Settings
    config: Path
    gpu: Gpu
    fm: list[bool | None]
    clock: Clock

    def service(self, **kw: object) -> WorkerService:
        return WorkerService(
            self.settings.paths.db_path,
            renderer_factory=self.gpu.renderer,
            tagger_factory=self.gpu.tagger,
            defaults_path=self.config,
            kits=load_kits(),
            fm_check=lambda: self.fm[0],
            clock=self.clock,
            conn=self.conn,
            **kw,  # type: ignore[arg-type]
        )

    @property
    def cid(self) -> int:
        career = load_career(self.conn)
        assert career is not None
        return career.id

    def drawn(self) -> list[tuple[int, str]]:
        """Pictures in the order they were drawn: (player, why)."""
        return [
            (r[0], r[1])
            for r in self.conn.execute("SELECT unique_id, reason FROM images ORDER BY id")
        ]

    def queued(self) -> list[tuple[int, str, str]]:
        return [
            tuple(r)
            for r in self.conn.execute(
                "SELECT unique_id, kind, group_name FROM jobs WHERE status = 'queued'"
                " ORDER BY tier, rank, id"
            )
        ]


@pytest.fixture
def env(config_file: Path) -> Iterator[Env]:
    conn = db.connect(load_settings(defaults_path=config_file).paths.db_path)
    # My league's cutoffs as numbers: these tests are not about league shares (Phase 16).
    settings = set_overrides(conn, MANUAL, config_file)
    run_sync(conn, snapshot(world()), settings, save_file="career.fm")
    yield Env(conn, settings, config_file, Gpu(), [False], Clock())
    conn.close()


def run_until_idle(svc: WorkerService, steps: int = 20) -> None:
    for _ in range(steps):
        if svc.step() > 0:
            return
    raise AssertionError("the worker never went idle")


def test_draws_my_club_first_publishes_and_goes_idle(env: Env) -> None:
    svc = env.service()
    assert svc.step() == 0.0
    assert env.drawn() == [(MINE_2, "initial")]  # the newest arrival at my club first
    assert svc.status().state == "running" and svc.status().done == 1
    run_until_idle(svc)
    # Queue order (spec section 2): my club newest first, then the rest of A, then B.
    assert env.drawn() == [
        (MINE_2, "initial"),
        (MINE, "initial"),
        (A_NEWGEN, "initial"),
        (B_NEWGEN, "initial"),
    ]
    status = svc.status()
    assert status.state == "idle" and status.done == 4 and status.models_loaded
    assert env.queued() == []
    pack = env.settings.paths.fm_user_dir / "graphics" / "facegen"
    assert len(list(pack.glob("*/*.png"))) == 4
    assert any("Reload Skin" in line for line in status.log)


def test_waits_while_fm27_runs_and_generate_now_overrides(env: Env) -> None:
    env.fm[0] = True
    svc = env.service()
    assert svc.step() == FM_POLL_SECONDS
    status = svc.status()
    assert status.state == "waiting" and "FM27 is running" in status.detail
    assert env.gpu.renderers == []  # the model was never loaded
    env.fm[0] = None  # cannot tell: the safe side is to wait as well
    assert svc.step() == FM_POLL_SECONDS and "cannot tell" in svc.status().detail

    svc.generate_now()
    run_until_idle(svc)
    assert {uid for uid, _ in env.drawn()} == {MINE, MINE_2, A_NEWGEN, B_NEWGEN}
    assert svc.status().generate_now is False  # lasts only until the queue is empty
    manual.generate(env.conn, env.cid, [STAR], env.settings)
    env.fm[0] = True
    assert svc.step() == FM_POLL_SECONDS  # a new request waits for FM27 again
    assert svc.status().models_loaded is False  # and the GPU was freed for the game
    assert env.gpu.renderers[0].closed == 1


def test_a_threshold_change_requeues_and_a_new_face_is_drawn_next(env: Env) -> None:
    """Phase 3 exit test, without the browser: both while the worker is running."""
    svc = env.service()
    svc.step()
    assert env.drawn() == [(MINE_2, "initial")]

    # Meanwhile, in the UI: lower group A's CA bar so the B newgen (CA 90) reaches A ...
    ui = db.connect(env.settings.paths.db_path)
    with transaction(ui):
        applied = set_overrides(ui, {"cutoffs.everywhere.a_ca": 85}, env.config)
        classify_and_plan(ui, env.cid, applied)
    # ... and ask for a new face for the player just drawn.
    manual.new_face(ui, env.cid, MINE_2, applied)
    ui.close()

    svc.step()
    assert env.drawn()[-1] == (MINE_2, "reroll")
    run_until_idle(svc)
    assert (B_NEWGEN, "initial") in env.drawn()
    group = env.conn.execute(
        "SELECT group_name FROM images WHERE unique_id = ? AND is_current = 1", (B_NEWGEN,)
    ).fetchone()[0]
    assert group == "a"


def test_hand_picked_players_of_any_group_are_drawn(env: Env) -> None:
    manual.generate(env.conn, env.cid, [STAR, B_NEWGEN], env.settings, "b")
    svc = env.service()
    svc.step()
    svc.step()
    assert env.drawn() == [(STAR, "initial"), (B_NEWGEN, "initial")]


def test_pause_frees_the_gpu_and_resume_goes_on(env: Env) -> None:
    svc = env.service()
    svc.step()
    svc.pause()
    assert svc.step() > 0
    status = svc.status()
    assert status.state == "paused" and not status.models_loaded
    assert env.gpu.renderers[0].closed == 1 and env.gpu.taggers[0].closed == 1
    assert len(env.drawn()) == 1
    svc.resume()
    svc.step()
    assert len(env.drawn()) == 2 and len(env.gpu.renderers) == 2  # loaded again


def test_an_error_pauses_the_worker_and_keeps_the_job(env: Env) -> None:
    env.gpu.fail_first_render = RuntimeError("CUDA out of memory")
    svc = env.service()
    svc.step()
    status = svc.status()
    assert status.state == "error" and status.paused
    assert "CUDA out of memory" in (status.last_error or "")
    assert status.reason == "job_error" and status.values["name"] == "Mine Two"
    assert "CUDA out of memory" in status.values["error"]
    assert env.drawn() == [] and env.queued()[0][0] == MINE_2
    error = env.conn.execute("SELECT error FROM jobs WHERE unique_id = ?", (MINE_2,)).fetchone()[0]
    assert error.startswith("interrupted: RuntimeError")
    assert svc.step() > 0 and env.drawn() == []  # stays paused
    svc.resume()
    svc.step()
    assert env.drawn() == [(MINE_2, "initial")]


def _illegal_access() -> RuntimeError:
    return RuntimeError("CUDA error: an illegal memory access was encountered")


def test_a_gpu_fault_stops_drawing_and_keeps_the_job(env: Env) -> None:
    env.gpu.fail_first_render = _illegal_access()
    svc = env.service()
    assert svc.step() > 0
    status = svc.status()
    assert status.state == "gpu-fault" and status.gpu_fault and status.paused
    assert "start it again" in status.detail and "trying again" not in status.detail
    assert status.reason == "gpu_fault" and "illegal memory access" in status.values["error"]
    assert "illegal memory access" in (status.last_error or "")
    assert not status.models_loaded
    assert env.drawn() == [] and env.queued()[0][0] == MINE_2  # back in the queue
    # Nothing draws again in this process, whatever is pressed.
    assert not svc.resume() and not svc.generate_now()
    assert svc.step() > 0 and env.drawn() == []
    status = svc.status()
    assert status.state == "gpu-fault" and not status.generate_now
    assert len(env.gpu.renderers) == 1  # no new model was loaded onto the dead card


def test_a_gpu_fault_whose_cleanup_fails_too_leaves_no_closed_model(env: Env) -> None:
    """The crash of 2026-11-13: freeing the card failed as well, and the next job was given
    the closed model (AttributeError: 'NoneType' object has no attribute 'text_encoder')."""
    env.gpu.fail_first_render = _illegal_access()
    svc = env.service()

    def broken_close() -> None:
        raise _illegal_access()

    original = env.gpu.renderer

    def renderer(model):  # type: ignore[no-untyped-def]
        r = original(model)
        r.close = broken_close  # type: ignore[method-assign]
        return r

    svc._renderer_factory = renderer
    assert svc.step() > 0  # does not raise
    assert svc.status().state == "gpu-fault"
    assert svc._worker is not None and not svc._worker.models_loaded
    assert any("could not free the GPU cleanly" in line for line in svc.status().log)


def test_a_gpu_fault_outside_a_job_is_not_retried(env: Env) -> None:
    svc = env.service()
    calls = []

    def step() -> float:
        calls.append(1)
        svc._stop.set()
        raise _illegal_access()

    svc.step = step  # type: ignore[method-assign]
    svc._loop()
    assert calls == [1] and svc.status().state == "gpu-fault"


def test_worker_errors_reach_the_log_file_with_their_traceback(env: Env) -> None:
    import logging

    from facegen.service import log_to_file

    path = log_to_file(env.settings.paths.data_dir)
    assert log_to_file(env.settings.paths.data_dir) == path  # asked twice: one handler
    try:
        env.gpu.fail_first_render = _illegal_access()
        env.service().step()
        text = path.read_text(encoding="utf-8")
    finally:
        log = logging.getLogger("facegen")
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()
    assert "Traceback" in text and "illegal memory access" in text and "GPU fault" in text
    assert text.count("Traceback") == 1


def test_a_job_cancelled_between_choosing_and_starting_is_skipped(env: Env) -> None:
    svc = env.service()
    original = svc._worker_for

    def cancel_first(conn: sqlite3.Connection):  # type: ignore[no-untyped-def]
        w = original(conn)
        runnable = w.runnable

        def runnable_then_cancel(f):  # type: ignore[no-untyped-def]
            ready = runnable(f)
            conn.execute("UPDATE jobs SET status = 'cancelled' WHERE id = ?", (ready[0].id,))
            return ready

        w.runnable = runnable_then_cancel  # type: ignore[method-assign]
        return w

    svc._worker_for = cancel_first  # type: ignore[method-assign]
    assert svc.step() == 0.0
    assert svc.status().state != "error" and env.drawn() == []


def test_an_empty_queue_frees_the_gpu_after_a_while(env: Env) -> None:
    svc = env.service()
    run_until_idle(svc)
    assert svc.status().models_loaded
    env.clock.now += IDLE_UNLOAD_SECONDS
    svc.step()
    assert not svc.status().models_loaded


def test_no_career_yet(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    svc = WorkerService(
        settings.paths.db_path,
        renderer_factory=FakeRenderer,
        tagger_factory=FakeTagger,
        defaults_path=config_file,
        kits=load_kits(),
    )
    assert svc.step() > 0
    assert svc.status().detail == "no career yet: sync a save"


def _wait_for(check, timeout: float = 10.0) -> None:  # type: ignore[no-untyped-def]
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_the_thread_draws_the_queue_and_stops(env: Env) -> None:
    svc = WorkerService(
        env.settings.paths.db_path,
        renderer_factory=env.gpu.renderer,
        tagger_factory=env.gpu.tagger,
        defaults_path=env.config,
        kits=load_kits(),
        fm_check=lambda: False,
    )
    svc.start()
    try:
        _wait_for(lambda: svc.status().state == "idle" and svc.status().done == 4)
    finally:
        svc.stop(timeout=10)
    assert svc.status().state == "stopped"
    with worker_lock(env.settings.paths.data_dir):  # released
        pass


def test_the_thread_waits_for_another_worker(env: Env) -> None:
    svc = env.service()
    with worker_lock(env.settings.paths.data_dir):
        svc.start()
        try:
            _wait_for(lambda: "another facegen worker" in svc.status().detail)
        finally:
            svc.stop(timeout=10)
    assert env.drawn() == []


class FakeSource:
    def __init__(self, snap: Snapshot) -> None:
        self.snap = snap

    def load(self, path: Path) -> Snapshot:
        return self.snap


def test_sync_now(env: Env) -> None:
    games = env.settings.paths.games_dir
    woken = []
    syncer = SyncService(
        env.settings.paths.db_path,
        lambda: FakeSource(snapshot(world())),
        defaults_path=env.config,
        on_done=lambda: woken.append(1),
    )
    syncer.run()
    assert syncer.status().error == f"No save found in {games}" and woken == [1]

    games.mkdir(parents=True)
    save = games / "career.fm"
    save.write_bytes(b"save")
    os.utime(save, (1_700_000_000, 1_700_000_000))
    assert syncer.start()
    syncer.join(10)
    status = syncer.status()
    assert not status.running and status.error is None
    assert status.message is not None and status.message.startswith("Synced career.fm")
    assert (env.settings.paths.snapshots_dir / "career.fm").is_file()  # read from a copy
    assert woken == [1, 1]
    assert status.notice is None  # a new save

    # FM27 has not saved since: Sync now says so rather than look as if it missed something.
    assert syncer.start()
    syncer.join(10)
    status = syncer.status()
    assert status.error is None and status.notice is not None
    assert status.notice.startswith("FM27 has not saved since career.fm was last synced")
    assert "save the game" in status.notice.lower()

    # A new save is news again.
    os.utime(save, (1_700_000_100, 1_700_000_100))
    assert syncer.start()
    syncer.join(10)
    assert syncer.status().notice is None


def test_an_error_outside_a_job_does_not_end_the_thread(env: Env) -> None:
    svc = WorkerService(
        env.settings.paths.db_path,
        renderer_factory=env.gpu.renderer,
        tagger_factory=env.gpu.tagger,
        defaults_path=env.config,
        kits=load_kits(),
        fm_check=lambda: False,
    )
    real_step = svc.step
    calls = []

    def flaky_step() -> float:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_step()

    svc.step = flaky_step  # type: ignore[method-assign]
    svc.start()
    try:
        _wait_for(lambda: "database is locked" in "\n".join(svc.status().log))
        svc.wake()  # skip the pause before the retry
        _wait_for(lambda: svc.status().state == "idle" and svc.status().done == 4)
    finally:
        svc.stop(timeout=10)


def test_an_idle_worker_says_what_it_holds_back(env: Env) -> None:
    """The user's report (2026-11-11): a group B job sat in the queue, the worker said
    "the queue is empty", and Generate now did nothing."""
    with transaction(env.conn):
        set_overrides(env.conn, {"schedule.auto_groups": ["a"]}, env.config)
    svc = env.service()
    run_until_idle(svc)
    status = svc.status()
    assert status.state == "idle"
    assert "1 job(s) in group B are not drawn automatically" in status.detail
    assert "Generate now" in status.detail
    assert (status.reason, status.values) == ("held", {"count": 1, "groups": "B"})
    svc.generate_now()
    svc.step()
    assert env.drawn()[-1] == (B_NEWGEN, "initial")
    run_until_idle(svc)
    assert svc.status().detail == "the queue is empty"
    assert svc.status().reason == "queue_empty"


# --- Phase 7b: in a worker process, the loop ends wherever the GPU must be let go of -----------


@dataclass
class RecordingLink:
    """Stands in for worker_process.ChildLink."""

    lines: list[str] = field(default_factory=list)
    statuses: list[WorkerStatus] = field(default_factory=list)

    def line(self, text: str) -> None:
        self.lines.append(text)

    def status(self, read) -> None:  # type: ignore[no-untyped-def]
        self.statuses.append(read())


def test_in_a_worker_process_a_gpu_fault_ends_the_loop(env: Env) -> None:
    env.gpu.fail_first_render = _illegal_access()
    link = RecordingLink()
    svc = env.service(link=link, generate_now=True)
    svc.step()
    assert svc.exit is not None and svc.exit.reason == "fault"
    assert svc.exit.job is not None and svc.exit.job.unique_id == MINE_2
    assert "illegal memory access" in (svc.exit.message or "")
    assert svc._stop.is_set()
    assert env.queued()[0][0] == MINE_2  # back in the queue for the next process
    status = svc.status()
    assert status.state == "gpu-fault" and "start it again" not in status.detail
    assert status.generate_now and not status.paused  # left for the next process
    assert any("GPU fault: " in line for line in link.lines)
    assert not any("drawing stopped" in line for line in link.lines)
    assert link.statuses[-1].state == "gpu-fault"


def test_in_a_worker_process_a_job_error_ends_the_loop(env: Env) -> None:
    env.gpu.fail_first_render = RuntimeError("CUDA out of memory")
    svc = env.service(link=RecordingLink())
    svc.step()
    assert svc.exit is not None and svc.exit.reason == "error"
    assert svc.exit.job is not None and svc.exit.job.unique_id == MINE_2
    assert "CUDA out of memory" in (svc.exit.message or "")
    assert env.queued()[0][0] == MINE_2


def test_in_a_worker_process_pause_publishes_and_ends_the_loop(env: Env) -> None:
    svc = env.service(link=RecordingLink())
    assert svc.step() == 0.0 and svc.exit is None
    svc.pause()
    svc.step()
    assert svc.exit is not None and svc.exit.reason == "paused"
    assert not svc.status().models_loaded
    pack = env.settings.paths.fm_user_dir / "graphics" / "facegen"
    assert len(list(pack.glob("*/*.png"))) == 1


def test_in_a_worker_process_freeing_the_gpu_ends_the_loop(env: Env) -> None:
    svc = env.service(link=RecordingLink())
    svc.step()
    env.fm[0] = True  # FM27 starts while the model is loaded
    svc.step()
    assert svc.exit is not None and svc.exit.reason == "freed"
    # The next process has loaded nothing: it waits for FM27 and stays.
    fresh = env.service(link=RecordingLink())
    assert fresh.step() == FM_POLL_SECONDS
    assert fresh.exit is None and fresh.status().state == "waiting"


def test_in_a_worker_process_an_empty_queue_ends_the_loop_after_a_while(env: Env) -> None:
    svc = env.service(link=RecordingLink())
    run_until_idle(svc)
    assert svc.exit is None
    env.clock.now += IDLE_UNLOAD_SECONDS
    svc.step()
    assert svc.exit is not None and svc.exit.reason == "freed"
    fresh = env.service(link=RecordingLink())
    env.clock.now += IDLE_UNLOAD_SECONDS
    fresh.step()
    fresh.step()
    assert fresh.exit is None  # nothing loaded, nothing to free


def test_in_a_worker_process_pause_ends_the_loop_with_no_career_or_another_worker(
    env: Env, config_file: Path
) -> None:
    settings = load_settings(defaults_path=config_file)
    empty = WorkerService(
        settings.paths.db_path.with_name("empty.db"),
        renderer_factory=FakeRenderer,
        tagger_factory=FakeTagger,
        defaults_path=config_file,
        kits=load_kits(),
        link=RecordingLink(),
    )
    empty.step()
    assert empty.exit is None
    empty.pause()
    empty.step()
    assert empty.exit is not None and empty.exit.reason == "paused"

    svc = env.service(link=RecordingLink())
    svc.pause()
    with worker_lock(env.settings.paths.data_dir):  # another worker holds the GPU
        svc.run()  # returns at once instead of waiting for it
    assert svc.exit is not None and svc.exit.reason == "paused"


def test_in_a_worker_process_a_crash_of_the_loop_says_so(env: Env) -> None:
    svc = env.service(link=RecordingLink())

    def broken(conn: sqlite3.Connection) -> None:
        raise sqlite3.DatabaseError("file is not a database")

    svc._requeue_stale = broken  # type: ignore[method-assign]
    svc.run()
    assert svc.exit is not None and svc.exit.reason == "crashed"
    assert "file is not a database" in (svc.exit.message or "")


def test_in_a_worker_process_the_link_hears_each_line_and_change_once(env: Env) -> None:
    link = RecordingLink()
    svc = env.service(link=link)
    svc.step()
    assert any(line.endswith("Mine Two (1002), initial A") for line in link.lines)
    assert all(line[2] == ":" for line in link.lines)  # time-stamped
    assert link.statuses[-1].done == 1
    # Without a link nothing is sent and nothing ends.
    plain = env.service()
    plain.step()
    plain.pause()
    plain.step()
    assert plain.exit is None and not plain._stop.is_set()


# --- Phase 4: the save watcher, career packs and the exit test -------------------------------


class SwitchableSource:
    """What the "game" has written: the snapshot to return, or an error to raise."""

    def __init__(self, snap: Snapshot | Exception) -> None:
        self.snap = snap
        self.loads = 0

    def load(self, path: Path) -> Snapshot:
        self.loads += 1
        if isinstance(self.snap, Exception):
            raise self.snap
        return self.snap


@dataclass
class Watch:
    env: Env
    source: SwitchableSource
    syncer: SyncService
    poller: PollerService
    log: list[str]

    def save(self, name: str = "career.fm", mtime: int = 1_800_000_000) -> Path:
        """The game writes a save."""
        games = self.env.settings.paths.games_dir
        games.mkdir(parents=True, exist_ok=True)
        path = games / name
        path.write_bytes(b"save " + str(mtime).encode())
        os.utime(path, (mtime, mtime))
        return path

    def poll_until_synced(self) -> None:
        """Look, wait out the settle time, look again."""
        assert self.poller.step() == 10.0, self.poller.status()
        assert self.poller.status().state == "settling"
        self.env.clock.now += 10
        self.poller.step()

    def runs(self) -> list[sqlite3.Row]:
        return self.env.conn.execute("SELECT * FROM sync_runs ORDER BY id").fetchall()


@pytest.fixture
def watch(env: Env) -> Watch:
    log: list[str] = []
    source = SwitchableSource(snapshot(world()))
    syncer = SyncService(
        env.settings.paths.db_path, lambda: source, defaults_path=env.config, log=log.append
    )
    poller = PollerService(
        env.settings.paths.db_path,
        syncer,
        defaults_path=env.config,
        settle_seconds=10.0,
        clock=env.clock,
        conn=env.conn,
    )
    return Watch(env, source, syncer, poller, log)


def mapped(env: Env, key: str) -> set[int]:
    import re

    path = env.settings.paths.fm_user_dir / "graphics" / "facegen" / key / "config.xml"
    return (
        {int(x) for x in re.findall(r'from="(\d+)"', path.read_text())} if path.exists() else set()
    )


def test_the_watcher_syncs_a_new_save_once_and_only_once(watch: Watch) -> None:
    assert watch.poller.step() == 60.0  # no save yet
    assert watch.poller.status().state == "no save"
    before = len(watch.runs())
    watch.save()
    watch.poll_until_synced()
    runs = watch.runs()
    assert len(runs) == before + 1
    assert (runs[-1]["trigger"], runs[-1]["save_mtime"], runs[-1]["error"]) == (
        "poller",
        1_800_000_000.0,
        None,
    )
    assert watch.poller.status().state == "synced"
    assert any(line.startswith("Synced career.fm") for line in watch.log)
    # Nothing new: later looks do not read it again, nor does a restarted watcher.
    for _ in range(3):
        assert watch.poller.step() == 60.0
    fresh = PollerService(
        watch.env.settings.paths.db_path,
        watch.syncer,
        defaults_path=watch.env.config,
        clock=watch.env.clock,
        conn=watch.env.conn,
    )
    fresh.step()
    assert len(watch.runs()) == before + 1 and watch.source.loads == 1
    assert "is synced" in fresh.status().detail
    # The next save is read again.
    watch.save(mtime=1_800_000_600)
    watch.poll_until_synced()
    assert len(watch.runs()) == before + 2


def test_the_watcher_leaves_a_save_alone_while_it_is_being_written(watch: Watch) -> None:
    watch.save(mtime=1_800_000_000)
    assert watch.poller.step() == 10.0
    watch.env.clock.now += 10
    watch.save(mtime=1_800_000_005)  # still being written
    assert watch.poller.step() == 10.0 and watch.source.loads == 0
    watch.env.clock.now += 10
    watch.poller.step()
    assert watch.source.loads == 1


def test_an_unsupported_build_is_reported_once_and_the_queue_is_kept(watch: Watch) -> None:
    from facegen.datasource.base import UnsupportedSaveError

    queued = watch.env.queued()
    watch.source.snap = UnsupportedSaveError(
        "FM27 build 27.2.0+999 cannot be read by fmsave 0.11.0", build="27.2.0+999"
    )
    watch.save()
    watch.poll_until_synced()
    run = watch.runs()[-1]
    assert (run["unsupported"], run["build"], run["save_mtime"]) == (1, "27.2.0+999", 1.8e9)
    status = watch.syncer.status()
    assert status.unsupported and "27.2.0+999" in (status.error or "")
    assert any("bump the pin" in line for line in watch.log)
    assert watch.poller.status().state == "error"
    # Not tried again until the game writes a new save.
    watch.poller.step()
    assert watch.source.loads == 1
    assert "its game build is not supported" in watch.poller.status().detail
    assert watch.env.queued() == queued  # existing jobs and pictures stay


def test_a_save_still_being_copied_is_tried_again(
    watch: Watch, monkeypatch: pytest.MonkeyPatch
) -> None:
    from facegen.datasource import snapshots
    from facegen.datasource.base import SaveBusyError

    real = snapshots.copy_snapshot
    calls: list[int] = []

    def busy_once(*args: object, **kw: object) -> Path:
        calls.append(1)
        if len(calls) == 1:
            raise SaveBusyError("career.fm kept changing while it was copied; is FM27 saving?")
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(snapshots, "copy_snapshot", busy_once)
    watch.save()
    watch.poll_until_synced()
    assert watch.runs()[-1]["save_mtime"] is None  # recorded, but not "dealt with"
    assert "trying again" in watch.poller.status().detail
    watch.poll_until_synced()
    assert watch.runs()[-1]["error"] is None and watch.source.loads == 1


def test_the_watcher_waits_for_a_sync_started_by_hand(watch: Watch) -> None:
    watch.save()
    watch.poller.step()
    watch.env.clock.now += 10
    assert watch.syncer._claim("sync now")  # "Sync now" is running
    assert watch.poller.step() == 10.0
    assert watch.source.loads == 0 and "by hand" in watch.poller.status().detail


def test_a_newer_save_of_another_career_switches_the_pack(watch: Watch) -> None:
    env = watch.env
    svc = env.service()
    run_until_idle(svc)
    one = load_career(env.conn)
    assert one is not None and mapped(env, one.key) == {MINE, MINE_2, A_NEWGEN, B_NEWGEN}

    # The user starts or loads another career and saves it.
    other = [player(5001, name="Other Mine", age=24), player(5002, name="Other Two", age=31)]
    watch.source.snap = snapshot(other, career="two", file_name="two.fm")
    watch.save("two.fm")
    watch.poll_until_synced()
    two = load_career(env.conn)
    assert two is not None and two.key != one.key
    assert mapped(env, one.key) == set()  # FM27 no longer shows career one's pictures
    assert any("unmapped" in line for line in watch.log)

    run_until_idle(svc)
    assert any("Now working on career" in line for line in svc.status().log)
    assert mapped(env, two.key) == {5001, 5002} and mapped(env, one.key) == set()
    pack_one = env.settings.paths.fm_user_dir / "graphics" / "facegen" / one.key
    assert (pack_one / f"{MINE}.png").is_file()  # kept: switching back is a config change

    # Back to career one: its save is written again.
    watch.source.snap = snapshot(world())
    watch.save("career.fm", mtime=1_800_001_000)
    watch.poll_until_synced()
    assert load_career(env.conn).key == one.key  # type: ignore[union-attr]
    assert mapped(env, one.key) == {MINE, MINE_2, A_NEWGEN, B_NEWGEN}
    assert mapped(env, two.key) == set()


def test_exit_test_play_save_quit_and_the_pictures_are_there(watch: Watch) -> None:
    """Phase 4 exit test, without the game: FM27 runs, the user saves after youth intake
    and quits; with no manual step the new players are drawn (the intake first) and
    published before the next launch."""
    from datetime import date

    env = watch.env
    svc = env.service()
    run_until_idle(svc)  # the career so far is drawn
    cid = env.cid
    env.fm[0] = True  # the user is playing

    youth = [NEWGEN + 50 + i for i in range(3)]
    star = NEWGEN + 60  # a new newgen elsewhere who now qualifies for B
    later = [
        *world(),
        *(player(uid, name=f"Youth {uid}", ca=40, pa=120, age=15) for uid in youth),
        player(star, name="Rival Star", ca=120, pa=160, club_id=RIVAL),
    ]
    watch.source.snap = snapshot(later, game_date=date(2026, 3, 20))
    watch.save(mtime=1_800_000_000)
    watch.poll_until_synced()  # synced while the game runs
    assert any("3 new at my club, drawn first" in line for line in watch.log)
    assert svc.step() == FM_POLL_SECONDS and svc.status().state == "waiting"
    drawn_before = len(env.drawn())

    env.fm[0] = False  # the user quits
    run_until_idle(svc)
    new = env.drawn()[drawn_before:]
    assert {uid for uid, _ in new[:3]} == set(youth)  # the intake first
    assert new[3] == (star, "initial")
    assert mapped(env, load_career(env.conn).key) >= {*youth, star}  # type: ignore[union-attr]
    assert env.cid == cid and env.queued() == []


def test_serve_publishes_once_at_start(env: Env) -> None:
    """Pictures drawn but not published (or forgotten by the 3:4 migration) are published by
    the first idle pass, without waiting for a new picture."""
    run_until_idle(env.service())
    key = load_career(env.conn).key  # type: ignore[union-attr]
    pack = env.settings.paths.fm_user_dir / "graphics" / "facegen" / key
    for png in pack.glob("*.png"):
        png.unlink()
    env.conn.execute("UPDATE images SET published_path = NULL")
    renders = env.gpu.renders
    again = env.service()
    assert again.step() > 0 and again.status().state == "idle"
    assert len(list(pack.glob("*.png"))) == 4 and env.gpu.renders == renders  # nothing drawn


# --- Phase 5: updates for tracked players -----------------------------------------------------


def test_exit_test_transfer_and_release_are_redrawn_with_no_manual_step(watch: Watch) -> None:
    """Phase 5 exit test, without the game. A group A player transferred to my club gets a
    new-kit picture from the same sheet, with the old picture as a reference; an academy
    newgen who is released stays tracked in group B and is re-drawn at B quality in the kit
    of the next club they join, and not while a free agent."""
    import json

    from factories import MY_CLUB, RIVAL_2
    from PIL import Image

    env = watch.env
    academy = NEWGEN + 70
    youngster = player(academy, name="Academy", ca=60, pa=130, age=18, home_grown=(MY_CLUB,))

    def save(players: list, mtime: int) -> None:
        watch.source.snap = snapshot(players)
        watch.save(mtime=mtime)
        watch.poll_until_synced()

    svc = env.service()
    save([*world(), youngster], 1_800_000_000)
    run_until_idle(svc)
    first = {uid: img for uid, img in current_pictures(env).items()}
    assert first[A_NEWGEN]["club_id"] == ABROAD and first[academy]["group_name"] == "a"

    # A_NEWGEN signs for my club; the academy player is released.
    moved = [p for p in world() if p.unique_id != A_NEWGEN]
    signed = player(A_NEWGEN, name="A Newgen", ca=160, pa=170, club_id=MY_CLUB)
    released = player(academy, name="Academy", ca=60, pa=130, age=18, club_id=None,
                      home_grown=(MY_CLUB,))  # fmt: skip
    save([*moved, signed, released], 1_800_000_100)
    run_until_idle(svc)
    now = current_pictures(env)
    kit = now[A_NEWGEN]
    assert (kit["reason"], kit["group_name"], kit["club_id"]) == ("kit", "a", MY_CLUB)
    assert kit["traits_version"] == first[A_NEWGEN]["traits_version"] == 1  # the same character
    reference = json.loads(kit["render"])["reference"]
    assert reference["used"] and reference["image_id"] == first[A_NEWGEN]["id"]
    row = env.conn.execute(
        "SELECT group_name, tracked FROM players WHERE unique_id = ?", (academy,)
    ).fetchone()
    assert tuple(row) == ("b", 1)  # dropped to B, still tracked
    assert now[academy]["id"] == first[academy]["id"]  # a free agent: no kit to draw yet

    # The academy player joins a club in my league.
    joined = player(academy, name="Academy", ca=60, pa=130, age=18, club_id=RIVAL_2,
                    home_grown=(MY_CLUB,))  # fmt: skip
    save([*moved, signed, joined], 1_800_000_200)
    run_until_idle(svc)
    redrawn = current_pictures(env)[academy]
    assert (redrawn["reason"], redrawn["group_name"], redrawn["club_id"]) == ("kit", "b", RIVAL_2)
    assert redrawn["traits_version"] == 1
    assert json.loads(redrawn["render"])["reference"] is None  # not at my club: no reference
    key = load_career(env.conn).key  # type: ignore[union-attr]
    with Image.open(
        env.settings.paths.fm_user_dir / "graphics" / "facegen" / key / f"{academy}.png"
    ) as im:
        assert im.size == (576, 768)  # group B's size, 3:4
    assert academy in mapped(env, key) and env.queued() == []


def current_pictures(env: Env) -> dict[int, sqlite3.Row]:
    return {
        r["unique_id"]: r for r in env.conn.execute("SELECT * FROM images WHERE is_current = 1")
    }


def test_one_encoder_trip_serves_the_jobs_drawn_one_step_at_a_time(env: Env) -> None:
    svc = env.service()
    svc.step()
    (renderer,) = env.gpu.renderers
    assert [len(trip) for trip in renderer.trips] == [4]  # this job and the three queued
    manual.generate(env.conn, env.cid, [STAR], env.settings)  # asked for meanwhile
    run_until_idle(svc)
    assert len(env.drawn()) == 5
    assert len(renderer.trips) == 2  # only the new request needed another trip
    assert renderer.trips[1][0] == renderer.prompts[3]  # drawn next: manual requests come first


def test_missing_models_are_a_setup_state_not_an_error(env: Env) -> None:
    """Phase 10b: with the models not downloaded the worker waits and loads nothing; once the
    download has finished it draws."""
    present = [False]
    asked: list[str] = []
    svc = env.service(models_ready=lambda model: asked.append(model) or present[0])
    assert svc.step() == FM_POLL_SECONDS
    status = svc.status()
    assert status.state == "setup" and "not downloaded" in status.detail
    assert "4 job(s) wait" in status.detail
    assert (status.reason, status.values) == ("models_missing", {"count": 4})
    assert status.last_error is None and not status.paused
    assert env.gpu.renderers == [] and env.drawn() == []
    present[0] = True
    assert svc.step() == 0.0
    assert env.drawn() == [(MINE_2, "initial")]
    assert set(asked) == {"animagine"}  # the selected model's files (Phase 15e)


def test_the_models_check_follows_the_selected_model(env: Env) -> None:
    """Phase 15e: the files asked about are those of `style.model` as saved before the job."""
    asked: list[str] = []
    svc = env.service(models_ready=lambda model: asked.append(model) or model == "animagine")
    set_overrides(env.conn, {"style.model": "realvis"}, env.config)
    assert svc.step() == FM_POLL_SECONDS
    assert asked == ["realvis"] and svc.status().reason == "models_missing"
    assert env.gpu.renderers == [] and env.drawn() == []
    set_overrides(env.conn, {"style.model": "animagine"}, env.config)
    assert svc.step() == 0.0
    assert asked == ["realvis", "animagine"] and len(env.drawn()) == 1


def test_an_empty_queue_does_not_ask_for_the_models(env: Env) -> None:
    asked = []
    svc = env.service(models_ready=lambda model: asked.append(1) or True)
    run_until_idle(svc)
    n = len(asked)
    assert svc.step() > 0 and svc.status().state == "idle"
    assert len(asked) == n  # nothing to draw: nothing asked


# --- Phase 12c: the state as a reason code the pages word in their language ----------------


def test_each_state_says_why_as_a_code_with_its_values_and_since_when(env: Env) -> None:
    env.fm[0] = True
    svc = env.service()
    svc.step()
    first = svc.status()
    assert (first.state, first.reason, first.values) == ("waiting", "fm_running", {"count": 4})
    assert first.since is not None and first.since.endswith("+00:00")
    svc.step()
    assert svc.status().since == first.since  # the same wait goes on: the same start
    env.fm[0] = None
    svc.step()
    assert svc.status().reason == "fm_unknown"
    svc.pause()
    svc.step()
    assert (svc.status().state, svc.status().reason) == ("paused", "paused")


def test_a_job_is_drawing_with_its_name_kind_and_group_then_next_job(env: Env) -> None:
    link = RecordingLink()
    svc = env.service(link=link)
    svc.step()
    drawing = [s for s in link.statuses if s.reason == "drawing"]
    assert drawing and drawing[0].values == {"name": "Mine Two", "kind": "initial", "group": "A"}
    assert drawing[0].current is not None and drawing[0].since is not None
    assert svc.status().reason == "next_job"


def test_every_reason_the_worker_gives_is_known() -> None:
    """Each `reason="..."` the worker and its process proxy set is in WORKER_REASONS, which the
    message catalog's contract test words in both languages."""
    used: set[str] = set()
    for name in ("worker_service.py", "worker_process.py"):
        source = (ROOT / "src" / "facegen" / "runtime" / name).read_text(encoding="utf-8")
        used |= set(re.findall(r'reason="([a-z_]+)"', source))
        used |= set(re.findall(r'_mark\(\s*"[a-z-]+",\s*[^,]+,\s*"([a-z_]+)"', source))
        used |= set(re.findall(r'"fm_running" if running else "([a-z_]+)"', source))
    assert used and used <= set(WORKER_REASONS), sorted(used - set(WORKER_REASONS))
    assert {"fm_running", "fm_unknown", "starting", "restarting", "gave_up"} <= used
