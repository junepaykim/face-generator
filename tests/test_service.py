"""`facegen serve`'s background work: the worker loop (driven step by step) and "sync now"."""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from factories import ABROAD, NEWGEN, RIVAL, player, snapshot
from fakes import FakeRenderer, FakeTagger

from facegen import db, manual
from facegen.datasource.records import Snapshot
from facegen.db import transaction
from facegen.kits import load_kits
from facegen.service import FM_POLL_SECONDS, IDLE_UNLOAD_SECONDS, SyncService, WorkerService
from facegen.settings import Settings, load_settings, set_overrides
from facegen.sync import classify_and_plan, run_sync
from facegen.worker import load_career, worker_lock

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

    def renderer(self) -> FakeRenderer:
        fail = self.fail_first_render if not self.renderers else None
        r = FakeRenderer(fail_on_call=1, exc=fail) if fail else FakeRenderer()
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
    settings = load_settings(defaults_path=config_file)
    conn = db.connect(settings.paths.db_path)
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
    assert {uid for uid, _ in env.drawn()} == {MINE, MINE_2, A_NEWGEN}
    status = svc.status()
    assert status.state == "idle" and status.done == 3 and status.models_loaded
    # Group B is not drawn automatically yet (schedule.auto_groups, Phase 4).
    assert env.queued() == [(B_NEWGEN, "initial", "b")]
    pack = env.settings.paths.fm_user_dir / "graphics" / "facegen"
    assert len(list(pack.glob("*/*.png"))) == 3
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
    # The whole queue, group B included, which is not drawn automatically yet.
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
        applied = set_overrides(ui, {"groups.a.ca_above": 85}, env.config)
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
    assert env.drawn() == [] and env.queued()[0][0] == MINE_2
    error = env.conn.execute("SELECT error FROM jobs WHERE unique_id = ?", (MINE_2,)).fetchone()[0]
    assert error.startswith("interrupted: RuntimeError")
    assert svc.step() > 0 and env.drawn() == []  # stays paused
    svc.resume()
    svc.step()
    assert env.drawn() == [(MINE_2, "initial")]


def test_a_job_cancelled_between_choosing_and_starting_is_skipped(env: Env) -> None:
    svc = env.service()
    original = svc._worker_for

    def cancel_first(conn: sqlite3.Connection):  # type: ignore[no-untyped-def]
        w = original(conn)
        runnable = w.runnable

        def runnable_then_cancel(f):  # type: ignore[no-untyped-def]
            ready, waiting = runnable(f)
            conn.execute("UPDATE jobs SET status = 'cancelled' WHERE id = ?", (ready[0].id,))
            return ready, waiting

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
        _wait_for(lambda: svc.status().state == "idle" and svc.status().done == 3)
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
        _wait_for(lambda: svc.status().state == "idle" and svc.status().done == 3)
    finally:
        svc.stop(timeout=10)


def test_an_idle_worker_says_what_it_holds_back(env: Env) -> None:
    """The user's report (2026-11-11): a group B job sat in the queue, the worker said
    "the queue is empty", and Generate now did nothing."""
    svc = env.service()
    run_until_idle(svc)
    status = svc.status()
    assert status.state == "idle"
    assert "1 job(s) in group B are not drawn automatically yet" in status.detail
    assert "Generate now" in status.detail
    svc.generate_now()
    svc.step()
    assert env.drawn()[-1] == (B_NEWGEN, "initial")
    run_until_idle(svc)
    assert svc.status().detail == "the queue is empty"
