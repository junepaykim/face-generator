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
from facegen.service import (
    FM_POLL_SECONDS,
    IDLE_UNLOAD_SECONDS,
    PollerService,
    SyncService,
    WorkerService,
)
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
    svc.generate_now()
    svc.step()
    assert env.drawn()[-1] == (B_NEWGEN, "initial")
    run_until_idle(svc)
    assert svc.status().detail == "the queue is empty"


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
    from facegen import poller as poller_module
    from facegen.datasource.base import SaveBusyError

    real = poller_module.copy_snapshot
    calls: list[int] = []

    def busy_once(*args: object, **kw: object) -> Path:
        calls.append(1)
        if len(calls) == 1:
            raise SaveBusyError("career.fm kept changing while it was copied; is FM27 saving?")
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(poller_module, "copy_snapshot", busy_once)
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
