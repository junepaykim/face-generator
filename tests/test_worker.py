"""The job runner with a fake renderer and tagger: no GPU, no save."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from factories import MY_CLUB, NEWGEN, RIVAL, player, snapshot
from fakes import FakeRenderer, FakeTagger
from PIL import Image

from facegen.gallery import write_gallery
from facegen.kits import load_kits
from facegen.prompt import CONTENT_NEGATIVE, YOUNG_NEGATIVE
from facegen.settings import Paths, Settings
from facegen.sync import run_sync
from facegen.worker import (
    MAX_ATTEMPTS,
    JobFilter,
    Worker,
    WorkerBusy,
    load_career,
    worker_lock,
)

OLD_HAND = 1001  # my club, 29: W08
KEEPER = 1002  # my club goalkeeper, 17: W02 (under 18)
STAR = NEWGEN + 1  # group A elsewhere
LEAGUE_B = NEWGEN + 2  # group B in my league

MINE = JobFilter(groups=frozenset({"a"}), my_club_only=True)


def world() -> list:
    return [
        player(OLD_HAND, name="Old Hand", age=29),
        player(KEEPER, name="Keeper", age=17, goalkeeper=True),
        player(STAR, name="Star", ca=160, pa=180, club_id=RIVAL, age=19),
        player(LEAGUE_B, name="League B", ca=90, pa=152, club_id=RIVAL, age=20),
    ]


@dataclass
class Env:
    conn: sqlite3.Connection
    settings: Settings
    fm: Path
    data: Path

    @property
    def pack(self) -> Path:
        career = load_career(self.conn)
        assert career is not None
        return self.fm / "graphics" / "facegen" / career.key

    def worker(
        self,
        renderer: FakeRenderer | None = None,
        tagger: FakeTagger | None = None,
        fm: bool | None = False,
        settings: Settings | None = None,
    ) -> tuple[Worker, FakeRenderer, FakeTagger]:
        r, t = renderer or FakeRenderer(), tagger or FakeTagger()
        career = load_career(self.conn)
        assert career is not None
        w = Worker(
            self.conn,
            settings or self.settings,
            load_kits(),
            career,
            renderer_factory=lambda: r,
            tagger_factory=lambda: t,
            fm_check=lambda: fm,
            log=lambda line: None,
        )
        return w, r, t

    def jobs(self) -> dict[int, str]:
        return {
            r["unique_id"]: r["status"]
            for r in self.conn.execute("SELECT unique_id, status FROM jobs ORDER BY id")
        }

    def current(self, uid: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM images WHERE unique_id = ? AND is_current = 1", (uid,)
        ).fetchone()


@pytest.fixture
def env(conn: sqlite3.Connection, settings: Settings, tmp_path: Path) -> Env:
    s = replace(settings, paths=Paths(fm_user_dir=tmp_path / "fm", data_dir=tmp_path / "data"))
    run_sync(conn, snapshot(world()), s, save_file="career.fm")
    return Env(conn, s, tmp_path / "fm", tmp_path / "data")


def test_my_club_run_generates_publishes_and_a_rerun_changes_nothing(env: Env) -> None:
    w, r, _t = env.worker()
    summary = w.run(MINE)
    assert {o.job.unique_id for o in summary.done} == {OLD_HAND, KEEPER}
    assert summary.failed == [] and summary.stopped is None
    # Group A: three candidates and a detail pass each.
    assert (r.renders, r.details) == (6, 2)
    assert env.jobs() == {OLD_HAND: "done", KEEPER: "done", STAR: "queued", LEAGUE_B: "queued"}

    for uid in (OLD_HAND, KEEPER):
        img = env.current(uid)
        assert img is not None and img["model"] == "fake/animagine" and img["group_name"] == "a"
        assert img["prompt"].startswith("1girl, solo, safe, ")
        assert img["negative_prompt"].startswith(CONTENT_NEGATIVE)
        master = env.data / img["master_path"]
        assert master.is_file() and master.parent.name == str(uid)
        assert master.parent.parent.name == env.pack.name  # data/masters/<career>/<uid>/
        with Image.open(master) as im:
            assert im.size == (96, 96)  # the detailed picture is the master
            assert "1girl" in im.info["facegen"]
        with Image.open(env.pack / f"{uid}.png") as im:
            assert im.size == (1024, 1024)  # group A's published size
        render = json.loads(img["render"])
        assert render["detail"]["used"] and render["my_club"]
        check = json.loads(img["check_result"])
        assert check["passed"] and len(check["candidates"]) == 3
    keeper = env.current(KEEPER)
    assert keeper["age_band"] == "W02" and YOUNG_NEGATIVE in keeper["negative_prompt"]
    assert "goalkeeper" in keeper["prompt"]
    assert env.current(OLD_HAND)["age_band"] == "W08"
    assert YOUNG_NEGATIVE not in env.current(OLD_HAND)["negative_prompt"]
    assert summary.publish is not None and summary.publish.written == 2
    xml = (env.pack / "config.xml").read_text()
    assert f'from="{OLD_HAND}"' in xml and f'from="{KEEPER}"' in xml and f'from="{STAR}"' not in xml

    # Re-running changes nothing: no jobs, no rewritten files, no new queue entries.
    before = {p.name: p.stat().st_mtime_ns for p in env.pack.iterdir()}
    again, r2, _ = env.worker()
    summary2 = again.run(MINE)
    assert summary2.done == [] and r2.renders == 0
    assert summary2.publish is not None and not summary2.publish.changed
    assert {p.name: p.stat().st_mtime_ns for p in env.pack.iterdir()} == before
    report = run_sync(env.conn, snapshot(world()), env.settings, save_file="career.fm")
    assert not report.enqueued and not report.updated and not report.cancelled


def test_same_job_gives_the_same_seeds_and_prompt(env: Env, tmp_path: Path) -> None:
    w, r, _ = env.worker()
    w.run(MINE, limit=1, publish=False)
    first = (r.calls, r.prompts[0])
    # Forget the picture and the sheet, then redo: same career key, same player.
    env.conn.execute("DELETE FROM images")
    env.conn.execute("DELETE FROM traits")
    env.conn.execute("UPDATE jobs SET status = 'queued'")
    w2, r2, _ = env.worker()
    w2.run(MINE, limit=1, publish=False)
    assert (r2.calls, r2.prompts[0]) == first


def test_the_best_passing_candidate_is_chosen(env: Env) -> None:
    w, r, _t = env.worker(tagger=FakeTagger(["sensitive", "pass", "good"]))
    outcome = w.run(MINE, limit=1).done[0]
    img = env.current(outcome.job.unique_id)
    check = json.loads(img["check_result"])
    assert check["chosen"]["index"] == 2
    assert img["seed"] == r.calls[2][1]
    assert [c["passed"] for c in check["candidates"]] == [False, True, True]
    assert check["candidates"][0]["failures"][0].startswith("rated sensitive")


def test_a_round_with_no_pass_retries_with_new_seeds(env: Env) -> None:
    w, r, _t = env.worker(tagger=FakeTagger(["sensitive", "two", "sensitive", "pass"]))
    outcome = w.run(MINE, limit=1).done[0]
    seeds = [s for kind, s in r.calls if kind == "render"]
    assert len(seeds) == 6 and len(set(seeds)) == 6
    check = json.loads(env.current(outcome.job.unique_id)["check_result"])
    assert check["attempts"] == 2 and check["chosen"] == {
        "attempt": 1,
        "index": 0,
        "seed": seeds[3],
    }


def test_three_failed_rounds_go_to_review_and_nothing_is_published(env: Env) -> None:
    w, r, _ = env.worker(tagger=FakeTagger(["sensitive"] * (3 * MAX_ATTEMPTS)))
    summary = w.run(MINE, limit=1)
    assert len(summary.failed) == 1 and summary.done == []
    uid = summary.failed[0].job.unique_id
    assert r.renders == 3 * MAX_ATTEMPTS and r.details == 0
    assert env.jobs()[uid] == "failed"
    assert env.current(uid) is None
    rejected = env.conn.execute("SELECT * FROM images WHERE unique_id = ?", (uid,)).fetchone()
    assert rejected["is_current"] == 0 and rejected["published_path"] is None
    assert (env.data / rejected["master_path"]).is_file()
    assert not json.loads(rejected["check_result"])["passed"]
    assert not (env.pack / f"{uid}.png").exists()
    error = env.conn.execute("SELECT error FROM jobs WHERE unique_id = ?", (uid,)).fetchone()[0]
    assert "rated sensitive" in error
    # A re-sync does not queue the same failed job again.
    report = run_sync(env.conn, snapshot(world()), env.settings, save_file="career.fm")
    assert not report.enqueued


def test_a_detail_pass_that_fails_the_check_keeps_the_original(env: Env) -> None:
    w, r, _ = env.worker(tagger=FakeTagger(["pass", "pass", "pass", "sensitive"]))
    outcome = w.run(MINE, limit=1).done[0]
    img = env.current(outcome.job.unique_id)
    render = json.loads(img["render"])
    assert r.details == 1 and render["detail"]["used"] is False
    with Image.open(env.data / img["master_path"]) as im:
        assert im.size == (64, 64)  # the undetailed winner


def test_group_b_draws_one_candidate_without_detail(env: Env) -> None:
    w, r, _ = env.worker()
    summary = w.run(JobFilter(groups=frozenset({"b"})))
    assert [o.job.unique_id for o in summary.done] == [LEAGUE_B]
    assert (r.renders, r.details) == (1, 0)
    with Image.open(env.pack / f"{LEAGUE_B}.png") as im:
        assert im.size == (768, 768)
    assert json.loads(env.current(LEAGUE_B)["render"])["my_club"] is False


def test_fm_running_stops_before_any_job(env: Env) -> None:
    loaded = []
    for state in (True, None):
        career = load_career(env.conn)
        assert career is not None
        w = Worker(
            env.conn,
            env.settings,
            load_kits(),
            career,
            renderer_factory=lambda: loaded.append(1) or FakeRenderer(),
            tagger_factory=FakeTagger,
            fm_check=lambda s=state: s,
            log=lambda line: None,
        )
        summary = w.run(MINE)
        assert summary.stopped and summary.done == []
    assert loaded == []  # the model was never loaded
    assert set(env.jobs().values()) == {"queued"}
    # "Generate now" overrides.
    w, _r, _ = env.worker(fm=True)
    assert len(w.run(MINE, generate_now=True).done) == 2


def test_an_interrupted_job_goes_back_to_the_queue(env: Env) -> None:
    w, r, _ = env.worker(renderer=FakeRenderer(fail_on_call=2))
    with pytest.raises(KeyboardInterrupt):
        w.run(MINE)
    statuses = env.jobs()
    assert statuses[OLD_HAND] == "queued" and statuses[KEEPER] == "queued"
    err = env.conn.execute("SELECT error FROM jobs WHERE error IS NOT NULL").fetchone()[0]
    assert err.startswith("interrupted")
    assert env.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
    # The next run redoes it from the same seeds.
    w2, r2, _ = env.worker()
    assert len(w2.run(MINE).done) == 2
    assert r2.calls[0] == r.calls[0]


def test_a_job_left_running_by_a_dead_worker_is_redone(env: Env) -> None:
    env.conn.execute("UPDATE jobs SET status = 'running' WHERE unique_id = ?", (OLD_HAND,))
    w, _, _ = env.worker()
    assert {o.job.unique_id for o in w.run(MINE).done} == {OLD_HAND, KEEPER}


def test_only_one_worker_at_a_time(env: Env) -> None:
    w, _, _ = env.worker()
    with worker_lock(env.data), pytest.raises(WorkerBusy):
        w.run(MINE)


def test_redraws_that_need_a_reference_wait_for_phase_5(env: Env) -> None:
    w, _, _ = env.worker()
    w.run(MINE)
    env.conn.execute(
        "INSERT INTO jobs (career_id, unique_id, kind, group_name, tier, rank, status,"
        " target_club_id, target_band, reason, created_at, updated_at)"
        " SELECT career_id, unique_id, 'kit', 'a', 1, 0, 'queued', ?, 'W08', 'test', '', ''"
        " FROM players WHERE unique_id = ?",
        (MY_CLUB, OLD_HAND),
    )
    w2, r2, _ = env.worker()
    summary = w2.run(MINE)
    assert summary.done == [] and summary.waiting_for_reference == 1 and r2.renders == 0


def test_preview_draws_and_stores_nothing(env: Env) -> None:
    w, r, _ = env.worker()
    previews = w.preview(MINE)
    assert {p.job.unique_id for p in previews} == {OLD_HAND, KEEPER}
    assert all(p.prompt is not None and "safe" in p.prompt.positive for p in previews)
    assert r.calls == []
    assert env.conn.execute("SELECT COUNT(*) FROM traits").fetchone()[0] == 0
    # The preview shows the sheet the run then stores.
    w.run(MINE, publish=False)
    stored = {row[0]: row[1] for row in env.conn.execute("SELECT unique_id, base_seed FROM traits")}
    assert {p.job.unique_id: p.sheet.base_seed for p in previews} == stored


def test_no_publish_leaves_fm_untouched(env: Env) -> None:
    w, _, _ = env.worker()
    summary = w.run(MINE, publish=False)
    assert len(summary.done) == 2 and summary.publish is None
    assert not env.fm.exists()
    report = w.publish()
    assert report.written == 2 and (env.pack / "config.xml").is_file()


def test_gallery_lists_pictures_and_review(env: Env) -> None:
    w, _, _ = env.worker(tagger=FakeTagger(["sensitive"] * 9))
    w.run(MINE)
    career = load_career(env.conn)
    assert career is not None
    page = write_gallery(env.conn, career.id, career.key, "Test career", env.data)
    text = page.read_text()
    assert page.parent == env.data / "masters" / career.key
    assert "Old Hand" in text and "Keeper" in text
    assert "1 current pictures, 1 sent to review" in text
    assert 'src="masters/' not in text  # links are relative to the page
