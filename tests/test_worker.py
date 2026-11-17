"""The job runner with a fake renderer and tagger: no GPU, no save."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from factories import GAME_DATE, MY_CLUB, NEWGEN, PLAIN_PERSONALITY, RIVAL, player, snapshot
from fakes import FakeRenderer, FakeTagger
from PIL import Image

from facegen.gallery import write_gallery
from facegen.kits import load_kits
from facegen.prompt import CONTENT_NEGATIVE, YOUNG_NEGATIVE
from facegen.settings import Paths, PersonalitySettings, Settings
from facegen.sync import run_sync
from facegen.traits import load_sheet, personality_to_json, sample_sheet
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
    s = replace(
        settings,
        paths=Paths(
            fm_user_dir=tmp_path / "fm",
            data_dir=tmp_path / "data",
            fm_install_dir=tmp_path / "fm-install",
        ),
    )
    run_sync(conn, snapshot(world()), s, save_file="career.fm")
    return Env(conn, s, tmp_path / "fm", tmp_path / "data")


def test_changed_selection_is_rejected_then_fresh_job_runs(env: Env) -> None:
    from facegen import manual
    from facegen.worker import JobGone

    w, renderer, _ = env.worker()
    selected = next(j for j in w.runnable(MINE) if j.unique_id == OLD_HAND)
    manual.new_face(env.conn, w.career.id, OLD_HAND, env.settings)
    with pytest.raises(JobGone):
        w.run_job(selected)
    assert renderer.calls == []
    fresh = next(j for j in w.runnable(MINE) if j.unique_id == OLD_HAND)
    assert fresh.kind == "reroll" and fresh.revision > selected.revision
    assert w.run_job(fresh).job.kind == "reroll"


def test_same_request_replaced_in_same_second_invalidates_selection(env: Env) -> None:
    from facegen import manual
    from facegen.worker import JobGone

    w, renderer, _ = env.worker()
    manual.new_face(env.conn, w.career.id, OLD_HAND, env.settings, now=123)
    selected = next(j for j in w.runnable(MINE) if j.unique_id == OLD_HAND)
    manual.new_face(env.conn, w.career.id, OLD_HAND, env.settings, now=123)
    with pytest.raises(JobGone):
        w.run_job(selected)
    assert renderer.calls == []


@pytest.mark.parametrize("column,value", [("present", 0), ("in_scope", 0), ("pin", "never")])
def test_claim_rechecks_player_eligibility(env: Env, column: str, value: object) -> None:
    from facegen.worker import JobGone

    w, renderer, _ = env.worker()
    selected = w.runnable(MINE)[0]
    env.conn.execute(
        f"UPDATE players SET {column} = ? WHERE unique_id = ?", (value, selected.unique_id)
    )
    with pytest.raises(JobGone):
        w.run_job(selected)
    assert env.jobs()[selected.unique_id] == "cancelled"
    assert renderer.calls == []


def test_interrupted_reroll_reuses_allocated_sheet_seed_and_preview(env: Env) -> None:
    from facegen import manual
    from facegen.traits import load_sheet, picture_seed

    w, _, _ = env.worker()
    initial = next(j for j in w.runnable(MINE) if j.unique_id == OLD_HAND)
    w.run_job(initial)
    original_sheet = load_sheet(env.conn, w.career.id, OLD_HAND)
    manual.new_face(env.conn, w.career.id, OLD_HAND, env.settings)
    seeds = []
    for _ in range(2):
        renderer = FakeRenderer(fail_on_call=1, exc=RuntimeError("interrupted"))
        w, _, _ = env.worker(renderer)
        job = next(j for j in w.runnable(MINE) if j.unique_id == OLD_HAND)
        with pytest.raises(RuntimeError, match="interrupted"):
            w.run_job(job)
        seeds.append(renderer.calls[0][1])
        assert env.jobs()[OLD_HAND] == "queued"
        assert load_sheet(env.conn, w.career.id, OLD_HAND) == original_sheet
    assert seeds[0] == seeds[1]
    preview = next(p for p in w.preview(MINE) if p.job.unique_id == OLD_HAND)
    assert preview.job.traits_version == 2
    assert picture_seed(preview.sheet.base_seed, 2, 0, 0) == seeds[0]
    assert (
        env.conn.execute("SELECT COUNT(*) FROM traits WHERE unique_id = ?", (OLD_HAND,)).fetchone()[
            0
        ]
        == 2
    )
    # Restart recovery uses the same allocation even when the last process died running.
    env.conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (preview.job.id,))
    w, renderer, _ = env.worker()
    assert w.requeue_stale() == 1
    w.run_job(next(j for j in w.runnable(MINE) if j.unique_id == OLD_HAND))
    assert renderer.calls[0][1] == seeds[0]
    assert env.current(OLD_HAND)["traits_version"] == 2


def test_restore_sheet_controls_next_redraw_and_reroll_versions(env: Env) -> None:
    from facegen import manual
    from facegen.traits import load_sheet
    from facegen.web.queries import player_detail

    w, _, _ = env.worker()

    def draw() -> None:
        w.run_job(next(j for j in w.runnable(JobFilter()) if j.unique_id == STAR))

    draw()
    original = env.current(STAR)
    manual.new_face(env.conn, w.career.id, STAR, env.settings)
    draw()
    assert env.current(STAR)["traits_version"] == 2
    manual.restore_picture(env.conn, w.career.id, w.career.key, STAR, original["id"], env.settings)
    assert load_sheet(env.conn, w.career.id, STAR)[0] == 1
    detail = player_detail(env.conn, w.career.id, STAR, env.data)
    assert detail["sheet"]["version"] == 1
    manual.new_picture(env.conn, w.career.id, STAR, env.settings)
    draw()
    assert env.current(STAR)["traits_version"] == 1
    manual.new_face(env.conn, w.career.id, STAR, env.settings)
    draw()
    assert env.current(STAR)["traits_version"] == 3


@pytest.mark.parametrize(
    "returncode,stdout,expected",
    [
        (1, b"", None),
        (1, b'"fm.exe"', None),
        (0, b'"FM.exe","12"', True),
        (0, b"INFO: No tasks are running", False),
    ],
)
@pytest.mark.parametrize("kind", ["windows", "wsl"])
def test_fm_process_check_respects_command_result(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    returncode: int,
    stdout: bytes,
    expected: bool | None,
) -> None:
    import subprocess

    from facegen import platform_info, procs
    from facegen.worker import fm_running

    calls: list[tuple[list[str], dict]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)

    monkeypatch.setattr(platform_info, "current", lambda: kind)
    monkeypatch.setattr(procs, "windows_tool", lambda name, k, env: f"/x/{name}")
    monkeypatch.setattr(subprocess, "run", run)
    assert fm_running() is expected
    assert calls[0][0] == procs.tasklist_command("/x/tasklist.exe")
    assert calls[0][1]["timeout"] == 30


def test_fm_process_check_without_tasklist_cannot_tell(monkeypatch: pytest.MonkeyPatch) -> None:
    from facegen import platform_info, procs
    from facegen.worker import fm_running

    monkeypatch.setattr(platform_info, "current", lambda: "wsl")
    monkeypatch.setattr(procs, "windows_tool", lambda name, k, env: None)
    assert fm_running() is None


def test_fm_process_check_on_linux_looks_for_wine_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from facegen import platform_info, worker

    monkeypatch.setattr(platform_info, "current", lambda: "linux")
    monkeypatch.setattr(worker, "_process_names", lambda: ["bash\n", "fm.exe\n"])
    assert worker.fm_running() is True
    monkeypatch.setattr(worker, "_process_names", lambda: ["bash\n"])
    assert worker.fm_running() is False


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
            # The detailed picture (x1.5) cropped to 3:4 is the master.
            assert im.size == (81, 108)
            assert "1girl" in im.info["facegen"]
        with Image.open(env.pack / f"{uid}.png") as im:
            assert im.size == (768, 1024)  # group A's published size: 3:4, 1024 high
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
    env.conn.execute("UPDATE players SET active_traits_version = NULL")
    env.conn.execute("UPDATE jobs SET status = 'queued', traits_version = NULL")
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
        assert im.size == (54, 72)  # the undetailed winner, cropped to 3:4


def test_group_b_draws_one_candidate_without_detail(env: Env) -> None:
    w, r, _ = env.worker()
    summary = w.run(JobFilter(groups=frozenset({"b"})))
    assert [o.job.unique_id for o in summary.done] == [LEAGUE_B]
    assert (r.renders, r.details) == (1, 0)
    with Image.open(env.pack / f"{LEAGUE_B}.png") as im:
        assert im.size == (576, 768)
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


def _queue_job(env: Env, uid: int, kind: str, club: int | None, band: str) -> None:
    env.conn.execute(
        "INSERT INTO jobs (career_id, unique_id, kind, group_name, tier, rank, status,"
        " target_club_id, target_band, reason, created_at, updated_at)"
        " SELECT career_id, unique_id, ?, 'a', 1, 0, 'queued', ?, ?, 'test', '', ''"
        " FROM players WHERE unique_id = ?",
        (kind, club, band, uid),
    )


def test_a_redraw_at_my_club_uses_the_current_picture_as_a_reference(env: Env) -> None:
    w, r, _ = env.worker()
    w.run(MINE)
    assert set(r.references) == {None}  # first pictures never use one
    first = env.current(OLD_HAND)
    _queue_job(env, OLD_HAND, "kit", RIVAL, "W08")
    w2, r2, _ = env.worker()
    summary = w2.run(MINE)
    assert [o.job.unique_id for o in summary.done] == [OLD_HAND]
    assert "reference v1 at 0.1" in summary.done[0].message

    # Every candidate and the detail pass got the face of v1, at the setting's strength.
    refs = r2.references
    assert len(refs) == 4 and all(ref is refs[0] for ref in refs)
    assert refs[0].scale == env.settings.reference.scale == 0.1
    crop = refs[0].image
    assert crop.width == crop.height == round(0.36 * 108)  # a square around the head of v1

    new = env.current(OLD_HAND)
    info = json.loads(new["render"])["reference"]
    assert info["used"] and info["image_id"] == first["id"] and info["version"] == 1
    assert info["scale"] == 0.1 and info["adapter"] == "fake/ip-adapter"
    assert info["master"] == first["master_path"]
    assert (env.data / info["crop"]).name == "v002-reference.webp"
    assert (env.data / info["crop"]).is_file()


def test_a_picture_of_a_man_is_never_a_reference(env: Env) -> None:
    w, _, _ = env.worker()
    w.run(MINE)
    # The current picture was drawn as a man (2026-11-13): a new picture of her uses no
    # reference, so no man's face carries over.
    env.conn.execute(
        "UPDATE images SET age_band = 'M2' WHERE unique_id = ? AND is_current = 1", (OLD_HAND,)
    )
    _queue_job(env, OLD_HAND, "redraw", MY_CLUB, "W08")
    w2, r2, _ = env.worker()
    done = w2.run(MINE).done
    assert [o.job.unique_id for o in done] == [OLD_HAND]
    assert set(r2.references) == {None}
    info = json.loads(env.current(OLD_HAND)["render"])["reference"]
    assert info["used"] is False and info["why"] == "the current picture is of a man"


def test_a_redraw_elsewhere_or_with_the_reference_off_uses_none(env: Env) -> None:
    group_a = JobFilter(groups=frozenset({"a"}))
    w, _, _ = env.worker()
    w.run(group_a)  # STAR, group A at another club, too
    _queue_job(env, STAR, "kit", MY_CLUB + 50, "W03")
    w2, r2, _ = env.worker()
    assert [o.job.unique_id for o in w2.run(group_a).done] == [STAR]
    assert set(r2.references) == {None}  # other_clubs is off by default

    _queue_job(env, OLD_HAND, "age", MY_CLUB, "W09")
    off = replace(env.settings, reference=replace(env.settings.reference, enabled=False))
    w3, r3, _ = env.worker(settings=off)
    assert [o.job.unique_id for o in w3.run(group_a).done] == [OLD_HAND]
    assert set(r3.references) == {None}
    assert json.loads(env.current(OLD_HAND)["render"])["reference"] is None

    on_everywhere = replace(
        env.settings, reference=replace(env.settings.reference, other_clubs=True)
    )
    _queue_job(env, STAR, "age", MY_CLUB + 50, "W04")
    w4, r4, _ = env.worker(settings=on_everywhere)
    assert [o.job.unique_id for o in w4.run(group_a).done] == [STAR]
    assert r4.references[0] is not None and r4.references[0].scale == 0.1


def test_a_missing_reference_file_draws_without_one(env: Env) -> None:
    w, _, _ = env.worker()
    w.run(MINE)
    (env.data / env.current(OLD_HAND)["master_path"]).unlink()
    _queue_job(env, OLD_HAND, "kit", RIVAL, "W08")
    w2, r2, _ = env.worker()
    summary = w2.run(MINE)
    assert len(summary.done) == 1 and set(r2.references) == {None}
    info = json.loads(env.current(OLD_HAND)["render"])["reference"]
    assert info["used"] is False and info["why"].startswith("picture file missing")


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


def test_the_sheet_is_weighted_by_the_players_personality(env: Env) -> None:
    quiet = replace(PLAIN_PERSONALITY, pressure=3, adaptability=3, sportsmanship=20, temperament=20)
    env.conn.execute(
        "UPDATE players SET personality = ? WHERE unique_id = ?",
        (personality_to_json(quiet), OLD_HAND),
    )
    career = load_career(env.conn)
    assert career is not None
    plain = sample_sheet(career.key, OLD_HAND, 1, height_cm=180, strength=12)
    weighted = sample_sheet(career.key, OLD_HAND, 1, height_cm=180, strength=12, personality=quiet)
    assert weighted != plain  # this player's draw changes, so the test can tell them apart

    off = replace(env.settings, personality=PersonalitySettings(weight=0))
    w_off, _, _ = env.worker(settings=off)
    preview = next(p for p in w_off.preview(MINE) if p.job.unique_id == OLD_HAND)
    assert preview.sheet == plain

    w, _, _ = env.worker()
    preview = next(p for p in w.preview(MINE) if p.job.unique_id == OLD_HAND)
    assert preview.job.personality == quiet and preview.sheet == weighted
    w.run(MINE, publish=False)
    assert load_sheet(env.conn, career.id, OLD_HAND) == (1, weighted)


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


# -- the prompts encoded ahead (facegen.prompt_cache) --

EVERY = JobFilter()


def test_one_trip_of_the_text_encoders_serves_the_whole_queue(env: Env) -> None:
    w, r, _ = env.worker()
    summary = w.run(EVERY, publish=False)
    assert len(summary.done) == 4
    # One trip, before the first render: every job's prompt as the job then drew it, in order.
    # (The fake renderer would have made a trip of its own for any prompt not prepared.)
    assert len(r.trips) == 1
    assert r.trips[0] == list(dict.fromkeys(r.prompts))


def test_the_trips_follow_the_window_and_the_run_limit(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    import facegen.worker as worker_module

    monkeypatch.setattr(worker_module, "PRE_ENCODE_JOBS", 3)
    w, r, _ = env.worker()
    w.run(EVERY, limit=1, publish=False)
    assert [len(trip) for trip in r.trips] == [1]  # nothing beyond the limit
    w.run(EVERY, publish=False)
    assert [len(trip) for trip in r.trips] == [1, 3]  # the other three in one window
    assert len(set(r.prompts)) == 4


def test_a_single_job_encodes_its_own_prompt_when_it_draws(env: Env) -> None:
    w, r, _ = env.worker()
    first, second, *_ = w.runnable(EVERY)
    w.run_job(first)  # no queue given: the single-job path
    assert r.trips == [[r.prompts[0]]]
    w.run_job(second, w.runnable(EVERY)[1:])
    assert len(r.trips) == 2 and r.trips[1][0] == r.prompts[-1]


def test_a_job_queued_after_the_trip_gets_a_trip_of_its_own(env: Env) -> None:
    from facegen import manual

    w, r, _ = env.worker()
    first, *rest = w.runnable(MINE)
    w.run_job(first, rest)
    assert [len(trip) for trip in r.trips] == [2]
    manual.new_face(env.conn, w.career.id, first.unique_id, env.settings)  # queued now
    w.run(MINE, publish=False)
    assert len(r.trips) == 2
    # The new face, encoded when its job came up: first in this run, as a manual request.
    assert r.trips[1][0] == r.prompts[3]


def test_a_job_that_cannot_be_previewed_does_not_stop_the_one_running(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    w, r, _ = env.worker()
    first, second, third, _ = w.runnable(EVERY)
    preview = w._preview

    def broken(job: object) -> object:
        if job is second:
            raise KeyError("W99")
        return preview(job)  # type: ignore[arg-type]

    monkeypatch.setattr(w, "_preview", broken)
    assert w.run_job(first, [second, third]).status == "done"
    assert len(r.trips) == 1 and len(r.trips[0]) == 2  # this job's prompt and the third's


def test_the_renderers_cache_holds_a_whole_window_of_prompts() -> None:
    from facegen.renderer import TEXT_CACHE_SIZE
    from facegen.worker import PRE_ENCODE_JOBS

    # A positive and a negative text per job, and the empty text that pads them.
    assert 2 * PRE_ENCODE_JOBS + 1 <= TEXT_CACHE_SIZE


def test_my_clubs_staff_are_drawn_at_a_quality_in_club_wear_for_their_job(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    from factories import staff_member

    from facegen.age import WORDINGS
    from facegen.prompt import STAFF_NEGATIVE, STAFF_POSITIVE
    from facegen.wardrobe import ROLES

    coach = NEWGEN + 100
    s = replace(
        settings,
        paths=Paths(
            fm_user_dir=tmp_path / "fm",
            data_dir=tmp_path / "data",
            fm_install_dir=tmp_path / "fm-install",
        ),
    )
    snap = snapshot(world(), staff=[staff_member(coach, age=30, job="coach", female=True)])
    run_sync(conn, snap, s, save_file="career.fm")
    env = Env(conn, s, tmp_path / "fm", tmp_path / "data")
    w, renderer, _ = env.worker()
    staff_job = next(j for j in w.runnable(MINE) if j.unique_id == coach)
    assert staff_job.staff and staff_job.target_band == "W03"  # 25-31 in the staff table
    assert staff_job.staff_job == "coach"
    w.run_job(staff_job)
    image = env.current(coach)
    assert image is not None and image["group_name"] == "a" and image["age_band"] == "W03"
    # Group A: three candidates and the detail pass, as for the players at my club.
    assert len([c for c in renderer.calls if c[0] == "render"]) == s.groups.a.candidates
    prompt = image["prompt"]
    render = json.loads(image["render"])
    assert render["staff"] is True and render["outfit"] in ROLES["coach"].looks
    assert "soccer uniform" not in prompt and "shorts" not in prompt
    assert WORDINGS["W03"].text in prompt
    # The content rule holds for staff: `safe`, long sleeves, the fixed negative, the
    # under-18 negatives for W03, and the playing kit kept out.
    assert "safe" in prompt and STAFF_POSITIVE in prompt
    assert CONTENT_NEGATIVE in image["negative_prompt"]
    assert YOUNG_NEGATIVE in image["negative_prompt"]
    assert STAFF_NEGATIVE in image["negative_prompt"]
    # Staff keep the old pose library on the grey backdrop: no player shoot.
    assert render["shoot"] is None and render["backdrop"] == "grey"
    assert "simple background, grey background" in prompt


def test_a_players_picture_is_a_shoot_recorded_with_it(env: Env) -> None:
    from facegen.shoot import BACKDROP_COLOURS, LOOKS, season_of

    w, _r, _t = env.worker()
    assert w.career.game_date == GAME_DATE
    w.run(MINE, publish=False)
    for uid in (OLD_HAND, KEEPER):
        image = env.current(uid)
        render = json.loads(image["render"])
        shoot = render["shoot"]
        assert shoot["look"] in LOOKS and shoot["season"] == season_of(GAME_DATE)
        assert render["kit_source"] == "fallback"  # the test club is in no table
        assert render["kit_known"] is False
        assert render["pose"] == f"{shoot['arms']}, {shoot['angle']}"
        assert render["backdrop"] == shoot["backdrop"]
        assert "simple background" in image["prompt"]
        colours = [c for c in BACKDROP_COLOURS if f"{c} background" in image["prompt"]]
        assert colours and set(colours) <= set(render["backdrop"].split())
        assert CONTENT_NEGATIVE in image["negative_prompt"]
    assert "gloves" in env.current(KEEPER)["prompt"]


def test_the_season_follows_the_in_game_date(env: Env) -> None:
    from datetime import date

    w, _, _ = env.worker()
    job = next(j for j in w.runnable(MINE) if j.unique_id == OLD_HAND)
    seasons = {}
    for day in (date(2026, 12, 20), date(2027, 7, 1), None):
        w.career = replace(w.career, game_date=day)
        sheet = sample_sheet(w.career.key, OLD_HAND, 1, height_cm=180, strength=10)
        seasons[day] = w._setup(job, sheet, 1).shoot.season
    assert list(seasons.values()) == ["winter", "summer", "unknown"]


def _staff_env(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, **snap: object
) -> tuple[Env, Settings]:
    s = replace(
        settings,
        paths=Paths(
            fm_user_dir=tmp_path / "fm",
            data_dir=tmp_path / "data",
            fm_install_dir=tmp_path / "fm-install",
        ),
    )
    run_sync(conn, snapshot(world(), **snap), s, save_file="career.fm")  # type: ignore[arg-type]
    return Env(conn, s, tmp_path / "fm", tmp_path / "data"), s


def test_a_man_on_the_staff_is_drawn_and_checked_as_a_woman(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    """Everyone is drawn as a woman (user, 2026-11-14), whatever the save says."""
    from factories import staff_member

    from facegen.prompt import COMPOSITION_NEGATIVE, SUBJECT

    scout = NEWGEN + 100
    env, _ = _staff_env(
        conn, settings, tmp_path, staff=[staff_member(scout, age=50, job="scout", female=False)]
    )
    w, _renderer, _tagger = env.worker()
    job = next(j for j in w.runnable(MINE) if j.unique_id == scout)
    assert job.target_band == "W06"
    w.run_job(job)
    image = env.current(scout)
    assert image is not None and image["age_band"] == "W06"
    assert image["prompt"].startswith(SUBJECT + ", ")
    assert COMPOSITION_NEGATIVE in image["negative_prompt"]
    assert json.loads(image["check_result"])["attempts"] == 1
    assert "male" not in json.loads(image["render"])


def test_a_journalist_is_dressed_for_work_on_the_grey_backdrop(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    from factories import media_person

    from facegen.wardrobe import ROLES

    reporter = 2_002_097_912
    env, s = _staff_env(
        conn, settings, tmp_path, media=[media_person(reporter, female=True, age=59)]
    )
    w, renderer, _ = env.worker()
    job = next(j for j in w.runnable(JobFilter()) if j.unique_id == reporter)
    assert (job.media_role, job.group, job.target_band) == ("journalist", "b", "W07")
    w.run_job(job)
    image = env.current(reporter)
    render = json.loads(image["render"])
    assert render["media"] == "journalist" and render["outfit"] in ROLES["journalist"].looks
    assert render["backdrop"] == "grey" and "male" not in render
    # Group B: one candidate, no detail pass.
    assert len([c for c in renderer.calls if c[0] == "render"]) == s.groups.b.candidates
    assert "soccer" not in image["prompt"]


# --- the detail pass's size and VAE tiling (Phase 10c, 10d) ----------------------------------


def _with(settings: Settings, **groups_a: object) -> Settings:
    return replace(
        settings, groups=replace(settings.groups, a=replace(settings.groups.a, **groups_a))
    )


def test_the_detail_size_reaches_the_renderer_and_is_recorded(env: Env) -> None:
    w, r, _ = env.worker(settings=_with(env.settings, detail_size=(1152, 1472)))
    outcome = w.run(MINE, limit=1).done[0]
    assert r.detail_sizes == [(1152, 1472)]
    img = env.current(outcome.job.unique_id)
    detail = json.loads(img["render"])["detail"]
    assert detail["target"] == [1152, 1472] and detail["used"] is True
    assert detail["size"] == [72, 92]  # the fake's 56x72 render scaled as 896x1152 would be
    with Image.open(env.data / img["master_path"]) as im:
        assert im.size == (69, 92)  # cropped to 3:4


def test_the_default_detail_size_is_one_and_a_half_times_the_render(env: Env) -> None:
    w, r, _ = env.worker()
    w.run(MINE, limit=1)
    assert r.detail_sizes == [(1344, 1728)]


def test_detail_size_off_skips_the_pass(env: Env) -> None:
    w, r, _ = env.worker(settings=_with(env.settings, detail_size=None))
    outcome = w.run(MINE, limit=1).done[0]
    assert (r.renders, r.details) == (3, 0)
    assert json.loads(env.current(outcome.job.unique_id)["render"])["detail"] is None


def test_vae_tiling_follows_the_setting_for_each_job(env: Env) -> None:
    from facegen.settings import RenderSettings

    tiled = replace(env.settings, render=RenderSettings(vae_tiling=True))
    w, r, _ = env.worker(settings=tiled)
    outcome = w.run(MINE, limit=1).done[0]
    assert r.vae_tiling is True
    assert json.loads(env.current(outcome.job.unique_id)["render"])["vae_tiling"] is True
    w.settings = env.settings  # changed in the UI: the next job draws without tiling
    w.run(MINE, limit=1)
    assert r.vae_tiling is False
