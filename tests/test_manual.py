"""Manual requests from the UI: generate, new face, new picture, pin, remove, retry, cancel."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from factories import ABROAD, NEWGEN, RIVAL, WOMENS, player, snapshot
from fakes import FakeRenderer, FakeTagger

from facegen import manual
from facegen.kits import load_kits
from facegen.manual import ManualError
from facegen.settings import Paths, Settings
from facegen.sync import run_sync
from facegen.web import queries
from facegen.worker import JobFilter, Worker, load_career

MINE = 1001  # pre-existing at my club: A
STAR = 1002  # pre-existing star at another club: in no group, picked by hand
OTHER = 1003  # pre-existing at a rival: in no group
B_NEWGEN = NEWGEN + 2  # rival in my league, PA 152: B
A_NEWGEN = NEWGEN + 4  # abroad, CA 160: A elsewhere
WOMAN = NEWGEN + 6  # women's club: ignored

ONLY_MANUAL = JobFilter(groups=frozenset(), manual_always=True)


def world() -> list:
    return [
        player(MINE, name="Mine", age=29),
        player(STAR, name="Star", ca=190, pa=195, club_id=ABROAD),
        player(OTHER, name="Other", ca=100, pa=110, club_id=RIVAL),
        player(B_NEWGEN, name="B Newgen", ca=90, pa=152, club_id=RIVAL),
        player(A_NEWGEN, name="A Newgen", ca=160, pa=170, club_id=ABROAD),
        player(WOMAN, name="Woman", ca=100, pa=190, club_id=WOMENS),
    ]


@dataclass
class Env:
    conn: sqlite3.Connection
    settings: Settings
    tmp: Path

    @property
    def cid(self) -> int:
        career = load_career(self.conn)
        assert career is not None
        return career.id

    @property
    def key(self) -> str:
        career = load_career(self.conn)
        assert career is not None
        return career.key

    def draw(self, f: JobFilter, tagger: FakeTagger | None = None) -> Worker:
        career = load_career(self.conn)
        assert career is not None
        renderer, tags = FakeRenderer(), tagger or FakeTagger()
        w = Worker(
            self.conn,
            self.settings,
            load_kits(),
            career,
            renderer_factory=lambda: renderer,
            tagger_factory=lambda: tags,
            fm_check=lambda: False,
            log=lambda line: None,
        )
        w.run(f)
        return w

    def queue(self) -> list[tuple[int, str, str, int, int]]:
        """Queued jobs in queue order: (player, kind, group, manual, tier)."""
        return [
            tuple(r)
            for r in self.conn.execute(
                "SELECT unique_id, kind, group_name, manual, tier FROM jobs"
                " WHERE status = 'queued' ORDER BY tier, rank, id"
            )
        ]

    def player(self, uid: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM players WHERE unique_id = ?", (uid,)).fetchone()

    def active(self, uid: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM jobs WHERE unique_id = ? AND status IN ('queued', 'running')", (uid,)
        ).fetchall()


@pytest.fixture
def env(conn: sqlite3.Connection, settings: Settings, tmp_path: Path) -> Env:
    s = replace(settings, paths=Paths(fm_user_dir=tmp_path / "fm", data_dir=tmp_path / "data"))
    run_sync(conn, snapshot(world()), s, save_file="career.fm")
    return Env(conn, s, tmp_path)


def test_generate_hand_picks_a_pre_existing_player_from_another_club(env: Env) -> None:
    r = manual.generate(env.conn, env.cid, [STAR], env.settings)
    assert r.queued == [STAR] and r.summary() == "1 queued"
    p = env.player(STAR)
    assert (p["pin"], p["group_name"], p["tracked"]) == ("b", "b", 0)
    assert "pinned to B" in p["reasons"]
    # Manual requests go before every automatic job.
    assert env.queue()[0] == (STAR, "initial", "b", 1, 0)
    assert all(tier > 0 for *_, tier in env.queue()[1:])


def test_generate_refuses_players_outside_mens_football(env: Env) -> None:
    r = manual.generate(env.conn, env.cid, [WOMAN, STAR], env.settings, "a")
    assert r.queued == [STAR]
    assert r.skipped == [(WOMAN, "not in men's football")]
    assert env.player(WOMAN)["pin"] is None and env.active(WOMAN) == []
    with pytest.raises(ManualError):
        manual.generate(env.conn, env.cid, [999], env.settings)
    with pytest.raises(ManualError):
        manual.generate(env.conn, env.cid, [STAR], env.settings, "z")


def test_generate_takes_over_the_queued_automatic_job(env: Env) -> None:
    assert [(j["manual"], j["tier"]) for j in env.active(A_NEWGEN)] == [(0, 2)]
    manual.generate(env.conn, env.cid, [A_NEWGEN], env.settings, "a")
    assert [(j["manual"], j["tier"], j["kind"]) for j in env.active(A_NEWGEN)] == [
        (1, 0, "initial")
    ]


def test_generate_upgrades_a_lower_picture_and_keeps_a_better_one(env: Env) -> None:
    env.draw(JobFilter(groups=frozenset({"a", "b"})))
    r = manual.generate(env.conn, env.cid, [B_NEWGEN], env.settings, "a")
    assert r.queued == [B_NEWGEN]
    assert env.queue()[0] == (B_NEWGEN, "upgrade", "a", 1, 0)
    r = manual.generate(env.conn, env.cid, [A_NEWGEN], env.settings, "c")
    assert r.pinned == [A_NEWGEN] and r.queued == []
    assert env.player(A_NEWGEN)["pin"] == "c" and env.active(A_NEWGEN) == []


def test_the_newest_request_runs_first(env: Env) -> None:
    manual.generate(env.conn, env.cid, [STAR, OTHER], env.settings, now=100.0)
    manual.new_face(env.conn, env.cid, B_NEWGEN, env.settings, now=200.0)
    assert [uid for uid, *_ in env.queue()[:3]] == [B_NEWGEN, STAR, OTHER]


def test_new_face_draws_from_a_new_character_sheet(env: Env) -> None:
    env.draw(JobFilter(groups=frozenset({"a"})))
    first = env.conn.execute(
        "SELECT traits_version, prompt FROM images WHERE unique_id = ? AND is_current = 1",
        (A_NEWGEN,),
    ).fetchone()
    manual.new_face(env.conn, env.cid, A_NEWGEN, env.settings)
    assert env.queue()[0][:2] == (A_NEWGEN, "reroll")
    env.draw(ONLY_MANUAL)
    now = env.conn.execute(
        "SELECT version, reason, traits_version FROM images WHERE unique_id = ? AND is_current = 1",
        (A_NEWGEN,),
    ).fetchone()
    assert tuple(now) == (2, "reroll", first["traits_version"] + 1)
    versions = [
        r[0]
        for r in env.conn.execute("SELECT version FROM traits WHERE unique_id = ?", (A_NEWGEN,))
    ]
    assert sorted(versions) == [1, 2]


def test_new_picture_keeps_the_sheet_and_uses_a_reference_at_my_club(env: Env) -> None:
    with pytest.raises(ManualError, match="no picture"):
        manual.new_picture(env.conn, env.cid, A_NEWGEN, env.settings)
    env.draw(JobFilter(groups=frozenset({"a"})))
    manual.new_picture(env.conn, env.cid, A_NEWGEN, env.settings)
    manual.new_picture(env.conn, env.cid, MINE, env.settings)
    env.draw(ONLY_MANUAL)
    pictures = {
        r["unique_id"]: (r["version"], r["reason"], r["traits_version"], json.loads(r["render"]))
        for r in env.conn.execute("SELECT * FROM images WHERE is_current = 1")
    }
    # Same sheet both times; at my club the first picture is the reference, elsewhere none.
    mine, elsewhere = pictures[MINE], pictures[A_NEWGEN]
    assert mine[:3] == (2, "redraw", 1) and elsewhere[:3] == (2, "redraw", 1)
    assert mine[3]["reference"]["used"] and mine[3]["reference"]["version"] == 1
    assert elsewhere[3]["reference"] is None


def test_new_picture_for_many_skips_who_cannot_have_one(env: Env) -> None:
    env.draw(JobFilter(groups=frozenset({"a"})))
    r = manual.new_pictures(env.conn, env.cid, [MINE, OTHER, A_NEWGEN, WOMAN], env.settings)
    assert r.queued == [MINE, A_NEWGEN]
    assert dict(r.skipped) == {
        OTHER: "no picture to draw again; use Generate",
        WOMAN: "not in men's football",
    }
    # Ticked together, they run in the order given, before every automatic job.
    assert [q[:2] for q in env.queue() if q[3]] == [(MINE, "redraw"), (A_NEWGEN, "redraw")]
    assert env.queue()[:2] == [(MINE, "redraw", "a", 1, 0), (A_NEWGEN, "redraw", "a", 1, 0)]


def test_new_face_needs_a_group_or_a_picture(env: Env) -> None:
    with pytest.raises(ManualError, match="no group"):
        manual.new_face(env.conn, env.cid, OTHER, env.settings)
    with pytest.raises(ManualError, match="men's football"):
        manual.new_face(env.conn, env.cid, WOMAN, env.settings)


def test_pin_to_never_cancels_and_clearing_the_pin_requeues(env: Env) -> None:
    manual.set_pin(env.conn, env.cid, [A_NEWGEN], "never", env.settings)
    p = env.player(A_NEWGEN)
    assert p["group_name"] is None and "pinned to never" in p["reasons"]
    assert env.active(A_NEWGEN) == []
    with pytest.raises(ManualError, match="never"):
        manual.new_face(env.conn, env.cid, A_NEWGEN, env.settings)
    manual.set_pin(env.conn, env.cid, [A_NEWGEN], None, env.settings)
    assert [(j["kind"], j["manual"]) for j in env.active(A_NEWGEN)] == [("initial", 0)]
    with pytest.raises(ManualError):
        manual.set_pin(env.conn, env.cid, [A_NEWGEN], "x", env.settings)


def test_remove_picture_brings_back_the_games_photo(env: Env) -> None:
    env.draw(JobFilter(groups=frozenset({"a"})))
    pack = env.settings.paths.fm_user_dir / "graphics" / "facegen" / env.key
    assert (pack / f"{MINE}.png").is_file()
    master = env.conn.execute(
        "SELECT master_path FROM images WHERE unique_id = ?", (MINE,)
    ).fetchone()[0]

    assert manual.remove_picture(env.conn, env.cid, env.key, MINE, env.settings)
    assert not (pack / f"{MINE}.png").exists()
    assert f'from="{MINE}"' not in (pack / "config.xml").read_text()
    assert f'from="{A_NEWGEN}"' in (pack / "config.xml").read_text()
    assert env.player(MINE)["pin"] == "never"
    current = env.conn.execute(
        "SELECT COUNT(*) FROM images WHERE unique_id = ? AND is_current = 1", (MINE,)
    ).fetchone()[0]
    assert current == 0
    assert (env.settings.paths.data_dir / master).is_file()  # the master is kept
    # A new sync does not queue the player again.
    report = run_sync(env.conn, snapshot(world()), env.settings, save_file="career.fm")
    assert env.active(MINE) == [] and not report.enqueued


def test_nothing_is_queued_for_a_player_being_drawn(env: Env) -> None:
    env.conn.execute("UPDATE jobs SET status = 'running' WHERE unique_id = ?", (A_NEWGEN,))
    r = manual.generate(env.conn, env.cid, [A_NEWGEN], env.settings, "b")
    assert r.skipped == [(A_NEWGEN, "being drawn right now")]
    with pytest.raises(ManualError, match="right now"):
        manual.new_face(env.conn, env.cid, A_NEWGEN, env.settings)
    with pytest.raises(ManualError, match="right now"):
        manual.remove_picture(env.conn, env.cid, env.key, A_NEWGEN, env.settings)


def test_retry_queues_a_review_job_again(env: Env) -> None:
    with pytest.raises(ManualError, match="nothing to retry"):
        manual.retry(env.conn, env.cid, B_NEWGEN, env.settings)
    env.draw(JobFilter(groups=frozenset({"b"})), FakeTagger(["sensitive"] * 3))
    assert [i["unique_id"] for i in queries.review_items(env.conn, env.cid)] == [B_NEWGEN]
    manual.retry(env.conn, env.cid, B_NEWGEN, env.settings)
    assert env.queue()[0] == (B_NEWGEN, "initial", "b", 1, 0)
    assert queries.review_items(env.conn, env.cid) == []
    env.draw(ONLY_MANUAL)
    assert (
        env.conn.execute(
            "SELECT version FROM images WHERE unique_id = ? AND is_current = 1",
            (B_NEWGEN,),
        ).fetchone()[0]
        == 2
    )  # v1 was the rejected try; the retry draws v2 with new seeds


def test_cancelling_a_generate_request_undoes_the_hand_pick(env: Env) -> None:
    manual.generate(env.conn, env.cid, [STAR], env.settings)
    assert manual.cancel(env.conn, env.cid, STAR, env.settings)
    p = env.player(STAR)
    assert (p["pin"], p["group_name"]) == (None, None) and env.active(STAR) == []
    assert not manual.cancel(env.conn, env.cid, STAR, env.settings)


def test_cancelling_a_new_face_keeps_the_pin(env: Env) -> None:
    manual.set_pin(env.conn, env.cid, [OTHER], "c", env.settings)
    manual.new_face(env.conn, env.cid, OTHER, env.settings)
    assert manual.cancel(env.conn, env.cid, OTHER, env.settings)
    assert env.player(OTHER)["pin"] == "c"
    # The rules queue the pinned player's own picture again.
    assert [(j["kind"], j["manual"]) for j in env.active(OTHER)] == [("initial", 0)]


def test_a_group_pin_is_a_hand_pick(env: Env) -> None:
    """The user's report (2026-11-11): Isco pinned to B got an automatic B job that the
    worker does not draw. A group pin queues a request, as Generate does."""
    r = manual.set_pin(env.conn, env.cid, [STAR], "b", env.settings)
    assert r.queued == [STAR]
    assert env.queue()[0] == (STAR, "initial", "b", 1, 0)
    # A player who already has a good enough picture is only pinned.
    env.draw(ONLY_MANUAL)
    assert manual.set_pin(env.conn, env.cid, [STAR], "c", env.settings).pinned == [STAR]
    # Players outside men's football are refused, as by Generate.
    assert manual.set_pin(env.conn, env.cid, [WOMAN], "a", env.settings).skipped


def test_pin_to_never_cancels_a_request_too(env: Env) -> None:
    manual.generate(env.conn, env.cid, [STAR], env.settings)
    manual.set_pin(env.conn, env.cid, [STAR], "never", env.settings)
    assert env.active(STAR) == []
    env.draw(ONLY_MANUAL)
    assert (
        env.conn.execute("SELECT COUNT(*) FROM images WHERE unique_id = ?", (STAR,)).fetchone()[0]
        == 0
    )


def test_clearing_a_pin_undoes_a_generate_request_but_not_a_new_face(env: Env) -> None:
    manual.generate(env.conn, env.cid, [STAR], env.settings)
    manual.set_pin(env.conn, env.cid, [STAR], None, env.settings)
    assert env.active(STAR) == [] and env.player(STAR)["group_name"] is None
    manual.new_face(env.conn, env.cid, B_NEWGEN, env.settings)
    manual.set_pin(env.conn, env.cid, [B_NEWGEN], None, env.settings)
    assert [(j["kind"], j["manual"]) for j in env.active(B_NEWGEN)] == [("reroll", 1)]


def _image(env: Env, uid: int, version: int) -> sqlite3.Row:
    return env.conn.execute(
        "SELECT * FROM images WHERE unique_id = ? AND version = ?", (uid, version)
    ).fetchone()


def test_restore_brings_back_an_earlier_picture_until_the_next_change(env: Env) -> None:
    env.draw(JobFilter(groups=frozenset({"a"})))
    manual.new_picture(env.conn, env.cid, A_NEWGEN, env.settings)
    env.draw(ONLY_MANUAL)
    v1, v2 = _image(env, A_NEWGEN, 1), _image(env, A_NEWGEN, 2)
    assert v2["is_current"] and not v1["is_current"]

    r = manual.restore_picture(env.conn, env.cid, env.key, A_NEWGEN, v1["id"], env.settings)
    assert r.version == 1 and not r.pinned_never
    assert r.publish.written == 1 and r.publish.active  # rewritten in FM27's pack
    v1, v2 = _image(env, A_NEWGEN, 1), _image(env, A_NEWGEN, 2)
    assert v1["is_current"] and v1["restored_at"] and v1["published_path"]
    assert not v2["is_current"] and v2["published_path"] is None
    assert (v1["restored_group"], v1["restored_club_id"], v1["restored_band"]) == (
        "a",
        ABROAD,
        "W03",
    )

    # The same save again: the restored picture counts as up to date.
    report = run_sync(env.conn, snapshot(world()), env.settings, save_file="career.fm")
    assert not report.enqueued and env.active(A_NEWGEN) == []
    # A transfer is a change: the tracked player is re-drawn in the new kit.
    moved = [p for p in world() if p.unique_id != A_NEWGEN]
    moved.append(player(A_NEWGEN, name="A Newgen", ca=160, pa=170, club_id=RIVAL))
    run_sync(env.conn, snapshot(moved), env.settings, save_file="career.fm")
    assert [(j["kind"], j["target_club_id"]) for j in env.active(A_NEWGEN)] == [("kit", RIVAL)]


def test_restore_refuses_what_must_not_be_shown(env: Env) -> None:
    env.draw(JobFilter(groups=frozenset({"a"})))
    v1 = _image(env, MINE, 1)
    with pytest.raises(ManualError, match="already the current"):
        manual.restore_picture(env.conn, env.cid, env.key, MINE, v1["id"], env.settings)
    with pytest.raises(ManualError, match="no picture"):
        manual.restore_picture(env.conn, env.cid, env.key, OTHER, v1["id"], env.settings)

    # A re-draw that failed the check three times is kept for review, never shown.
    manual.new_picture(env.conn, env.cid, MINE, env.settings)
    env.draw(ONLY_MANUAL, FakeTagger(["sensitive"] * 9))
    rejected = _image(env, MINE, 2)
    assert rejected["master_path"].endswith("-rejected.png")
    with pytest.raises(ManualError, match="failed the quality check"):
        manual.restore_picture(env.conn, env.cid, env.key, MINE, rejected["id"], env.settings)

    # A picture whose file is gone from data/ cannot come back either.
    manual.new_picture(env.conn, env.cid, MINE, env.settings)
    env.draw(ONLY_MANUAL)
    (env.settings.paths.data_dir / v1["master_path"]).unlink()
    with pytest.raises(ManualError, match="missing"):
        manual.restore_picture(env.conn, env.cid, env.key, MINE, v1["id"], env.settings)
    assert _image(env, MINE, 3)["is_current"]


def test_restore_after_remove_keeps_the_never_pin(env: Env) -> None:
    env.draw(JobFilter(groups=frozenset({"a"})))
    manual.remove_picture(env.conn, env.cid, env.key, MINE, env.settings)
    v1 = _image(env, MINE, 1)
    r = manual.restore_picture(env.conn, env.cid, env.key, MINE, v1["id"], env.settings)
    assert r.pinned_never and _image(env, MINE, 1)["is_current"]
    assert env.player(MINE)["pin"] == "never" and env.active(MINE) == []
