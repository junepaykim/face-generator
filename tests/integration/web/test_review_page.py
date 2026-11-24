"""The Review page (Phase 12i): act on many players at once, and compare a player's candidates
side by side with the picture FM27 shows now."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import pytest
from factories import NEWGEN, RIVAL, player
from fakes import FakeRenderer, FakeTagger
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, post_player_action, synced

from kitshot.datasource.kits import load_kits
from kitshot.i18n import Translator
from kitshot.persistence import db
from kitshot.persistence.settings import load_settings
from kitshot.pipeline import manual
from kitshot.pipeline.worker import JobFilter, Worker, load_career
from kitshot.web.messages import check_failure_text

ONE, TWO, THREE = NEWGEN + 2, NEWGEN + 4, NEWGEN + 6
MINE = 1001
WORLD = [
    player(MINE, name="Mine Player"),
    player(ONE, name="Kid One", ca=135, pa=160, club_id=RIVAL),
    player(TWO, name="Kid Two", ca=136, pa=160, club_id=RIVAL),
    player(THREE, name="Kid Three", ca=137, pa=160, club_id=RIVAL),
]


@dataclass
class Env:
    client: TestClient
    config: Path
    db_path: Path

    def draw(self, verdicts: list[str], groups: frozenset[str] = frozenset({"b"})) -> None:
        """Run the jobs of these groups (B) with the tagger answering these verdicts, then
        "pass"."""
        conn = db.connect(self.db_path)
        try:
            career = load_career(conn)
            assert career is not None
            tagger = FakeTagger(verdicts)
            Worker(
                conn, load_settings(conn, self.config), load_kits(), career,
                renderer_factory=FakeRenderer, tagger_factory=lambda: tagger,
                fm_check=lambda: False, log=lambda *_: None,
            ).run(JobFilter(groups=groups))  # fmt: skip
        finally:
            conn.close()

    def jobs(self, uid: int) -> list[tuple[str, str]]:
        conn = db.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT kind, status FROM jobs WHERE unique_id = ? ORDER BY id", (uid,)
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        finally:
            conn.close()


@pytest.fixture
def env(config_file: Path) -> Iterator[Env]:
    db_path = synced(config_file, WORLD)
    client = client_for(config_file, db_path)
    e = Env(client, config_file, db_path)
    e.draw(["sensitive"] * 9)  # every candidate of the three kids fails, three times each
    yield e
    client.close()


def listed(page: str) -> list[int]:
    return [int(u) for u in re.findall(r'<input type="checkbox" name="uid" value="(\d+)"', page)]


def review_badge(page: str) -> str | None:
    """The count on the sidebar's Review link, or None without one."""
    nav = r'<a href="/review" class="[^"]*" title="Review">(.*?)</a>'
    link = re.search(nav, page, re.DOTALL)
    assert link is not None
    count = re.search(r'class="count badge badge-count">([^<]+)<', link[1])
    return count[1] if count else None


def post(env: Env, action: str, *uids: int) -> str:
    r = post_player_action(
        env.client,
        "/players/action",
        data={"action": action, "uid": [str(u) for u in uids], "next": "/review", "pin": "never"},
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/review?msg=")
    return unescape(env.client.get(r.headers["location"]).text)


def test_each_player_can_be_ticked_for_the_bars_buttons(env: Env) -> None:
    page = unescape(env.client.get("/review").text)
    assert sorted(listed(page)) == [ONE, TWO, THREE]
    assert '<form method="post" action="/players/action" id="bulk">' in page
    for action in ("retry", "new-face", "pin"):
        assert re.search(rf'name="action" value="{action}" data-needs-tick', page), action
    # Each player's own buttons still act on that player alone.
    assert f'formaction="/players/{ONE}/retry"' in page
    assert f'formaction="/players/{ONE}/new-face"' in page
    assert (
        "Select all" in page
        and "모두 선택" in env.client.get("/review", headers=KOREAN_WINDOWS).text
    )


def test_retry_new_face_and_never_for_many_at_once(env: Env) -> None:
    page = post(env, "retry", ONE, TWO)
    assert "Retry: 2 queued." in page
    assert listed(page) == [THREE]
    assert env.jobs(ONE)[-1] == ("initial", "queued")
    page = post(env, "new-face", THREE)
    assert "New face: 1 queued." in page and listed(page) == []
    assert env.jobs(THREE)[-1] == ("reroll", "queued")
    # A player with nothing to retry is skipped with the reason.
    page = post(env, "retry", MINE)
    assert "Retry: 1 skipped (no failed picture to retry)." in page
    # Never draw takes failed players off Review, its badge and the Players page's filter: no
    # retry is coming (user, 2026-11-24).
    env.draw(["sensitive"] * 9)  # the retries and the new face fail too
    page = unescape(env.client.get("/review").text)
    assert sorted(listed(page)) == [ONE, TWO, THREE] and review_badge(page) == "3"
    page = post(env, "pin", ONE, TWO)
    assert "2 player(s) pinned to never." in page
    assert listed(page) == [THREE] and review_badge(page) == "1"
    players = unescape(env.client.get("/players?picture=review&shown=everyone").text)
    assert f'href="/players/{THREE}"' in players and f'href="/players/{ONE}"' not in players
    own = unescape(env.client.get(f"/players/{ONE}").text)
    assert ">in review</a>" not in own and "pinned: Never draw" in own
    conn = db.connect(env.db_path)
    try:
        assert conn.execute("SELECT pin FROM players WHERE unique_id = ?", (ONE,)).fetchone()[
            0
        ] == ("never")
    finally:
        conn.close()
    # Clearing the pin lists the player again: the failed job is still their latest.
    r = post_player_action(env.client, f"/players/{ONE}/pin", data={"pin": "", "next": "/review"})
    page = unescape(env.client.get(r.headers["location"]).text)
    assert sorted(listed(page)) == [ONE, THREE] and review_badge(page) == "2"


def test_the_reasons_are_in_plain_words(env: Env) -> None:
    page = unescape(env.client.get("/review").text)
    assert '<span class="b bad">Not rated safe (sensitive)</span>' in page
    assert 'title="rated sensitive (general' in page  # the check's own words on hover
    korean = unescape(env.client.get("/review", headers=KOREAN_WINDOWS).text)
    assert "안전 등급이 아님(sensitive)" in korean and "나란히 비교" in korean
    en = Translator("en")
    for raw, said in (
        ("content: nude", "Content not allowed: nude"),
        ("hair colour not seen", "Hair colour not seen"),
        ("eye colour not seen", "Eye colour not seen"),
        ("not one girl: no solo, 2girls", "Not one person: no solo, 2girls"),
        ("no figure found against the background", "No person found against the background"),
        ("head starts 41% down the picture", "Head too low (41% down)"),
        ("something new", "something new"),
    ):
        assert check_failure_text(raw, en) == said


def test_the_candidates_side_by_side_with_the_picture_fm27_shows(env: Env) -> None:
    page = env.client.get("/review").text
    dialog = re.search(rf'<dialog class="compare" id="compare-{ONE}".*?</dialog>', page, re.DOTALL)
    assert dialog is not None
    assert page.count(f'data-compare="compare-{ONE}"') == 1  # its Compare button opens it
    # Group B draws one candidate a try, and a job has three tries.
    assert len(re.findall(r'<img data-src="/files\?path=', dialog[0])) == 3
    # Each picture opens in the lightbox (Phase 17b): the thumbnails as one group, the
    # dialog's pictures as another.
    outside = page.replace(dialog[0], "")
    assert (
        len(
            re.findall(
                rf'data-lightbox="review-{ONE}" data-caption="[^"]+ · Candidate \d"', outside
            )
        )
        == 3
    )
    assert len(re.findall(rf'data-lightbox="compare-{ONE}"', dialog[0])) == 3
    assert 'target="_blank"' not in page
    assert "Now in FM27" not in dialog[0]  # a first picture: nothing to compare with yet

    # A new picture for a player who has one fails its checks: the current one is shown too.
    post(env, "retry", ONE)
    env.draw([])  # passes
    conn = db.connect(env.db_path)
    try:
        career = load_career(conn)
        assert career is not None
        manual.new_picture(conn, career.id, ONE, load_settings(conn, env.config))
        current = conn.execute(
            "SELECT id FROM images WHERE unique_id = ? AND is_current = 1", (ONE,)
        ).fetchone()[0]
    finally:
        conn.close()
    env.draw(["sensitive"] * 3)
    page = env.client.get("/review").text
    dialog = re.search(rf'<dialog class="compare" id="compare-{ONE}".*?</dialog>', page, re.DOTALL)
    assert dialog is not None
    assert f'<img data-src="/pictures/{current}" alt="Now in FM27">' in dialog[0]
    assert f'<a href="/pictures/{current}" data-lightbox="compare-{ONE}"' in dialog[0]
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    assert "dialog.showModal()" in script and "img.dataset.src" in script


# -- Pictures sent to Review (Phase 17m, D7) ---------------------------------------------------


def section(page: str, uid: int) -> str:
    found = re.search(
        rf'<section class="card review" id="review-{uid}">.*?</section>', page, re.DOTALL
    )
    assert found is not None, uid
    return found[0]


def test_a_picture_sent_here_is_listed_with_its_own_actions(env: Env) -> None:
    env.draw([], frozenset({"a"}))  # Mine Player's picture passes
    r = post_player_action(env.client, f"/players/{MINE}/flag", data={"next": "/review"})
    page = unescape(env.client.get(r.headers["location"]).text)
    assert "Sent to Review; the picture stays in FM27 until a new one passes." in page
    assert sorted(listed(page)) == sorted([ONE, TWO, THREE, MINE])
    mine = section(page, MINE)
    assert (
        f'<a href="/players/{MINE}">Mine Player</a> <span class="badge tone-accent review-flag"'
        ' title="You sent this picture here; it stays in FM27 until a new one passes">'
        "Sent by you</span>"
    ) in mine
    assert f'formaction="/players/{MINE}/new-picture"' in mine  # Redraw, not Retry
    assert f'formaction="/players/{MINE}/keep"' in mine and ">Keep picture</button>" in mine
    assert f"/players/{MINE}/retry" not in mine and f"/players/{MINE}/pin" in mine
    assert re.search(r'<img src="/pictures/\d+\?size=192"[^>]*alt="Now in FM27">', mine)
    assert '<span class="b ok">Now in FM27</span>' in mine and "data-compare" not in mine
    one = section(page, ONE)  # a failed picture keeps its own
    assert f"/players/{ONE}/retry" in one and "/keep" not in one and "Sent by you" not in one
    # The bar can redraw while such a picture is listed.
    assert re.search(r'name="action" value="new-picture" data-needs-tick', page)
    korean = env.client.get("/review", headers=KOREAN_WINDOWS).text
    assert ">직접 보냄</span>" in korean and ">그림 유지</button>" in korean
    # The Players list's "in review" filter and the player's page say so too.
    players = unescape(env.client.get("/players?picture=review&shown=everyone").text)
    assert f'href="/players/{MINE}"' in players and ">sent to Review</a>" in players
    own = unescape(env.client.get(f"/players/{MINE}").text)
    assert ">sent to Review</a>" in own and f'action="/players/{MINE}/keep"' in own

    # With only such pictures listed, the bar has nothing to retry.
    post(env, "new-face", ONE, TWO, THREE)
    page = unescape(env.client.get("/review").text)
    assert listed(page) == [MINE] and 'value="retry"' not in page
    assert '<button class="primary" name="action" value="new-picture" data-needs-tick' in page

    r = post_player_action(env.client, f"/players/{MINE}/keep", data={"next": "/review"})
    page = unescape(env.client.get(r.headers["location"]).text)
    assert "Picture kept and taken off Review." in page and listed(page) == []
    own = unescape(env.client.get(f"/players/{MINE}").text)
    assert f'action="/players/{MINE}/flag"' in own and ">Send to Review</button>" in own


def test_a_redraw_takes_a_picture_off_the_list_while_it_is_queued(env: Env) -> None:
    env.draw([], frozenset({"a"}))
    post_player_action(env.client, f"/players/{MINE}/flag", data={"next": "/review"})
    page = post(env, "new-picture", MINE, ONE)  # the bar's Redraw
    assert "Redraw: 1 queued, 1 skipped (no picture to draw again; use Draw)." in page
    assert MINE not in listed(page) and env.jobs(MINE)[-1] == ("redraw", "queued")
    # Nothing to send: no picture yet.
    r = post_player_action(env.client, f"/players/{ONE}/flag", data={"next": "/review"})
    assert "Kid One has no picture." in unescape(env.client.get(r.headers["location"]).text)
