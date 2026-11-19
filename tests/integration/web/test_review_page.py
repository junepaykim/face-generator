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

from facegen.datasource.kits import load_kits
from facegen.i18n import Translator
from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline import manual
from facegen.pipeline.worker import JobFilter, Worker, load_career
from facegen.web.messages import check_failure_text

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

    def draw(self, verdicts: list[str]) -> None:
        """Run the group B jobs with the tagger answering these verdicts, then "pass"."""
        conn = db.connect(self.db_path)
        try:
            career = load_career(conn)
            assert career is not None
            tagger = FakeTagger(verdicts)
            Worker(
                conn, load_settings(conn, self.config), load_kits(), career,
                renderer_factory=FakeRenderer, tagger_factory=lambda: tagger,
                fm_check=lambda: False, log=lambda line: None,
            ).run(JobFilter(groups=frozenset({"b"})))  # fmt: skip
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
    assert "1 player(s) pinned to never." in post(env, "pin", ONE)
    conn = db.connect(env.db_path)
    try:
        assert conn.execute("SELECT pin FROM players WHERE unique_id = ?", (ONE,)).fetchone()[
            0
        ] == ("never")
    finally:
        conn.close()


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
    assert f'data-compare="compare-{ONE}"' in page  # the button and each thumbnail open it
    # Group B draws one candidate a try, and a job has three tries.
    assert len(re.findall(r'<img data-src="/files\?path=', dialog[0])) == 3
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
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    assert "dialog.showModal()" in script and "img.dataset.src" in script
