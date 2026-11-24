"""The picture wall (Phase 12h): everyone's current picture, newest first, with filters, and
new pictures arriving without a reload."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

import pytest
from factories import ABROAD, NEWGEN, RIVAL, player, snapshot
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline import manual
from facegen.pipeline.sync import run_sync
from facegen.runtime.worker_service import WorkerService
from facegen.web import live, queries
from facegen.web.app import AppContext
from facegen.web.queries import GalleryFilter

WORLD = [
    player(1001, name="Mine Alpha", ca=150, pa=160),
    player(1002, name="Mine Beta", ca=120, pa=140),
    player(NEWGEN + 2, name="Rival Kid", ca=135, pa=160, club_id=RIVAL),
    player(NEWGEN + 4, name="Abroad Star", ca=170, pa=185, club_id=ABROAD),
]


@dataclass
class Env:
    client: TestClient
    worker: WorkerService
    db_path: Path
    config: Path

    def draw(self, n: int) -> None:
        for _ in range(n):
            self.worker.step()

    def conn(self):
        return db.connect(self.db_path)

    def jobs(self, uid: int) -> list[tuple[Any, ...]]:
        """The player's requested jobs still waiting, in every career."""
        with closing(self.conn()) as conn:
            return [
                tuple(r)
                for r in conn.execute(
                    "SELECT career_id, kind FROM jobs WHERE unique_id = ? AND manual = 1"
                    " AND status = 'queued' ORDER BY id",
                    (uid,),
                )
            ]


@pytest.fixture
def env(config_file: Path) -> Iterator[Env]:
    db_path = synced(config_file, WORLD)
    worker = fake_worker(config_file, db_path)
    client = client_for(config_file, db_path, worker)
    yield Env(client, worker, db_path, config_file)
    client.close()


def cards(page: str) -> list[str]:
    """The names on the wall, in order."""
    return re.findall(r'<a class="thumb-name" href="/players/\d+">([^<]+)</a>', unescape(page))


def ids(page: str) -> list[int]:
    return [int(i) for i in re.findall(r'class="thumb-card" data-id="(\d+):', page)]


def test_before_any_picture_the_wall_says_so(env: Env) -> None:
    page = env.client.get("/gallery").text
    assert "No pictures yet." in page and 'href="/">See the progress on the Dashboard' in page
    assert 'href="/gallery" class="on"' in page  # in the sidebar, after the dashboard
    korean = env.client.get("/gallery", headers=KOREAN_WINDOWS).text
    assert "아직 그림이 없습니다." in korean and "갤러리" in korean


def test_every_current_picture_newest_first(env: Env) -> None:
    env.draw(4)
    page = env.client.get("/gallery").text
    assert sorted(cards(page)) == ["Abroad Star", "Mine Alpha", "Mine Beta", "Rival Kid"]
    assert ids(page) == sorted(ids(page), reverse=True)
    assert "4 pictures" in page
    assert re.search(
        r'srcset="/pictures/(\d+)\?size=192&amp;v=(v\d+\.png) 1x, '
        r'/pictures/\1\?size=384&amp;v=\2 2x"',
        page,
    )
    assert "그림 4장" in env.client.get("/gallery", headers=KOREAN_WINDOWS).text
    # The dashboard's link goes here.
    assert 'href="/gallery">View all in Gallery →</a>' in env.client.get("/status").text

    # A picture FM27 no longer shows is not on the wall.
    with env.conn() as conn:
        conn.execute("UPDATE images SET is_current = 0 WHERE unique_id = 1002")
    assert "Mine Beta" not in cards(env.client.get("/gallery").text)


def test_filters_and_sorts(env: Env) -> None:
    env.draw(4)
    get = lambda query: cards(env.client.get(f"/gallery?{query}").text)
    assert sorted(get("where=mine")) == ["Mine Alpha", "Mine Beta"]
    assert sorted(get("where=league")) == ["Mine Alpha", "Mine Beta", "Rival Kid"]
    assert get("where=other") == ["Abroad Star"]
    assert get("q=kid") == ["Rival Kid"]
    assert get(f"q={NEWGEN + 4}") == ["Abroad Star"]
    assert get("sort=name") == ["Abroad Star", "Mine Alpha", "Mine Beta", "Rival Kid"]
    assert get("sort=ca") == ["Abroad Star", "Mine Alpha", "Rival Kid", "Mine Beta"]
    groups = {g: get(f"group={g}") for g in ("a", "b", "c")}
    assert sorted(name for names in groups.values() for name in names) == sorted(get(""))
    assert "Mine Alpha" in groups["a"]
    assert get("role=staff") == []
    page = env.client.get("/gallery?role=staff").text
    assert "No pictures match." in page and 'href="/gallery">Clear' in page
    # Anything malformed falls back to its default.
    f = GalleryFilter.from_query({"where": "moon", "group": "z", "sort": "x", "page": "-3"})
    assert f == GalleryFilter()


def test_the_search_is_the_players_pages(env: Env) -> None:
    env.draw(4)

    def get(q: str) -> list[str]:
        response = env.client.get("/gallery", params={"q": q})
        assert response.status_code == 200
        return cards(response.text)

    assert get("_") == [] and get("%") == []  # plain characters, not LIKE's wildcards
    assert get("9" * 25) == []  # beyond SQLite's integers: no ID, so a name search
    assert get("²") == []  # a digit, but no number
    assert get(f"00{NEWGEN + 4}") == ["Abroad Star"]


def test_pages_of_pictures(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.draw(4)
    monkeypatch.setattr(queries, "GALLERY_PER_PAGE", 3)
    first = env.client.get("/gallery?sort=name").text
    assert cards(first) == ["Abroad Star", "Mine Alpha", "Mine Beta"]
    assert 'href="/gallery?sort=name&amp;page=2" rel="next"' in first
    assert cards(env.client.get("/gallery?sort=name&page=2").text) == ["Rival Kid"]
    # The pager beside the count and under the wall (Phase 17b): its page box belongs to a GET
    # form carrying the filters. A page past the last shows the last; the live part agrees.
    boxes = r'<input type="number" name="page" form="(jump-\w+)" min="1" max="(\d+)" value="(\d+)"'
    assert re.findall(boxes, first) == [("jump-top", "2", "1"), ("jump-bottom", "2", "1")]
    assert re.search(r'4 pictures</span><form id="jump-top" method="get" action="/gallery"', first)
    hidden = re.findall(r'<form id="jump-top"[^>]*>(.*?)</form>', first)[0]
    assert hidden == '<input type="hidden" name="sort" value="name">'
    for url in ("/gallery?sort=name&page=50", "/gallery/part?sort=name&page=50"):
        page = env.client.get(url).text
        assert cards(page) == ["Rival Kid"], url
        assert re.findall(boxes, page)[0] == ("jump-top", "2", "2"), url
    assert cards(env.client.get("/gallery?sort=name&page=0").text) == cards(first)
    gallery = env.client.get("/gallery?sort=name&page=50").text
    assert 'data-refresh="/gallery/part?sort=name&amp;page=2"' in gallery


def test_a_new_picture_reaches_an_open_wall(env: Env) -> None:
    env.draw(2)
    page = env.client.get("/gallery?where=mine").text
    assert 'data-refresh="/gallery/part?where=mine" data-topic="pictures"' in page
    rev = re.search(r'data-rev="([^"]+)"', page)[1]
    settings = load_settings(defaults_path=env.config)
    ctx = AppContext(env.db_path, env.config, env.worker)
    conn = db.connect(settings.paths.db_path)
    try:
        assert live.snapshot(ctx, conn)["pictures"]["rev"] == rev
        env.draw(1)
        assert live.snapshot(ctx, conn)["pictures"]["rev"] != rev
    finally:
        conn.close()
    part = env.client.get("/gallery/part?where=mine").text
    assert part.lstrip().startswith('<div class="list-head" id="wall-count">')
    assert '<section id="wall-grid" data-pictures>' in part and "<html" not in part
    # The page's script merges the wall like the dashboard's latest pictures.
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    assert 'old.hasAttribute("data-pictures")' in script
    assert '["status", "log", "pictures"]' in script
    assert "form[data-autosubmit]" in script


# The page's script asks for JSON; its form token goes in a header (the client holds it).
BY_SCRIPT = {"Accept": "application/json"}


def card_form(page: str, uid: int) -> str:
    """The quick actions over a player's card."""
    found = re.search(
        rf'<form class="thumb-acts" method="post" action="/players/{uid}/new-picture"'
        r".*?</form>",
        page,
        re.DOTALL,
    )
    assert found is not None
    return found[0]


def test_new_picture_and_new_face_from_a_card(env: Env) -> None:
    """Phase 17f: each card on the wall and in the dashboard's latest pictures has Redraw (was
    New picture, renamed in 17g) and New face over its picture. The script posts them (the token in a header, the career the
    card was drawn for) and gets JSON back, so the card says what happened and the page stays;
    without the script the form posts as usual and comes back to the same wall."""
    env.draw(4)
    for url, back in (
        ("/gallery?where=mine&sort=name", "/gallery?where=mine&sort=name"),
        ("/", "/"),
    ):
        page = unescape(env.client.get(url).text)
        form = card_form(page, 1001)
        assert 'name="csrf_token"' in form and 'name="career_id"' in form, url
        assert f'name="next" value="{back}"' in form, url
        assert 'role="group" aria-label="Mine Alpha"' in form, url
        assert '<button title="A new picture of the same character">Redraw</button>' in form
        assert (
            '<button formaction="/players/1001/new-face"'
            ' title="A new character, then a picture of it">New face</button>'
        ) in form, url
        assert '</form><span class="thumb-note" role="status"></span></div>' in page, url
    korean = env.client.get("/gallery", headers=KOREAN_WINDOWS).text
    assert ">다시 그리기</button>" in korean and ">새 얼굴</button>" in korean
    career = re.search(r'name="career_id" value="(\d+)"', form)[1]

    r = env.client.post("/players/1001/new-picture", data={"career_id": career}, headers=BY_SCRIPT)
    assert r.status_code == 200 and r.json() == {"ok": True, "message": "Redraw queued."}
    r = env.client.post(
        "/players/1002/new-face", data={"career_id": career}, headers=BY_SCRIPT | KOREAN_WINDOWS
    )
    assert r.json() == {
        "ok": True,
        "message": "새 얼굴을 대기열에 넣었습니다. 새 캐릭터로 바로 다음에 생성합니다.",
    }
    assert env.jobs(1001) == [(int(career), "redraw")]
    assert env.jobs(1002) == [(int(career), "reroll")]

    # Without the script: back to the wall it came from, with the message.
    r = env.client.post(
        f"/players/{NEWGEN + 2}/new-picture",
        data={"career_id": career, "next": "/gallery?where=league"},
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/gallery?where=league&msg=Redraw%20queued."
    assert env.jobs(NEWGEN + 2) == [(int(career), "redraw")]


def test_send_to_review_from_a_card(env: Env) -> None:
    """Phase 17m (D7): the card's third action sends its picture to Review. The note says so
    (not "Queued"), nothing is queued, the picture stays, and Review lists the player."""
    env.draw(4)
    for url in ("/gallery", "/"):
        form = card_form(unescape(env.client.get(url).text), 1001)
        assert (
            '<button formaction="/players/1001/flag" title="List it in Review; the picture stays'
            ' in FM27 until a new one passes">Send to Review</button></form>'
        ) in form, url
    assert ">검토로 보내기</button>" in env.client.get("/gallery", headers=KOREAN_WINDOWS).text
    career = re.search(r'name="career_id" value="(\d+)"', env.client.get("/gallery").text)[1]
    r = env.client.post("/players/1001/flag", data={"career_id": career}, headers=BY_SCRIPT)
    assert r.json() == {
        "ok": True,
        "message": "Sent to Review; the picture stays in FM27 until a new one passes.",
        "note": "Sent to Review",
    }
    r = env.client.post(
        "/players/1001/flag", data={"career_id": career}, headers=BY_SCRIPT | KOREAN_WINDOWS
    )
    assert r.json()["ok"] and r.json()["note"] == "검토로 보냄"  # again: still there
    assert env.jobs(1001) == []
    review = unescape(env.client.get("/review").text)
    assert '<a href="/players/1001">Mine Alpha</a> <span class="badge tone-accent' in review
    assert '<span class="count badge badge-count">1</span>' in review  # the sidebar's count
    page = env.client.get("/gallery").text  # the card shows the answer's own note
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    assert 'say(data.note || text("js.queued"), "ok", data.message)' in script


def test_a_quick_action_says_why_it_was_not_done(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.draw(4)
    career = re.search(r'name="career_id" value="(\d+)"', env.client.get("/gallery").text)[1]
    with closing(env.conn()) as conn:
        conn.execute("UPDATE players SET pin = 'never' WHERE unique_id = 1002")
    r = env.client.post("/players/1002/new-picture", data={"career_id": career}, headers=BY_SCRIPT)
    assert r.status_code == 200
    assert r.json() == {"ok": False, "error": "Mine Beta: set to never draw; clear the pin first"}
    r = env.client.post(
        "/players/1002/new-face", data={"career_id": career}, headers=BY_SCRIPT | KOREAN_WINDOWS
    )
    assert r.json()["ok"] is False and "그리지 않음" in r.json()["error"]
    assert env.jobs(1002) == []

    # An unexpected error is logged and worded as a page's problem, with its own text.
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(manual, "new_picture", broken)
    r = env.client.post("/players/1001/new-picture", data={"career_id": career}, headers=BY_SCRIPT)
    assert r.status_code == 500 and r.json()["ok"] is False
    assert "RuntimeError: the disk went away" in r.json()["error"]
    assert env.jobs(1001) == []


def test_a_quick_action_needs_the_token_and_the_cards_career(env: Env) -> None:
    env.draw(4)
    career = re.search(r'name="career_id" value="(\d+)"', env.client.get("/gallery").text)[1]
    token = env.client.headers.pop("X-CSRF-Token")
    try:
        for headers in ({}, {"X-CSRF-Token": "wrong"}):
            r = env.client.post(
                "/players/1001/new-picture",
                data={"career_id": career},
                headers=BY_SCRIPT | headers,
            )
            assert r.status_code == 403  # the script then asks for a reload
        # The header alone is enough: the script sends the card's form as it is.
        r = env.client.post(
            "/players/1001/new-picture",
            data={"career_id": career},
            headers=BY_SCRIPT | {"X-CSRF-Token": token},
        )
        assert r.json()["ok"] is True
    finally:
        env.client.headers["X-CSRF-Token"] = token
    assert env.jobs(1001) == [(int(career), "redraw")]

    # A card drawn for another career (a newer save was read meanwhile) changes neither.
    with closing(env.conn()) as conn:
        run_sync(
            conn,
            snapshot(WORLD, career="two", file_name="two.fm"),
            load_settings(conn, env.config),
            save_file="two.fm",
            save_mtime=1_800_000_000.0,
        )
    for data in ({"career_id": career}, {}):
        r = env.client.post("/players/1002/new-picture", data=data, headers=BY_SCRIPT)
        assert r.json() == {
            "ok": False,
            "error": "This form no longer matches the active career. Reload the page and try again.",
        }
    assert env.jobs(1002) == []


def test_a_card_names_the_role_and_keeps_the_time_for_its_tooltip(env: Env) -> None:
    """Phase 17f: a role badge where the date was; when the picture was made is the picture's
    tooltip and the lightbox's caption, in this PC's time (the script)."""
    env.draw(4)
    with closing(env.conn()) as conn:
        conn.execute(
            "UPDATE players SET is_staff = 1, staff_job = 'director of football'"
            " WHERE unique_id = 1002"
        )
        conn.execute(
            "UPDATE players SET media_role = 'journalist', local_media = 1, club_id = NULL"
            " WHERE unique_id = ?",
            (NEWGEN + 2,),
        )
        conn.execute(
            "UPDATE players SET media_role = 'press officer', club_id = NULL WHERE unique_id = ?",
            (NEWGEN + 4,),
        )
        made = conn.execute(
            "SELECT created_at FROM images WHERE unique_id = 1001 AND is_current = 1"
        ).fetchone()[0]
    badge = r'<span class="badge thumb-role" title="(?P<role>[^"]+)">(?P=role)</span>'
    card = (
        r'<a class="thumb-name" href="/players/\d+">([^<]+)</a>\s*'
        r'<span class="thumb-club muted small">([^<]+)</span>\s*'
        r'<span class="thumb-meta"><span class="grp[^>]*>[^<]+</span> ' + badge + "</span>"
    )
    page = unescape(env.client.get("/gallery?sort=name").text)
    assert [(name, club, role) for name, club, role in re.findall(card, page)] == [
        ("Abroad Star", "elsewhere", "Press officer"),
        ("Mine Alpha", "Club 100", "Player"),
        ("Mine Beta", "Club 100", "Director of football"),
        ("Rival Kid", "my club's country", "Journalist"),
    ]
    korean = unescape(env.client.get("/gallery?sort=name", headers=KOREAN_WINDOWS).text)
    assert [role for _, _, role in re.findall(card, korean)] == [
        "홍보 담당자",
        "선수",
        "Director of football",
        "기자",
    ]
    assert "<time" not in page[page.index('id="wall-grid"') :]
    assert f'data-made="{made}" title="Made {made[:16].replace("T", " ")}"' in page
    assert f'data-made="{made}" title="{made[:16].replace("T", " ")} 생성"' in korean
    texts = re.search(r'<script type="application/json" id="texts">(.*?)</script>', page)[1]
    assert '"gallery.made": "Made {time}"' in texts and '"js.queued": "Queued"' in texts
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    for piece in ("[data-made][title]", "link.dataset.made", '"lb-when"'):
        assert piece in script, piece


def test_clear_is_a_button_while_a_filter_is_on(env: Env) -> None:
    env.draw(1)
    clear = '<a class="button filter-clear" href="/gallery{}">Clear</a>'
    page = env.client.get("/gallery?where=mine&sort=name").text
    assert clear.format("?sort=name") in page  # the sort is not a filter: it stays
    assert clear.format("") in env.client.get("/gallery?group=a").text
    assert "filter-clear" not in env.client.get("/gallery?sort=name").text
    assert ">초기화</a>" in env.client.get("/gallery?role=staff", headers=KOREAN_WINDOWS).text


def test_quick_actions_come_with_the_wall_and_keep_through_live_updates(env: Env) -> None:
    """A live update sends the cards with their actions; the script keeps a card shown (and its
    note) until its picture changes, and keeps the keyboard's focus on its buttons."""
    env.draw(2)
    page = env.client.get("/gallery?where=mine").text
    part = env.client.get("/gallery/part?where=mine").text
    forms = re.findall(r'<form class="thumb-acts".*?</form>', part, re.DOTALL)
    assert len(forms) == 2 and forms == re.findall(
        r'<form class="thumb-acts".*?</form>', page, re.DOTALL
    )
    assert all('name="csrf_token"' in f and 'name="career_id"' in f for f in forms)
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    for piece in (
        'form.matches("form[data-quick]")',
        '"X-CSRF-Token": form.querySelector(\'input[name="csrf_token"]\').value',
        '"Accept": "application/json"',
        "focused.focus({ preventScroll: true })",
        '"js.quick_reload"',
    ):
        assert piece in script, piece
    style = env.client.get(re.search(r'href="(/static/style.css[^"]*)"', page)[1]).text
    assert ".thumb-frame:hover .thumb-acts, .thumb-acts:focus-within { opacity: 1; }" in style
