"""The picture wall (Phase 12h): everyone's current picture, newest first, with filters, and
new pictures arriving without a reload."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import pytest
from factories import ABROAD, NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.persistence import db
from facegen.persistence.settings import load_settings
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


@pytest.fixture
def env(config_file: Path) -> Iterator[Env]:
    db_path = synced(config_file, WORLD)
    worker = fake_worker(config_file, db_path)
    client = client_for(config_file, db_path, worker)
    yield Env(client, worker, db_path, config_file)
    client.close()


def cards(page: str) -> list[str]:
    """The names on the wall, in order."""
    return re.findall(r'<span class="thumb-name">([^<]+)</span>', unescape(page))


def ids(page: str) -> list[int]:
    return [
        int(i) for i in re.findall(r'class="thumb-card" href="/players/\d+" data-id="(\d+)"', page)
    ]


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
    assert re.search(r'srcset="/pictures/\d+\?size=192 1x, /pictures/\d+\?size=384 2x"', page)
    assert "그림 4장" in env.client.get("/gallery", headers=KOREAN_WINDOWS).text
    # The dashboard's link goes here.
    assert 'href="/gallery">Gallery</a>' in env.client.get("/status").text

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
    assert "No pictures match." in page and 'href="/gallery">clear' in page
    # Anything malformed falls back to its default.
    f = GalleryFilter.from_query({"where": "moon", "group": "z", "sort": "x", "page": "-3"})
    assert f == GalleryFilter()


def test_pages_of_pictures(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.draw(4)
    monkeypatch.setattr(queries, "GALLERY_PER_PAGE", 3)
    first = env.client.get("/gallery?sort=name").text
    assert cards(first) == ["Abroad Star", "Mine Alpha", "Mine Beta"]
    assert "page 1 of 2" in first and 'href="/gallery?sort=name&amp;page=2"' in first
    assert cards(env.client.get("/gallery?sort=name&page=2").text) == ["Rival Kid"]


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
    assert part.lstrip().startswith('<p class="muted small" id="wall-count">')
    assert '<section id="wall-grid" data-pictures>' in part and "<html" not in part
    # The page's script merges the wall like the dashboard's latest pictures.
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    assert 'old.hasAttribute("data-pictures")' in script
    assert '["status", "log", "pictures"]' in script
    assert "form[data-autosubmit]" in script
