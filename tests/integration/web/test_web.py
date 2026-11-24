"""The web UI through FastAPI's test client, with the worker driven step by step."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from html import unescape
from pathlib import Path
from typing import Any

import pytest
from factories import (
    ABROAD,
    MANUAL_MODE,
    NEWGEN,
    PLAIN_PERSONALITY,
    RIVAL,
    WOMENS,
    player,
    snapshot,
)
from fakes import FakeRenderer, FakeTagger
from fastapi.testclient import TestClient
from support.paths import ROOT
from support.web import KOREAN_WINDOWS, post_player_action

from facegen.core.records import Snapshot
from facegen.datasource.kits import load_kits
from facegen.datasource.photos import PhotoIndex, PhotoScan
from facegen.persistence import db
from facegen.persistence.settings import (
    Settings,
    is_fixed,
    load_defaults,
    load_settings,
    set_overrides,
)
from facegen.pipeline.sync import run_sync
from facegen.pipeline.worker import JobFilter, Worker, load_career
from facegen.service import SyncService, WorkerService
from facegen.web.app import AppContext, SetupState, create_app

MINE = 1001  # my club: A
STAR = 1002  # pre-existing at another club: hand-picked only
B_NEWGEN = NEWGEN + 2  # rival in my league, CA 90, PA 152: B
WOMAN = NEWGEN + 6  # women's club
ABROAD_KID = NEWGEN + 8  # abroad, CA 120: in no group until B's bar drops

BASE = "http://127.0.0.1:8765"


def world() -> list:
    return [
        player(MINE, name="Mine Player", age=29),
        player(
            STAR,
            name="Famous Striker",
            ca=185,
            pa=190,
            club_id=ABROAD,
            personality=replace(PLAIN_PERSONALITY, pressure=17, temperament=4),
        ),
        player(B_NEWGEN, name="Rival Kid", ca=90, pa=152, club_id=RIVAL),
        player(WOMAN, name="Womens Player", ca=100, pa=190, club_id=WOMENS),
        player(ABROAD_KID, name="Abroad Kid", ca=120, pa=130, club_id=ABROAD),
    ]


class FakeSource:
    def load(self, path: Path) -> Snapshot:
        return snapshot(world(), file_name="career.fm")


@dataclass
class Env:
    client: TestClient
    worker: WorkerService
    syncer: SyncService
    settings: Settings
    config: Path

    def conn(self) -> sqlite3.Connection:
        return db.connect(self.settings.paths.db_path)

    def one(self, sql: str, *args: Any) -> Any:
        c = self.conn()
        try:
            row = c.execute(sql, args).fetchone()
            return None if row is None else row[0]
        finally:
            c.close()

    def drain(self) -> None:
        for _ in range(20):
            if self.worker.step() > 0:
                return
        raise AssertionError("the worker never went idle")


def make_env(config: Path, *, synced: bool) -> Env:
    settings = load_settings(defaults_path=config)
    if synced:
        conn = db.connect(settings.paths.db_path)
        # My league's cutoffs as numbers: the made-up league holds too few candidates for shares.
        set_overrides(conn, MANUAL_MODE, config)
        run_sync(conn, snapshot(world()), load_settings(conn, config), save_file="career.fm")
        conn.close()
    worker = WorkerService(
        settings.paths.db_path,
        renderer_factory=FakeRenderer,
        tagger_factory=FakeTagger,
        defaults_path=config,
        kits=load_kits(),
        fm_check=lambda: False,
    )
    syncer = SyncService(settings.paths.db_path, FakeSource, defaults_path=config)
    app = create_app(AppContext(settings.paths.db_path, config, worker, syncer))
    client = TestClient(app, base_url=BASE, follow_redirects=False)
    token = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/").text)
    assert token is not None
    client.headers["X-CSRF-Token"] = token[1]
    return Env(client, worker, syncer, settings, config)


@pytest.fixture
def env(config_file: Path) -> Iterator[Env]:
    e = make_env(config_file, synced=True)
    yield e
    e.client.close()


def follow(env: Env, response: Any) -> str:
    assert response.status_code == 303, response.text
    page = env.client.get(response.headers["location"])
    assert page.status_code == 200
    return page.text


def test_every_page_works_before_the_first_sync(config_file: Path) -> None:
    e = make_env(config_file, synced=False)
    for url in ("/", "/status", "/players", "/review", "/settings"):
        r = e.client.get(url)
        assert r.status_code == 200, url
    assert "no career yet" in e.client.get("/").text
    assert "No careers yet" not in e.client.get("/players").text
    text = follow(e, e.client.post("/sync"))
    assert "Sync started" in text
    e.syncer.join(10)
    # There is no save in the (test) games folder.
    assert "No save found" in e.client.get("/").text


def test_the_status_block_names_the_career_so_the_page_reloads_after_the_first_sync(
    config_file: Path,
) -> None:
    e = make_env(config_file, synced=False)
    assert 'data-career=""' in e.client.get("/status").text  # app.js reloads when it changes
    e.client.close()
    e = make_env(config_file, synced=True)
    status = e.client.get("/status").text
    assert re.search(r'data-career="\d+"', status)
    assert 'data-career="' in e.client.get("/").text  # the page starts with the same mark
    e.client.close()


def test_dashboard(env: Env) -> None:
    text = env.client.get("/").text
    assert re.search(r'<span class="grp grp-a" data-tip="[^"]+">Detailed</span>', text)
    assert "Mine Player" in text  # queued
    assert 'data-refresh="/status"' in text
    status = env.client.get("/status").text
    assert "<html" not in status and 'id="now"' in status
    # The Rust code is part of every install now: no row about it (user, 2026-11-17).
    assert "Rust code" not in env.client.get("/settings/advanced").text
    env.drain()
    text = env.client.get("/status").text
    # Group A, then group B: both drawn without a request since Phase 4.
    assert "Mine Player" in text and "Latest pictures" in text
    assert "2 made, 0 sent to review" in env.client.get("/settings/advanced").text


def test_players_list_filters(env: Env) -> None:
    def names(query: str) -> set[str]:
        text = env.client.get(f"/players?shown=everyone&{query}").text
        return set(re.findall(r'<a href="/players/\d+">([^<]+)</a>', text))

    # By default only the picture targets (Phase 17h): in a group or queued here.
    listed = env.client.get("/players").text
    assert set(re.findall(r'<a href="/players/\d+">([^<]+)</a>', listed)) == {
        "Mine Player",
        "Rival Kid",
    }
    assert names("") == {"Mine Player", "Famous Striker", "Rival Kid", "Abroad Kid"}
    assert names("scope=all") >= {"Womens Player"}
    assert names("q=striker") == {"Famous Striker"}
    assert names(f"q={STAR}") == {"Famous Striker"}
    assert names("origin=newgen") == {"Rival Kid", "Abroad Kid"}
    assert names("origin=existing") == {"Mine Player", "Famous Striker"}
    assert names("league=mine") == {"Mine Player", "Rival Kid"}
    assert names("league=other") == {"Famous Striker", "Abroad Kid"}
    assert names(f"club_id={ABROAD}") == {"Famous Striker", "Abroad Kid"}
    assert names(f"club=Club {RIVAL}") == {"Rival Kid"}
    assert names("ca_min=100&ca_max=150") == {"Abroad Kid"}
    assert names("pa_min=150&pa_max=160") == {"Rival Kid"}
    assert names("group=a") == {"Mine Player"}
    assert names("group=none") == {"Famous Striker", "Abroad Kid"}
    assert names("picture=queued") == {"Mine Player", "Rival Kid"}
    assert names("picture=yes") == set()
    assert names("sort=name&page=99") == names("")  # past the last page: the last (Phase 17b)
    assert env.client.get("/players?ca_min=abc&league=bogus").status_code == 200


def test_players_pager_preserves_filter_and_sort(env: Env) -> None:
    people = [player(2000 - i, name=f"Trial player {i:03d}", club_id=ABROAD) for i in range(101)]
    conn = env.conn()
    try:
        run_sync(conn, snapshot(people), env.settings, save_file="career.fm")
    finally:
        conn.close()

    first_url = "/players?q=Trial+player&shown=everyone&sort=name"
    second_url = f"{first_url}&page=2"
    rows = r'<a href="/players/(\d+)">([^<]+)</a>'
    expected = [(str(p.unique_id), p.name) for p in people]
    first = unescape(env.client.get(first_url).text)
    assert "101 players" in first
    assert page_boxes(first) == [("jump-top", "2", "1"), ("jump-bottom", "2", "1")]
    assert re.findall(rows, first) == expected[:100]
    assert f'href="{second_url}" rel="next"' in first and 'rel="prev"' not in first
    second = unescape(env.client.get(second_url).text)
    assert page_boxes(second) == [("jump-top", "2", "2"), ("jump-bottom", "2", "2")]
    assert re.findall(rows, second) == expected[100:]
    assert f'href="{first_url}" rel="prev"' in second and 'rel="next"' not in second


def page_boxes(page: str) -> list[tuple[str, str, str]]:
    """Each pager's page box on a page: its form, the last page and the page shown."""
    return re.findall(
        r'<input type="number" name="page" form="([^"]+)" min="1" max="(\d+)" value="(\d+)"', page
    )


def test_players_pager_jumps_to_a_typed_page(env: Env) -> None:
    """Phase 17b: ‹ [ 1 ] / 2 › beside the count and under the list. The page box belongs to a
    GET form carrying the filters, outside the bulk form (forms cannot nest); a page past the
    last shows the last one, and one below the first the first."""
    people = [player(2000 - i, name=f"Trial player {i:03d}", club_id=ABROAD) for i in range(101)]
    conn = env.conn()
    try:
        run_sync(conn, snapshot(people), env.settings, save_file="career.fm")
    finally:
        conn.close()
    rows = r'<a href="/players/(\d+)">([^<]+)</a>'
    expected = [(str(p.unique_id), p.name) for p in people]
    page = unescape(
        env.client.get("/players?q=Trial+player&sort=name&group=any&shown=everyone").text
    )
    forms = dict(
        re.findall(
            r'<form id="(jump-\w+)" method="get" action="/players" novalidate data-pager hidden>'
            r"(.*?)</form>",
            page,
        )
    )
    assert set(forms) == {"jump-top", "jump-bottom"}
    hidden = r'<input type="hidden" name="([^"]+)" value="([^"]*)">'
    for fields in forms.values():  # the filters as set, defaults left out, no page
        assert re.findall(hidden, fields) == [
            ("q", "Trial player"),
            ("shown", "everyone"),
            ("sort", "name"),
        ]
    bulk = page.index('id="bulk"')
    assert page.index('<form id="jump-top"') < bulk < page.index('form="jump-top"')
    assert page.index("</form>", bulk) < page.index('<form id="jump-bottom"')
    assert re.search(r'101 players</span>\s*<nav class="pager"', page)

    def jump(number: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        typed = dict(re.findall(hidden, forms["jump-top"]), page=number)
        text = unescape(env.client.get("/players", params=typed).text)
        return re.findall(rows, text), page_boxes(text)

    assert jump("2") == (expected[100:], [("jump-top", "2", "2"), ("jump-bottom", "2", "2")])
    assert jump("9999") == jump("2")  # past the last page: the last
    assert jump("100001")[0] == expected[100:]
    assert jump("0")[0] == expected[:100] and jump("-4")[0] == expected[:100]
    assert jump("x")[0] == expected[:100]
    # One page: no pager.
    single = env.client.get("/players?q=Trial+player+000&shown=everyone").text
    assert "1 players" in single and page_boxes(single) == [] and "data-pager" not in single
    korean = env.client.get("/players?q=Trial+player&shown=everyone", headers=KOREAN_WINDOWS).text
    assert 'aria-label="다음 쪽"' in korean and 'aria-label="쪽 번호"' in korean


# A link to a whole picture: the master by image ID or a file (candidate, reference).
PICTURE_LINK = re.compile(r'<a [^>]*href="(?:/pictures/\d+|/files\?path=[^"]+)"[^>]*>')


def test_every_picture_opens_in_the_lightbox(env: Env) -> None:
    """Phase 17b: a picture opens over the page, never in the web browser (the app's window
    hands a target="_blank" link to it). The link keeps its address, so it still opens the
    picture without the script; the script groups links by their data-lightbox value."""
    env.drain()
    follow(env, post_player_action(env.client, f"/players/{MINE}/new-picture"))
    env.drain()
    v1 = env.one("SELECT id FROM images WHERE unique_id = ? AND version = 1", MINE)
    v2 = env.one("SELECT id FROM images WHERE unique_id = ? AND is_current = 1", MINE)
    for url in ("/", "/players", "/gallery", f"/players/{MINE}", "/review"):
        page = env.client.get(url).text
        assert 'target="_blank"' not in page, url
        assert page.count('<dialog class="lightbox" id="lightbox"') == 1, url
        links = PICTURE_LINK.findall(page)
        assert links or url == "/review", url
        assert [link for link in links if "data-lightbox=" not in link] == [], url

    page = unescape(env.client.get(f"/players/{MINE}").text)
    # The current picture and the history (the script counts the current one once), the
    # candidates and the reference.
    versions = r'<a href="/pictures/(\d+)" data-lightbox="versions" data-caption="([^"]+)">'
    assert re.findall(versions, page) == [
        (str(v2), "Mine Player · v2"),
        (str(v2), "Mine Player · v2"),
        (str(v1), "Mine Player · v1"),
    ]
    assert re.search(r'<a href="/files\?path=[^"]+" data-lightbox="reference"', page)
    candidates = re.findall(r'data-lightbox="candidates" data-caption="([^"]+)"', page)
    assert candidates and candidates[0] == "Mine Player · v2 · Candidate 1"
    players = unescape(env.client.get("/players").text)
    assert (
        f'<a class="face-link" href="/pictures/{v2}" data-lightbox="players"'
        f' data-caption="Mine Player" data-player="/players/{MINE}">'
    ) in players
    assert f'<a href="/players/{MINE}">Mine Player</a>' in players  # the name: the player page

    korean = env.client.get(f"/players/{MINE}", headers=KOREAN_WINDOWS).text
    assert 'aria-label="그림 크게 보기"' in korean and ">원래 크기</button>" in korean
    script = env.client.get(re.search(r'src="(/static/app.js[^"]*)"', page)[1]).text
    for piece in (
        'closest("a[data-lightbox]")',
        "{ passive: false }",  # the wheel zooms the picture, not the page
        "setPointerCapture",
        'e.key === "0"',
        '"dblclick"',
        "opener.focus(",
        "form[data-pager]",
    ):
        assert piece in script, piece


def test_only_outside_pages_open_in_the_web_browser() -> None:
    """Phase 17b: in the app's window a target="_blank" link opens the web browser; only the
    update's "What's new" (on GitHub) may do that."""
    templates = ROOT / "src" / "facegen" / "web" / "templates"
    blank = {p.name for p in templates.glob("*.html") if "_blank" in p.read_text(encoding="utf-8")}
    assert blank == {"_update.html"}


def test_find_tick_and_generate_a_pre_existing_player(env: Env) -> None:
    """Phase 3 exit test: a pre-existing player from another club, found, ticked, drawn. Not a
    picture target, so the search offers everyone in the save first (Phase 17h)."""
    targets = env.client.get("/players?q=famous").text
    assert f'name="uid" value="{STAR}"' not in targets
    assert "Showing picture targets only." in targets
    link = '<a href="/players?q=famous&amp;shown=everyone">Show everyone in the save (1)</a>'
    assert link in targets
    found = env.client.get("/players?q=famous&shown=everyone").text
    assert f'name="uid" value="{STAR}"' in found
    back = "/players?q=famous&shown=everyone"
    r = post_player_action(
        env.client,
        "/players/action",
        data={"uid": [str(STAR)], "action": "generate", "group": "b", "next": back},
    )
    assert r.headers["location"].startswith(f"{back}&msg=")
    assert "Draw in the Standard group: 1 queued" in follow(env, r)
    env.worker.step()  # manual requests first
    image_id = env.one(
        "SELECT id FROM images WHERE unique_id = ? AND is_current = 1 AND published_path NOTNULL",
        STAR,
    )
    assert image_id is not None
    page = env.client.get(f"/players/{STAR}").text
    assert "pinned: Standard" in page and f"/pictures/{image_id}?size=384" in page
    assert "in FM27" in page
    thumb = env.client.get(f"/pictures/{image_id}?size=96")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/webp"
    assert env.client.get(f"/pictures/{image_id}").headers["content-type"] == "image/png"
    assert env.client.get(f"/pictures/{image_id}?size=50").status_code == 400


@pytest.mark.parametrize("detail", [False, True], ids=["list", "detail"])
def test_stale_player_form_never_writes_to_either_career(env: Env, detail: bool) -> None:
    path = f"/players/{STAR}" if detail else "/players"
    field = re.search(r'name="career_id" value="(\d+)"', env.client.get(path).text)
    assert field is not None
    conn = env.conn()
    try:
        run_sync(
            conn,
            snapshot(world(), career="two", file_name="two.fm"),
            env.settings,
            save_file="two.fm",
            save_mtime=1_800_000_000.0,
        )
        target = load_career(conn)
        assert target is not None and str(target.id) != field[1]
        action = f"{path}/pin" if detail else "/players/action"
        for token in ({"career_id": field[1]}, {}):  # includes forms opened before this update
            response = env.client.post(
                action, data={**token, "uid": str(STAR), "action": "pin", "pin": "b"}
            )
            assert (
                "This form no longer matches the active career. Reload the page and try again"
                in follow(env, response)
            )
        assert [
            r[0] for r in conn.execute("SELECT pin FROM players WHERE unique_id = ?", (STAR,))
        ] == [None, None]
        assert not conn.execute(
            "SELECT 1 FROM jobs WHERE unique_id = ? AND manual = 1", (STAR,)
        ).fetchone()
    finally:
        conn.close()


def test_generate_needs_a_tick_and_skips_womens_players(env: Env) -> None:
    text = follow(
        env, post_player_action(env.client, "/players/action", data={"action": "generate"})
    )
    assert "Tick at least one player" in text
    r = post_player_action(
        env.client,
        "/players/action",
        data={"uid": [str(WOMAN)], "action": "generate", "group": "a"},
    )
    assert "not in men&#39;s football" in follow(env, r)


def test_bulk_pin(env: Env) -> None:
    r = post_player_action(
        env.client,
        "/players/action",
        data={"uid": [str(STAR), str(B_NEWGEN)], "action": "pin", "pin": "c"},
    )
    assert "Pin to Light: 2 queued" in follow(env, r)
    assert env.one("SELECT COUNT(*) FROM players WHERE pin = 'c'") == 2
    assert env.one("SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND manual = 1") == 2
    r = post_player_action(
        env.client,
        "/players/action",
        data={"uid": [str(STAR), str(B_NEWGEN)], "action": "pin", "pin": "never"},
    )
    assert "2 player(s) pinned to never" in follow(env, r)
    assert env.one("SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND manual = 1") == 0


def test_player_actions_read_the_settings_again_under_the_write_lock(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from facegen.pipeline import manual

    read: list[bool] = []
    real = manual._action_transaction

    def seen(conn, career_id, active_only, settings, read_settings):  # type: ignore[no-untyped-def]
        # The request's own connection, open until it returns.
        read.append(read_settings is not None and isinstance(read_settings(), Settings))
        return real(conn, career_id, active_only, settings, read_settings)

    monkeypatch.setattr(manual, "_action_transaction", seen)
    for action, data in (
        ("generate", {"group": "c"}),
        ("new-picture", {}),
        ("retry", {}),
        ("new-face", {}),
        ("pin", {"pin": "never"}),
    ):
        post_player_action(
            env.client, "/players/action", data={"uid": [str(STAR)], "action": action, **data}
        )
    for action, data in (
        ("pin", {"pin": ""}),
        ("generate", {"group": "c"}),
        ("cancel", {}),
        ("new-face", {}),
        ("new-picture", {}),
        ("retry", {}),
        ("restore", {"image_id": "999"}),
        ("pick", {"attempt": "1", "index": "1", "version": "1"}),
        ("remove", {}),
    ):
        post_player_action(env.client, f"/players/{STAR}/{action}", data=data)
    assert read == [True] * 14  # every action the two pages post


def test_player_page_actions(env: Env) -> None:
    env.drain()
    assert env.client.get(f"/players/{MINE}").status_code == 200
    assert env.client.get("/players/424242").status_code == 404

    text = follow(env, post_player_action(env.client, f"/players/{STAR}/new-picture"))
    assert "no picture to draw again" in text
    text = follow(env, post_player_action(env.client, f"/players/{MINE}/new-face"))
    assert "New face queued" in text
    assert "queued: new face" in env.client.get(f"/players/{MINE}").text
    text = follow(env, post_player_action(env.client, f"/players/{MINE}/cancel"))
    assert "Request cancelled" in text

    text = follow(env, post_player_action(env.client, f"/players/{STAR}/pin", data={"pin": "a"}))
    assert "Pinned to Detailed and queued" in text
    text = follow(env, post_player_action(env.client, f"/players/{STAR}/pin", data={"pin": ""}))
    assert "Pin cleared" in text

    text = follow(env, post_player_action(env.client, f"/players/{MINE}/remove"))
    assert "Picture removed" in text and "Reload Skin" in text
    assert env.one("SELECT pin FROM players WHERE unique_id = ?", MINE) == "never"
    assert post_player_action(env.client, f"/players/{MINE}/bogus").status_code == 303


def test_player_page_shows_personality_and_what_it_weights(env: Env) -> None:
    text = env.client.get(f"/players/{STAR}").text
    assert "Pressure 17" in text and "Temperament 4" in text
    assert "high pressure → confident" in unescape(text)
    assert "temperament →" not in unescape(text)  # low temperament weights nothing
    mine = env.client.get(f"/players/{MINE}").text
    assert "unknown until the next sync" in mine and "Personality favours" not in mine
    env.drain()  # MINE gets a sheet; at my club the page states the round-eye share
    assert "20% of sharp eyes are drawn round" in env.client.get(f"/players/{MINE}").text


def test_restore_an_earlier_picture_from_the_history(env: Env) -> None:
    env.drain()
    follow(env, post_player_action(env.client, f"/players/{MINE}/new-picture"))
    env.drain()
    v1 = env.one("SELECT id FROM images WHERE unique_id = ? AND version = 1", MINE)
    page = env.client.get(f"/players/{MINE}").text
    assert "History" in page and f'name="image_id" value="{v1}"' in page
    assert "Drawn with v1's face as a reference" in page  # at my club: v1 was the reference

    text = follow(
        env, post_player_action(env.client, f"/players/{MINE}/restore", data={"image_id": str(v1)})
    )
    assert "v1 is the current picture again" in text and "Reload Skin" in text
    assert env.one("SELECT version FROM images WHERE unique_id = ? AND is_current = 1", MINE) == 1
    assert "current (restored)" in env.client.get(f"/players/{MINE}").text
    text = follow(
        env, post_player_action(env.client, f"/players/{MINE}/restore", data={"image_id": str(v1)})
    )
    assert "already the current picture" in text
    text = follow(
        env, post_player_action(env.client, f"/players/{MINE}/restore", data={"image_id": "x"})
    )
    assert "Bad picture ID" in text


def test_male_portrait_opens_with_its_own_sheet_and_gender(env: Env) -> None:
    conn = env.conn()
    try:
        settings = set_overrides(conn, {"style.gender": "man"}, env.config)
        career = load_career(conn)
        assert career is not None
        worker = Worker(
            conn,
            settings,
            load_kits(),
            career,
            renderer_factory=FakeRenderer,
            tagger_factory=lambda: FakeTagger(default="boy"),
            fm_check=lambda: False,
            log=lambda *_: None,
        )
        result = worker.run(JobFilter(groups=frozenset({"a"})), limit=1)
        assert result.done[0].job.unique_id == MINE
        set_overrides(conn, {"style.gender": "woman"}, env.config)
        for staff, media in [(0, None), (1, None), (0, "journalist")]:
            conn.execute(
                "UPDATE players SET is_staff = ?, media_role = ? WHERE unique_id = ?",
                (staff, media, MINE),
            )
            page = env.client.get(f"/players/{MINE}")
            assert page.status_code == 200 and "facial hair" in page.text.lower()
            assert "drawn as a woman" not in page.text
            if staff or media:
                assert "drawn as a man" in page.text
    finally:
        conn.close()


def test_pick_another_candidate_on_the_player_page(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.drain()
    page = env.client.get(f"/players/{MINE}").text
    forms = re.findall(
        rf'<form method="post" action="/players/{MINE}/pick">.*?</form>', page, re.DOTALL
    )
    assert len(forms) == 2  # group A's other two candidates passed; the chosen one is a link
    found = re.search(
        r'name="version" value="(\d+)".*?name="attempt" value="(\d+)".*?name="index" value="(\d+)"',
        forms[0],
        re.DOTALL,
    )
    assert found is not None and found[1] == "1"  # the picture the page shows
    pick = {"version": found[1], "attempt": found[2], "index": found[3]}
    image_id = env.one("SELECT id FROM images WHERE unique_id = ? AND is_current = 1", MINE)
    first = env.client.get(f"/pictures/{image_id}", params={"size": 96})
    assert first.headers["cache-control"] == "private, no-cache"
    etag = first.headers["etag"]
    asked = {"If-None-Match": etag}
    assert env.client.get(f"/pictures/{image_id}?size=96", headers=asked).status_code == 304

    def revisions() -> tuple[str, ...]:
        pages = [env.client.get(path).text for path in ("/", "/gallery")]
        return tuple(
            re.search(r'data-topic="(?:status|pictures)" data-rev="([^"]+)"', p)[1] for p in pages
        )

    before = revisions()

    def disk_full(*args: object) -> None:
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr("facegen.pipeline.publish._write_atomic", disk_full)
        failure = follow(env, post_player_action(env.client, f"/players/{MINE}/pick", data=pick))
    assert "Publishing failed" in failure and "The disk is full." in failure
    retry_form = next(
        form
        for form in re.findall(
            rf'<form method="post" action="/players/{MINE}/pick">.*?</form>', failure, re.DOTALL
        )
        if "Retry publishing this picture" in form
    )
    assert all(f'name="{key}" value="{value}"' in retry_form for key, value in pick.items())
    updated = revisions()
    assert all(old != new for old, new in zip(before, updated, strict=True))
    version = f"v001-pick-a{pick['attempt']}c{pick['index']}.png"
    for url in ("/status", "/gallery/part"):
        part = env.client.get(url).text
        assert f'data-id="{image_id}:{version}"' in part
        assert f"/pictures/{image_id}?size=192&amp;v={version}" in part

    text = follow(env, post_player_action(env.client, f"/players/{MINE}/pick", data=pick))
    assert "v1 now uses the picture you picked" in text and "Reload Skin" in text
    assert "Retry publishing this picture" not in text
    assert "picked by hand" in text and "automatic pick" in text
    again = env.client.get(f"/pictures/{image_id}?size=96", headers=asked)
    assert again.status_code == 200 and again.headers["etag"] != etag  # the picked file

    text = follow(env, post_player_action(env.client, f"/players/{MINE}/pick", data=pick))
    assert "already in use for v1" in text
    stale = {**pick, "version": "0"}  # a page opened before the picture now shown was drawn
    text = follow(env, post_player_action(env.client, f"/players/{MINE}/pick", data=stale))
    assert "v1 has been drawn since this page was opened" in text
    bad = {"attempt": "x", "index": "0", "version": "1"}
    text = follow(env, post_player_action(env.client, f"/players/{MINE}/pick", data=bad))
    assert "Bad picture ID" in text


def test_new_picture_for_ticked_players(env: Env) -> None:
    env.drain()
    r = post_player_action(
        env.client,
        "/players/action",
        data={"action": "new-picture", "uid": [str(MINE), str(STAR)], "next": "/players"},
    )
    text = follow(env, r)
    assert "1 queued, 1 skipped (no picture to draw again; use Draw)" in text
    dashboard = env.client.get("/").text
    assert "reference picture" in dashboard  # MINE's re-draw uses the current picture


def test_reference_badges_follow_current_picture_model_gender_and_file(env: Env) -> None:
    env.drain()
    post_player_action(env.client, f"/players/{MINE}/new-picture")
    with closing(env.conn()) as conn:
        for model, gender, expected in (
            ("realvis", "woman", False),
            ("animagine", "man", False),
            ("animagine", "woman", True),
        ):
            set_overrides(conn, {"style.model": model, "style.gender": gender}, env.config)
            dashboard = env.client.get("/").text
            page = env.client.get(f"/players/{MINE}").text
            assert (">reference picture</span>" in dashboard) is expected
            assert ("with the current face as a reference" in page) is expected
            assert ("keeping the current face" in page) is expected
        master = conn.execute(
            "SELECT master_path FROM images WHERE unique_id = ? AND is_current = 1", (MINE,)
        ).fetchone()[0]
        (env.settings.paths.data_dir / master).unlink()
    assert ">reference picture</span>" not in env.client.get("/").text
    assert "with the current face as a reference" not in env.client.get(f"/players/{MINE}").text


def test_review_page_and_retry(env: Env) -> None:
    conn = env.conn()
    career = load_career(conn)
    assert career is not None
    tagger = FakeTagger(["sensitive"] * 3)
    Worker(
        conn,
        env.settings,
        load_kits(),
        career,
        renderer_factory=FakeRenderer,
        tagger_factory=lambda: tagger,
        fm_check=lambda: False,
        log=lambda *_: None,
    ).run(JobFilter(groups=frozenset({"b"})))
    conn.close()
    page = env.client.get("/review").text
    assert "Rival Kid" in page and "rated sensitive" in page
    candidate = re.search(r'src="/files\?path=([^"&]+)&amp;size=192"', page)
    assert candidate is not None
    assert env.client.get(f"/files?path={candidate.group(1)}&size=192").status_code == 200
    assert '<span class="count badge badge-count">1</span>' in env.client.get("/").text
    r = post_player_action(env.client, f"/players/{B_NEWGEN}/retry", data={"next": "/review"})
    assert r.headers["location"].startswith("/review?msg=")
    assert "No players to review right now." in follow(env, r)


def _settings_form(env: Env, **changes: Any) -> dict[str, Any]:
    """The settings form as a browser would post it: every editable field, as stored."""
    values = dict(load_settings(env.conn(), env.config).values)
    values.update(changes)
    form: dict[str, Any] = {}
    for key, default in load_defaults(env.config).items():
        if is_fixed(key):
            continue
        value = values[key]
        if isinstance(default, bool):
            if value:
                form[key] = "on"
        elif isinstance(default, list):
            form[key] = ", ".join(str(v) for v in value)
        else:
            form[key] = str(value)
    return form


def autosave(env: Env, **fields: str) -> dict[str, Any]:
    """What the settings page's script posts when fields change, and the JSON answer."""
    r = env.client.post("/settings/save", data=fields)
    assert r.status_code == 200, r.text
    return r.json()


def test_settings_are_saved_as_they_change(env: Env) -> None:
    page = env.client.get("/settings/advanced").text
    assert 'name="cutoffs.everywhere.a_ca"' in page and 'name="ui.host"' not in page
    assert 'name="paths.fm_user_dir"' not in page
    # FM27's folders are shown, with where each came from, but not as fields.
    assert "FM27&#39;s folders" in page or "FM27's folders" in page
    assert (
        f'<span class="path">{env.settings.paths.fm_user_dir}</span> '
        '<span class="muted small origin">from ' in page
    )
    assert "Install</th><td>" in page
    assert "data-autosave" in page and 'value="preview"' not in page
    # The Save button is there only for a browser with JavaScript off.
    assert re.search(r"<noscript>.*value=\"apply-some\".*</noscript>", page, re.DOTALL)
    # Five tabs, each with its own form; Undo and Recent changes stay outside them.
    for tab in ("cutoffs", "quality", "targets", "ages", "system"):
        assert f'id="tab-{tab}" data-tab-panel="{tab}"' in page and f'href="#tab-{tab}"' in page
    assert page.count("data-autosave") == 5
    assert re.search(r"data-undo\s+disabled", page)  # nothing to undo yet

    data = autosave(env, **{"cutoffs.everywhere.a_ca": "85"})
    assert data["ok"] and data["changed"]
    assert data["message"].startswith("Saved: Everywhere · Detailed CA 165+ → 85+.")
    # Rival Kid's queued B job becomes an A job, and Abroad Kid (CA 120) is queued at A.
    assert data["message"].endswith(
        "Detailed 1 → 3, Standard 1 → 0, auto-update 1 → 3; queue: 1 queued, 1 updated."
    )
    assert data["fields"] == {
        "cutoffs.everywhere.a_ca": {"value": 85, "text": "85", "changed": True}
    }
    assert data["can_undo"] and not data["can_redo"]
    assert (
        "Everywhere · Detailed CA" in data["history_html"]
        and 'name="through"' in data["history_html"]
    )
    assert 'name="csrf_token"' in data["history_html"]
    assert env.one("SELECT group_name FROM players WHERE unique_id = ?", B_NEWGEN) == "a"
    assert env.one("SELECT group_name FROM jobs WHERE unique_id = ?", B_NEWGEN) == "a"
    assert load_settings(env.conn(), env.config).cutoffs.everywhere["a_ca"] == 85

    again = autosave(env, **{"cutoffs.everywhere.a_ca": "85"})
    assert again["ok"] and not again["changed"] and again["message"] == "Nothing changed."
    # The page lists the change and offers to undo it.
    page = env.client.get("/settings/advanced").text
    assert "Recent changes" in page and "165+ → <strong>85+</strong>" in page
    assert not re.search(r"data-undo\s+disabled", page)


def test_autosave_refuses_bad_values_and_names_the_field(env: Env) -> None:
    data = autosave(env, **{"cutoffs.everywhere.a_ca": "999"})
    assert not data["ok"] and data["errors"]["cutoffs.everywhere.a_ca"] == "must be from 1 to 200"
    data = autosave(env, **{"cutoffs.everywhere.a_ca": "many"})
    assert data["errors"]["cutoffs.everywhere.a_ca"] == "needs a number, or - to switch it off"
    data = autosave(env, **{"groups.b.candidates": "two"})
    assert data["errors"] == {"groups.b.candidates": "needs a number"}
    # Values that only fail together blame the fields named; nothing is stored.
    data = autosave(env, **{"body.short_below_cm": "200"})
    assert list(data["errors"]) == ["body.short_below_cm"]
    data = autosave(env, **{"age.staff.W03": "17"})  # below W02's 18
    assert list(data["errors"]) == ["age.staff.W03"]
    assert "must start at an older age than the one before" in data["error"]
    assert env.one("SELECT COUNT(*) FROM settings WHERE key != 'cutoffs.mode'") == 0
    # Fixing the field that blocked another saves both in one change.
    data = autosave(env, **{"age.staff.W02": "16", "age.staff.W03": "17"})
    assert data["ok"] and env.one("SELECT COUNT(*) FROM settings_history") == 1
    # A checkbox is posted either way: empty is off.
    autosave(env, **{"reference.enabled": ""})
    assert load_settings(env.conn(), env.config).reference.enabled is False
    autosave(env, **{"reference.enabled": "on"})
    assert load_settings(env.conn(), env.config).reference.enabled is True
    assert not autosave(env, **{"reference.enabled": "maybe"})["ok"]
    # The detail pass size is one of a few choices (Phase 10c).
    data = autosave(env, **{"groups.a.detail_size": "smaller"})
    assert data["ok"] and data["fields"]["groups.a.detail_size"]["text"] == "smaller"
    assert load_settings(env.conn(), env.config).groups.a.detail_size() == (1152, 1472)
    data = autosave(env, **{"groups.a.detail_size": "1152x1472"})
    assert not data["ok"] and "x1.5, smaller, off" in data["errors"]["groups.a.detail_size"]


def test_undo_redo_and_undo_back_several_steps(env: Env) -> None:
    autosave(env, **{"cutoffs.everywhere.a_ca": "150"})  # moves nobody in this world
    autosave(env, **{"cutoffs.everywhere.b_ca": "110"})  # Abroad Kid (CA 120) joins B
    assert env.one("SELECT group_name FROM players WHERE unique_id = ?", ABROAD_KID) == "b"

    text = follow(env, env.client.post("/settings", data={"action": "undo"}))
    assert "Undone: Everywhere · Standard CA 110+ → 145+." in text
    assert env.one("SELECT group_name FROM players WHERE unique_id = ?", ABROAD_KID) is None
    text = follow(env, env.client.post("/settings", data={"action": "redo"}))
    assert "Redone: Everywhere · Standard CA 145+ → 110+." in text
    assert env.one("SELECT group_name FROM players WHERE unique_id = ?", ABROAD_KID) == "b"

    first = env.one("SELECT MIN(id) FROM settings_history")
    text = follow(env, env.client.post("/settings", data={"action": "undo", "through": first}))
    assert "Everywhere · Detailed CA 150+ → 165+" in text
    assert "Everywhere · Standard CA 110+ → 145+" in text
    assert env.one("SELECT COUNT(*) FROM settings WHERE key != 'cutoffs.mode'") == 0
    assert env.one("SELECT group_name FROM players WHERE unique_id = ?", B_NEWGEN) == "b"
    assert "Nothing to undo." in follow(env, env.client.post("/settings", data={"action": "undo"}))


def test_reset_all_and_the_save_button_without_javascript(env: Env) -> None:
    form = _settings_form(env, **{"cutoffs.everywhere.a_ca": 85})
    text = follow(env, env.client.post("/settings", data={**form, "action": "apply"}))
    assert "Saved: Everywhere · Detailed CA 165+ → 85+." in text
    assert "Nothing changed" in follow(
        env, env.client.post("/settings", data={**form, "action": "apply"})
    )
    # An unticked checkbox is not posted with the whole form: it is "off".
    del form["reference.enabled"]
    follow(env, env.client.post("/settings", data={**form, "action": "apply"}))
    assert load_settings(env.conn(), env.config).reference.enabled is False
    form["groups.b.candidates"] = "two"
    r = env.client.post("/settings", data={**form, "action": "apply"})
    assert r.status_code == 200 and "needs a number" in r.text

    text = follow(env, env.client.post("/settings", data={"action": "defaults"}))
    assert "every setting back to its default" in text
    assert env.one("SELECT COUNT(*) FROM settings") == 0
    follow(env, env.client.post("/settings", data={"action": "undo"}))
    s = load_settings(env.conn(), env.config)
    assert s.cutoffs.everywhere["a_ca"] == 85 and s.reference.enabled is False


def test_my_clubs_staff_on_the_players_pages(env: Env) -> None:
    from factories import staff_member

    conn = env.conn()
    run_sync(
        conn,
        snapshot(world(), staff=[staff_member(NEWGEN + 100, name="Coach Rowe", job="coach")]),
        env.settings,
        save_file="career.fm",
    )
    conn.close()
    page = env.client.get("/players?role=staff").text
    assert "Coach Rowe" in page and "Mine Player" not in page
    assert '<span class="tag">coach</span>' in page
    assert "Coach Rowe" not in env.client.get("/players?role=players").text
    detail = env.client.get(f"/players/{NEWGEN + 100}").text
    assert "on my club's staff (coach)" in unescape(detail)
    assert "1 staff" in env.client.get("/status").text


def test_youth_and_media_on_the_players_pages(env: Env) -> None:
    from factories import media_person, player, staff_member

    conn = env.conn()
    run_sync(
        conn,
        snapshot(
            [*world(), player(1050, name="Kid Hale", age=17, team_slot=2)],
            staff=[staff_member(NEWGEN + 100, name="Coach Rowe", job="coach", female=False)],
            media=[media_person(2_002_097_912, female=False)],
        ),
        env.settings,
        save_file="career.fm",
    )
    conn.close()
    youth = env.client.get("/players?role=youth").text
    assert "Kid Hale" in youth and "Coach Rowe" not in youth
    assert '<span class="tag">youth team</span>' in youth
    media = env.client.get("/players?role=media").text
    assert "Journalist 2002097912" in media and "Kid Hale" not in media
    assert '<span class="tag">journalist</span>' in media
    assert "Journalist 2002097912" not in env.client.get("/players?role=players").text
    detail = env.client.get("/players/2002097912").text
    assert "journalist from my club's country" in unescape(detail)
    assert "a man in the save, drawn as a woman" in detail
    assert "Pin a regular at my club to" in detail
    coach = env.client.get(f"/players/{NEWGEN + 100}").text
    assert "a man in the save, drawn as a woman" in coach
    assert "Pin a regular at my club to" not in coach
    status = env.client.get("/status").text
    assert "1 in my youth teams" in status and "1 journalists and press officers" in status


def test_photos_on_the_players_pages_and_the_dashboard(env: Env) -> None:
    def names(query: str) -> set[str]:
        text = env.client.get(f"/players?shown=everyone&{query}").text
        return set(re.findall(r'<a href="/players/\d+">([^<]+)</a>', text))

    assert "Who has a photo is not known yet" in env.client.get("/players?photo=none").text
    assert "No photo</th>" not in env.client.get("/status").text
    conn = env.conn()
    scan = PhotoScan(
        state={"catalog": "v1", "configs": [], "folders": {}},
        index=PhotoIndex(frozenset({MINE}), frozenset({ABROAD_KID})),
    )
    run_sync(conn, snapshot(world()), env.settings, save_file="career.fm", photos=scan)
    conn.close()

    assert names("photo=game") == {"Mine Player"}
    assert names("photo=facepack") == {"Abroad Kid"}
    assert names("photo=none") == {"Famous Striker", "Rival Kid"}
    assert names("origin=existing&photo=none") == {"Famous Striker"}
    page = env.client.get("/players?shown=everyone").text
    assert "FM27 photo</span>" in page and "facepack</span>" in page
    assert page.count("no photo</span>") == 1  # only pre-existing players are tagged
    assert "not known yet" not in env.client.get("/players?photo=none").text

    # Famous Striker (CA 185) has no photo, so the newgen thresholds now put them in A.
    detail = env.client.get(f"/players/{STAR}").text
    assert "none: FM27 shows a silhouette" in detail
    assert "Pre-existing, no photo: CA 185 (165+)" in detail  # worded from its codes
    assert re.search(r"Photo</th><td>\s*FM27's\s*</td>", env.client.get(f"/players/{MINE}").text)

    status = env.client.get("/status").text
    assert "1 of them in a group" in status
    # FM27's photo list itself is in Settings' About this run (Phase 12 feedback).
    assert "FM27's photo list</th><td>1 photos" in unescape(
        env.client.get("/settings/advanced").text
    )


def test_the_no_photo_setting_is_on_the_settings_page(env: Env) -> None:
    text = env.client.get("/settings/advanced").text
    assert "Pre-existing players without a photo" in text
    assert 'name="no_photo.enabled"' in text


def test_worker_and_sync_buttons(env: Env) -> None:
    follow(env, env.client.post("/worker/pause"))
    assert env.worker.status().paused
    follow(env, env.client.post("/worker/resume"))
    assert not env.worker.status().paused
    follow(env, env.client.post("/worker/generate-now"))
    assert env.worker.status().generate_now
    assert env.client.post("/worker/explode").status_code == 404


def test_after_a_gpu_fault_the_buttons_are_refused_with_what_to_do(env: Env) -> None:
    env.worker._fault("AcceleratorError: CUDA error: an illegal memory access was encountered")
    for action in ("resume", "generate-now"):
        page = follow(env, env.client.post(f"/worker/{action}"))
        assert "start it again" in page
    status = env.worker.status()
    assert status.gpu_fault and status.paused and not status.generate_now
    page = env.client.get("/status").text
    assert "gpu-fault" in page and "/worker/resume" not in page
    assert "/worker/generate-now" not in page


def test_only_local_requests_are_answered(env: Env) -> None:
    assert env.client.get("/", headers={"host": "evil.example"}).status_code == 403
    assert env.client.get("/", headers={"host": "127.0.0.1.evil.example:8765"}).status_code == 403
    assert env.client.get("/", headers={"host": "localhost:8765"}).status_code == 200
    assert env.client.get("/", headers={"host": "[::1]:8765"}).status_code == 200
    # A form posted from another site is refused; one from this page is not.
    bad = env.client.post("/worker/pause", headers={"origin": "http://evil.example"})
    assert bad.status_code == 403 and not env.worker.status().paused
    ok = env.client.post("/worker/pause", headers={"origin": BASE})
    assert ok.status_code == 303 and env.worker.status().paused


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "http://127.0.0.1:9999",
        "http://localhost:8765",
        "https://127.0.0.1:8765",
        "http://[",
        "http://127.0.0.1:bad",
        "http://user@127.0.0.1:8765",
        BASE + "/path",
        "http://evil.example",
    ],
)
def test_foreign_and_malformed_origins_cannot_mutate(env: Env, origin: str) -> None:
    r = env.client.post("/worker/pause", headers={"Origin": origin})
    assert r.status_code == 403
    assert not env.worker.status().paused


@pytest.mark.parametrize("host", ["", "[", "127.0.0.1:bad", "user@localhost:8765"])
def test_malformed_host_fails_closed(env: Env, host: str) -> None:
    assert env.client.get("/", headers={"Host": host}).status_code == 403


def test_csrf_is_required_even_with_no_origin_or_same_origin(env: Env) -> None:
    del env.client.headers["X-CSRF-Token"]
    for headers in ({}, {"Origin": BASE}, {"Origin": BASE, "X-CSRF-Token": "wrong"}):
        assert env.client.post("/worker/pause", headers=headers).status_code == 403
    assert not env.worker.status().paused
    page = env.client.get("/")
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)[1]
    assert page.headers["X-Frame-Options"] == "DENY"
    assert page.headers["Cache-Control"] == "no-store"
    # Real browser form submission, including downstream body parsing, without a header.
    r = post_player_action(
        env.client,
        "/players/action",
        data={
            "csrf_token": token,
            "uid": str(STAR),
            "action": "generate",
            "group": "b",
        },
        headers={"Origin": BASE},
    )
    assert r.status_code == 303
    assert env.one("SELECT pin FROM players WHERE unique_id = ?", STAR) == "b"


def test_every_post_form_carries_the_token(env: Env) -> None:
    for url in ("/", "/status", "/gallery", "/players", f"/players/{MINE}", "/settings", "/review"):
        for form in re.findall(
            r'<form\b[^>]*method="post"[^>]*>.*?</form>', env.client.get(url).text, re.DOTALL
        ):
            assert 'name="csrf_token"' in form, url
            if 'action="/players/' in form:
                assert 'name="career_id"' in form, url


def test_pictures_come_only_from_the_masters_folder(env: Env) -> None:
    for path in (
        "masters/../facegen.db",
        "/etc/passwd",
        "facegen.db",
        "masters/x/../../facegen.db",
    ):
        assert env.client.get("/files", params={"path": path}).status_code == 404
    assert env.client.get("/pictures/12345").status_code == 404


def test_redirects_stay_on_this_site(env: Env) -> None:
    for target in ("//evil.example/x", "http://evil.example/", "javascript:alert(1)"):
        r = post_player_action(
            env.client,
            "/players/action",
            data={"uid": [str(STAR)], "action": "pin", "pin": "b", "next": target},
        )
        assert r.headers["location"].startswith("/players?msg=")


def test_dashboard_reports_an_unsupported_build(env: Env) -> None:
    from facegen.pipeline.sync import record_failed_sync

    assert "cannot be read" not in env.client.get("/").text
    c = env.conn()
    record_failed_sync(
        c,
        "career.fm",
        "FM27 build 27.2.0+999 cannot be read by fmsave 0.11.0: players: schema differs",
        "27.2.0+999",
        save_mtime=1.0,
        unsupported=True,
        trigger="poller",
    )
    c.close()
    text = env.client.get("/status").text
    assert "The newest save cannot be read" in text
    assert "FM27 build <strong>27.2.0+999</strong>" in text
    assert "bump the pin" in text
    assert "Started by</th><td>the save watcher" in env.client.get("/log").text


def test_dashboard_checks_setup_once_per_request(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    original = AppContext.setup
    checks = 0

    def counted(ctx: AppContext, settings: Settings) -> SetupState:
        nonlocal checks
        checks += 1
        return original(ctx, settings)

    monkeypatch.setattr(AppContext, "setup", counted)
    assert env.client.get("/").status_code == 200
    assert checks == 1
    assert env.client.get("/status").status_code == 200
    assert checks == 2


def test_dashboard_names_the_career_fm27_is_shown(env: Env) -> None:
    c = env.conn()
    run_sync(
        c,
        snapshot([player(5001)], career="two", file_name="two.fm"),
        env.settings,
        save_file="two.fm",
        save_mtime=1_800_000_000.0,
    )
    c.close()
    text = env.client.get("/").text
    # Phase 17e: the active pack on a line of its own, the other careers muted below it.
    assert (
        '<p class="small careers-active">Active pack: <strong>two</strong> (newest save)</p>'
        in text
    )
    assert '<p class="muted small careers-others">Other careers: career</p>' in text
    korean = env.client.get("/", headers=KOREAN_WINDOWS).text
    assert (
        "FM27 적용: <strong>two</strong> (최신 세이브)" in korean
        and "다른 커리어: career" in korean
    )


def test_pages_link_the_script_and_styles_by_version(env: Env) -> None:
    from facegen.web.app import HERE, static_version

    version = static_version(HERE / "static")
    page = env.client.get("/settings/advanced").text
    assert f'/static/app.js?v={version}"' in page and f'/static/style.css?v={version}"' in page
    assert env.client.get(f"/static/app.js?v={version}").status_code == 200


def test_pages_carry_the_kitshot_name_and_mark(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 17c: the name Kitshot in titles and the sidebar, and the user's SVG as the
    favicon, the sidebar's mark and the dashboard's empty picture. It is served as an SVG
    even where Windows's registry maps .svg to another type ("nosniff" would refuse it)."""
    import mimetypes

    from facegen.web.app import HERE, static_version

    mimetypes.init()
    monkeypatch.setitem(mimetypes.types_map, ".svg", "text/xml")
    e = make_env(config_file, synced=True)
    try:
        mark = f"/static/kitshot.svg?v={static_version(HERE / 'static')}"
        page = e.client.get("/gallery").text
        assert "<title>Gallery · Kitshot</title>" in page
        assert f'<link rel="icon" type="image/svg+xml" href="{mark}">' in page
        brand = re.search(r'<a class="brand"[^>]*>.*?</a>', page, re.DOTALL)
        assert brand is not None and 'title="Kitshot"' in brand[0]
        assert f'<img class="logo" src="{mark}" alt=""' in brand[0]
        assert '<span class="label">Kitshot</span>' in brand[0] and ">fg<" not in page
        dashboard = e.client.get("/").text
        assert f'<img class="now-pic-mark" src="{mark}"' in dashboard and ">fg<" not in dashboard
        svg = e.client.get(mark)
        assert svg.status_code == 200 and svg.headers["content-type"] == "image/svg+xml"
        assert svg.text.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"')
        missing = e.client.get("/nowhere", headers={"accept": "text/html"})
        assert missing.status_code == 404 and " · Kitshot</title>" in missing.text
        assert f'<link rel="icon" type="image/svg+xml" href="{mark}">' in missing.text
    finally:
        e.client.close()


def test_every_page_starts_with_the_same_header(env: Env) -> None:
    """Phase 17a: one header first on every page, above any message, so switching pages keeps
    the title and what is below it in place. Settings keeps its two levels in it."""
    titles = {
        "/": "Dashboard",
        "/gallery": "Gallery",
        "/players": "Players",
        "/review": "Review",
        "/log": "Log",
        "/settings": "Settings",
        "/settings/advanced": "Settings",
        "/hardware": "Auto-configure for my GPU",
        f"/players/{MINE}": "Mine Player",
    }
    for url, title in titles.items():
        main = re.split(
            r"<main(?: class=\"[\w -]+\")?>", env.client.get(f"{url}?msg=Done").text, 1
        )[1]
        assert re.match(rf'\s*<header class="page-head">\s*<h1>{re.escape(title)}\b', main), url
        assert main.count('class="page-head"') == 1 and main.count("<h1>") == 1, url
        assert main.index("</header>") < main.index('<div class="flash ok" role="status">Done'), url
    page = env.client.get("/settings").text
    head = page[page.index('<header class="page-head">') : page.index("</header>")]
    assert 'class="page-actions"' in head and 'class="tabs"' in head and 'class="info"' in head
    page = env.client.get("/hardware").text
    head = page[page.index('<header class="page-head">') : page.index("</header>")]
    assert '<a href="/settings">← Settings</a>' in head


def test_option_labels_start_with_a_capital(env: Env) -> None:
    """Phase 17a: "Any group", not "any group", in the lists of choices (English)."""
    for url in ("/players", "/gallery", f"/players/{MINE}"):
        options = re.findall(r"<option[^>]*>([^<]+)</option>", unescape(env.client.get(url).text))
        assert options, url
        assert [o for o in options if o[0].islower()] == [], url
    assert (
        '<a class="button filter-clear" href="/players">Clear</a>'
        in env.client.get("/players?group=a").text
    )
