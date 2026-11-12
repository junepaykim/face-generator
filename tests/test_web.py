"""The web UI through FastAPI's test client, with the worker driven step by step."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from factories import ABROAD, NEWGEN, RIVAL, WOMENS, player, snapshot
from fakes import FakeRenderer, FakeTagger
from fastapi.testclient import TestClient

from facegen import db
from facegen.datasource.records import Snapshot
from facegen.kits import load_kits
from facegen.service import SyncService, WorkerService
from facegen.settings import Settings, is_fixed, load_defaults, load_settings
from facegen.sync import run_sync
from facegen.web.app import AppContext, create_app
from facegen.worker import JobFilter, Worker, load_career

MINE = 1001  # my club: A
STAR = 1002  # pre-existing at another club: hand-picked only
B_NEWGEN = NEWGEN + 2  # rival in my league, CA 90, PA 152: B
WOMAN = NEWGEN + 6  # women's club
ABROAD_KID = NEWGEN + 8  # abroad, CA 120: in no group until B's bar drops

BASE = "http://127.0.0.1:8765"


def world() -> list:
    return [
        player(MINE, name="Mine Player", age=29),
        player(STAR, name="Famous Striker", ca=185, pa=190, club_id=ABROAD),
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
        run_sync(conn, snapshot(world()), settings, save_file="career.fm")
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


def test_dashboard(env: Env) -> None:
    text = env.client.get("/").text
    assert "Group A" in text and "Mine Player" in text  # in the queue
    assert 'data-refresh="/status"' in text
    status = env.client.get("/status").text
    assert "<html" not in status and "Worker" in status
    env.drain()
    text = env.client.get("/status").text
    assert "1 drawn" in text and "Mine Player" in text


def test_players_list_filters(env: Env) -> None:
    def names(query: str) -> set[str]:
        text = env.client.get(f"/players?{query}").text
        return set(re.findall(r'<a href="/players/\d+">([^<]+)</a>', text))

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
    assert names("sort=name&page=99") == set()
    assert env.client.get("/players?ca_min=abc&league=bogus").status_code == 200


def test_find_tick_and_generate_a_pre_existing_player(env: Env) -> None:
    """Phase 3 exit test: a pre-existing player from another club, found, ticked, drawn."""
    found = env.client.get("/players?q=famous").text
    assert f'name="uid" value="{STAR}"' in found
    r = env.client.post(
        "/players/action",
        data={"uid": [str(STAR)], "action": "generate", "group": "b", "next": "/players?q=famous"},
    )
    assert r.headers["location"].startswith("/players?q=famous&msg=")
    assert "Generate as group B: 1 queued" in follow(env, r)
    env.worker.step()  # manual requests first
    image_id = env.one(
        "SELECT id FROM images WHERE unique_id = ? AND is_current = 1 AND published_path NOTNULL",
        STAR,
    )
    assert image_id is not None
    page = env.client.get(f"/players/{STAR}").text
    assert "pinned to b" in page and f"/pictures/{image_id}?size=384" in page
    assert "published" in page
    thumb = env.client.get(f"/pictures/{image_id}?size=96")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/webp"
    assert env.client.get(f"/pictures/{image_id}").headers["content-type"] == "image/png"
    assert env.client.get(f"/pictures/{image_id}?size=50").status_code == 400


def test_generate_needs_a_tick_and_skips_womens_players(env: Env) -> None:
    text = follow(env, env.client.post("/players/action", data={"action": "generate"}))
    assert "Tick at least one player" in text
    r = env.client.post(
        "/players/action", data={"uid": [str(WOMAN)], "action": "generate", "group": "a"}
    )
    assert "not in men&#39;s football" in follow(env, r)


def test_bulk_pin(env: Env) -> None:
    r = env.client.post(
        "/players/action", data={"uid": [str(STAR), str(B_NEWGEN)], "action": "pin", "pin": "c"}
    )
    assert "Pin to C (a hand-pick, as Generate): 2 queued" in follow(env, r)
    assert env.one("SELECT COUNT(*) FROM players WHERE pin = 'c'") == 2
    assert env.one("SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND manual = 1") == 2
    r = env.client.post(
        "/players/action",
        data={"uid": [str(STAR), str(B_NEWGEN)], "action": "pin", "pin": "never"},
    )
    assert "2 player(s) pinned to never" in follow(env, r)
    assert env.one("SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND manual = 1") == 0


def test_player_page_actions(env: Env) -> None:
    env.drain()
    assert env.client.get(f"/players/{MINE}").status_code == 200
    assert env.client.get("/players/424242").status_code == 404

    text = follow(env, env.client.post(f"/players/{STAR}/new-picture"))
    assert "no picture to draw again" in text
    text = follow(env, env.client.post(f"/players/{MINE}/new-face"))
    assert "New face queued" in text
    assert "queued: new face" in env.client.get(f"/players/{MINE}").text
    text = follow(env, env.client.post(f"/players/{MINE}/cancel"))
    assert "Request cancelled" in text

    text = follow(env, env.client.post(f"/players/{STAR}/pin", data={"pin": "a"}))
    assert "Pinned to A and queued as a request" in text
    text = follow(env, env.client.post(f"/players/{STAR}/pin", data={"pin": ""}))
    assert "Pin cleared" in text

    text = follow(env, env.client.post(f"/players/{MINE}/remove"))
    assert "Picture removed" in text and "Reload Skin" in text
    assert env.one("SELECT pin FROM players WHERE unique_id = ?", MINE) == "never"
    assert env.client.post(f"/players/{MINE}/bogus").status_code == 303


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
        log=lambda line: None,
    ).run(JobFilter(groups=frozenset({"b"})))
    conn.close()
    page = env.client.get("/review").text
    assert "Rival Kid" in page and "rated sensitive" in page
    candidate = re.search(r'src="/files\?path=([^"&]+)&amp;size=192"', page)
    assert candidate is not None
    assert env.client.get(f"/files?path={candidate.group(1)}&size=192").status_code == 200
    assert '<span class="count">1</span>' in env.client.get("/").text
    r = env.client.post(f"/players/{B_NEWGEN}/retry", data={"next": "/review"})
    assert r.headers["location"].startswith("/review?msg=")
    assert "Nothing to review" in follow(env, r)


def _settings_form(env: Env, **changes: Any) -> dict[str, Any]:
    """The settings form as a browser would post it: every editable field."""
    values = dict(load_settings(defaults_path=env.config).values)
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


def test_settings_preview_then_apply(env: Env) -> None:
    page = env.client.get("/settings").text
    assert 'name="groups.a.ca_above"' in page and 'name="ui.host"' not in page
    assert 'name="paths.fm_user_dir"' not in page

    form = _settings_form(env, **{"groups.a.ca_above": 85})
    r = env.client.post("/settings", data={**form, "action": "preview"})
    assert r.status_code == 200
    assert "Preview" in r.text and "Rival Kid" in r.text  # moves from B to A
    assert env.one("SELECT COUNT(*) FROM settings") == 0  # nothing stored yet
    assert env.one("SELECT group_name FROM players WHERE unique_id = ?", B_NEWGEN) == "b"

    text = follow(env, env.client.post("/settings", data={**form, "action": "apply"}))
    assert "Applied 1 change(s)" in text and "1 updated" in text
    assert env.one("SELECT group_name FROM players WHERE unique_id = ?", B_NEWGEN) == "a"
    assert env.one("SELECT group_name FROM jobs WHERE unique_id = ?", B_NEWGEN) == "a"
    assert load_settings(env.conn(), env.config).groups.a.ca_above == 85
    assert "Nothing changed" in follow(env, env.client.post("/settings", data={**form}))


def test_settings_preview_lists_what_would_be_queued(env: Env) -> None:
    form = _settings_form(env, **{"groups.b.any_ca_above": 110})
    r = env.client.post("/settings", data={**form, "action": "preview"})
    assert "1 first picture B would be queued" in r.text
    assert "Abroad Kid" in r.text


def test_settings_errors_and_booleans(env: Env) -> None:
    form = _settings_form(env, **{"groups.a.ca_above": 999})
    r = env.client.post("/settings", data={**form, "action": "apply"})
    assert r.status_code == 200 and "between 0 and 200" in r.text
    form = _settings_form(env)
    form["groups.b.candidates"] = "two"
    r = env.client.post("/settings", data={**form, "action": "apply"})
    assert r.status_code == 200 and "needs a number" in r.text
    assert env.one("SELECT COUNT(*) FROM settings") == 0
    # An unticked checkbox is "off".
    form = _settings_form(env)
    del form["groups.a.detail_pass"]
    follow(env, env.client.post("/settings", data={**form, "action": "apply"}))
    assert load_settings(env.conn(), env.config).groups.a.detail_pass is False
    # Load defaults previews going back.
    r = env.client.post("/settings", data={"action": "defaults"})
    assert r.status_code == 200 and "groups.a.detail_pass" in r.text


def test_worker_and_sync_buttons(env: Env) -> None:
    follow(env, env.client.post("/worker/pause"))
    assert env.worker.status().paused
    follow(env, env.client.post("/worker/resume"))
    assert not env.worker.status().paused
    follow(env, env.client.post("/worker/generate-now"))
    assert env.worker.status().generate_now
    assert env.client.post("/worker/explode").status_code == 404


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
        r = env.client.post(
            "/players/action",
            data={"uid": [str(STAR)], "action": "pin", "pin": "b", "next": target},
        )
        assert r.headers["location"].startswith("/players?msg=")
