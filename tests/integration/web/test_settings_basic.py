"""Settings in two levels (Phase 12g): Basic shows who gets a picture, the quality preset and
when to draw; Advanced has every setting; one history and one Undo serve both."""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlencode

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.core import presets
from facegen.persistence import db
from facegen.persistence.settings import load_defaults, load_settings
from facegen.pipeline import settings_history
from facegen.setup import hardware
from facegen.setup.hardware import Hardware
from facegen.web.app import AppContext, create_app

GIB = 2**30


@pytest.fixture
def client(config_file: Path) -> Iterator[TestClient]:
    # A rival newgen just below group B's "any club" PA line: "More" brings them in.
    world = [player(1001, name="Mine"), player(NEWGEN + 2, ca=90, pa=150, club_id=RIVAL)]
    db_path = synced(config_file, world)
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    yield c
    c.close()


def values(config_file: Path) -> dict:
    settings = load_settings(defaults_path=config_file)
    conn = db.connect(settings.paths.db_path)
    try:
        return dict(load_settings(conn, config_file).values)
    finally:
        conn.close()


def checked(page: str, kind: str) -> list[str]:
    return re.findall(rf'name="preset\.{kind}" value="(\w+)" data-preset checked', page)


def test_basic_holds_who_quality_and_when_and_advanced_holds_everything(
    client: TestClient,
) -> None:
    page = unescape(client.get("/settings").text)
    for title in ("Who gets a picture", "Picture quality", "Drawing on its own", "FM27's folder"):
        assert title in page, title
    assert checked(page, "reach") == ["default"] and checked(page, "quality") == ["high"]
    for key in (
        "staff.enabled", "media.journalists", "media.press_officers", "no_photo.enabled",
        "schedule.generate_while_fm_running", "schedule.auto_groups",
        "notifications.new_pictures",
    ):  # fmt: skip
        assert f'name="{key}" data-key' in page, key
    assert 'name="groups.a.ca_above"' not in page and "About this run" not in page
    assert 'href="/settings/basic" class="on" aria-current="page">Basic' in page
    advanced = client.get("/settings/advanced").text
    assert 'name="groups.a.ca_above"' in advanced and "Recent changes" in advanced
    assert 'href="/settings/advanced" class="on" aria-current="page">Advanced' in advanced
    korean = unescape(client.get("/settings/basic", headers=KOREAN_WINDOWS).text)
    for text in ("그림 대상", "다른 구단 선수", "적게", "보통", "많이", "그림 품질", "메모리 절약"):
        assert text in korean, text


def test_a_preset_is_one_named_change_that_one_undo_reverts(
    client: TestClient, config_file: Path
) -> None:
    defaults = load_defaults(config_file)
    data = client.post("/settings/save", data={"preset.reach": "more"}).json()
    assert data["ok"] and data["changed"] and data["presets"]["reach"] == "more"
    assert data["message"].startswith("Saved: Players at other clubs: More. ")
    assert "Standard 0 → 1" in data["message"]  # the rival newgen now qualifies
    now = values(config_file)
    assert all(now[k] == defaults[k] - 10 for k in presets.REACH_KEYS)
    advanced = unescape(client.get("/settings/advanced").text)
    assert "· Players at other clubs: More" in advanced  # the history names the preset
    assert checked(client.get("/settings/basic").text, "reach") == ["more"]

    r = client.post("/settings", data={"action": "undo", "next": "/settings"})
    assert r.status_code == 303 and r.headers["location"].startswith("/settings?msg=")
    assert "Undone: Players at other clubs: More." in unquote(r.headers["location"])
    assert all(values(config_file)[k] == defaults[k] for k in presets.REACH_KEYS)
    # Undo from Advanced goes back to Advanced, and its history can undo too.
    r = client.post("/settings", data={"action": "redo"})
    assert r.headers["location"].startswith("/settings/advanced?msg=")


def test_a_threshold_set_by_hand_shows_no_preset(client: TestClient) -> None:
    assert client.post("/settings/save", data={"groups.b.any_pa_above": "149"}).json()["ok"]
    page = client.get("/settings").text
    assert checked(page, "reach") == []
    assert re.search(r'data-custom="reach"\s*>Set by hand in <a href="/settings/advanced">', page)
    assert re.search(r'data-custom="quality" hidden>', page)


def test_basic_switches_and_quality_save_like_advanced_fields(
    client: TestClient, config_file: Path
) -> None:
    data = client.post("/settings/save", data={"staff.enabled": ""}).json()
    assert data["ok"] and data["fields"]["staff.enabled"]["changed"]
    data = client.post("/settings/save", data={"notifications.new_pictures": ""}).json()
    assert data["ok"] and values(config_file)["notifications.new_pictures"] is False
    assert load_settings(defaults_path=config_file).notifications.new_pictures is True
    data = client.post("/settings/save", data={"preset.quality": "low"}).json()
    assert data["ok"] and data["presets"]["quality"] == "low"
    assert {k: values(config_file)[k] for k in presets.QUALITY["low"]} == hardware.LOW.changes
    bad = client.post("/settings/save", data={"preset.quality": "ultra"}).json()
    assert not bad["ok"] and "preset.quality" in bad["errors"]
    conn = db.connect(load_settings(defaults_path=config_file).paths.db_path)
    try:
        labels = [e.label for e in settings_history.entries(conn)]
    finally:
        conn.close()
    assert labels == ["preset.quality.low", "edit", "edit"]


def test_without_the_script_basic_saves_what_it_shows(
    client: TestClient, config_file: Path
) -> None:
    # As a browser posts it: hidden "off" first, the ticked box after it.
    form = [
        ("action", "apply-some"), ("next", "/settings"),
        ("staff.enabled", "off"), ("staff.enabled", "on"),
        ("media.journalists", "off"),
        ("preset.reach", "fewer"),
    ]  # fmt: skip
    r = client.post(
        "/settings",
        content=urlencode(form),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/settings?msg=")
    now = values(config_file)
    assert now["staff.enabled"] is True and now["media.journalists"] is False
    assert now["groups.a.ca_above"] == 165 and now["render.vae_tiling"] is False


def test_the_quality_this_card_suits_is_marked(config_file: Path) -> None:
    db_path = synced(config_file, [player(1001, name="Mine")])
    ctx = AppContext(db_path, config_file, None, hardware=lambda: Hardware("Fake 8 GB", 8 * GIB))
    with TestClient(create_app(ctx), base_url=BASE) as client:
        for _ in range(50):
            page = client.get("/settings").text
            if "Recommended" in page:
                break
    match = re.search(r'value="(\w+)" data-preset[^>]*>\s*<span class="choice-name">[^<]+'
                      r'<span class="tag ok" data-tip="Fake 8 GB">Recommended', page)  # fmt: skip
    assert match is not None and match[1] == "low"


def test_the_settings_link_opens_the_level_shown_last_even_after_a_restart(
    client: TestClient, config_file: Path
) -> None:
    advanced_on = 'href="/settings/advanced" class="on" aria-current="page">Advanced'
    assert "Who gets a picture" in client.get("/settings").text  # Basic at first
    client.get("/settings/advanced")
    assert advanced_on in client.get("/settings").text
    client.get("/gallery")  # another page in between
    page = client.get("/settings").text
    assert advanced_on in page and 'name="groups.a.ca_above"' in page
    # A new run of the app on the same data folder.
    db_path = load_settings(defaults_path=config_file).paths.db_path
    again = client_for(config_file, db_path)
    try:
        assert advanced_on in again.get("/settings").text
        again.get("/settings/basic")
        assert "Who gets a picture" in again.get("/settings").text
    finally:
        again.close()
    assert (load_settings(defaults_path=config_file).paths.data_dir / "settings-level").exists()
