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
from support.config import write_config
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, synced

from kitshot.core import presets
from kitshot.persistence import db
from kitshot.persistence.settings import load_defaults, load_settings
from kitshot.pipeline import settings_history
from kitshot.setup import hardware
from kitshot.setup.hardware import Hardware
from kitshot.web.app import AppContext, create_app

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
        "staff.enabled", "staff.include_human_manager", "media.journalists",
        "media.press_officers", "no_photo.enabled",
        "schedule.generate_while_fm_running", "schedule.auto_groups",
        "notifications.new_pictures",
    ):  # fmt: skip
        assert f'name="{key}" data-key' in page, key
    assert 'name="cutoffs.everywhere.a_ca"' not in page and "About this run" not in page
    assert 'href="/settings/basic" class="on" aria-current="page">Basic' in page
    advanced = client.get("/settings/advanced").text
    assert 'name="cutoffs.everywhere.a_ca"' in advanced and "Recent changes" in advanced
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
    more = presets.values_of("reach", "more", defaults)
    assert all(now[k] == more[k] for k in presets.REACH_KEYS)
    assert now["cutoffs.everywhere.a_ca"] == defaults["cutoffs.everywhere.a_ca"] - 10
    assert now["cutoffs.share.b_ca"] == defaults["cutoffs.share.b_ca"] * 1.5
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
    assert client.post("/settings/save", data={"cutoffs.everywhere.b_pa": "149"}).json()["ok"]
    page = client.get("/settings").text
    assert checked(page, "reach") == []
    assert re.search(r'data-custom="reach"\s*>Set by hand in <a href="/settings/advanced">', page)
    assert re.search(r'data-custom="quality" hidden>', page)


def test_basic_switches_and_quality_save_like_advanced_fields(
    client: TestClient, config_file: Path
) -> None:
    key = "staff.include_human_manager"
    control = f'name="{key}" data-key'
    assert control + " checked" not in client.get("/settings/basic").text
    assert client.post("/settings/save", data={key: "on"}).json()["ok"]
    assert values(config_file)[key] is True
    assert control + " checked" in client.get("/settings/basic").text
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
    assert labels == ["preset.quality.low", "edit", "edit", "edit"]


def test_the_picture_style_is_one_model_and_one_gender_for_everyone(tmp_path: Path) -> None:
    """User, 2026-11-20: one model and one gender, on Basic and Advanced alike. Z-Image-Turbo
    is offered since the worker draws its re-draws without a reference (Phase 15g). RealVisXL
    and men by default, the models listed RealVisXL, Z-Image-Turbo, Animagine (user,
    2026-11-25)."""
    config_file = write_config(tmp_path, style=True)
    db_path = synced(config_file, [player(1001, name="Mine")])
    client = client_for(config_file, db_path, fake_worker(config_file, db_path))
    page = unescape(client.get("/settings").text)
    assert "Picture style" in page
    assert '<option value="realvis" selected>Photo · Standard (RealVisXL)</option>' in page
    assert re.search(
        r'<option value="zimage_turbo"\s*>Photo · High quality \(Z-Image-Turbo\)</option>', page
    )
    assert re.search(r'<option value="animagine"\s*>Anime \(Animagine XL\)</option>', page)
    assert '<option value="man" selected>Man</option>' in page
    assert re.search(r'<option value="woman"\s*>Woman</option>', page)
    order = [page.index(f'<option value="{m}"') for m in ("realvis", "zimage_turbo", "animagine")]
    assert order == sorted(order)
    for key in ("style.model", "style.gender"):
        assert f'<select name="{key}" data-key>' in page, key
        assert (
            f'<select name="{key}" data-key data-default=' in client.get("/settings/advanced").text
        )

    data = client.post("/settings/save", data={"style.model": "animagine", "style.gender": "woman"})
    assert data.json()["ok"]
    now = values(config_file)
    assert (now["style.model"], now["style.gender"]) == ("animagine", "woman")
    page = unescape(client.get("/settings").text)
    assert '<option value="animagine" selected>Anime (Animagine XL)</option>' in page
    for key, value in (("style.model", "flux"), ("style.gender", "other")):
        bad = client.post("/settings/save", data={key: value}).json()
        assert not bad["ok"] and key in bad["errors"], key
    assert client.post("/settings/save", data={"style.model": "zimage_turbo"}).json()["ok"]
    assert values(config_file)["style.model"] == "zimage_turbo"
    korean = unescape(client.get("/settings/basic", headers=KOREAN_WINDOWS).text)
    for text in (
        "그림 스타일", "실사·표준 (RealVisXL)", "실사·고퀄 (Z-Image-Turbo)", "애니 (Animagine XL)",
        "남성", "여성",
    ):  # fmt: skip
        assert text in korean, text


def test_the_picture_style_says_z_image_turbo_needs_10_gb(client: TestClient) -> None:
    """Its render needs about 6 GiB in every preset (Phase 15i): too much for an 8 GB card
    beside Windows. Said on Basic and Advanced alike (user, 2026-11-21)."""
    note = "Z-Image-Turbo needs a graphics card with 10 GB or more."
    for path in ("/settings", "/settings/advanced"):
        page = unescape(client.get(path).text)
        assert page.count(note) == 1, path
        korean = unescape(client.get(path, headers=KOREAN_WINDOWS).text)
        assert korean.count("Z-Image-Turbo는 그래픽카드 메모리가 10 GB 이상 필요합니다.") == 1, path
    advanced = unescape(client.get("/settings/advanced").text)
    assert "Journalists and press officers are drawn in the" in advanced
    assert "drawn as women" not in advanced


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
    assert now["cutoffs.everywhere.a_ca"] == 175 and now["render.vae_tiling"] is False  # 165+10


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
    assert advanced_on in page and 'name="cutoffs.everywhere.a_ca"' in page
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
