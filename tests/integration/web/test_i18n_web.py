"""Korean and English pages, the language and theme switches, and the sidebar (Phase 12a/b)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker, post_player_action, synced

MINE = 1001
KID = NEWGEN + 2
PAGES = ("/", "/status", "/players", f"/players/{MINE}", "/review", "/settings", "/hardware")


@pytest.fixture
def client(config_file: Path) -> Iterator[TestClient]:
    world = [
        player(MINE, name="Mine Player", age=29),
        player(KID, name="Rival Kid", ca=90, pa=152, club_id=RIVAL),
    ]
    db_path = synced(config_file, world)
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    yield c
    c.close()


def save(client: TestClient, **values: str) -> dict:
    r = client.post("/settings/save", data=values, headers={"Accept": "application/json"})
    assert r.status_code == 200
    return r.json()


def test_english_by_default_and_korean_for_a_korean_window(client: TestClient) -> None:
    page = client.get("/").text
    assert '<html lang="en" data-theme="auto">' in page and ">Dashboard<" in page
    korean = client.get("/", headers=KOREAN_WINDOWS).text
    assert '<html lang="ko" data-theme="auto">' in korean
    assert ">대시보드<" in korean and ">대기열" in korean and ">Dashboard<" not in korean


@pytest.mark.parametrize("url", PAGES)
def test_every_page_renders_in_korean_without_raw_keys(client: TestClient, url: str) -> None:
    r = client.get(url, headers=KOREAN_WINDOWS)
    assert r.status_code == 200, url
    text = re.sub(r"<(script|style)\b.*?</\1>", "", r.text, flags=re.DOTALL)
    visible = re.sub(r"<[^>]+>", " ", text)
    # A key no catalog holds shows as itself: "dash.queue.title" and the like.
    raw = re.findall(
        r"\b(?:nav|dash|setup|players|player|settings|msg|col|tag|kind|tier)\.[a-z_]+", visible
    )
    assert raw == [], url


def test_the_language_switch_wins_over_the_window_and_can_be_undone(client: TestClient) -> None:
    data = save(client, **{"display.language": "ko"})
    assert data["ok"] and data["changed"] and data["reload"]
    assert ">대시보드<" in client.get("/").text  # no Accept-Language: the setting decides
    assert ">대시보드<" in client.get("/", headers={"Accept-Language": "en-US"}).text
    settings = client.get("/settings/advanced").text
    assert "언어와 화면" in settings and "최근 변경" in settings  # listed, and undoable
    r = client.post("/settings", data={"action": "undo"})
    assert r.status_code == 303
    assert ">Dashboard<" in client.get("/").text


def test_flash_messages_follow_the_language(client: TestClient) -> None:
    r = post_player_action(
        client, "/players/action", data={"action": "pin", "pin": "never"}, headers=KOREAN_WINDOWS
    )
    assert r.status_code == 303
    page = client.get(r.headers["location"], headers=KOREAN_WINDOWS).text
    assert "먼저 선수를 선택하세요." in page
    r = post_player_action(
        client,
        "/players/action",
        data={"action": "pin", "pin": "never", "uid": [str(MINE), str(KID)]},
        headers=KOREAN_WINDOWS,
    )
    assert (
        "선수 2명을 그리지 않음으로 고정했습니다."
        in client.get(r.headers["location"], headers=KOREAN_WINDOWS).text
    )


def test_settings_errors_and_outcomes_are_worded_in_the_language(client: TestClient) -> None:
    bad = client.post(
        "/settings/save",
        data={"groups.a.ca_above": "lots"},
        headers={"Accept": "application/json", **KOREAN_WINDOWS},
    ).json()
    assert not bad["ok"] and bad["errors"]["groups.a.ca_above"] == "숫자를 입력하세요"
    good = client.post(
        "/settings/save",
        data={"groups.a.ca_above": "150"},
        headers={"Accept": "application/json", **KOREAN_WINDOWS},
    ).json()
    assert good["message"].startswith("저장함: 정밀 그룹 · CA 초과 155 → 150.")
    assert not good["reload"]


def test_display_settings_are_checked(client: TestClient) -> None:
    for key, value in (("display.language", "fr"), ("display.theme", "blue")):
        data = save(client, **{key: value})
        assert not data["ok"] and key in data["errors"]
    assert '<html lang="en" data-theme="auto">' in client.get("/").text


def test_the_theme_switch(client: TestClient) -> None:
    assert save(client, **{"display.theme": "dark"})["reload"]
    page = client.get("/").text
    assert 'data-theme="dark"' in page
    assert re.search(r'<option value="dark" selected>Dark</option>', page)


def test_the_sidebar_names_the_pages_and_marks_the_current_one(client: TestClient) -> None:
    page = client.get("/players").text
    assert '<aside class="side">' in page and '<header class="top">' not in page
    assert re.search(r'<a href="/players" class="on"[^>]*>', page)
    assert 'select data-display="display.language"' in page
    assert 'select data-display="display.theme"' in page
    assert 'id="csrf"' in page  # the switches post with the form token
