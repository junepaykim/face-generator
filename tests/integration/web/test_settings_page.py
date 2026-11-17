"""The settings page after the user's Phase 12 review (2026-11-17): plain words, group names
as chips, age looks by name with "-" to leave one out, ⓘ tips instead of long notes, and
the run's details in About this run."""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import unescape
from pathlib import Path

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.core.age import WORDINGS
from facegen.web.forms import BAND_TAGS


@pytest.fixture
def client(config_file: Path) -> Iterator[TestClient]:
    world = [player(1001, name="Mine"), player(NEWGEN + 2, ca=90, pa=152, club_id=RIVAL)]
    db_path = synced(config_file, world)
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    yield c
    c.close()


def save(client: TestClient, **values: str) -> dict:
    return client.post("/settings/save", data=values).json()


def test_each_band_tag_is_part_of_its_wording() -> None:
    assert list(BAND_TAGS) == list(WORDINGS)
    for wording, tags in BAND_TAGS.items():
        for tag in tags.split(", "):
            assert tag in WORDINGS[wording].text, wording


def test_age_looks_are_named_with_a_tag_or_two_and_a_dash_leaves_one_out(
    client: TestClient,
) -> None:
    page = client.get("/settings").text
    assert "Child face (child)" in page and "Late youthful (teenage girl)" in page
    assert re.search(r'name="age.b.W03" data-key data-default="-" value="-"', page)
    korean = client.get("/settings", headers=KOREAN_WINDOWS).text
    assert "어린 얼굴 (child)" in korean and "앳된 얼굴 (young girl)" in korean
    data = save(client, **{"age.b.W03": "17"})
    assert data["ok"] and data["fields"]["age.b.W03"]["text"] == "17"
    assert "Age looks: Standard · Late teens not used → 17" in data["message"]
    back = save(client, **{"age.b.W03": "-"})
    assert back["ok"] and back["fields"]["age.b.W03"]["text"] == "-"
    bad = save(client, **{"age.b.W03": "soon"})
    assert not bad["ok"] and "- to leave it out" in bad["errors"]["age.b.W03"]


def test_group_names_show_as_chips_in_labels_and_titles(client: TestClient) -> None:
    page = client.get("/settings").text
    standard = '<span class="grp grp-b" data-tip="One picture each at standard quality: strong and promising players in my league, continental competitions and my youth teams, and the press.">Standard</span>'
    assert f"Academy players stay in the {standard} group after they leave" in page
    assert f"<legend>{standard} group" in page
    korean = client.get("/settings", headers=KOREAN_WINDOWS).text
    assert re.search(
        r'내 구단 출신 유스는 이적해도 <span class="grp grp-b" data-tip="[^"]+">표준</span> 그룹에 남기기',
        korean,
    )


def test_groups_drawn_automatically_are_tick_boxes(client: TestClient) -> None:
    page = client.get("/settings").text
    assert '<input type="hidden" name="schedule.auto_groups" data-key' in page
    for group in "abc":
        assert f'data-group-of="schedule.auto_groups" value="{group}" checked' in page
    data = save(client, **{"schedule.auto_groups": "a, c"})
    assert data["ok"]
    assert (
        "Groups drawn automatically Detailed, Standard, Light → Detailed, Light"
        in (data["message"])
    )
    page = client.get("/settings").text
    assert 'data-group-of="schedule.auto_groups" value="b" >' in page


def test_long_notes_are_tips_and_the_obvious_ones_are_gone(client: TestClient) -> None:
    page = unescape(client.get("/settings").text)
    assert '<p class="muted">Settings save' not in page  # in the heading's ⓘ (user, 2026-11-17)
    for gone in (
        "Thresholds re-classify",
        "The worker and the save watcher use it at once",
        "The install folder is found on this PC",
        "Without a photo means",
    ):
        assert gone not in page, gone
    tips = re.findall(r'<span class="tip" aria-hidden="true">(.*?)</span></span>', page, re.DOTALL)
    assert any(tip.startswith("Settings save automatically and apply at once") for tip in tips)
    assert any(tip.startswith("How much a new character's expression") for tip in tips)
    assert any(tip.startswith("When a picture is drawn again") for tip in tips)
    assert any(tip.startswith("Auto-configure sets the") for tip in tips)
    assert any(tip.startswith("The age each look starts at") for tip in tips)
    assert "Pictures with these words become much less likely." in tips
    korean = unescape(client.get("/settings", headers=KOREAN_WINDOWS).text)
    assert "금지어에 추가된 키워드를 포함하는 이미지가 생성될 가능성이 대폭 줄어듭니다." in korean


def test_about_this_run_has_what_left_the_dashboard(client: TestClient) -> None:
    page = unescape(client.get("/settings").text)
    assert "About this run" in page
    assert "Image model</th><td>not loaded" in page
    assert "This run</th><td>0 made, 0 sent to review" in page
    assert "Save watcher</th><td>off (" in page  # no watcher in this app
    status = client.get("/status").text
    assert "Rust code" not in status and "--no-watch" not in status
    assert "The worker draws manual requests first" not in status


def test_review_says_why_in_a_tip_and_an_empty_list_plainly(client: TestClient) -> None:
    page = client.get("/review").text
    assert re.search(r'<h1>Review <span class="info"', page)
    assert "No players to review right now." in page
    korean = client.get("/review", headers=KOREAN_WINDOWS).text
    assert "현재 검토 대상인 선수가 없습니다." in korean
