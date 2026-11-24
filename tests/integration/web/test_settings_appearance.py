"""The Appearance tab (Phase 17l): age looks as one table of the players' groups side by side and
one of staff and media, each column Auto (an even spread) or Custom, an empty cell for a look not
used, All -1 / All +1 as one change; then players at my club, height and build, personality."""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import unescape
from pathlib import Path

import pytest
from factories import MANUAL_MODE, MY_CLUB, media_person, player, snapshot, staff_member
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker

from kitshot.core.age import TABLES
from kitshot.i18n import Translator
from kitshot.persistence import db
from kitshot.persistence.settings import load_settings, set_overrides
from kitshot.pipeline.sync import run_sync
from kitshot.web.forms import describe_key, label_parts, value_text

MINE = 1030  # at my club, 19: my club's W03 (19-21)


@pytest.fixture
def client(config_file: Path) -> Iterator[TestClient]:
    db_path = load_settings(defaults_path=config_file).paths.db_path
    conn = db.connect(db_path)
    settings = set_overrides(conn, MANUAL_MODE, config_file)
    people = snapshot(
        [player(MINE, age=19)],
        staff=[staff_member(3000 + n, age=age) for n, age in enumerate((25, 30, 45))],
        media=[
            media_person(4000, age=29),
            media_person(4001, age=40),
            media_person(4002, age=55),
            media_person(4003, age=61, nation=99),  # from abroad: no picture, not counted
        ],
    )
    run_sync(conn, people, settings, save_file="career.fm")
    # MINE already has a picture at my club's W03: a new look re-draws them.
    conn.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, club_id, age_band,"
        " model, prompt, negative_prompt, seed, is_current, created_at)"
        " VALUES (1, ?, 1, 'initial', 'a', ?, 'W03', 'm', 'p', 'n', 1, 1, 'now')",
        (MINE, MY_CLUB),
    )
    conn.execute("UPDATE jobs SET status = 'done' WHERE unique_id = ?", (MINE,))
    conn.close()
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    yield c
    c.close()


def save(client: TestClient, **values: str) -> dict:
    r = client.post("/settings/save", data=values)
    assert r.status_code == 200, r.text
    return r.json()


def appearance(client: TestClient, **headers: str) -> str:
    page = unescape(client.get("/settings/advanced", headers=headers).text)
    found = re.search(
        r'<section class="tab-panel" id="tab-appearance".*?</section>', page, re.DOTALL
    )
    assert found is not None
    return found[0]


def shown(page: str, key: str) -> str:
    found = re.search(rf'data-effective="{re.escape(key)}">([^<]*)</span>', page)
    assert found is not None, key
    return found[1]


def columns(table_html: str) -> list[str]:
    head = re.search(r"<thead>.*?</thead>", table_html, re.DOTALL)[0]
    return [re.sub(r"<[^>]+>", "", th) for th in re.findall(r'<th scope="col">(.*?)</th>', head)]


def test_appearance_holds_age_looks_then_how_players_look(client: TestClient) -> None:
    tab = appearance(client)
    assert '<h2 class="tab-title">Appearance</h2>' in tab
    order = ['id="section-age"', 'id="section-my_club"', 'id="section-body"']
    order.append('id="section-personality"')
    places = [tab.index(marker) for marker in order]
    assert places == sorted(places)
    # Two tables, no longer six stacked cards.
    players = re.search(
        r'<table class="age-table cols-4" id="age-players">.*?</table>', tab, re.DOTALL
    )
    others = re.search(
        r'<table class="age-table cols-2" id="age-others">.*?</table>', tab, re.DOTALL
    )
    assert players and others
    assert columns(players[0]) == [
        "Look", "Detailed · my club", "Detailed · elsewhere", "Standard", "Light",
    ]  # fmt: skip
    assert columns(others[0]) == ["Look", "My club's staff", "Journalists and press officers"]
    for table in TABLES:
        assert f'id="section-age-{table}"' not in tab
    # Every row of both tables: the look's number, name and prompt words (staff's too).
    for html in (players[0], others[0]):
        assert re.findall(r'<span class="look-no">(\d+)</span>', html) == [
            str(n) for n in range(1, 11)
        ]
        assert '<span class="look-name">Veteran look</span>' in html
        assert ">mature female</span>" in html
    korean = appearance(client, **KOREAN_WINDOWS)
    assert '<h2 class="tab-title">외형</h2>' in korean and "<legend>나이 표현" in korean


def test_players_tables_are_custom_and_the_media_table_is_spread(client: TestClient) -> None:
    tab = appearance(client)
    for table in ("a_my_club", "a_other_club", "b", "c"):
        assert re.search(rf'name="age.{table}.mode" value="manual"[^>]* checked', tab)
        assert f'name="age.{table}.W05"' in tab and f'name="age.{table}.start"' not in tab
        assert f'data-shift="{table}" data-step="-1"' in tab  # All −1 / All +1, with the script
    for table in ("staff", "media"):
        assert re.search(rf'name="age.{table}.mode" value="auto"[^>]* checked', tab)
        assert f'name="age.{table}.start"' in tab and f'name="age.{table}.W05"' not in tab
        assert f'data-shift="{table}"' not in tab
    # Its own labels, not the cutoffs' "Auto (top %)".
    assert "<span>Auto</span>" in tab and "<span>Custom</span>" in tab and "top %" not in tab
    # The journalists' ages greyed: W04 to W10 over 30-60; how many with a picture have each.
    media = {look: shown(tab, f"ages.media.{look}") for look in ("W03", "W04", "W05", "W10")}
    assert media == {
        "W03": "Not used", "W04": "30 and under", "W05": "31–36", "W10": "60 and over",
    }  # fmt: skip
    counts = {look: shown(tab, f"ages.media.{look}.n") for look in ("W04", "W06", "W09", "W10")}
    assert counts == {"W04": "1 now", "W06": "1 now", "W09": "1 now", "W10": "0 now"}  # 61: abroad
    korean = appearance(client, **KOREAN_WINDOWS)
    assert shown(korean, "ages.media.W04.n") == "현재 1명"
    assert shown(korean, "ages.media.W04") == "30세 이하"
    assert "ages.b.W05.n" not in tab  # the players' tables are not counted
    # An empty cell is a look not used.
    assert re.search(
        r'name="age.b.W03" data-key data-default="" value="" placeholder="Not used"', tab
    )


def test_a_players_table_switched_to_auto_and_back(client: TestClient) -> None:
    data = save(client, **{"age.a_my_club.mode": "auto"})
    assert data["ok"] and data["reload"]
    # The player of 19 moves from W03 to W02 (17-19): re-drawn, and the message says so.
    assert "1 queued (1 for a new age look)" in data["message"]
    assert data["effective"]["ages.a_my_club.W02"] == "17–19"
    assert data["effective"]["ages.a_my_club.W01"] == "16 and under"
    tab = appearance(client)
    assert 'name="age.a_my_club.start"' in tab and 'name="age.a_my_club.W03"' not in tab
    assert shown(tab, "ages.a_my_club.W10") == "34 and over"
    assert 'data-shift="a_my_club"' not in tab
    # A spread input changes the greyed ages in place.
    data = save(client, **{"age.a_my_club.end": "40"})
    assert data["ok"] and not data["reload"]
    assert data["effective"]["ages.a_my_club.W10"] == "40 and over"
    # History names the table and its mode in its own words.
    page = unescape(client.get("/settings/advanced").text)
    assert "Age looks: Detailed at my club · Ages: Custom → <strong>Auto</strong>" in page

    back = save(client, **{"age.a_my_club.mode": "manual"})
    assert back["ok"] and back["reload"]
    # The spread copied: W02 stays at 17, W03 starts at 20 (16-40 since the edit above).
    assert "age.a_my_club.W02" not in back["fields"]
    assert back["fields"]["age.a_my_club.W03"]["text"] == "20"
    assert back["fields"]["age.a_my_club.W01"]["text"] == "0"


def test_all_minus_one_is_one_change_with_its_own_name(client: TestClient) -> None:
    before = load_settings_values(client)
    data = save(client, **{"shift.age.b": "-1"})
    assert data["ok"] and data["changed"]
    assert data["message"].startswith("Saved: Standard: every age -1.")
    assert data["fields"]["age.b.W02"] == {"value": 14, "text": "14", "changed": True}
    assert data["fields"]["age.b.W09"]["text"] == "28"
    assert "age.b.W03" not in data["fields"]  # not used: left as it is
    assert "Standard: every age -1" in unescape(data["history_html"])
    assert data["history_count"] == 1
    undone = client.post("/settings", data={"action": "undo"})
    assert undone.status_code == 303
    assert load_settings_values(client) == before
    # A spread table has nothing to shift; a shift that is not a whole number is refused.
    assert not save(client, **{"shift.age.media": "1"})["changed"]
    bad = save(client, **{"shift.age.b": "soon"})
    assert not bad["ok"] and "shift.age.b" in bad["errors"]
    korean = client.post("/settings/save", data={"shift.age.c": "1"}, headers=KOREAN_WINDOWS).json()
    assert korean["message"].startswith("저장함: 간편: 모든 나이 +1.")


def load_settings_values(client: TestClient) -> dict[str, object]:
    page = client.get("/settings/advanced").text
    return dict(
        re.findall(
            r'name="(age\.[a-z_]+\.W\d\d)" data-key data-default="[^"]*" value="([^"]*)"', page
        )
    )


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_every_tables_mode_and_spread_have_labels(lang: str) -> None:
    t = Translator(lang)
    for table in TABLES:
        for leaf in ("mode", "from", "to", "start", "end"):
            key = f"age.{table}.{leaf}"
            label, values = label_parts(key, t)
            assert label and t.has(label), key
            assert describe_key(key, t).endswith(t(label, **values))
        assert value_text("auto", t, f"age.{table}.mode") == t("settings.choice.age_mode.auto")
    assert value_text("manual", Translator("en"), "age.b.mode") == "Custom"
    assert value_text("auto", Translator("en"), "cutoffs.mode") == "Auto (top %)"
