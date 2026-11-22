"""The Advanced page in five tabs, with live cutoff values and counts, and my club's staff
spread over an age span (Phase 16e, 16f, 16h)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from html import unescape
from pathlib import Path

import pytest
from factories import MY_CLUB, NEWGEN, RIVAL, RIVAL_2, player, snapshot, staff_member
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker

from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline.sync import run_sync

TABS = ("cutoffs", "quality", "targets", "ages", "system")


def world() -> list:
    """Newgens in my league with CA 60 to 150 by 10 and PA 100 to 190."""
    clubs = (RIVAL, RIVAL_2)
    return [
        player(NEWGEN + i, ca=60 + 10 * i, pa=100 + 10 * i, club_id=clubs[i % 2]) for i in range(10)
    ]


def sync(config_file: Path, *, staff: tuple = ()) -> Path:
    conn = db.connect(load_settings(defaults_path=config_file).paths.db_path)
    settings = load_settings(conn, config_file)
    run_sync(conn, snapshot(world(), staff=staff), settings, save_file="career.fm")
    conn.close()
    return settings.paths.db_path


@pytest.fixture
def client(config_file: Path) -> Iterator[TestClient]:
    staff = tuple(staff_member(3000 + n, age=age) for n, age in enumerate((25, 30, 31, 45, 70)))
    db_path = sync(config_file, staff=staff)
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    yield c
    c.close()


def save(client: TestClient, **values: str) -> dict:
    r = client.post("/settings/save", data=values)
    assert r.status_code == 200, r.text
    return r.json()


def test_five_tabs_each_with_its_own_form_and_the_bar_outside(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    nav = re.search(r'<nav class="pills".*?</nav>', page, re.DOTALL)
    assert nav is not None
    assert re.findall(r'data-tab="(\w+)"', nav[0]) == list(TABS)
    panels = re.findall(r'<section class="tab-panel" id="tab-(\w+)" data-tab-panel="\w+">', page)
    assert panels == list(TABS)
    # Each tab saves on its own; Undo, Redo and Recent changes are outside every tab.
    assert page.count("data-autosave") == 5
    assert page.index("data-undo") < page.index('id="tab-cutoffs"')
    assert page.index('id="history"') > page.index('id="tab-system"')
    # Without the script every tab shows, each with its title.
    assert 'class="tab-panel" id="tab-quality" data-tab-panel="quality" hidden' not in page
    assert '<h2 class="tab-title">Quality</h2>' in page


def test_each_section_is_in_its_tab_and_old_anchors_open_it(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    anchors = json.loads(unescape(re.search(r'data-anchors="([^"]+)"', page)[1]))
    assert anchors["about"] == "system" and anchors["graphics-card"] == "quality"
    assert anchors["age"] == "ages" and anchors["groups"] == "quality"
    assert anchors["eligibility"] == "system" and anchors["staff"] == "targets"

    def tab_of(marker: str) -> str:
        at = page.index(marker)
        starts = [(page.index(f'id="tab-{t}"'), t) for t in TABS]
        return max((pos, t) for pos, t in starts if pos < at)[1]

    assert tab_of('name="cutoffs.share.a_ca"') == "cutoffs"
    assert tab_of('name="cutoffs.everywhere.b_pa"') == "cutoffs"
    assert tab_of('name="groups.b.candidates"') == "quality"
    assert tab_of('name="picture.aspect"') == "quality"
    assert (
        tab_of('name="style.model"') == "quality"
        and tab_of('name="my_club.soften_skin"') == "quality"
    )
    assert tab_of('name="media.journalists"') == "targets"
    assert tab_of('name="age.b.W04"') == "ages" and tab_of('name="age.staff.mode"') == "ages"
    assert (
        tab_of('id="about"') == "system"
        and tab_of('name="eligibility.continental_competition_ids"') == "system"
    )
    assert tab_of('name="newgen.fallback_min_unique_id"') == "system"
    # A tab's form comes back to its tab without the script.
    assert 'name="next" value="/settings/advanced#tab-ages"' in page


def test_the_cutoffs_tab_shows_live_values_counts_and_the_leagues(client: TestClient) -> None:
    page = unescape(client.get("/settings/advanced").text)
    # My league's 10 candidates, CA 60-150: top 2.5% is rank 1 (150), top 12% rank 2 (140).
    assert 'data-effective="cutoffs.share.a_ca">≈ CA 150+' in page
    assert 'data-effective="cutoffs.share.b_ca">≈ CA 140+' in page
    assert 'data-effective="cutoffs.share.above_b_ca">–' in page  # no league above found
    summary = re.search(r'data-part="tier-summary".*?</div>', page, re.DOTALL)[0]
    assert "Now:" in summary and re.search(r"Standard</span>\s*<strong>\d+</strong>", summary)
    assert "My league:" in page and "Club 101" in page
    # The everywhere cuts are folded, their summary shows them.
    assert re.search(
        r'<details class="everywhere" id="section-cutoffs-everywhere">\s*<summary>', page
    )
    assert "CA 165+ or PA 180+" in page

    # Light: top 80% is CA 80+ (four players: CA 80-110, the rest are Detailed or Standard by
    # PA 160+ and 180+ everywhere); top 40% is CA 120+ (none left).
    data = save(client, **{"cutoffs.share.c_ca": "40"})
    assert data["ok"] and data["changed"]
    assert data["effective"]["cutoffs.share.c_ca"] == "≈ CA 120+"
    assert "Light 4 → 0" in data["message"]
    summary = data["parts"]["tier-summary"]
    assert re.search(r'Light</span>\s*<span class="muted">4</span> → <strong>0</strong>', summary)
    undone = client.post("/settings", data={"action": "undo"})
    assert undone.status_code == 303
    page = unescape(client.get("/settings/advanced").text)
    assert 'data-effective="cutoffs.share.c_ca">≈ CA 80+' in page
    assert re.search(r"Light</span>\s*<strong>4</strong>", page)


def test_switching_to_manual_converts_and_redraws_the_page(client: TestClient) -> None:
    data = save(client, **{"cutoffs.mode": "manual"})
    assert data["ok"] and data["reload"]
    assert data["fields"]["cutoffs.manual.b_ca"]["value"] == 140  # converted from top 12%
    assert data["effective"]["cutoffs.manual.b_ca"] == "≈ top 20%"  # 2 of 10 at 140 or more
    page = client.get("/settings/advanced").text
    assert 'name="cutoffs.manual.b_ca"' in page and 'name="cutoffs.share.b_ca"' not in page
    back = save(client, **{"cutoffs.mode": "auto"})
    assert back["ok"] and back["reload"]
    assert 'name="cutoffs.share.b_ca"' in client.get("/settings/advanced").text


def test_before_any_sync_the_live_values_are_dashes(config_file: Path) -> None:
    db_path = load_settings(defaults_path=config_file).paths.db_path
    db.connect(db_path).close()
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    page = unescape(c.get("/settings/advanced").text)
    assert 'data-effective="cutoffs.share.a_ca">–' in page
    assert "Values show once a save is synced." in page
    assert re.search(r"Detailed</span>\s*–", page)
    c.close()


def test_the_staff_looks_list_ages_and_counts_and_auto_spreads_them(client: TestClient) -> None:
    page = unescape(client.get("/settings/advanced").text)
    looks = re.search(r'data-part="staff-looks".*?</table>', page, re.DOTALL)[0]
    # Auto by default (user, 2026-11-22): W02 to W10 over 28-64, staff aged 25, 30, 31, 45, 70.
    assert re.search(r"Young face</td>\s*<td>28 and under</td>\s*<td class=\"num\">1", looks)
    assert re.search(r"Late teens</td>\s*<td>29–33</td>\s*<td class=\"num\">2", looks)
    assert re.search(r"Veteran look</td>\s*<td>64 and over</td>\s*<td class=\"num\">1", looks)
    assert 'name="age.staff.start"' in page and 'name="age.staff.W03"' not in page
    refused = save(client, **{"age.staff.end": "32"})  # 3 years for 7 looks between
    assert not refused["ok"] and "age.staff.end" in refused["errors"]

    # Manual: the spread is copied into the looks, as one change.
    data = save(client, **{"age.staff.mode": "manual"})
    assert data["ok"] and data["reload"]
    assert data["fields"]["age.staff.W03"]["value"] == 29
    assert data["fields"]["age.staff.W10"]["value"] == 64
    assert data["fields"]["age.staff.W02"]["value"] == 0  # the first look: any age up to 28
    looks = unescape(data["parts"]["staff-looks"])
    assert re.search(r"Late teens</td>\s*<td>29–33</td>\s*<td class=\"num\">2", looks)
    page = client.get("/settings/advanced").text
    assert 'name="age.staff.W03"' in page and 'name="age.staff.start"' not in page
    data = save(client, **{"age.staff.W03": "25"})  # a look's first age, set by hand
    looks = unescape(data["parts"]["staff-looks"])
    assert re.search(r"Young face</td>\s*<td>24 and under</td>\s*<td class=\"num\">0", looks)
    assert re.search(r"Late teens</td>\s*<td>25–33</td>\s*<td class=\"num\">3", looks)
    client.post("/settings", data={"action": "undo"})
    client.post("/settings", data={"action": "undo"})  # one Undo per change: back to the spread
    assert 'name="age.staff.start"' in client.get("/settings/advanced").text


def test_the_tabs_in_korean(client: TestClient) -> None:
    page = unescape(client.get("/settings/advanced", headers=KOREAN_WINDOWS).text)
    for word in ("선정 기준", "품질", "대상", "나이 표현", "시스템", "내 리그: CA", "상위 리그"):
        assert word in page, word
    assert "≈ CA 150 이상" in page


def test_my_club_tiers_and_the_shape_are_choices(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    assert re.search(r'<select name="cutoffs.my_club.loans" data-key data-default="b"', page)
    assert '<option value="b" selected>Standard</option>' in page
    assert '<option value="1:1" selected>1:1 square</option>' in page  # by default (2026-11-22)
    assert '<option value="3:4" >3:4 portrait</option>' in page
    data = save(client, **{"cutoffs.my_club.youth": "c", "picture.aspect": "3:4"})
    assert data["ok"] and "My club · Youth and reserve teams Standard → Light" in data["message"]
    assert MY_CLUB == 100
