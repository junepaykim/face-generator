"""The Advanced page in five tabs, with live cutoff values and counts, and my club's staff
spread over an age span (Phase 16e, 16f, 16h)."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from html import unescape
from pathlib import Path

import pytest
from factories import (
    MY_CLUB,
    MY_LEAGUE_COMP,
    NEWGEN,
    RIVAL,
    RIVAL_2,
    player,
    snapshot,
    staff_member,
)
from fastapi.testclient import TestClient
from support.paths import ROOT
from support.web import KOREAN_WINDOWS, client_for, fake_worker

from kitshot.i18n import Translator
from kitshot.persistence import db
from kitshot.persistence.settings import load_settings
from kitshot.pipeline.sync import run_sync
from kitshot.platform.procs import run_quiet
from kitshot.web.forms import describe_key, label_parts

TABS = ("cutoffs", "quality", "targets", "appearance", "system")


def test_autosave_waits_for_edits_before_reload() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise browser JavaScript")
    run_quiet(
        [
            node,
            Path(__file__).with_name("settings_autosave.cjs"),
            ROOT / "src/kitshot/web/static/app.js",
        ],
        check=True,
        timeout=15,
    )


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_staff_spread_uses_translated_field_and_history_labels(lang: str) -> None:
    t = Translator(lang)
    for leaf in ("from", "to", "start", "end"):
        key = f"age.staff.{leaf}"
        label, values = label_parts(key, t)
        expected = t(f"settings.label.{leaf}")
        assert t(label, **values) == expected
        assert describe_key(key, t).endswith(expected)


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


def test_everywhere_summary_matches_the_saved_cutoffs(client: TestClient) -> None:
    data = save(client, **{"cutoffs.everywhere.a_ca": "175"})
    summary = data["parts"]["everywhere-summary"]
    assert "CA 175+ or PA 180+" in summary
    page = client.get("/settings/advanced").text
    assert f'<summary data-part="everywhere-summary">{summary}</summary>' in page


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
    assert anchors["age"] == "appearance" and anchors["groups"] == "quality"
    # The Age looks tab's old address and the looks' sections open Appearance (Phase 17l).
    assert anchors["ages"] == anchors["my_club"] == anchors["body"] == "appearance"
    assert anchors["staff"] == "targets"
    # Who counts opens Cutoffs (Phase 17k); the sections whose settings moved open System.
    assert anchors["eligibility"] == "cutoffs"
    assert anchors["newgen"] == anchors["notifications"] == anchors["database_ids"] == "system"

    def tab_of(marker: str) -> str:
        at = page.index(marker)
        starts = [(page.index(f'id="tab-{t}"'), t) for t in TABS]
        return max((pos, t) for pos, t in starts if pos < at)[1]

    assert tab_of('name="cutoffs.share.a_ca"') == "cutoffs"
    assert tab_of('name="cutoffs.everywhere.b_pa"') == "cutoffs"
    assert tab_of('name="groups.b.candidates"') == "quality"
    assert tab_of('name="picture.aspect"') == "quality"
    assert tab_of('name="style.model"') == "quality"
    assert tab_of('name="my_club.round_eyes"') == "appearance"
    assert tab_of('name="body.tall_from_cm"') == tab_of('name="personality.weight"') == "appearance"
    assert tab_of('name="media.journalists"') == "targets"
    assert tab_of('name="staff.include_human_manager"') == "targets"
    assert tab_of('name="age.b.W04"') == "appearance"
    assert tab_of('name="age.staff.mode"') == "appearance"
    assert tab_of('id="about"') == "system"
    assert tab_of('name="eligibility.continental_competition_ids"') == "system"
    assert tab_of('name="newgen.fallback_min_unique_id"') == "system"
    assert tab_of('name="eligibility.mens_football_only"') == "cutoffs"
    assert tab_of('name="eligibility.loaned_out_at_my_club"') == "cutoffs"
    assert tab_of('name="notifications.new_pictures"') == "system"
    # A tab's form comes back to its tab without the script.
    assert 'name="next" value="/settings/advanced#tab-appearance"' in page


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
        r'<details class="everywhere" id="section-cutoffs-everywhere">\s*'
        r'<summary data-part="everywhere-summary">',
        page,
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


def off(page: str, key: str) -> bool:
    """Whether the setting's control is shown turned off (disabled, so never sent)."""
    found = re.findall(rf'<input type="(?:text|checkbox)"[^>]* name="{re.escape(key)}"[^>]*>', page)
    assert len(found) == 1, found
    return found[0].endswith(" disabled>")


def test_the_league_above_and_below_are_off_when_the_save_gives_none(client: TestClient) -> None:
    """The test save holds no division list, so no league above or below: their cells and Light
    switches are turned off, and the list says so; my league's are not."""
    page = unescape(client.get("/settings/advanced").text)
    for key in ("above_b_ca", "above_c_ca", "below_c_ca"):
        assert off(page, f"cutoffs.share.{key}")
    assert off(page, "cutoffs.light_above") and off(page, "cutoffs.light_below")
    assert re.search(r'name="cutoffs.light_above" value="off" disabled>', page)
    assert not off(page, "cutoffs.share.b_ca") and not off(page, "cutoffs.share.continental_b_ca")
    assert re.search(r'Above:</span> none</li>\s*<li><span class="muted">Below:</span> none', page)


def test_my_league_is_off_when_its_season_conflicts_with_another(config_file: Path) -> None:
    """Just promoted: the old division's season has the most fixtures, but another domestic
    competition my club plays in is a season newer. My league's cells are turned off."""
    new_division = 777
    db_path = load_settings(defaults_path=config_file).paths.db_path
    conn = db.connect(db_path)
    fixtures = {
        MY_LEAGUE_COMP: {c: 46 for c in (MY_CLUB, RIVAL, RIVAL_2)},
        new_division: {MY_CLUB: 3, RIVAL: 3},
    }
    shot = snapshot(
        world(),
        tables=(),
        fixture_counts=fixtures,
        fixture_seasons={MY_LEAGUE_COMP: 2026, new_division: 2027},
    )
    run_sync(conn, shot, load_settings(conn, config_file), save_file="career.fm")
    conn.close()
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    page = unescape(c.get("/settings/advanced").text)
    for key in ("a_ca", "a_pa", "b_ca", "b_pa", "c_ca", "c_pa"):
        assert off(page, f"cutoffs.share.{key}")
    assert 'data-effective="cutoffs.share.b_ca">–' in page
    assert not off(page, "cutoffs.share.continental_b_ca")
    assert re.search(r"My league:</span> none", page) or "No league was found" in page
    c.close()


def test_switching_to_manual_converts_and_redraws_the_page(client: TestClient) -> None:
    data = save(client, **{"cutoffs.mode": "manual"})
    assert data["ok"] and data["reload"]
    # The top 12% are the two highest players. CA 131 already selects them, PA 151 selects four.
    assert "cutoffs.manual.b_ca" not in data["fields"]
    assert data["fields"]["cutoffs.manual.b_pa"]["value"] == 180
    assert data["effective"]["cutoffs.manual.b_ca"] == "≈ top 20%"  # 2 of 10 at 140 or more
    page = client.get("/settings/advanced").text
    assert re.search(r'name="cutoffs.manual.b_ca"[^>]* value="131"', page)
    assert 'name="cutoffs.share.b_ca"' not in page
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
    assert not off(page, "cutoffs.share.above_b_ca") and not off(page, "cutoffs.light_below")
    assert re.search(r"Detailed</span>\s*–", page)
    c.close()


def shown(page: str, key: str) -> str:
    """The text an age table shows for a look ("ages.staff.W03") or its count ("….n")."""
    found = re.search(rf'data-effective="{re.escape(key)}">([^<]*)</span>', page)
    assert found is not None, key
    return found[1]


def test_the_staff_looks_list_ages_and_counts_and_auto_spreads_them(client: TestClient) -> None:
    page = unescape(client.get("/settings/advanced").text)
    # Auto by default (user, 2026-11-22): W02 to W10 over 28-64, staff aged 25, 30, 31, 45, 70.
    assert (shown(page, "ages.staff.W02"), shown(page, "ages.staff.W02.n")) == (
        "28 and under", "1 now",
    )  # fmt: skip
    assert (shown(page, "ages.staff.W03"), shown(page, "ages.staff.W03.n")) == ("29–33", "2 now")
    assert (shown(page, "ages.staff.W10"), shown(page, "ages.staff.W10.n")) == (
        "64 and over", "1 now",
    )  # fmt: skip
    assert shown(page, "ages.staff.W01") == "Not used"
    assert 'name="age.staff.start"' in page and 'name="age.staff.W03"' not in page
    refused = save(client, **{"age.staff.end": "32"})  # 3 years for 7 looks between
    assert not refused["ok"] and "age.staff.end" in refused["errors"]

    # Custom: the spread is copied into the looks, as one change.
    data = save(client, **{"age.staff.mode": "manual"})
    assert data["ok"] and data["reload"]
    assert data["fields"]["age.staff.W03"]["value"] == 29
    assert data["fields"]["age.staff.W10"]["value"] == 64
    assert data["fields"]["age.staff.W02"]["value"] == 0  # the first look: any age up to 28
    assert (data["effective"]["ages.staff.W03"], data["effective"]["ages.staff.W03.n"]) == (
        "29–33", "2 now",
    )  # fmt: skip
    page = unescape(client.get("/settings/advanced").text)
    assert 'name="age.staff.W03"' in page and 'name="age.staff.start"' not in page
    assert shown(page, "ages.staff.W03.n") == "2 now"  # counted in Custom too
    data = save(client, **{"age.staff.W03": "25"})  # a look's first age, set by hand
    assert (data["effective"]["ages.staff.W02"], data["effective"]["ages.staff.W02.n"]) == (
        "24 and under", "0 now",
    )  # fmt: skip
    assert (data["effective"]["ages.staff.W03"], data["effective"]["ages.staff.W03.n"]) == (
        "25–33", "3 now",
    )  # fmt: skip
    client.post("/settings", data={"action": "undo"})
    client.post("/settings", data={"action": "undo"})  # one Undo per change: back to the spread
    assert 'name="age.staff.start"' in client.get("/settings/advanced").text


def test_the_tabs_in_korean(client: TestClient) -> None:
    page = unescape(client.get("/settings/advanced", headers=KOREAN_WINDOWS).text)
    for word in (
        "선정 기준",
        "품질",
        "대상",
        "외형",
        "나이 표현",
        "시스템",
        "내 리그: CA",
        "상위 리그",
    ):
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
