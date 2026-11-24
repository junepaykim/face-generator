"""The Advanced page's Cutoffs and System tabs (Phase 17k): who counts on Cutoffs, FM27's folders
across the page with no Change button and Open folder on Windows only, the notification with
Running, the group ticks on one line, a hint under drawing while FM27 runs, and the database IDs
folded at the bottom of System."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS, fake_worker, synced

from kitshot.i18n import Translator
from kitshot.persistence.db import RENAMED_SETTINGS
from kitshot.persistence.settings import load_defaults, load_settings
from kitshot.web.app import AppContext, create_app
from kitshot.web.forms import KEY_SECTIONS, describe_key, settings_sections, settings_tabs

MOVED = (
    "eligibility.mens_football_only",
    "eligibility.loaned_out_at_my_club",
    "eligibility.continental_competition_ids",
    "newgen.fallback_min_unique_id",
    "notifications.new_pictures",
)


@dataclass
class Opener:
    opened: list[Path] = field(default_factory=list)

    def __call__(self, path: Path) -> None:
        self.opened.append(path)


def make_client(config_file: Path, **kw: Any) -> TestClient:
    world = [player(1001, name="Mine"), player(NEWGEN + 2, ca=90, pa=152, club_id=RIVAL)]
    db_path = synced(config_file, world)
    ctx = AppContext(db_path, config_file, fake_worker(config_file, db_path), **kw)
    c = TestClient(create_app(ctx), base_url=BASE, follow_redirects=False)
    token = re.search(r'name="csrf_token" value="([^"]+)"', c.get("/settings").text)
    assert token is not None
    c.headers["X-CSRF-Token"] = token[1]
    return c


@pytest.fixture
def client(config_file: Path) -> Iterator[TestClient]:
    c = make_client(config_file)
    yield c
    c.close()


def panel(page: str, tab: str) -> str:
    return re.search(rf'<section class="tab-panel" id="tab-{tab}".*?</section>', page, re.DOTALL)[0]


def settings_form(part: str) -> str:
    return re.search(
        r'<form method="post" action="/settings" class="settings".*?</form>', part, re.DOTALL
    )[0]


def row(page: str, key: str) -> str:
    """The settings row (a `.field`) holding the key's control."""
    rows = re.findall(r'<(label|div) class="field [^"]*" data-field>(.*?)</\1>', page, re.DOTALL)
    found = [html for _, html in rows if f'name="{key}"' in html]
    assert len(found) == 1, key
    return found[0]


# --- 17k-1: who counts on Cutoffs, by where each setting is shown -------------------------------


def test_moved_settings_keep_their_keys_and_are_shown_where_they_belong() -> None:
    defaults = load_defaults()
    sections = settings_sections(dict(defaults), defaults)
    where = {f.key: section for section, _, _, fs in sections for f in fs}
    assert where["eligibility.mens_football_only"] == "eligibility"
    assert where["eligibility.loaned_out_at_my_club"] == "cutoffs.my_club"
    assert where["notifications.new_pictures"] == "schedule"
    assert where["eligibility.continental_competition_ids"] == "database_ids"
    assert where["newgen.fallback_min_unique_id"] == "database_ids"
    tabs = settings_tabs(sections)
    in_tab = {f.key: tab for tab, secs in tabs.items() for _, _, _, fs in secs for f in fs}
    assert in_tab["eligibility.mens_football_only"] == "cutoffs"
    assert in_tab["eligibility.loaned_out_at_my_club"] == "cutoffs"
    assert in_tab["notifications.new_pictures"] == "system"
    assert in_tab["newgen.fallback_min_unique_id"] == "system"
    # Only where they are shown: the keys, and so stored values, did not change.
    assert set(KEY_SECTIONS) <= set(defaults) and not set(MOVED) & set(RENAMED_SETTINGS)
    # The database IDs are the last section of System, in the defaults file's order.
    system = [section for section, *_ in tabs["system"]]
    assert system[-1] == "database_ids" and "notifications" not in system
    ids = [f.key for section, _, _, fs in tabs["system"] if section == "database_ids" for f in fs]
    assert ids == ["eligibility.continental_competition_ids", "newgen.fallback_min_unique_id"]
    # A history line still names the setting by its own section.
    assert describe_key("eligibility.loaned_out_at_my_club", Translator("en")) == (
        "Who counts · Count players I loaned out as my club's"
    )


def test_who_counts_opens_cutoffs_and_loaned_out_sits_by_its_group(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    cutoffs = settings_form(panel(page, "cutoffs"))
    system = panel(page, "system")
    for key in MOVED:
        assert (
            page.count(f'type="checkbox" name="{key}"')
            + page.count(f'type="text" name="{key}"')
            + page.count(f'type="number" name="{key}"')
            == 1
        ), key
    # Who counts is the Cutoffs form's first card, with its note; nothing of it stays in System.
    first = re.search(r"<fieldset id=\"([^\"]+)\">\s*<legend>([^<]*)", cutoffs)
    assert first is not None and first.groups() == ("section-eligibility", "Who counts")
    assert "Applies from the next sync." in cutoffs.split("</fieldset>")[0]
    assert 'name="eligibility.mens_football_only"' in cutoffs.split("</fieldset>")[0]
    assert "eligibility." not in system.split('id="section-database_ids"')[0]
    # My club: loaned-out players right after "Players I loaned out".
    my_club = re.search(
        r'<fieldset id="section-cutoffs-my_club">.*?</fieldset>', cutoffs, re.DOTALL
    )[0]
    names = re.findall(r'name="((?:cutoffs\.my_club|eligibility)\.\w+)"', my_club)
    assert list(dict.fromkeys(names)) == [
        "cutoffs.my_club.first_team",
        "cutoffs.my_club.youth",
        "cutoffs.my_club.loans",
        "eligibility.loaned_out_at_my_club",
        "cutoffs.my_club.academy_alumni",
    ]


def test_a_moved_setting_saves_from_its_new_tab(client: TestClient) -> None:
    r = client.post("/settings/save", data={"eligibility.loaned_out_at_my_club": ""})
    data = r.json()
    assert data["ok"] and data["changed"]
    assert "Who counts · Count players I loaned out as my club's on → off" in data["message"]
    assert "Applies from the next sync." in data["message"]
    # Without the script, the Cutoffs tab's Save sends it with that tab's fields only.
    r = client.post(
        "/settings",
        data={
            "action": "apply-some",
            "next": "/settings/advanced#tab-cutoffs",
            "eligibility.loaned_out_at_my_club": "on",
            "eligibility.mens_football_only": "on",
        },
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/settings/advanced?msg=")
    page = client.get("/settings/advanced").text
    assert 'name="eligibility.loaned_out_at_my_club" data-key data-default="on" checked' in page


# --- 17k-2, 17k-3: FM27's folders -----------------------------------------------------------


def test_the_folders_card_spans_the_page_with_no_change_button(
    client: TestClient, config_file: Path
) -> None:
    page = client.get("/settings/advanced").text
    system = panel(page, "system")
    # Across the page, above About this run; no longer beside it in a row of cards.
    assert 'class="cards"' not in system
    assert system.index('id="folders"') < system.index('id="about"') < system.index("data-autosave")
    card = re.search(r'<div class="card" id="folders" data-notes>.*?</form>', system, re.DOTALL)[0]
    assert "Change" not in card and "<button>Change</button>" not in page
    # A typed folder applies on Enter or when the box is left; without the script, a button.
    form = re.search(r'<form method="post" action="/setup/folder"[^>]*>', card)[0]
    assert "data-folder-apply" in form
    assert "<noscript><button>Use this folder</button></noscript>" in card
    # The path whole, where it came from beside it, never split.
    fm = load_settings(defaults_path=config_file).paths.fm_user_dir
    assert f'<span class="path">{fm}</span> <span class="muted small origin">from ' in card


def test_a_folder_not_set_and_where_one_came_from_in_korean(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8")
    install = re.search(r"fm_install_dir = .*", text)
    assert install is not None
    config_file.write_text(text.replace(install[0], 'fm_install_dir = ""'), encoding="utf-8")
    with make_client(config_file) as c:
        page = c.get("/settings/advanced").text
        assert (
            '<th>Install</th><td><span class="muted small origin">'
            "not set, and not found on this PC</span>" in page
        )
        korean = c.get("/settings/advanced", headers=KOREAN_WINDOWS).text
        assert re.search(
            r'<th>세이브와 그래픽</th><td><span class="path">[^<]+</span> '
            r'<span class="muted small origin">\S+에서 지정</span>',
            korean,
        )
        assert "지정되지 않았고 이 PC에서도 찾지 못함" in korean and "바꾸기" not in korean


def test_open_folder_only_where_explorer_opens(config_file: Path) -> None:
    with make_client(config_file) as c:  # WSL's `kitshot serve`, Linux: no Explorer
        for url in ("/settings/advanced", "/settings/basic"):
            assert "/folders/open" not in c.get(url).text, url
    opener = Opener()
    with make_client(config_file, open_folder=opener) as c:
        page = c.get("/settings/advanced").text
        card = re.search(r'<div class="card" id="folders" data-notes>.*?\n</div>', page, re.DOTALL)[
            0
        ]
        forms = re.findall(
            r'<form method="post" action="/folders/open" data-open-folder>(.*?)</form>', card
        )
        assert [re.search(r'name="name" value="(\w+)"', f)[1] for f in forms] == ["fm", "pack"]
        assert all('name="csrf_token"' in f for f in forms)
        assert "<button>Open FM27's folder</button>" in forms[0]
        assert "<button>Open pack folder</button>" in forms[1]
        assert 'data-note role="status"' in card
        assert "/folders/open" in c.get("/settings/basic").text
        korean = c.get("/settings/advanced", headers=KOREAN_WINDOWS).text
        assert (
            "<button>FM27 폴더 열기</button>" in korean
            and "<button>그림 팩 폴더 열기</button>" in korean
        )
        fm = load_settings(defaults_path=config_file).paths.fm_user_dir
        assert fm is not None
        fm.mkdir(parents=True, exist_ok=True)
        r = c.post("/folders/open", data={"name": "fm"}, headers={"Accept": "application/json"})
        assert r.json() == {"ok": True, "error": None} and opener.opened == [fm]


# --- 17k-4 to 17k-7: Running, the hint, the group ticks, the database IDs ------------------------


def test_running_holds_the_notification_and_the_hint(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    form = settings_form(panel(page, "system"))
    running = re.search(r'<fieldset id="section-schedule">.*?</fieldset>', form, re.DOTALL)[0]
    names = re.findall(r'name="((?:schedule|notifications)\.\w+)"', running)
    assert list(dict.fromkeys(names)) == [
        "schedule.poll_seconds",
        "schedule.generate_while_fm_running",
        "schedule.auto_groups",
        "notifications.new_pictures",
    ]
    assert 'id="section-notifications"' not in page and "<legend>Notifications" not in page
    hint = '<span class="field-hint muted small">Off: drawing pauses while FM27 is open.</span>'
    assert hint in row(page, "schedule.generate_while_fm_running")
    assert page.count("field-hint") == 1
    korean = client.get("/settings/advanced", headers=KOREAN_WINDOWS).text
    assert "끄면 FM27이 켜져 있는 동안 그리기를 멈춥니다." in korean


def test_the_group_ticks_take_one_line_with_their_default(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    ticks = re.search(
        r'<div class="field[^"]*" data-field>(?:(?!</div>).)*?schedule\.auto_groups.*?</div>',
        page,
        re.DOTALL,
    )[0]
    assert re.match(r'<div class="field +wide"', ticks)
    control = re.search(r'<span class="control">(.*)', ticks, re.DOTALL)[1]
    # The default and ↺ are inside the control's span, after the three ticks.
    assert control.index('value="c"') < control.index('class="field-note"')
    assert 'class="field-note"' not in ticks.split('<span class="control">')[0]


def test_the_database_ids_are_folded_at_the_bottom_of_system(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    form = settings_form(panel(page, "system"))
    fold = re.search(
        r'<details class="fold-card" id="section-database_ids">\s*<summary>([^<]*)</summary>(.*?)</details>',
        form,
        re.DOTALL,
    )
    assert fold is not None and fold[1] == "FM27 database IDs"  # closed by default
    after = form[form.index("</details>") :]
    assert "data-field" not in after  # the last card of the tab
    ids = re.findall(r'name="((?:eligibility|newgen)\.\w+)"', fold[2])
    assert list(dict.fromkeys(ids)) == [
        "eligibility.continental_competition_ids",
        "newgen.fallback_min_unique_id",
    ]
    box = re.search(
        r'<input type="text" name="eligibility.continental_competition_ids"[^>]*>', fold[2]
    )[0]
    assert 'placeholder="e.g. 1301394, 1301396"' in box
    tips = unescape(fold[2])
    assert "Used only for a career's first sync" in tips and "Leave empty to find them" in tips
    korean = client.get("/settings/advanced", headers=KOREAN_WINDOWS).text
    assert "<summary>FM27 데이터베이스 ID</summary>" in korean
    assert 'placeholder="예: 1301394, 1301396"' in korean


def test_a_wrong_database_id_opens_the_fold(client: TestClient) -> None:
    page = client.post(
        "/settings",
        data={"action": "apply", "newgen.fallback_min_unique_id": "soon"},
    ).text
    assert '<details class="fold-card" id="section-database_ids" open>' in page
    assert "needs a number" in row(page, "newgen.fallback_min_unique_id")


def test_a_folder_applied_from_system_comes_back_to_system(
    client: TestClient, config_file: Path
) -> None:
    page = client.get("/settings/advanced").text
    card = re.search(r'<div class="card" id="folders" data-notes>.*?</form>', page, re.DOTALL)[0]
    assert '<input type="hidden" name="next" value="/settings/advanced#tab-system">' in card
    basic = client.get("/settings/basic").text
    assert '<input type="hidden" name="next" value="/settings/basic">' in basic
    back = "/settings/advanced#tab-system"
    wrong = config_file.parent / "nowhere"
    r = client.post("/setup/folder", data={"folder": str(wrong), "next": back})
    location = r.headers["location"]
    assert r.status_code == 303 and location.startswith("/settings/advanced?err=Not%20FM27")
    assert location.endswith("#tab-system") and "&folder=" in location
    fm = load_settings(defaults_path=config_file).paths.fm_user_dir
    assert fm is not None
    (fm / "games").mkdir(parents=True, exist_ok=True)
    r = client.post("/setup/folder", data={"folder": str(fm), "next": back})
    assert r.headers["location"].startswith("/settings/advanced?msg=")
    assert r.headers["location"].endswith("#tab-system")
