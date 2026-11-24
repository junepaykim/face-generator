"""The Advanced page's frame and Quality tab (Phase 17j): one grid for every row, compact
controls that keep custom values, Recent changes in a drawer, a quieter bar, the Quality tab's
order with the picture's shape in Picture style, the default with ↺, and the card's memory."""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import unescape
from pathlib import Path

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.core.settings_values import CANDIDATE_RANGE, PICTURE_HEIGHTS
from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.setup.hardware import Hardware
from facegen.web.app import AppContext, create_app

GIB = 2**30


@pytest.fixture
def client(config_file: Path) -> Iterator[TestClient]:
    world = [player(1001, name="Mine"), player(NEWGEN + 2, ca=90, pa=152, club_id=RIVAL)]
    db_path = synced(config_file, world)
    c = client_for(config_file, db_path, fake_worker(config_file, db_path))
    yield c
    c.close()


def save(client: TestClient, **values: str) -> dict:
    r = client.post("/settings/save", data=values)
    assert r.status_code == 200, r.text
    return r.json()


def stored(config_file: Path) -> dict:
    settings = load_settings(defaults_path=config_file)
    conn = db.connect(settings.paths.db_path)
    try:
        return dict(load_settings(conn, config_file).values)
    finally:
        conn.close()


def panel(page: str, tab: str) -> str:
    return re.search(rf'<section class="tab-panel" id="tab-{tab}".*?</section>', page, re.DOTALL)[0]


def fields_of(page: str) -> list[str]:
    """Each settings row (a `.field`), as its HTML."""
    return re.findall(r'<(label|div) class="field [^"]*" data-field>(.*?)</\1>', page, re.DOTALL)


def test_every_row_is_label_control_and_default_on_one_grid(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    rows = fields_of(page)
    # Every card's settings; the age looks are cells of their tables since Phase 17l.
    assert len(rows) > 30
    for _, row in rows:
        name = row.index('<span class="name">')
        control = row.index('<span class="control">')
        note = row.index('<span class="field-note">')
        assert name < control < note, row[:200]
    # Checkboxes sit in the control column too.
    assert re.search(
        r'<span class="control">\s*<input type="hidden" name="render.vae_tiling" value="off">',
        page,
    )


def test_the_quality_tab_reads_style_quality_rendering_reference(client: TestClient) -> None:
    quality = panel(client.get("/settings/advanced").text, "quality")
    order = [
        'id="graphics-card"',
        'id="section-style"',
        'id="section-groups"',
        'id="section-render"',
        'id="section-reference"',
    ]
    places = [quality.index(marker) for marker in order]
    assert places == sorted(places)
    # The picture's shape is one more row of Picture style, not a card of its own.
    style = re.search(r'id="section-style".*?</fieldset>', quality, re.DOTALL)[0]
    assert 'name="style.model"' in style and 'name="picture.aspect"' in style
    assert 'id="section-picture"' not in quality
    assert unescape(quality).count("Picture shape") == 1
    # How people look moved to the Appearance tab (Phase 17l, D10).
    for gone in ("section-my_club", "section-body", "section-personality"):
        assert f'id="{gone}"' not in quality
    anchors = unescape(
        re.search(r'data-anchors="([^"]+)"', client.get("/settings/advanced").text)[1]
    )
    assert '"picture": "quality"' in anchors  # an old #section-picture link opens Quality


def test_the_quality_table_has_a_height_list_a_stepper_and_short_names(
    client: TestClient,
) -> None:
    quality = unescape(panel(client.get("/settings/advanced").text, "quality"))
    heights = re.search(
        r'<select name="groups.b.published_long_edge".*?</select>', quality, re.DOTALL
    )
    assert re.findall(r'<option value="(\d+)"', heights[0]) == [str(h) for h in PICTURE_HEIGHTS]
    assert '<option value="768" selected>768</option>' in heights[0]
    low, high = CANDIDATE_RANGE
    assert re.search(
        rf'<span class="stepper">\s*<button type="button" data-step-of="groups.a.candidates" '
        rf'data-step="-1"[^>]* hidden>−</button>\s*<input type="number" name="groups.a.candidates" '
        rf'data-key data-default="3" value="3" step="1" min="{low}" max="{high}"',
        quality,
    )
    assert 'data-step-of="groups.a.candidates" data-step="1"' in quality
    detail = re.search(r'<select name="groups.a.detail_size".*?</select>', quality, re.DOTALL)[0]
    assert re.findall(r">([^<]+)</option>", detail) == ["x1.5 (best)", "Smaller (low VRAM)", "Off"]
    korean = unescape(client.get("/settings/advanced", headers=KOREAN_WINDOWS).text)
    assert ">1.5배(최고 품질)</option>" in korean and ">작게(VRAM 절약)</option>" in korean

    data = save(client, **{"groups.b.candidates": "2", "groups.c.published_long_edge": "1024"})
    assert data["ok"] and data["fields"]["groups.b.candidates"]["text"] == "2"
    assert "Candidates per picture 1 → 2" in data["message"]
    assert "Picture height (px) 512 → 1024" in data["message"]
    too_many = save(client, **{"groups.b.candidates": str(high + 1)})
    assert not too_many["ok"] and "groups.b.candidates" in too_many["errors"]


def test_a_stored_value_outside_the_list_stays_selectable(client: TestClient) -> None:
    data = save(client, **{"groups.a.published_long_edge": "900"})
    assert data["ok"] and data["fields"]["groups.a.published_long_edge"]["text"] == "900"
    page = client.get("/settings/advanced").text
    heights = re.search(r'<select name="groups.a.published_long_edge".*?</select>', page, re.DOTALL)
    assert '<option value="900" selected>900 (custom)</option>' in heights[0]
    assert '<option value="1024" >1024</option>' in heights[0]
    korean = client.get("/settings/advanced", headers=KOREAN_WINDOWS).text
    assert '<option value="900" selected>900 (직접 정한 값)</option>' in korean
    # Picking a listed height again leaves the custom one behind.
    assert save(client, **{"groups.a.published_long_edge": "1024"})["ok"]
    assert "(custom)" not in client.get("/settings/advanced").text


def test_skin_lightening_is_a_row_of_buttons(client: TestClient) -> None:
    page = unescape(client.get("/settings/advanced").text)
    row = re.search(
        r'<div class="field[^"]*" data-field>\s*<span class="name">Draw tan.*?</div>',
        page,
        re.DOTALL,
    )
    assert row is not None  # a div: a label would pass a click on its name to the first button
    assert "Draw tan and dark skin lighter <" in row[0]  # the range left the label
    assert re.findall(r'<input type="radio" name="my_club.soften_skin" value="(\d)"', row[0]) == [
        "0",
        "1",
        "2",
    ]
    assert re.findall(r"<span>([^<]+)</span></label>", row[0]) == ["0 (Off)", "1", "2"]
    assert 'value="2" data-key data-default="2" checked>' in row[0]
    assert 'role="radiogroup" aria-label="Draw tan and dark skin lighter"' in row[0]
    data = save(client, **{"my_club.soften_skin": "0"})
    assert data["ok"] and "Draw tan and dark skin lighter 2 → 0 (Off)" in data["message"]
    korean = unescape(client.get("/settings/advanced", headers=KOREAN_WINDOWS).text)
    assert "<span>0 (끔)</span>" in korean and "탄 피부와 어두운 피부를 밝게 <" in korean


def test_the_default_is_quiet_and_a_reset_button_puts_it_back(client: TestClient) -> None:
    page = client.get("/settings/advanced").text
    shape = re.search(
        r'<label class="field[^"]*" data-field>\s*<span class="name">Picture shape.*?</label>',
        page,
        re.DOTALL,
    )[0]
    assert '<span class="muted small">Default: 1:1 square</span>' in shape
    assert re.search(
        r'<button type="button" class="reset-one" data-use-default title="Use default" '
        r'aria-label="Use default" hidden>↺</button>',
        shape,
    )
    assert "use default</button>" not in page.lower()
    save(client, **{"picture.aspect": "3:4"})
    page = client.get("/settings/advanced").text
    shape = re.search(
        r'<label class="field[^"]*" data-field>\s*<span class="name">Picture shape.*?</label>',
        page,
        re.DOTALL,
    )[0]
    assert 'class="field changed' in shape
    assert 'aria-label="Use default" >↺</button>' in shape
    korean = client.get("/settings/advanced", headers=KOREAN_WINDOWS).text
    assert "기본값: 1:1 정사각형" in korean


def test_the_legend_shows_only_while_a_setting_differs(
    client: TestClient, config_file: Path
) -> None:
    db_path = load_settings(defaults_path=config_file).paths.db_path
    plain = client_for(config_file, db_path)  # the fixture's save sets Manual cutoffs
    plain.post("/settings", data={"action": "defaults"})
    page = plain.get("/settings/advanced").text
    assert re.search(r'<span class="changed-legend" data-changed-legend hidden>', page)
    save(plain, **{"render.vae_tiling": "on"})
    page = plain.get("/settings/advanced").text
    assert re.search(
        r'<span class="changed-legend" data-changed-legend >Changed from the default:', page
    )
    plain.close()


def test_recent_changes_open_from_the_bar_and_stay_below_without_script(
    client: TestClient, config_file: Path
) -> None:
    page = client.get("/settings/advanced").text
    count = len(re.findall(r'<input type="hidden" name="through"', page))
    count += len(re.findall(r'<li class="undone">', page))
    bar = re.search(r'<div class="settings-bar">.*?<div class="tab-panels">', page, re.DOTALL)[0]
    # The tabs, Undo, Redo, History and Reset in the bar's one row; the status under them.
    row = re.search(
        r'<div class="settings-bar-row">.*?<div class="settings-bar-note">', bar, re.DOTALL
    )[0]
    for marker in (
        "data-tab-pills",
        "data-undo",
        "data-redo",
        "data-history-open",
        'value="defaults"',
    ):
        assert marker in row, marker
    assert "data-save-status" not in row
    assert re.search(
        rf'<a class="button" href="#history" data-history-open aria-controls="history-drawer" '
        rf'aria-expanded="false">History \({count}\)</a>',
        bar,
    )
    # The drawer is after every tab; the list in it is what each save sends again.
    drawer = page.index('<aside class="card history-drawer" id="history-drawer"')
    assert drawer > page.index('id="tab-system"')
    assert '<h2 id="history-title">Recent changes</h2>' in page
    assert re.search(r'data-history-close aria-label="Close" title="Close" hidden>×</button>', page)
    data = save(client, **{"render.vae_tiling": "on"})
    assert data["history_count"] == count + 1
    assert "Recent changes" not in data["history_html"]  # the drawer keeps its title
    assert "render.vae_tiling" not in data["history_html"] and "VAE tiling" in data["history_html"]
    page = client.get("/settings/advanced").text
    assert f"History ({count + 1})</a>" in page
    korean = client.get("/settings/advanced", headers=KOREAN_WINDOWS).text
    assert f"변경 기록 ({count + 1})</a>" in korean
    # Undo back to an entry still works from the list.
    assert stored(config_file)["render.vae_tiling"] is True
    entry = re.search(r'name="through" value="(\d+)"', page)[1]
    undone = client.post("/settings", data={"action": "undo", "through": entry})
    assert undone.status_code == 303
    assert stored(config_file)["render.vae_tiling"] is False


def test_the_graphics_card_shows_its_memory(config_file: Path) -> None:
    db_path = synced(config_file, [player(1001, name="Mine")])
    card = Hardware("NVIDIA GeForce RTX 4080", int(15.99 * GIB))
    ctx = AppContext(db_path, config_file, None, hardware=lambda: card)
    with TestClient(create_app(ctx), base_url=BASE) as client:
        client.get("/hardware")  # asks the card and waits for the answer
        page = unescape(client.get("/settings/advanced").text)
    shown = "NVIDIA GeForce RTX 4080 · 16 GB"
    graphics = re.search(r'id="graphics-card".*?</div>\s*</div>', page, re.DOTALL)[0]
    assert shown in graphics
    assert f"Graphics card</th><td>{shown}</td>" in page  # About this run
    assert card.memory_gb == 16 and Hardware("Small", int(7.6 * GIB)).memory_gb == 8
    assert Hardware(reason="none").memory_gb is None


def test_both_notes_say_the_same_applies_sentence(client: TestClient) -> None:
    looks = unescape(panel(client.get("/settings/advanced").text, "appearance"))
    my_club = re.search(r'id="section-my_club".*?</fieldset>', looks, re.DOTALL)[0]
    assert '<p class="muted small">Applies to pictures drawn from now on.</p>' in my_club
    body = re.search(r'id="section-body".*?</fieldset>', looks, re.DOTALL)[0]
    assert "Applies to new characters." in body  # another thing: who is sampled anew
    korean = unescape(client.get("/settings/advanced", headers=KOREAN_WINDOWS).text)
    assert "<legend>내 구단 선수 외모</legend>" in korean
    assert korean.count("앞으로 그리는 그림부터 적용됩니다.") >= 1
