"""The Players page's layout (Phase 17g): the main filters with the rest folded under "More
filters", Clear, the bulk bar's two groups of actions, the unique ID under each name, the
outline for no picture yet and the numbers' columns. Who is listed and why (Phase 17h): the
picture targets by default, and the reasons as chips in the page's language."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import pytest
from factories import ABROAD, NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker, post_player_action, synced

from kitshot.runtime.worker_service import WorkerService
from kitshot.web.queries import PlayerPageFilter

MINE = 1001
STAR = 1002
KID = NEWGEN + 2
WORLD = [
    player(MINE, name="Mine Player", age=29),
    player(STAR, name="Famous Striker", ca=185, pa=190, club_id=ABROAD),
    player(KID, name="Rival Kid", ca=90, pa=152, club_id=RIVAL),
]
MAIN = ("q", "club", "group", "picture", "photo", "sort")
FOLDED = ("shown", "league", "ca_min", "ca_max", "pa_min", "pa_max", "origin", "role", "scope")


@dataclass
class Env:
    client: TestClient
    worker: WorkerService
    db_path: Path


@pytest.fixture
def env(config_file: Path) -> Iterator[Env]:
    db_path = synced(config_file, WORLD)
    worker = fake_worker(config_file, db_path)
    client = client_for(config_file, db_path, worker)
    yield Env(client, worker, db_path)
    client.close()


def filter_form(page: str) -> str:
    start = page.index('<form class="filters players-filters"')
    return page[start : page.index("</form>", start)]


def bulk_bar(page: str) -> str:
    start = page.index('<div class="bulk sticky bulk-players" data-bulk>')
    return page[start : page.index('<div class="table-wrap"', start)]


def names(html: str) -> list[str]:
    """The fields' names in the order they come."""
    return re.findall(r'<(?:input|select) [^>]*name="(\w+)"', html)


def test_the_main_filters_come_first_and_the_rest_fold_away(env: Env) -> None:
    form = filter_form(env.client.get("/players").text)
    main, fold = form.split('<details class="filters-more">', 1)
    assert names(main) == list(MAIN)
    assert names(fold) == list(FOLDED)
    # Folded while none of them is set; the button carries no count.
    assert '<summary class="button">More filters</summary>' in fold
    assert fold.index("</details>") < fold.index('<button class="primary">Filter</button>')
    # A main filter set leaves it folded.
    assert '<details class="filters-more">' in env.client.get("/players?group=a&q=x").text


@pytest.mark.parametrize(
    ("query", "count"),
    [
        ("league=mine", 1),
        ("ca_min=100", 1),
        ("ca_min=100&ca_max=150", 1),  # a range counts once
        ("pa_max=150&role=staff", 2),
        ("league=continental&ca_max=120&pa_min=140&origin=newgen&role=media&scope=all", 6),
        ("league=mine&group=a&sort=name&page=2", 1),  # main filters, sort and page do not count
        ("shown=everyone", 1),
        ("shown=targets&scope=all", 1),  # the picture targets are the default
    ],
)
def test_more_filters_opens_while_one_is_set_and_counts_them(
    env: Env, query: str, count: int
) -> None:
    form = filter_form(env.client.get(f"/players?{query}").text)
    assert '<details class="filters-more" open>' in form
    assert (
        f'<summary class="button">More filters <span class="badge badge-count">{count}</span>'
        "</summary>" in form
    )
    assert PlayerPageFilter.from_query(dict(p.split("=") for p in query.split("&"))).folded == count


def test_folded_filters_keep_their_values(env: Env) -> None:
    form = filter_form(env.client.get("/players?ca_min=100&scope=all&league=mine").text)
    fold = form.split('<details class="filters-more" open>', 1)[1]
    assert re.search(r'name="ca_min" min="0" max="200" value="100"', fold)
    assert re.search(r'<option value="all" selected>All players</option>', fold)
    assert re.search(r'<option value="mine" selected>My league</option>', fold)


def test_clear_is_a_button_while_a_filter_is_on_and_keeps_the_sort(env: Env) -> None:
    for query in ("", "?sort=name", "?page=2"):
        assert "filter-clear" not in env.client.get(f"/players{query}").text, query
    cleared = {
        "group=a": "/players",
        "q=striker&sort=pa": "/players?sort=pa",
        "ca_min=100&league=mine&sort=name": "/players?sort=name",
        f"club_id={ABROAD}": "/players",
    }
    for query, href in cleared.items():
        form = filter_form(env.client.get(f"/players?{query}").text)
        clear = f'<a class="button filter-clear" href="{href}">Clear</a>'
        assert form.index('<button class="primary">Filter</button>') < form.index(clear), query
    assert PlayerPageFilter(q="x", league="mine", ca_min=3, sort="age", page=4).cleared() == (
        "sort=age"
    )
    korean = env.client.get("/players?group=a", headers=KOREAN_WINDOWS).text
    assert '<a class="button filter-clear" href="/players">초기화</a>' in korean
    assert "상세 필터" in korean


def test_the_bulk_bar_has_two_groups_shown_once_a_player_is_ticked(env: Env) -> None:
    bar = unescape(bulk_bar(env.client.get("/players").text))
    # The count alone until a tick: the actions and Clear selection are hidden (app.js).
    assert '<span class="ticked" data-ticked>0 ticked</span>' in bar
    assert (
        '<button type="button" class="link" data-clear-ticks hidden>Clear selection</button>' in bar
    )
    acts = bar[bar.index('<div class="bulk-acts" data-when-ticked hidden>') :]
    draw, pin = acts.split('<span class="sep" aria-hidden="true"></span>', 1)
    assert re.search(r"<label>Draw in\s*<select name=\"group\">", draw)
    assert re.search(r'<option value="b" selected>Standard</option>', draw)
    assert re.findall(
        r'<button[^>]*name="action" value="([\w-]+)"[^>]*>([^<]+)</button>', draw
    ) == [
        ("generate", "Draw"),
        ("new-picture", "Redraw"),
    ]
    assert 'title="Pins them to this group. Anyone without a picture, or with one from a' in draw
    assert "title=\"A new picture of the same character, in each one's own group." in draw
    assert re.search(r"<label>Pin as\s*<select name=\"pin\">", pin)
    assert re.findall(r"<option value=\"(\w*)\">([^<]+)</option>", pin) == [
        ("never", "Never draw"),
        ("", "No pin"),
    ]
    assert re.findall(r'<button[^>]*name="action" value="([\w-]+)"[^>]*>([^<]+)</button>', pin) == [
        ("pin", "Pin"),
    ]
    # Without the script the actions show.
    assert (
        "<noscript><style>.bulk-acts[data-when-ticked][hidden] { display: flex !important; }" in bar
    )
    assert '<div class="table-wrap" data-fit>' in env.client.get("/players").text
    korean = unescape(bulk_bar(env.client.get("/players", headers=KOREAN_WINDOWS).text))
    for words in (
        ">선택 해제<",
        "<label>그룹",
        ">그리기<",
        ">다시 그리기<",
        "<label>고정",
        ">적용<",
    ):
        assert words in korean, words


def test_draw_and_redraw_say_what_they_did(env: Env) -> None:
    env.worker.step()  # Mine Player (my club, group A) gets a picture
    r = post_player_action(
        env.client,
        "/players/action",
        data={"action": "new-picture", "uid": [str(MINE), str(STAR)], "next": "/players"},
    )
    text = unescape(env.client.get(r.headers["location"]).text)
    assert "Redraw: 1 queued, 1 skipped (no picture to draw again; use Draw)." in text
    r = post_player_action(
        env.client,
        "/players/action",
        data={"action": "generate", "group": "c", "uid": [str(STAR)], "next": "/players"},
    )
    assert "Draw in the Light group: 1 queued." in unescape(
        env.client.get(r.headers["location"]).text
    )
    r = post_player_action(
        env.client,
        "/players/action",
        data={"action": "new-picture", "uid": [str(KID)], "next": "/players"},
        headers=KOREAN_WINDOWS,
    )
    korean = unescape(env.client.get(r.headers["location"], headers=KOREAN_WINDOWS).text)
    assert "다시 그리기: " in korean and "‘그리기’를 쓰세요" in korean


def rows(page: str) -> dict[str, str]:
    """Each player's row by name."""
    found = re.findall(r'<tr>\s*<td><input type="checkbox" name="uid".*?</tr>', page, re.DOTALL)
    return {re.search(r'<a href="/players/\d+">([^<]+)</a>', r)[1]: r for r in found}


def test_each_name_has_its_unique_id_to_copy(env: Env) -> None:
    page = env.client.get("/players?shown=everyone").text
    for uid, row in ((MINE, rows(page)["Mine Player"]), (STAR, rows(page)["Famous Striker"])):
        assert re.search(
            rf'<td class="col-name"><a href="/players/{uid}">[^<]+</a>.*?'
            rf'<button type="button" class="uid-copy" data-copy="{uid}" title="Copy unique ID">'
            rf"{uid}</button></td>",
            row,
            re.DOTALL,
        )
    assert texts(page)["js.copied"] == "Copied"
    korean = env.client.get("/players", headers=KOREAN_WINDOWS).text
    assert 'title="고유 ID 복사"' in korean and texts(korean)["js.copied"] == "복사됨"


def texts(page: str) -> dict[str, str]:
    """The catalog texts handed to the page's script."""
    found = re.search(r'<script type="application/json" id="texts">(.*?)</script>', page, re.DOTALL)
    assert found is not None
    return json.loads(found[1])


def test_no_picture_yet_shows_an_outline(env: Env) -> None:
    page = env.client.get("/players").text
    outline = (
        '<span class="face none" role="img" title="No picture yet" aria-label="No picture yet">'
        '<svg viewBox="0 0 40 40" aria-hidden="true" focusable="false">'
    )
    assert all(outline in row for row in rows(page).values())
    env.worker.step()
    row = rows(env.client.get("/players").text)["Mine Player"]
    assert outline not in row and '<img class="face"' in row
    assert 'aria-label="아직 그림 없음"' in env.client.get("/players", headers=KOREAN_WINDOWS).text


def test_age_ca_and_pa_line_up_on_the_right(env: Env) -> None:
    page = env.client.get("/players?shown=everyone").text
    head = page[page.index("<thead>") : page.index("</thead>")]
    assert re.findall(r'<th class="num">([^<]+)</th>', head) == ["Age", "CA", "PA"]
    row = rows(page)["Famous Striker"]
    numbers = re.findall(r'<td class="num">([^<]+)</td>', row)
    assert numbers[0].isdigit() and numbers[1:] == ["185", "190"]


def test_the_player_page_says_draw_and_redraw_too(env: Env) -> None:
    """Draw and Redraw on a player's page as on the list's bar (Phase 17g, at the user's
    request); the Gallery's cards say Redraw too (test_gallery.py)."""
    page = unescape(env.client.get(f"/players/{STAR}").text)  # no picture yet
    assert re.search(
        rf'<form method="post" action="/players/{STAR}/generate" class="inline">.*?'
        r'<label>Draw in <select name="group">.*?<button class="primary">Draw</button>',
        page,
        re.DOTALL,
    )
    assert f'action="/players/{STAR}/new-picture"' not in page
    env.worker.step()  # Mine Player gets a picture
    page = unescape(env.client.get(f"/players/{MINE}").text)
    assert f'action="/players/{MINE}/generate"' not in page
    assert re.search(
        rf'<form method="post" action="/players/{MINE}/new-picture">.*?'
        r'<button title="A new picture of the same character[^"]*">Redraw</button></form>',
        page,
        re.DOTALL,
    )
    korean = unescape(env.client.get(f"/players/{MINE}", headers=KOREAN_WINDOWS).text)
    assert ">다시 그리기</button></form>" in korean
    korean = unescape(env.client.get(f"/players/{STAR}", headers=KOREAN_WINDOWS).text)
    assert "<label>그룹 <select" in korean and '<button class="primary">그리기</button>' in korean
    # The messages name the buttons.
    r = post_player_action(env.client, f"/players/{MINE}/new-picture")
    assert "Redraw queued." in unescape(env.client.get(r.headers["location"]).text)
    r = post_player_action(env.client, f"/players/{STAR}/new-face")
    text = unescape(env.client.get(r.headers["location"]).text)
    assert "Famous Striker: in no group; use Draw and choose one" in text
    r = post_player_action(env.client, f"/players/{STAR}/new-face", headers=KOREAN_WINDOWS)
    korean = unescape(env.client.get(r.headers["location"], headers=KOREAN_WINDOWS).text)
    assert "그룹이 없습니다. ‘그리기’로 그룹을 골라 주세요" in korean


# --- who is listed and why (Phase 17h) -------------------------------------------------------


def listed(page: str) -> set[str]:
    return set(rows(page))


def test_the_list_shows_the_picture_targets_and_everyone_is_one_choice_away(env: Env) -> None:
    """D6: by default the people this app draws; a pre-existing star abroad with no group
    (Famous Striker) only under "Everyone in the save", which a search offers with its count."""
    page = env.client.get("/players").text
    assert listed(page) == {"Mine Player", "Rival Kid"} and "2 players" in page
    shown = re.search(r'<select name="shown"[^>]*>(.*?)</select>', filter_form(page))
    assert shown is not None
    assert re.findall(r'<option value="(\w+)" ?(selected)?>([^<]+)</option>', shown[1]) == [
        ("targets", "selected", "Picture targets"),
        ("everyone", "", "Everyone in the save"),
    ]
    assert (
        'title="Picture targets: in a group, with a picture, queued, in review or pinned"' in page
    )
    assert "Showing picture targets only." not in page  # no search: nothing hidden to offer
    everyone = env.client.get("/players?shown=everyone").text
    assert listed(everyone) == {"Mine Player", "Famous Striker", "Rival Kid"}
    assert "3 players" in everyone and "Showing picture targets only." not in everyone

    # A search that finds someone who is not a target offers everyone, with the count.
    searched = env.client.get("/players?q=famous").text
    assert listed(searched) == set() and "No players match." in searched
    offer = (
        '<p class="muted small">Showing picture targets only. <a href="/players?q=famous'
        '&amp;shown=everyone">Show everyone in the save (1)</a></p>'
    )
    assert offer in searched
    assert "Show everyone" not in env.client.get("/players?q=mine").text  # nothing more there
    club = env.client.get(f"/players?club_id={ABROAD}").text
    assert "Show everyone in the save (1)" in club
    korean = env.client.get("/players?q=famous", headers=KOREAN_WINDOWS).text
    assert "그림 대상만 보고 있습니다." in korean and "세이브 전체 보기(1명)" in korean
    korean = env.client.get("/players", headers=KOREAN_WINDOWS).text
    assert '<option value="targets" selected>그림 대상</option>' in korean
    assert '<option value="everyone" >세이브 전체</option>' in korean


def why_cell(row: str) -> str:
    found = re.search(r'<td class="why">(.*?)</td>', row, re.DOTALL)
    assert found is not None
    return found[1]


def test_why_is_a_row_of_chips_with_the_sentence_as_tooltip(env: Env) -> None:
    page = rows(env.client.get("/players?shown=everyone").text)
    assert why_cell(page["Mine Player"]) == (
        '<span class="why-chip" title="In my club&#39;s first team">First team</span>'
    )
    assert why_cell(page["Rival Kid"]) == (
        '<span class="why-chip" title="Newgen, in my league: PA 152 (151+)">'
        "My league · PA 151+</span>"
    )
    assert why_cell(page["Famous Striker"]) == ""  # no rule picks them
    # Why they are kept up to date goes with the auto-update tag.
    assert (
        '<span class="tag" title="Drawn again when they change club or look older, so the picture'
        ' stays up to date.">auto-update</span>' in page["Mine Player"]
    )
    korean = rows(env.client.get("/players", headers=KOREAN_WINDOWS).text)
    assert (
        why_cell(korean["Mine Player"]) == '<span class="why-chip" title="내 구단 1군">1군</span>'
    )
    assert why_cell(korean["Rival Kid"]) == (
        '<span class="why-chip" title="생성 선수, 내 리그: PA 152 (151+)">내 리그 · PA 151+</span>'
    )
    detail = env.client.get(f"/players/{KID}", headers=KOREAN_WINDOWS).text
    assert "생성 선수, 내 리그: PA 152 (151+)" in detail


def test_a_row_not_classified_since_the_update_shows_its_stored_text(env: Env) -> None:
    """Rows written before the reasons were stored as codes show the English text until the
    next classification (a sync or a settings change) fills them in."""
    with sqlite3.connect(env.db_path) as conn:
        conn.execute("UPDATE players SET why = NULL WHERE unique_id = ?", (KID,))
    stored = "newgen in my league, PA 152 (151+)"
    korean = rows(env.client.get("/players", headers=KOREAN_WINDOWS).text)
    assert why_cell(korean["Rival Kid"]) == stored
    assert stored in env.client.get(f"/players/{KID}", headers=KOREAN_WINDOWS).text
    with sqlite3.connect(env.db_path) as conn:
        conn.execute('UPDATE players SET why = \'[{"code": "from a later version"}]\'')
    page = rows(env.client.get("/players").text)
    assert why_cell(page["Mine Player"]) == "in my club&#39;s first team"
