"""The dashboard says what the worker is doing and when the queue will be done (Phase 12c)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from html import unescape
from pathlib import Path
from unittest.mock import Mock

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, synced

from kitshot.core.job_records import Job
from kitshot.i18n import Translator
from kitshot.persistence import db
from kitshot.runtime.contracts import SyncControl, SyncStatus, WorkerStatus
from kitshot.web.app import AppContext, create_app
from kitshot.web.messages import clock, duration, worker_sentence

MINE = [1001, 1002, 1003]  # my club: group A
KIDS = [NEWGEN + 2 * i for i in range(1, 7)]  # rivals in my league, CA 90, PA 152: group B


def world() -> list:
    return [player(uid, name=f"Mine {i}", age=20 + i) for i, uid in enumerate(MINE)] + [
        player(uid, name=f"Kid {i}", ca=90, pa=152, club_id=RIVAL) for i, uid in enumerate(KIDS)
    ]


@dataclass
class Env:
    config: Path
    db_path: Path

    def times(self, group: str, seconds: float) -> None:
        """Every picture of a group took this long to draw."""
        conn = db.connect(self.db_path)
        conn.execute(
            "UPDATE images SET render = json_set(render, '$.seconds', ?) WHERE group_name = ?",
            (seconds, group),
        )
        conn.close()


@pytest.fixture
def env(config_file: Path) -> Env:
    return Env(config_file, synced(config_file, world()))


@pytest.fixture
def drawn(env: Env) -> Iterator[TestClient]:
    """The three at my club and three kids drawn; three kids still queued."""
    worker = fake_worker(env.config, env.db_path)
    for _ in range(6):
        assert worker.step() == 0.0
    env.times("a", 20.0)
    env.times("b", 7.0)
    client = client_for(env.config, env.db_path, worker)
    yield client
    client.close()


def test_the_pictures_just_drawn_show_newest_first(drawn: TestClient) -> None:
    page = drawn.get("/").text
    assert "Latest pictures" in page
    wall = page[page.index("Latest pictures") :]
    names = re.findall(r'<a class="thumb-name" href="/players/\d+">([^<]+)</a>', wall)
    assert names[:3] == ["Kid 2", "Kid 1", "Kid 0"]  # my club was drawn first
    assert sorted(names[3:]) == ["Mine 0", "Mine 1", "Mine 2"]
    first = re.search(r'<img src="(/pictures/\d+\?size=192&amp;v=v\d+\.png)"', wall)
    assert first is not None and drawn.get(first[1].replace("&amp;", "&")).status_code == 200
    # The picture opens in the lightbox (Phase 17b); the name opens the player's page.
    card = re.search(
        r'<a class="thumb-pic" href="/pictures/(\d+)" data-lightbox="recent"'
        r' data-caption="Kid 2 · v1" data-player="/players/(\d+)" data-made="[^"]+">',
        wall,
    )
    assert card is not None and f'<a class="thumb-name" href="/players/{card[2]}">Kid 2' in wall
    assert "갤러리에서 모두 보기 →" in drawn.get("/", headers=KOREAN_WINDOWS).text


def test_the_queue_is_how_many_are_left_and_when_it_is_done(drawn: TestClient) -> None:
    status = drawn.get("/status").text
    assert '<strong class="big">3 pictures left</strong>' in status
    assert "about 21 s" in status  # 3 × 7 s
    # One bar of what is left per group, and the list of who folded away.
    assert '<span class="qseg qseg-b" style="flex-grow: 3"></span>' in status
    assert re.search(
        r'<span><span class="grp grp-b" data-tip="[^"]+">Standard</span> 3</span>', status
    )
    assert '<details id="queue-list">' in status and "Kid 3" in status
    korean = drawn.get("/status", headers=KOREAN_WINDOWS).text
    assert "남은 그림 3장" in korean and "약 21초 예상" in korean


def test_groups_not_drawn_automatically_wait_for_generate_now(env: Env, drawn: TestClient) -> None:
    r = drawn.post("/settings/save", data={"schedule.auto_groups": "a"})
    assert r.json()["ok"]
    status = drawn.get("/status").text
    assert "3 of them wait for Generate now" in status
    assert 'class="qseg qseg-b held"' in status


def test_waiting_for_fm27_is_said_in_plain_words(env: Env) -> None:
    worker = fake_worker(env.config, env.db_path, fm_check=lambda: True)
    worker.step()
    client = client_for(env.config, env.db_path, worker)
    english = client.get("/status").text
    wait = "Waiting for FM27 to close, so the game keeps the graphics card. 9 pictures wait."
    assert wait in english
    korean = client.get("/status", headers=KOREAN_WINDOWS).text
    assert (
        "FM27이 실행 중이라 그래픽카드를 게임에 양보하고 있습니다. 그림 9장이 대기 중입니다."
        in korean
    )
    assert "the time shows after a few pictures" in english
    client.close()


class StubWorker:
    """A worker caught in the middle of a picture."""

    def __init__(self, status: WorkerStatus) -> None:
        self.s = status

    def status(self) -> WorkerStatus:
        return self.s

    def wake(self) -> None:
        pass


def drawing(
    uid: int, name: str, seconds_ago: float, kind: str = "redraw", group: str = "a"
) -> WorkerStatus:
    job = Job(
        id=99, unique_id=uid, kind=kind, group=group, target_club_id=None, target_band=None,
        name=name, at_my_club=True, goalkeeper=False, height_cm=180, strength=12, age=21,
    )  # fmt: skip
    since = (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat(timespec="seconds")
    return WorkerStatus(
        "running", f"{name}: {kind} {group.upper()}", job, False, False, False, True, 5, 0, None,
        (), reason="drawing", values={"name": name, "kind": kind, "group": group.upper()},
        since=since,
    )  # fmt: skip


def test_the_picture_being_drawn_with_how_far_it_is(env: Env, drawn: TestClient) -> None:
    worker = StubWorker(drawing(MINE[1], "Mine 1", 10))
    client = client_for(env.config, env.db_path, worker)
    status = client.get("/status").text
    assert (
        f'Generating <a href="/players/{MINE[1]}">Mine 1</a> · new picture ·'
        ' <span class="grp grp-a" data-tip="'
    ) in status
    # Their current picture beside it, labelled and shimmering while its replacement is drawn
    # (Phase 17e).
    assert re.search(
        r'<div class="now-pic">\s*(?:\{#.*?#\}\s*)?<a class="now-current shimmer"'
        r' href="/players/1002"><img src="/pictures/\d+\?size=192" alt="[^"]+">'
        r'<span class="badge now-label">Current</span></a>',
        status,
    )
    # No percentage, as no end is known: a shimmer, and 10 s of a usual 20 s, ticking on in the
    # page from when it started.
    assert "<progress" not in status
    assert '<span class="shimmer shimmer-bar" aria-hidden="true"></span>' in status
    timing = re.search(
        r'<span class="muted small now-clock" data-since="([^"]+)">'
        r"<span data-elapsed>(\d+) s</span> / avg 20 s</span>",
        status,
    )
    assert timing is not None and timing[1] == worker.s.since and 10 <= int(timing[2]) <= 12
    korean = client.get("/status", headers=KOREAN_WINDOWS).text
    assert re.search(r"<span data-elapsed>\d+초</span> / 평균 20초", korean)
    assert "생성 중" in korean and '<span class="badge now-label">현재 그림</span>' in korean
    client.close()


def test_someone_drawn_for_the_first_time_shimmers_in_the_placeholder(
    env: Env, drawn: TestClient
) -> None:
    client = client_for(
        env.config, env.db_path, StubWorker(drawing(KIDS[4], "Kid 4", 3, "initial", "b"))
    )
    status = client.get("/status").text
    assert '<div class="now-pic-none shimmer" aria-hidden="true">' in status
    assert "now-current" not in status and "busy" not in status
    assert f'Generating <a href="/players/{KIDS[4]}">Kid 4</a> · first picture ·' in status
    client.close()
    idle = drawn.get("/status").text  # nothing is drawn: a still placeholder
    assert '<div class="now-pic-none" aria-hidden="true">' in idle


def test_an_upgrade_names_its_group_once(env: Env, drawn: TestClient) -> None:
    """User, 2026-11-22: "upgrade to Detailed · [Detailed]" said it twice (Phase 17e)."""
    client = client_for(
        env.config, env.db_path, StubWorker(drawing(MINE[0], "Mine 0", 3, "upgrade"))
    )
    status = client.get("/status").text
    assert (
        f'Upgrading <a href="/players/{MINE[0]}">Mine 0</a> to <span class="grp grp-a"'
    ) in status
    assert "upgrade to" not in status.split('id="queue"')[0]
    korean = client.get("/status", headers=KOREAN_WINDOWS).text
    assert re.search(
        rf'<a href="/players/{MINE[0]}">Mine 0</a> <span class="grp grp-a"[^>]*>정밀</span>로 승격 중',
        korean,
    )
    client.close()
    # The same sentence where the worker's reason is worded without a link.
    upgrade = drawing(MINE[0], "Mine 0", 3, "upgrade")
    assert worker_sentence(upgrade, Translator("en")) == "Upgrading Mine 0 to Detailed"
    assert worker_sentence(upgrade, Translator("ko")) == "Mine 0 정밀로 승격 중"
    first = drawing(KIDS[4], "Kid 4", 3, "initial", "b")
    assert worker_sentence(first, Translator("en")) == "Generating Kid 4 · first picture · Standard"
    # A kind the catalog does not word keeps the general sentence.
    odd = drawing(KIDS[4], "Kid 4", 3, "sketch", "b")
    assert worker_sentence(odd, Translator("en")) == "Generating Kid 4 · kind.sketch · Standard"


def generate_now(page: str) -> str | None:
    """Generate now's button tip, or None when the page has no such button."""
    found = re.search(
        r'<form method="post" action="/worker/generate-now">.*?<button title="([^"]*)">'
        r"(?:Generate now|생성 시작)</button>",
        page,
    )
    return found[1] if found else None


def test_generate_now_shows_only_when_it_would_do_something(env: Env, drawn: TestClient) -> None:
    """User, 2026-11-22: Generate now sat beside Pause while the queue was being drawn."""
    client = client_for(
        env.config, env.db_path, StubWorker(drawing(KIDS[3], "Kid 3", 3, "initial", "b"))
    )
    # Drawing, and every group is drawn automatically: it would change nothing.
    assert generate_now(client.get("/status").text) is None
    # Pictures held back in groups not drawn automatically: it draws them too.
    assert client.post("/settings/save", data={"schedule.auto_groups": "a"}).json()["ok"]
    status = client.get("/status").text
    assert generate_now(status) == "Also generate the 3 pictures in groups not drawn automatically"
    korean = client.get("/status", headers=KOREAN_WINDOWS).text
    assert generate_now(korean) == "자동으로 생성하지 않는 그룹의 그림 3장도 생성합니다"
    client.close()


def test_generate_now_while_waiting_for_fm27(env: Env) -> None:
    worker = fake_worker(env.config, env.db_path, fm_check=lambda: True)
    worker.step()
    client = client_for(env.config, env.db_path, worker)
    assert generate_now(client.get("/status").text) == "Generate now, even while FM27 is running"
    assert client.post("/settings/save", data={"schedule.auto_groups": "a"}).json()["ok"]
    worker.step()
    assert generate_now(client.get("/status").text) == (
        "Generate the whole queue now, every group, even while FM27 is running"
    )
    # Pressed: it shows no more while it does its work.
    assert worker.generate_now()
    assert generate_now(client.get("/status").text) is None
    client.close()


def test_paused_with_nothing_held_offers_resume_only(env: Env) -> None:
    worker = fake_worker(env.config, env.db_path)
    worker.pause()
    worker.step()
    client = client_for(env.config, env.db_path, worker)
    status = client.get("/status").text
    assert 'action="/worker/resume"' in status and generate_now(status) is None
    client.close()


def test_when_the_queue_will_be_done_is_left_to_the_page_clock(env: Env, drawn: TestClient) -> None:
    """The time of day is the PC's own: the page works it out (Phase 17e)."""
    client = client_for(
        env.config, env.db_path, StubWorker(drawing(KIDS[3], "Kid 3", 3, "initial", "b"))
    )
    status = client.get("/status").text
    assert '<span class="muted">about 21 s<span data-done-in="21"></span></span>' in status
    client.close()
    # Not drawing (the queue waits for FM27 to close): no time of day is promised.
    waiting = fake_worker(env.config, env.db_path, fm_check=lambda: True)
    waiting.step()
    client = client_for(env.config, env.db_path, waiting)
    status = client.get("/status").text
    assert "about 21 s" in status and "data-done-in" not in status
    client.close()


@pytest.mark.parametrize(
    ("seconds", "en", "ko"),
    [
        (0.4, "0 s", "0초"),
        (19.9, "19 s", "19초"),
        (65, "1 min 5 s", "1분 5초"),
        (3725, "1 h 2 min", "1시간 2분"),
    ],
)
def test_the_clock_counts_whole_seconds_gone(seconds: float, en: str, ko: str) -> None:
    assert clock(seconds, Translator("en")) == en
    assert clock(seconds, Translator("ko")) == ko


def synced_client(env: Env, status: SyncStatus) -> TestClient:
    syncer = Mock(spec=SyncControl)
    syncer.status.return_value = status
    return TestClient(create_app(AppContext(env.db_path, env.config, syncer=syncer)), base_url=BASE)


def facts(page: str) -> str:
    return page[page.index('id="facts"') :]


def chips(page: str) -> list[str]:
    return re.findall(r'<span class="badge">([^<]+)</span>', facts(page))


def test_the_last_sync_says_what_it_did_in_chips(env: Env) -> None:
    """User, 2026-11-22: the save and the in-game date were said three times, the last in an
    English sentence even in Korean (Phase 17e)."""
    numbers = {"queued": 1552, "new_at_my_club": 1, "new_staff": 8, "updated": 0, "cancelled": 0}
    status = SyncStatus(
        message="Synced career.fm (in-game 2026-03-01): 1552 queued, 0 updated, 0 cancelled.",
        numbers=numbers,
        warnings=["a club could not be read", "the managed club was not found"],
    )
    with synced_client(env, status) as client:
        page = client.get("/").text
        assert chips(page) == ["1,552 queued", "1 new at my club", "8 new staff"]
        assert "Synced career.fm" not in page
        # The table keeps the save and when; the in-game date is in the sidebar.
        assert "<th>Save</th><td>career.fm</td>" in facts(page) and "In-game" not in facts(page)
        # The warnings: a badge that opens their list.
        assert (
            '<details class="problem-detail sync-warnings" id="sync-warnings"><summary>'
            '<span class="badge badge-warn">2 warnings</span></summary>'
        ) in page
        korean = client.get("/", headers=KOREAN_WINDOWS).text
        assert chips(korean) == ["대기열 1,552장 추가", "내 구단 새 선수 1명", "새 스태프 8명"]
        assert '<span class="badge badge-warn">경고 2개</span>' in korean
    nothing = SyncStatus(message="Synced career.fm", numbers=dict.fromkeys(numbers, 0))
    with synced_client(env, nothing) as client:
        assert chips(client.get("/").text) == ["Queue unchanged"]
        assert chips(client.get("/", headers=KOREAN_WINDOWS).text) == ["대기열 변화 없음"]
    recovered = SyncStatus(
        trigger="publish recovery", message="Recovered the active career's FM27 pack."
    )
    with synced_client(env, recovered) as client:
        korean = client.get("/", headers=KOREAN_WINDOWS).text
        assert "중단됐던 FM27 그림 적용을 마쳤습니다." in korean and "Recovered" not in korean


def test_every_players_card_row_links_to_who_it_counts(drawn: TestClient) -> None:
    """User, 2026-11-22: rows in groups with a line between them, every one a link."""
    page = drawn.get("/status").text
    card = page[page.index("<h2>Players</h2>") :]
    card = card[: card.index("</table>")]
    groups = re.findall(r"<tbody>(.*?)</tbody>", card, re.DOTALL)
    assert len(groups) == 3
    links = [re.findall(r'<td><a href="([^"]+)">([\d,]+)</a>', g) for g in groups]
    assert [[href for href, _ in group] for group in links] == [
        ["/players?group=a", "/players?group=b", "/players?group=c"],
        ["/players?group=tracked", "/players?group=pinned"],
        ["/players?picture=yes", "/review"],  # "No photo" only once FM27's photos are read
    ]
    rows = re.findall(r"<tr>.*?</tr>", card, re.DOTALL)
    assert all('<a href="' in row for row in rows)
    # Auto-update lists exactly those it counts.
    tracked = dict(links[1])["/players?group=tracked"]
    assert int(tracked) > 0
    assert f"{tracked} players" in drawn.get("/players?group=tracked").text
    korean = drawn.get("/players?group=tracked", headers=KOREAN_WINDOWS).text
    assert '<option value="tracked" selected>연동</option>' in korean


def test_the_log_has_a_page_of_its_own(drawn: TestClient) -> None:
    assert "Loading the image model" not in drawn.get("/status").text
    log = drawn.get("/log")
    assert log.status_code == 200 and "Loading the image model" in log.text
    assert 'data-refresh="/log/part"' in log.text
    part = drawn.get("/log/part").text
    assert "<html" not in part and "Loading the image model" in part
    assert '<a href="/log" class="on"' in log.text


def test_a_restart_and_a_missing_value_are_worded_or_fall_back() -> None:
    restarting = WorkerStatus(
        "restarting", "the graphics driver reported an error; Kid 0 went to review",
        None, False, False, False, False, 0, 0, None, (), reason="restarting",
        values={"cause": "gpu", "name": "Kid 0", "review": True, "error": "x", "seconds": 12},
    )  # fmt: skip
    assert worker_sentence(restarting, Translator("en")) == (
        "The graphics driver reported an error; Kid 0 went to review. Starting again in 12 s."
    )
    assert worker_sentence(restarting, Translator("ko")) == (
        "그래픽 드라이버에서 오류가 났습니다. Kid 0은(는) 검토로 보냈습니다. 12초 뒤 다시 시작합니다."
    )
    unknown = WorkerStatus(
        "restarting", "the English line", None, False, False, False, False, 0, 0, None, (),
        reason="restarting", values={"cause": "gpu"},
    )  # fmt: skip
    assert worker_sentence(unknown, Translator("ko")) == "the English line"


@pytest.mark.parametrize(
    ("seconds", "en", "ko"),
    [
        (0.2, "1 s", "1초"),
        (45, "45 s", "45초"),
        (61, "2 min", "2분"),
        (3600, "1 h", "1시간"),
        (3700, "1 h 2 min", "1시간 2분"),
    ],
)
def test_durations(seconds: float, en: str, ko: str) -> None:
    assert duration(seconds, Translator("en")) == en
    assert duration(seconds, Translator("ko")) == ko


def test_groups_and_terms_say_what_they_mean_on_hover(drawn: TestClient) -> None:
    """User, 2026-11-17: hovering 정밀, 표준, 간편, 연동 or 고정 says what it is."""
    korean = drawn.get("/status", headers=KOREAN_WINDOWS).text
    tips = dict(re.findall(r'data-tip="([^"]+)">([^<]+)<', korean))
    named = {name: tip for tip, name in tips.items()}
    assert named["정밀"].startswith("가장 공들여 그립니다.")
    assert named["표준"].startswith("표준 품질로 한 장씩 그립니다.")
    assert named["간편"].startswith("작은 크기로 가볍게 그립니다.")
    assert named["연동"].startswith("구단을 옮기거나 나이 표현이 바뀌면")
    assert named["고정"].startswith("규칙 대신 직접 정한 그룹")
    english = drawn.get("/status").text
    assert (
        'data-tip="Set by hand to a group, or to never draw, instead of following the rules.">Pinned<'
        in english
    )
    script = drawn.get(re.search(r'src="(/static/app.js[^"]*)"', drawn.get("/").text)[1]).text
    # ⓘ tips share the floating box, kept inside the window (user, 2026-11-22).
    assert 'closest("[data-tip], .info")' in script and 'tip.className = "hover-tip"' in script


# -- Skip current (Phase 17m, D8) --------------------------------------------------------------


class SkippingWorker(StubWorker):
    """A worker caught in the middle of a picture, noting the jobs it is asked to skip."""

    def __init__(self, status: WorkerStatus) -> None:
        super().__init__(status)
        self.skipped: list[int] = []

    def skip(self, job_id: int) -> bool:
        self.skipped.append(job_id)
        return True


def skip_button(page: str) -> tuple[str, str] | None:
    """Skip's job and tip, or None when the page has no such button."""
    found = re.search(
        r'<form method="post" action="/worker/skip"><input type="hidden" name="csrf_token"'
        r' value="[^"]+"><input type="hidden" name="job" value="(\d+)"><button title="([^"]*)">'
        r"(?:Skip|건너뛰기)</button>",
        page,
    )
    return (found[1], found[2]) if found else None


def test_skip_shows_while_drawing_with_another_picture_to_draw(env: Env, drawn: TestClient) -> None:
    worker = SkippingWorker(drawing(KIDS[3], "Kid 3", 3, "initial", "b"))
    client = client_for(env.config, env.db_path, worker)
    assert skip_button(client.get("/status").text) == (
        "99",
        "Stop generating Kid 3 now; it is generated again after the rest of the queue",
    )
    korean = client.get("/status", headers=KOREAN_WINDOWS).text
    assert skip_button(korean) == (
        "99",
        "Kid 3 생성을 지금 멈추고, 대기열의 나머지를 먼저 생성한 뒤 다시 생성합니다",
    )
    # The rest wait for Generate now: the skipped picture would come straight back.
    assert client.post("/settings/save", data={"schedule.auto_groups": "a"}).json()["ok"]
    assert skip_button(client.get("/status").text) is None
    worker.s = replace(worker.s, generate_now=True)
    assert skip_button(client.get("/status").text) is not None
    # Not drawing: nothing to skip.
    worker.s = replace(worker.s, state="idle", current=None, reason="held")
    assert skip_button(client.get("/status").text) is None
    client.close()


def test_skip_asks_for_the_job_the_page_showed(env: Env, drawn: TestClient) -> None:
    worker = SkippingWorker(drawing(KIDS[3], "Kid 3", 3, "initial", "b"))
    client = client_for(env.config, env.db_path, worker)
    r = client.post("/worker/skip", data={"job": "99"})
    assert r.status_code == 303 and r.headers["location"].startswith("/?msg=Skipped%20Kid%203")
    assert worker.skipped == [99]
    # Another picture is drawn by now, or none: the page's job is not skipped.
    for job in ("98", "", "x"):
        r = client.post("/worker/skip", data={"job": job})
        assert "err=Not%20skipped" in r.headers["location"], job
    assert worker.skipped == [99]
    page = client.get(r.headers["location"]).text
    assert "Not skipped: that picture is already finished." in page
    r = client.post("/worker/skip", data={"job": "99"}, headers={"X-CSRF-Token": "wrong"})
    assert r.status_code == 403 and worker.skipped == [99]
    client.close()


def test_a_skipped_job_is_tagged_in_the_queue_list(env: Env, drawn: TestClient) -> None:
    conn = db.connect(env.db_path)
    conn.execute("UPDATE jobs SET skipped = 1 WHERE unique_id = ?", (KIDS[3],))
    conn.close()
    status = unescape(drawn.get("/status").text)
    rows = re.findall(r"<tr class=\"\">.*?</tr>", status, re.DOTALL)
    assert "Kid 3" in rows[-1] and '<span class="tag">skipped</span>' in rows[-1]
    assert all("skipped" not in row for row in rows[:-1])
    assert '<span class="tag">건너뜀</span>' in drawn.get("/status", headers=KOREAN_WINDOWS).text
