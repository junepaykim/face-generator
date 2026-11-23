"""The dashboard says what the worker is doing and when the queue will be done (Phase 12c)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.core.job_records import Job
from facegen.i18n import Translator
from facegen.persistence import db
from facegen.runtime.contracts import WorkerStatus
from facegen.web.messages import duration, worker_sentence

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
        r' data-caption="Kid 2 · v1" data-player="/players/(\d+)">',
        wall,
    )
    assert card is not None and f'<a class="thumb-name" href="/players/{card[2]}">Kid 2' in wall
    assert "그림 갤러리" in drawn.get("/", headers=KOREAN_WINDOWS).text


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


def drawing(uid: int, name: str, seconds_ago: float) -> WorkerStatus:
    job = Job(
        id=99, unique_id=uid, kind="redraw", group="a", target_club_id=None, target_band=None,
        name=name, at_my_club=True, goalkeeper=False, height_cm=180, strength=12, age=21,
    )  # fmt: skip
    since = (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat(timespec="seconds")
    return WorkerStatus(
        "running", f"{name}: redraw A", job, False, False, False, True, 5, 0, None, (),
        reason="drawing", values={"name": name, "kind": "redraw", "group": "A"}, since=since,
    )  # fmt: skip


def test_the_picture_being_drawn_with_how_far_it_is(env: Env, drawn: TestClient) -> None:
    client = client_for(env.config, env.db_path, StubWorker(drawing(MINE[1], "Mine 1", 10)))
    status = client.get("/status").text
    assert (
        f'Generating <a href="/players/{MINE[1]}">Mine 1</a> · new picture ·'
        ' <span class="grp grp-a" data-tip="'
    ) in status
    # Their current picture beside it, and 10 s of a usual 20 s.
    assert re.search(
        r'<div class="now-pic">\s*<a href="/players/1002"><img src="/pictures/\d+', status
    )
    progress = re.search(
        r'<progress max="100" value="(\d+)" data-since="([^"]+)" data-each="20.0">', status
    )
    assert progress is not None and 45 <= int(progress[1]) <= 60
    assert "About 20 s per picture" in status
    korean = client.get("/status", headers=KOREAN_WINDOWS).text
    assert "한 장에 약 20초 소요" in korean and "생성 중" in korean
    client.close()


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
