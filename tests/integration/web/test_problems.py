"""Errors on the pages as "what happened, what to do" (Phase 12e): the error's own text is
folded away under the technical details, never the first thing shown."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, post_player_action, synced

from facegen.persistence import db
from facegen.runtime.contracts import WorkerStatus
from facegen.web import queries
from facegen.web.app import AppContext, create_app

MINE = 1001
KID = NEWGEN + 2
CUDA = "RuntimeError: CUDA error: out of memory. Tried to allocate 2.00 GiB"
ATTRIBUTE = "AttributeError: 'NoneType' object has no attribute 'unet'"


def outside_details(html: str) -> str:
    """The page as first seen: everything but the folded details."""
    return re.sub(r"<details.*?</details>", "", html, flags=re.DOTALL)


@dataclass
class Env:
    config: Path
    db_path: Path


@pytest.fixture
def env(config_file: Path) -> Env:
    world = [player(MINE, name="Mine"), player(KID, name="Kid", ca=90, pa=152, club_id=RIVAL)]
    return Env(config_file, synced(config_file, world))


class StubWorker:
    def __init__(self, status: WorkerStatus) -> None:
        self.s = status

    def status(self) -> WorkerStatus:
        return self.s

    def wake(self) -> None:
        pass


def paused_by(reason: str, error: str, **values: object) -> WorkerStatus:
    return WorkerStatus(
        "error", f"Mine: {error}", None, True, False, False, False, 0, 0, f"Mine: {error}", (),
        reason=reason, values={"error": error, **values},
    )  # fmt: skip


@pytest.fixture
def stub(env: Env) -> Iterator[tuple[TestClient, StubWorker]]:
    worker = StubWorker(paused_by("job_error", CUDA, name="Mine"))
    client = client_for(env.config, env.db_path, worker)
    yield client, worker
    client.close()


def test_a_cuda_error_is_what_happened_and_what_to_do(
    stub: tuple[TestClient, StubWorker],
) -> None:
    client, _ = stub
    status = client.get("/status").text
    seen = outside_details(status)
    assert "CUDA" not in seen and "RuntimeError" not in seen
    assert "Paused: Mine could not be generated." in seen
    assert "The graphics card ran out of memory." in seen
    assert "or let Auto-configure use less memory, then press Resume." in seen
    assert '<a href="/hardware">Auto-configure for my GPU</a>' in seen
    assert '<a href="/log">Open the log</a>' in seen
    assert '<form method="post" action="/worker/resume">' in seen  # the button
    assert CUDA in status  # one click away
    korean = outside_details(client.get("/status", headers=KOREAN_WINDOWS).text)
    assert "그래픽카드 메모리가 부족했습니다." in korean and "CUDA" not in korean


def test_an_attribute_error_is_unexpected_and_folded(
    stub: tuple[TestClient, StubWorker],
) -> None:
    client, worker = stub
    worker.s = paused_by("paused_error", ATTRIBUTE)
    status = client.get("/status").text
    seen = outside_details(status)
    assert "AttributeError" not in seen and "NoneType" not in seen
    assert "Paused after an error." in seen and "Something unexpected went wrong." in seen
    assert re.search(
        r'<details class="problem-detail" id="worker-detail">.*AttributeError', status, re.DOTALL
    )


def test_a_failed_sync_says_what_to_do(env: Env) -> None:
    conn = db.connect(env.db_path)
    conn.execute(
        "INSERT INTO sync_runs (started_at, finished_at, save_file, error)"
        " VALUES ('2026-11-17T10:00:00+00:00', '2026-11-17T10:00:01+00:00', '-', ?)",
        ("No save found in /fm/games",),
    )
    conn.close()
    client = client_for(env.config, env.db_path, fake_worker(env.config, env.db_path))
    seen = outside_details(client.get("/status").text)
    assert "The save could not be read" in seen and "There is no save to read yet." in seen
    assert "Save the game in FM27, then press Sync now." in seen
    assert "No save found" not in seen
    client.close()


def test_review_says_why_in_words_and_folds_the_rest(env: Env) -> None:
    worker = fake_worker(env.config, env.db_path)
    worker.step()  # Mine's picture is drawn; Kid's job is queued
    conn = db.connect(env.db_path)
    conn.execute(
        "UPDATE jobs SET status = 'failed', error = ? WHERE unique_id = ? AND status = 'queued'",
        ("failed the quality check 3 times: rating questionable", KID),
    )
    conn.close()
    client = client_for(env.config, env.db_path, worker)
    page = client.get("/review").text
    assert "Failed the quality check three times." in outside_details(page)
    assert "rating questionable" not in outside_details(page) and "rating questionable" in page
    assert (
        "품질 검사를 세 번 통과하지 못했습니다."
        in client.get("/review", headers=KOREAN_WINDOWS).text
    )
    client.close()


def test_a_refused_request_is_worded_in_the_pages_language(env: Env) -> None:
    client = client_for(env.config, env.db_path, fake_worker(env.config, env.db_path))
    r = post_player_action(
        client, f"/players/{KID}/new-picture", headers=KOREAN_WINDOWS
    )  # no picture yet
    page = client.get(r.headers["location"], headers=KOREAN_WINDOWS).text
    assert "Kid: 다시 그릴 그림이 없습니다. ‘그리기’를 쓰세요" in page
    r = post_player_action(client, f"/players/{KID}/new-picture")
    assert "Kid: no picture to draw again; use Draw" in client.get(r.headers["location"]).text
    bad = client.post(
        "/settings/save",
        data={"cutoffs.everywhere.a_ca": "300"},
        headers={"Accept": "application/json", **KOREAN_WINDOWS},
    ).json()
    assert bad["errors"]["cutoffs.everywhere.a_ca"] == "1~200 사이여야 합니다"
    client.close()


def test_a_missing_page_and_an_unexpected_error_are_pages(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = client_for(env.config, env.db_path, fake_worker(env.config, env.db_path))
    html = {"Accept": "text/html", **KOREAN_WINDOWS}
    missing = client.get("/players/123456", headers=html)
    assert missing.status_code == 404 and "찾을 수 없습니다" in missing.text
    assert '<a class="button" href="/">대시보드로</a>' in missing.text
    assert client.get("/players/123456").json() == {"detail": "No player 123456 in this career."}
    client.close()

    def broken(*args: object, **kwargs: object) -> None:
        raise AttributeError("'NoneType' object has no attribute 'unet'")

    monkeypatch.setattr(queries, "counts", broken)
    app = create_app(AppContext(env.db_path, env.config, fake_worker(env.config, env.db_path)))
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as raw:
        page = raw.get("/", headers={"Accept": "text/html"})
        assert page.status_code == 500 and "Something went wrong" in page.text
        assert "AttributeError" not in outside_details(page.text) and "AttributeError" in page.text
        plain = raw.get("/", headers={"Accept": "*/*"})
        assert plain.status_code == 500 and plain.text == "Internal Server Error"
