"""Live pages (Phase 12d): /events says when a page part changed, instead of polling.

Starlette's test client reads a response to its end, so these streams are made short
(live.STREAM_SECONDS); a real server and an open stream are in runtime/test_runtime.py.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.service import WorkerService
from facegen.web import live
from facegen.web.app import close_streams


@pytest.fixture(autouse=True)
def short_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "STREAM_SECONDS", 0.5)
    monkeypatch.setattr(live, "TICK_SECONDS", 0.05)
    monkeypatch.setattr(live, "PING_SECONDS", 0.2)


@pytest.fixture
def setup(config_file: Path) -> Iterator[tuple[TestClient, WorkerService]]:
    world = [player(1001, name="Mine"), player(NEWGEN + 2, ca=90, pa=152, club_id=RIVAL)]
    db_path = synced(config_file, world)
    worker = fake_worker(config_file, db_path)
    client = client_for(config_file, db_path, worker)
    yield client, worker
    client.close()


def events(body: str) -> list[tuple[str, dict]]:
    found = []
    for block in body.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            found.append((lines["event"], json.loads(lines["data"])))
    return found


def test_each_part_is_sent_once_then_only_pings(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    r = client.get("/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.text.startswith(f"retry: {live.RETRY_MS}\n\n")
    sent = events(r.text)
    # Nothing changed meanwhile: each topic once.
    assert [topic for topic, _ in sent] == ["status", "log", "pictures"]
    assert ": ping" in r.text
    status = sent[0][1]
    assert (status["state"], status["label"], status["review"]) == ("starting", "Starting", 0)
    # The page carries the same fingerprint, so the first event fetches nothing again.
    page = client.get("/").text
    assert f'data-topic="status" data-rev="{status["rev"]}"' in page
    assert f'data-topic="log" data-rev="{sent[1][1]["rev"]}"' in client.get("/log").text


def test_a_change_sends_the_part_again(setup: tuple[TestClient, WorkerService]) -> None:
    client, worker = setup
    later = threading.Timer(0.15, worker.step)  # a picture is drawn while the page is open
    later.start()
    body = client.get("/events").text
    later.join()
    statuses = [data for topic, data in events(body) if topic == "status"]
    logs = [data for topic, data in events(body) if topic == "log"]
    assert len(statuses) >= 2 and statuses[0]["rev"] != statuses[-1]["rev"]
    assert statuses[-1]["state"] == "running"
    assert len(logs) >= 2  # the worker's log grew


def test_the_state_is_named_in_the_pages_language(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    sent = events(client.get("/events", headers=KOREAN_WINDOWS).text)
    assert sent[0][1]["label"] == "시작 중"


def test_stopping_the_server_ends_the_streams(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    close_streams(client.app)  # type: ignore[arg-type]
    assert client.get("/events").text == f"retry: {live.RETRY_MS}\n\n"


def test_the_progress_bar_moves_in_the_page(setup: tuple[TestClient, WorkerService]) -> None:
    """The page has the script that moves it; the server sends no event for it."""
    client, _ = setup
    script = client.get(re.search(r'src="(/static/app.js[^"]*)"', client.get("/").text)[1]).text
    assert 'new EventSource("/events")' in script
    assert "progress[data-since][data-each]" in script
    assert "setInterval(refresh, 3000)" in script  # only without EventSource
