"""Live pages (Phase 12d): /events says when a page part changed, instead of polling.

Starlette's test client reads a response to its end, so these streams are made short
(live.STREAM_SECONDS); a real server and an open stream are in runtime/test_runtime.py.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from itertools import chain, repeat
from pathlib import Path
from unittest.mock import Mock

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, synced

from facegen.runtime.contracts import SyncControl, SyncStatus
from facegen.service import WorkerService
from facegen.web import live
from facegen.web.app import AppContext, close_streams, create_app


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


def test_the_status_event_carries_the_sidebar(setup: tuple[TestClient, WorkerService]) -> None:
    """Phase 17d: pictures left, the card's words and its buttons' state on every page."""
    client, worker = setup
    later = threading.Timer(0.15, worker.step)  # a picture is drawn while the page is open
    later.start()
    body = client.get("/events").text
    later.join()
    statuses = [data for topic, data in events(body) if topic == "status"]
    first, last = statuses[0], statuses[-1]
    left = first["queued"]
    assert left > 0 and first["short"] == f"Starting · {left:,} left"
    assert (first["paused"], first["fault"], first["sync_warn"], first["review"]) == (
        False, False, False, 0,
    )  # fmt: skip
    assert last["state"] == "running" and last["queued"] == left - 1
    assert last["short"] == (f"Generating · {left - 1:,} left" if left > 1 else "Generating")


def test_a_sync_that_went_wrong_reaches_the_sidebar(config_file: Path) -> None:
    db_path = synced(config_file, [player(1001)])
    syncer = Mock(spec=SyncControl)
    warned = SyncStatus(message="Sync finished.", warnings=["a club could not be read"])
    syncer.status.side_effect = chain([SyncStatus(message="Sync finished.")], repeat(warned))
    app = create_app(AppContext(db_path, config_file, syncer=syncer))
    with TestClient(app, base_url=BASE) as client:
        statuses = [data for topic, data in events(client.get("/events").text) if topic == "status"]
    assert [data["sync_warn"] for data in statuses] == [False, True]
    assert "short" not in statuses[0]  # no worker here: the card has no state line


def test_sync_is_read_once_per_render_and_again_on_each_live_tick(config_file: Path) -> None:
    db_path = synced(config_file, [player(1001)])
    running = SyncStatus(running=True, started="12:34:56")
    finished = SyncStatus(message="Sync finished.")
    syncer = Mock(spec=SyncControl)
    # The initial render and first event see the same state. The next event sees it finish.
    syncer.status.side_effect = chain([running, running], repeat(finished))
    app = create_app(AppContext(db_path, config_file, syncer=syncer))
    with TestClient(app, base_url=BASE) as client:
        page = client.get("/").text
        syncer.status.assert_called_once_with()
        assert "12:34:56" in page

        revisions = [
            data["rev"] for topic, data in events(client.get("/events").text) if topic == "status"
        ]
        assert len(revisions) == 2 and revisions[0] != revisions[1]
        assert f'data-topic="status" data-rev="{revisions[0]}"' in page

        syncer.status.reset_mock()
        assert "Sync finished." in client.get("/status").text
        syncer.status.assert_called_once_with()


def test_the_state_is_named_in_the_pages_language(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    sent = events(client.get("/events", headers=KOREAN_WINDOWS).text)
    assert sent[0][1]["label"] == "시작 중"
    assert sent[0][1]["short"] == f"시작 중 · {sent[0][1]['queued']:,}장 남음"


def test_stopping_the_server_ends_the_streams(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    close_streams(client.app)  # type: ignore[arg-type]
    assert client.get("/events").text == f"retry: {live.RETRY_MS}\n\n"


def test_the_clocks_tick_in_the_page(setup: tuple[TestClient, WorkerService]) -> None:
    """The time a picture has taken and when the queue will be done (Phase 17e) move in the
    page itself; the server sends no event for them, and they are not a change of the part."""
    client, _ = setup
    script = client.get(re.search(r'src="(/static/app.js[^"]*)"', client.get("/").text)[1]).text
    assert 'new EventSource("/events")' in script
    assert '"[data-since] [data-elapsed]"' in script and '"[data-done-in]"' in script
    assert 'querySelectorAll("[data-elapsed], [data-done-in]")' in script  # key() blanks them
    assert "new Intl.DateTimeFormat(LANG, shown)" in script
    assert "progress[data-since]" not in script
    assert "setInterval(refresh, 3000)" in script  # only without EventSource
