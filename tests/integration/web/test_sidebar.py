"""The sidebar on every page (Phase 17d): the career and status card with Pause/Resume, the
two menu groups, Review's count badge, Log's dot after a sync that went wrong, and the version
with an Update badge (the update tests cover the badge itself)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, synced

from kitshot.persistence import db
from kitshot.runtime.contracts import SyncControl, SyncStatus, WorkerControl, WorkerStatus
from kitshot.service import WorkerService
from kitshot.web.app import AppContext, create_app

PAGES = ("/", "/gallery", "/players", "/review", "/log", "/settings", "/settings/advanced")


@pytest.fixture
def db_path(config_file: Path) -> Path:
    world = [player(1001, name="Mine"), player(NEWGEN + 2, ca=90, pa=152, club_id=RIVAL)]
    return synced(config_file, world)


@pytest.fixture
def setup(config_file: Path, db_path: Path) -> Iterator[tuple[TestClient, WorkerService]]:
    worker = fake_worker(config_file, db_path)
    client = client_for(config_file, db_path, worker)
    yield client, worker
    client.close()


def queued(db_path: Path) -> int:
    conn = db.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0]
    finally:
        conn.close()


def side(page: str) -> str:
    found = re.search(r'<aside class="side">(.*?)</aside>', page, re.DOTALL)
    assert found is not None
    return found[1]


def card(page: str) -> str:
    found = re.search(r'<div class="side-card">(.*?)\n    </div>\n', side(page), re.DOTALL)
    assert found is not None
    return found[1]


def shown(part: str, marker: str) -> bool:
    """Whether the element carrying `marker` is there and not hidden."""
    tag = re.search(rf"<[^>]*\b{marker}\b[^>]*>", part)
    assert tag is not None, marker
    return " hidden" not in tag[0]


def status(state: str, reason: str, **kw: Any) -> WorkerStatus:
    base = WorkerStatus(
        state=state, detail="", current=None, paused=False, gpu_fault=False, generate_now=False,
        models_loaded=False, done=0, failed=0, last_error=None, log=(), reason=reason,
    )  # fmt: skip
    return replace(base, **kw)


def client_with(config_file: Path, db_path: Path, **ctx: Any) -> TestClient:
    return TestClient(create_app(AppContext(db_path, config_file, **ctx)), base_url=BASE)


@pytest.mark.parametrize("url", PAGES)
def test_every_page_has_the_card(setup: tuple[TestClient, WorkerService], url: str) -> None:
    client, _ = setup
    page = client.get(url).text
    assert "data-display" not in page  # Language and Theme are in Settings only
    part = card(page)
    assert (
        '<p class="side-career" title="career · in-game 2026-03-01"><strong>career</strong>' in part
    )
    assert '<span class="muted when">2026-03-01</span>' in part
    assert '<span class="dot-state state-starting" aria-hidden="true"></span>' in part
    # Pause comes back to the page it was pressed on; Resume waits for a pause.
    assert f'<input type="hidden" name="next" value="{url}">' in part
    assert shown(part, "data-side-pause") and not shown(part, "data-side-resume")


def test_the_card_says_the_state_and_how_many_are_left(
    setup: tuple[TestClient, WorkerService], db_path: Path
) -> None:
    client, _ = setup
    left = queued(db_path)
    assert left > 0
    part = card(client.get("/").text)
    assert f'<span class="side-state-text" data-side-text>Starting · {left:,} left</span>' in part
    korean = card(client.get("/", headers=KOREAN_WINDOWS).text)
    assert f"data-side-text>시작 중 · {left:,}장 남음</span>" in korean


def test_pause_and_resume_from_the_card_come_back_to_the_page(
    setup: tuple[TestClient, WorkerService],
) -> None:
    client, worker = setup
    r = client.post("/worker/pause", data={"next": "/gallery?page=2"})
    assert (r.status_code, r.headers["location"]) == (303, "/gallery?page=2")
    assert worker.status().paused
    part = card(client.get("/gallery").text)
    assert not shown(part, "data-side-pause") and shown(part, "data-side-resume")
    assert 'action="/worker/resume"' in part

    r = client.post("/worker/resume", data={"next": "/players"})
    assert (r.status_code, r.headers["location"]) == (303, "/players")
    assert not worker.status().paused
    # Another site, or no "next" (the dashboard's own buttons), goes to the dashboard.
    for target in ("https://example.com/", "//example.com/", ""):
        r = client.post("/worker/pause", data={"next": target})
        assert r.headers["location"] == "/"
    assert client.post("/worker/sleep").status_code == 404


def test_the_card_hides_its_button_after_a_gpu_fault(config_file: Path, db_path: Path) -> None:
    worker = Mock(spec=WorkerControl)
    worker.status.return_value = status("gpu-fault", "gpu_fault", gpu_fault=True, paused=True)
    with client_with(config_file, db_path, worker=worker) as client:
        part = card(client.get("/players").text)
    assert not shown(part, "data-side-pause") and not shown(part, "data-side-resume")
    assert "data-side-text>Graphics error ·" in part


def test_without_a_worker_or_a_career(config_file: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    db.connect(empty).close()
    with client_with(config_file, empty) as client:
        page = client.get("/gallery").text
    part = card(page)
    assert '<p class="side-career muted">no career yet</p>' in part
    assert "side-state" not in part and "dot-state" not in side(page)


def test_two_menu_groups(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    nav = side(client.get("/").text)
    links = re.findall(r'<a href="(/[a-z]*)"|<span class="(nav-sep)"', nav)
    order = [href or sep for href, sep in links]
    assert order == ["/", "/gallery", "/players", "/review", "nav-sep", "/log", "/settings"]


def test_review_badge_only_from_one(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    nav = side(client.get("/").text)
    assert "badge-count" not in nav  # nothing to review: no badge at 0
    # At 1 or more: test_web.py::test_review_page_and_retry; the live count in test_live.py.


def test_log_gets_a_dot_when_the_last_sync_went_wrong(config_file: Path, db_path: Path) -> None:
    syncer = Mock(spec=SyncControl)
    syncer.status.return_value = SyncStatus(message="Sync finished.")
    with client_with(config_file, db_path, syncer=syncer) as client:
        assert not shown(side(client.get("/log").text), "data-sync-warn")
        syncer.status.return_value = SyncStatus(warnings=["a club could not be read"])
        nav = side(client.get("/log").text)
        assert shown(nav, "data-sync-warn")
        assert 'title="The last sync failed or had warnings"' in nav
        syncer.status.return_value = SyncStatus(running=True)  # a new sync: wait for it
        assert not shown(side(client.get("/log").text), "data-sync-warn")
        # A failure is also in the database, after a restart.
        conn = db.connect(db_path)
        conn.execute(
            "UPDATE sync_runs SET error = 'no save' WHERE id = (SELECT MAX(id) FROM sync_runs)"
        )
        conn.close()
        syncer.status.return_value = SyncStatus()
        assert shown(side(client.get("/log").text), "data-sync-warn")


def test_the_dashboard_reads_the_sync_state_once_for_the_sidebar_too(
    config_file: Path, db_path: Path
) -> None:
    syncer = Mock(spec=SyncControl)
    syncer.status.return_value = SyncStatus(error="the save is still being written")
    with client_with(config_file, db_path, syncer=syncer) as client:
        page = client.get("/").text
    syncer.status.assert_called_once_with()
    assert shown(side(page), "data-sync-warn")


def test_no_version_line_without_the_update_check(setup: tuple[TestClient, WorkerService]) -> None:
    client, _ = setup
    assert "side-version" not in client.get("/").text  # covered in test_update_web.py
