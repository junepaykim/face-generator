"""The first run as one guided flow on the dashboard (Phase 12f)."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from factories import player, snapshot
from fakes import FakeRenderer, FakeTagger
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS

from facegen.core.records import Snapshot
from facegen.datasource.kits import load_kits
from facegen.persistence.settings import DEFAULTS_PATH, load_defaults, load_settings
from facegen.runtime.contracts import WorkerStatus
from facegen.service import SyncService, WorkerService
from facegen.setup.hardware import Hardware
from facegen.setup.models import FILES, ModelsStatus
from facegen.web import guide
from facegen.web.app import AppContext, create_app

GIB = 2**30
NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def status(reason: str, **values: object) -> WorkerStatus:
    return WorkerStatus(
        "idle", "", None, False, False, False, False, 0, 0, None, (), reason=reason, values=values
    )


def facts(**changes: object) -> guide.Facts:
    """A fresh install: no folder, the card not asked yet, the models missing, no save read."""
    base = guide.Facts(
        folder_ok=False,
        folder=None,
        folder_gone=False,
        models=ModelsStatus(Path("hub"), FILES, 10 * GIB, free_bytes=100 * GIB),
        download=None,
        card_available=True,
        card=None,
        current_values=dict(load_defaults()),
        career=None,
        game_date=None,
        syncing=False,
        can_sync=True,
        watching=True,
        worker=status("no_career"),
        pictures=0,
        first_picture_at=None,
        closed=False,
        now=NOW,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def states(g: guide.Guide | None) -> dict[str, str]:
    assert g is not None
    return {s.key: s.state for s in g.steps}


def test_a_fresh_install_asks_only_what_it_cannot_work_out() -> None:
    g = guide.build(facts())
    assert states(g) == {
        "folder": "ask", "gpu": "working", "models": "ask", "sync": "wait", "pictures": "wait",
    }  # fmt: skip
    assert g is not None and g.current == "folder" and not g.complete


def test_the_card_is_fine_needs_a_change_or_cannot_draw() -> None:
    big = guide.build(facts(card=Hardware("RTX 4080", 16 * GIB)))
    assert states(big)["gpu"] == "done"
    small = guide.build(facts(card=Hardware("Fake 8 GB", 8 * GIB)))
    step = next(s for s in small.steps if s.key == "gpu")  # type: ignore[union-attr]
    assert step.state == "ask" and {c.key for c in step.data["changes"]} == {
        "groups.a.candidates", "groups.a.detail_size", "render.vae_tiling",
    }  # fmt: skip
    none = guide.build(facts(card=Hardware(reason="no CUDA device")))
    assert states(none)["gpu"] == "problem"
    assert "gpu" not in states(guide.build(facts(card_available=False)))


def test_the_first_pictures_step_says_what_holds_it() -> None:
    ready = facts(folder_ok=True, models=None, career="Colchester", card_available=False)
    for reason, expected in (
        ("drawing", ("working", {"name": "Mine"})),
        ("fm_running", ("ask", {"why": "fm_running"})),
        ("held", ("ask", {"why": "held"})),
        ("paused", ("ask", {"why": "paused"})),
        ("queue_empty", ("ask", {"why": "nobody"})),
        ("job_error", ("problem", {})),
    ):
        g = guide.build(replace(ready, worker=status(reason, name="Mine")))
        step = next(s for s in g.steps if s.key == "pictures")  # type: ignore[union-attr]
        assert (step.state, step.data) == expected, reason


def test_finished_it_stays_a_day_unless_closed_and_never_for_an_old_install() -> None:
    done = facts(
        folder_ok=True, folder="/fm", models=None, career="Colchester", pictures=3,
        card=Hardware("Fake 8 GB", 8 * GIB),  # a suggestion left open holds nothing up
        first_picture_at=(NOW - timedelta(hours=2)).isoformat(),
    )  # fmt: skip
    g = guide.build(done)
    assert g is not None and g.complete and states(g)["gpu"] == "ask"
    assert guide.build(replace(done, closed=True)) is None
    old = (NOW - timedelta(days=5)).isoformat()
    assert guide.build(replace(done, first_picture_at=old)) is None
    # A folder that went missing on an old install brings the guide back for it.
    assert (
        states(guide.build(replace(done, first_picture_at=old, folder_ok=False)))["folder"] == "ask"
    )


# --- Through the web app ---------------------------------------------------------------------


class Source:
    def load(self, path: Path) -> Snapshot:
        return snapshot([player(1001, name="Mine Player", age=29)], file_name=path.name)


def make_client(tmp: Path, card: Hardware | None) -> tuple[TestClient, Path, WorkerService]:
    text = DEFAULTS_PATH.read_text(encoding="utf-8")
    text = text.replace('data_dir = "data"', f'data_dir = "{(tmp / "data").as_posix()}"')
    config = tmp / "config" / "test.toml"
    config.parent.mkdir()
    config.write_text(text, encoding="utf-8")
    settings = load_settings(defaults_path=config)
    worker = WorkerService(
        settings.paths.db_path, renderer_factory=FakeRenderer, tagger_factory=FakeTagger,
        defaults_path=config, kits=load_kits(), fm_check=lambda: False,
    )  # fmt: skip
    syncer = SyncService(settings.paths.db_path, Source, defaults_path=config)
    ctx = AppContext(
        settings.paths.db_path, config, worker, syncer,
        hardware=(lambda: card) if card is not None else None, platform="linux",
    )  # fmt: skip
    client = TestClient(create_app(ctx), base_url=BASE, follow_redirects=False)
    token = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/settings/advanced").text)
    assert token is not None
    client.headers["X-CSRF-Token"] = token[1]
    return client, settings.paths.data_dir, worker


def fm27(root: Path) -> Path:
    folder = root / "Documents" / "Sports Interactive" / "Football Manager 27"
    (folder / "games").mkdir(parents=True)
    save = folder / "games" / "career.fm"
    save.write_bytes(b"save")
    os.utime(save, (1_700_000_000, 1_700_000_000))
    return folder


def step_head(page: str, title: str) -> str:
    match = re.search(rf"<strong>{re.escape(title)}</strong>(.*?)</p>", page, re.DOTALL)
    assert match is not None, title
    return " ".join(match[1].split())


def test_from_nothing_to_the_first_picture_on_one_page(tmp_path: Path) -> None:
    client, data_dir, worker = make_client(tmp_path, Hardware("RTX 4080", 16 * GIB))
    page = client.get("/").text
    assert "Getting started" in page and 'id="now"' not in page  # the guide stands alone
    assert step_head(page, "FM27's folder") == "Not found. Choose it below."
    page = client.get("/status").text  # the card has answered in the background meanwhile
    for _ in range(50):
        if "the settings suit it" in page:
            break
        page = client.get("/status").text
    assert step_head(page, "Graphics card") == "RTX 4080, 16 GB: the settings suit it."

    folder = fm27(tmp_path)
    r = client.post("/setup/folder", data={"folder": str(folder), "next": "/"})
    assert r.status_code == 303
    # The folder starts a sync in a thread; the page says when the save was read.
    for _ in range(100):
        page = client.get("/status").text
        if "in-game" in step_head(page, "Your save"):
            break
    assert step_head(page, "Your save") == "career · in-game 2026-03-01"
    worker.step()  # the first picture
    page = client.get("/status").text
    assert "Ready" in page and step_head(page, "First pictures").startswith("1 picture made.")
    assert 'id="now"' in page  # the dashboard is back below the guide
    korean = client.get("/status", headers=KOREAN_WINDOWS).text
    assert "준비 완료" in korean and "그림 1장을 만들었습니다." in korean

    r = client.post("/setup/guide")
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert (data_dir / guide.CLOSED).exists()
    assert "Getting started" not in client.get("/").text and "Ready" not in client.get("/").text


def test_the_cards_suggestion_is_applied_from_the_guide(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, Hardware("Fake 8 GB", 8 * GIB))
    for _ in range(50):
        page = client.get("/status").text
        if "should change" in page:
            break
    assert "Fake 8 GB, 8 GB: 3 settings should change for this card." in page
    assert 'action="/settings/auto-configure"' in page and 'name="next" value="/"' in page
    r = client.post("/settings/auto-configure", data={"next": "/"})
    assert r.status_code == 303 and r.headers["location"].startswith("/?msg=")
    assert "Fake 8 GB, 8 GB: the settings suit it." in client.get("/status").text


def test_a_half_typed_folder_survives_a_live_update(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, None)
    script = client.get(re.search(r'src="(/static/app.js[^"]*)"', client.get("/").text)[1]).text
    assert "input.value !== input.defaultValue" in script and "refocus" in script


@pytest.mark.parametrize("closed", [False, True])
def test_the_live_fingerprint_follows_the_card_and_the_close(tmp_path: Path, closed: bool) -> None:
    client, data_dir, _ = make_client(tmp_path, Hardware("RTX 4080", 16 * GIB))
    if closed:
        guide.close(data_dir)
    first = re.search(r'data-rev="([^"]+)"', client.get("/").text)[1]
    for _ in range(50):
        again = re.search(r'data-rev="([^"]+)"', client.get("/").text)[1]
        if again != first:
            break
    assert again != first  # the card answered: the guide's part changed
