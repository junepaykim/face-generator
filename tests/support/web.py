"""The web app on made-up players for HTTP tests, with a worker driven step by step."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from factories import snapshot
from fakes import FakeRenderer, FakeTagger
from fastapi.testclient import TestClient

from facegen.core.records import PlayerRecord
from facegen.datasource.kits import load_kits
from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline.sync import run_sync
from facegen.runtime.worker_service import WorkerService
from facegen.web.app import AppContext, create_app

BASE = "http://127.0.0.1:8765"
KOREAN_WINDOWS = {"Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"}


def synced(config_file: Path, world: Sequence[PlayerRecord]) -> Path:
    """The database of a career synced from `world`."""
    settings = load_settings(defaults_path=config_file)
    conn = db.connect(settings.paths.db_path)
    run_sync(conn, snapshot(list(world)), settings, save_file="career.fm")
    conn.close()
    return settings.paths.db_path


def fake_worker(config_file: Path, db_path: Path, **kw: Any) -> WorkerService:
    """A worker with fake rendering and FM27 closed unless `fm_check` says otherwise."""
    return WorkerService(
        db_path,
        renderer_factory=FakeRenderer,
        tagger_factory=FakeTagger,
        defaults_path=config_file,
        kits=load_kits(),
        fm_check=kw.pop("fm_check", lambda: False),
        **kw,
    )


def client_for(config_file: Path, db_path: Path, worker: Any = None) -> TestClient:
    """A loopback client holding the app's form token. The caller closes it."""
    app = create_app(AppContext(db_path, config_file, worker))
    client = TestClient(app, base_url=BASE, follow_redirects=False)
    token = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/settings").text)
    assert token is not None
    client.headers["X-CSRF-Token"] = token[1]
    return client
