"""HTTP compatibility and isolation between independently composed web applications."""

from __future__ import annotations

import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest
from factories import player, snapshot
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from PIL import Image
from support.config import write_config
from support.desktop import FakeSyncer, FakeWorker

from kitshot.persistence import db
from kitshot.persistence.settings import load_settings
from kitshot.pipeline.sync import run_sync
from kitshot.runtime.policy import BUSY_TIMEOUT_MS
from kitshot.setup.hardware import Hardware
from kitshot.web.app import AppContext, create_app
from kitshot.web.routes.assets import make_thumb

BASE = "http://127.0.0.1:8765"


def test_concurrent_thumbnail_misses_publish_the_same_complete_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, cache = tmp_path / "portrait.png", tmp_path / "thumbs"
    Image.new("RGB", (150, 200), "red").save(source)
    both_saved = threading.Barrier(2)
    save = Image.Image.save

    def overlapping_save(image, *args, **kwargs):
        save(image, *args, **kwargs)
        both_saved.wait(timeout=5)

    monkeypatch.setattr(Image.Image, "save", overlapping_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = [pool.submit(make_thumb, source, 96, cache) for _ in range(2)]
        first, second = (request.result(timeout=10) for request in requests)
    assert first == second
    with Image.open(first) as thumb:
        thumb.load()
        assert thumb.size == (72, 96) and thumb.format == "WEBP"
    assert list(cache.iterdir()) == [first]


def token(client: TestClient) -> str:
    response = client.get("/")
    assert response.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
    assert match is not None
    return match[1]


def test_http_surface_and_slash_redirects(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    app = create_app(AppContext(settings.paths.db_path, config_file))
    routes = list(iter_route_contexts(app.routes))
    assert [(r.path, r.name) for r in routes if not isinstance(r.original_route, APIRoute)] == [
        ("/static", "static")
    ]
    surface = []
    connections = []
    for route in routes:
        if isinstance(route.original_route, APIRoute):
            response_class = getattr(route.response_class, "value", route.response_class)
            surface.append(
                (tuple(sorted(route.methods)), route.path, route.name, response_class.__name__)
            )
            assert route.status_code is None
            connections.extend(
                dep.call for dep in route.dependant.dependencies if dep.name == "conn"
            )
    assert connections and all(connection is connections[0] for connection in connections)
    assert surface == [
        (("GET",), "/", "dashboard", "HTMLResponse"),
        (("GET",), "/status", "status", "HTMLResponse"),
        (("POST",), "/worker/{action}", "worker_action", "JSONResponse"),
        (("POST",), "/sync", "sync_now", "JSONResponse"),
        (("GET",), "/log", "log", "HTMLResponse"),
        (("GET",), "/log/part", "log_part", "HTMLResponse"),
        (("GET",), "/log/report", "log_report", "PlainTextResponse"),
        (("POST",), "/folders/open", "open_folder", "JSONResponse"),
        (("POST",), "/setup/folder", "setup_folder", "JSONResponse"),
        (("POST",), "/setup/browse", "setup_browse", "JSONResponse"),
        (("POST",), "/setup/guide", "setup_guide_closed", "JSONResponse"),
        (("POST",), "/setup/models", "setup_models", "JSONResponse"),
        (("GET",), "/gallery", "gallery", "HTMLResponse"),
        (("GET",), "/gallery/part", "gallery_part", "HTMLResponse"),
        (("GET",), "/players", "players", "HTMLResponse"),
        (("POST",), "/players/action", "players_action", "JSONResponse"),
        (("GET",), "/players/{uid}", "player", "HTMLResponse"),
        (("POST",), "/players/{uid}/{action}", "player_action", "JSONResponse"),
        (("GET",), "/review", "review", "HTMLResponse"),
        (("GET",), "/settings", "settings_get", "HTMLResponse"),
        (("GET",), "/settings/basic", "settings_basic", "HTMLResponse"),
        (("GET",), "/settings/advanced", "settings_advanced", "HTMLResponse"),
        (("GET",), "/hardware", "hardware_page", "HTMLResponse"),
        (("POST",), "/settings/auto-configure", "auto_configure", "JSONResponse"),
        (("POST",), "/settings", "settings_post", "HTMLResponse"),
        (("POST",), "/settings/save", "settings_save", "JSONResponse"),
        (("POST",), "/update/check", "check", "JSONResponse"),
        (("POST",), "/update/later", "later", "JSONResponse"),
        (("POST",), "/update/install", "install", "HTMLResponse"),
        (("GET",), "/pictures/{image_id}", "picture", "JSONResponse"),
        (("GET",), "/files", "files", "JSONResponse"),
        (("GET",), "/events", "events", "JSONResponse"),
    ]
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        client.headers["X-CSRF-Token"] = token(client)
        for method, path in (
            ("GET", "/gallery"),
            ("GET", "/players"),
            ("GET", "/review"),
            ("GET", "/settings"),
            ("GET", "/settings/advanced"),
            ("POST", "/players/action"),
            ("POST", "/settings/save"),
        ):
            response = client.request(method, path + "/")
            assert response.status_code == 307
            assert response.headers["location"] == BASE + path
        # /action is a static batch POST; the GET still binds the typed player ID.
        assert client.get("/players/action").status_code == 422
        assert client.post("/players/action").status_code == 303


@dataclass
class CardProbe:
    name: str
    calls: int = 0

    def __call__(self) -> Hardware:
        self.calls += 1
        return Hardware(name=f"{self.name} card {self.calls}", total_bytes=8 * 2**30)


def test_apps_keep_tokens_data_services_and_hardware_separate(tmp_path: Path) -> None:
    workers = [FakeWorker(), FakeWorker()]
    syncers = [FakeSyncer(), FakeSyncer()]
    probes = [CardProbe("First"), CardProbe("Second")]
    clients = []
    settings = []
    pictures = []
    with ExitStack() as stack:
        for index, name in enumerate(("First", "Second")):
            root = tmp_path / name
            root.mkdir()
            config = write_config(root)
            current = load_settings(defaults_path=config)
            settings.append(current)
            conn = db.connect(current.paths.db_path)
            try:
                run_sync(
                    conn,
                    snapshot([player(1001, name=f"{name} Player")]),
                    current,
                    save_file="career.fm",
                )
            finally:
                conn.close()
            masters = current.paths.data_dir / "masters"
            masters.mkdir(parents=True, exist_ok=True)
            picture = masters / "portrait.png"
            Image.new("RGB", (24, 32), (index * 200, 40, 60)).save(picture)
            pictures.append(picture.read_bytes())
            app = create_app(
                AppContext(
                    current.paths.db_path,
                    config,
                    workers[index],
                    syncers[index],
                    hardware=probes[index],
                )
            )
            clients.append(
                stack.enter_context(TestClient(app, base_url=BASE, follow_redirects=False))
            )

        tokens = [token(client) for client in clients]
        assert tokens[0] != tokens[1]
        for index, client in enumerate(clients):
            client.headers["X-CSRF-Token"] = tokens[index]
            page = client.get("/players").text
            assert ("First Player" in page) == (index == 0)
            assert ("Second Player" in page) == (index == 1)
            assert (
                client.post(
                    "/worker/pause", headers={"X-CSRF-Token": tokens[1 - index]}
                ).status_code
                == 403
            )
            assert (
                client.get("/files", params={"path": "masters/portrait.png"}).content
                == pictures[index]
            )
            thumb = client.get("/files", params={"path": "masters/portrait.png", "size": 96})
            assert thumb.status_code == 200
            assert list((settings[index].paths.data_dir / "thumbs").glob("*.webp"))
        assert workers[0].calls == workers[1].calls == []

        assert clients[0].post("/worker/pause").status_code == 303
        assert workers[0].calls == ["pause"]
        assert workers[1].calls == []
        assert clients[1].post("/sync").status_code == 303
        assert syncers[0].starts == 0
        assert syncers[1].starts == 1

        for index, client in enumerate(clients):
            for _ in range(2):
                page = client.get("/hardware")
                assert page.status_code == 200
                assert f"{probes[index].name} card 1" in page.text
                assert f"{probes[1 - index].name} card" not in page.text
            assert probes[index].calls == 1
        assert "First card 2" in clients[0].get("/hardware?again=1").text
        assert "Second card 1" in clients[1].get("/hardware").text
        assert [probe.calls for probe in probes] == [2, 1]
        # Constructing/rendering the second app must not replace the first template token.
        assert [token(client) for client in clients] == tokens


def test_request_connections_close_after_reads_mutations_and_errors(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(defaults_path=config_file)
    app = create_app(AppContext(settings.paths.db_path, config_file))
    connections: list[sqlite3.Connection] = []
    connect = db.connect

    def tracked_connect(path: str | Path, **kwargs: int) -> sqlite3.Connection:
        assert path == settings.paths.db_path
        assert kwargs["busy_timeout_ms"] == BUSY_TIMEOUT_MS
        conn = connect(path, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(db, "connect", tracked_connect)
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        client.headers["X-CSRF-Token"] = token(client)
        for method, path, data, expected in (
            ("GET", "/status", None, 200),
            ("GET", "/players/1001", None, 409),
            ("GET", "/hardware", None, 200),
            ("POST", "/settings/save", {"groups.a.candidates": "2"}, 200),
            ("POST", "/settings/save", {"groups.a.candidates": "invalid"}, 200),
            ("POST", "/settings", {"action": "unknown"}, 400),
        ):
            before = len(connections)
            response = client.request(method, path, data=data)
            assert response.status_code == expected
            assert len(connections) == before + 1
            for conn in connections:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    conn.execute("SELECT 1")
