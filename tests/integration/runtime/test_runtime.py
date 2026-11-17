"""What serve, tray and app run (runtime.py, Phase 11): built the same way, stopped in the same
order; the web server in a thread of its own for the desktop app."""

from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

import pytest
from fakes import FakeRenderer, FakeTagger

from facegen.cli import Gpu
from facegen.persistence.settings import load_settings
from facegen.runtime.composition import (
    CannotStart,
    ServerStartError,
    ServerThread,
    build_runtime,
    port_answers,
)


class NoSource:
    def load(self, path: Path):  # type: ignore[no-untyped-def]
        raise AssertionError("no sync in these tests")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def fake_gpu(**kw: object) -> Gpu:
    return Gpu(renderer=FakeRenderer, tagger=FakeTagger, fm_check=lambda: True, **kw)  # type: ignore[arg-type]


def test_the_parts_stop_in_order(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    rt = build_runtime(settings, NoSource, fake_gpu(), config=config_file, paused=True)
    assert [name for name, _ in rt.stop_steps()] == ["save watcher", "worker"]
    assert rt.url == f"http://127.0.0.1:{settings.ui.port}/"
    assert rt.context.browse is None or callable(rt.context.browse)
    # With the real worker's model check, the download is stopped first.
    rt = build_runtime(
        settings, NoSource, fake_gpu(models_ready=lambda: True), config=config_file, paused=True
    )
    assert [name for name, _ in rt.stop_steps()] == ["model download", "save watcher", "worker"]
    rt = build_runtime(settings, NoSource, fake_gpu(), worker=False, watch=False, browse=None)
    assert rt.stop_steps() == [] and rt.context.browse is None


def test_an_address_beyond_this_pc_is_refused(config_file: Path) -> None:
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace('host = "127.0.0.1"', 'host = "0.0.0.0"'),
        encoding="utf-8",
    )
    settings = load_settings(defaults_path=config_file)
    with pytest.raises(CannotStart, match="this PC only"):
        build_runtime(settings, NoSource, fake_gpu())


def test_the_server_thread_answers_and_stops(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    rt = build_runtime(settings, NoSource, fake_gpu(), worker=False, watch=False, browse=None)
    port = free_port()
    assert not port_answers("127.0.0.1", port)
    server = ServerThread(rt.app, "127.0.0.1", port)
    server.start()
    try:
        assert server.running and port_answers("127.0.0.1", port)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            assert response.status == 200 and b"facegen" in response.read()
    finally:
        server.stop()
    assert not server.running and not port_answers("127.0.0.1", port)


def test_a_taken_port_is_a_clear_error(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    rt = build_runtime(settings, NoSource, fake_gpu(), worker=False, watch=False, browse=None)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = int(taken.getsockname()[1])
        assert port_answers("127.0.0.1", port)
        server = ServerThread(rt.app, "127.0.0.1", port)
        with pytest.raises(ServerStartError, match=f"could not listen on 127.0.0.1:{port}"):
            server.start(timeout=10)


def test_an_open_page_does_not_hold_the_server_when_it_stops(config_file: Path) -> None:
    """Phase 12d: a page's live event stream never ends by itself. Stopping the server ends it
    first (web.app.close_streams), so Quit does not wait for the window's stream."""
    import http.client
    import time

    settings = load_settings(defaults_path=config_file)
    rt = build_runtime(settings, NoSource, fake_gpu(), worker=False, watch=False, browse=None)
    port = free_port()
    server = ServerThread(rt.app, "127.0.0.1", port)
    server.start()
    page = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        page.request("GET", "/events", headers={"Accept": "text/event-stream"})
        response = page.getresponse()
        assert response.status == 200
        assert response.getheader("content-type", "").startswith("text/event-stream")
        assert response.readline().startswith(b"retry: ")  # the stream is open
        started = time.monotonic()
        server.stop()
        # Well inside GRACEFUL_SHUTDOWN_SECONDS (2 s), uvicorn's own fallback, and far from
        # SERVER_STOP_SECONDS: the stream noticed the server stopping.
        assert time.monotonic() - started < 1.5
        assert not server.running
    finally:
        page.close()
