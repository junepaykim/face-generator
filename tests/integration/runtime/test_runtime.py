"""What serve, tray and app run (runtime.py, Phase 11): built the same way, stopped in the same
order; the web server in a thread of its own for the desktop app."""

from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path

import pytest
from fakes import FakeRenderer, FakeTagger

from kitshot.cli import Gpu
from kitshot.persistence import db
from kitshot.persistence.settings import load_settings, set_override
from kitshot.runtime.composition import (
    CannotStart,
    ServerStartError,
    ServerThread,
    build_runtime,
    pictures_notice_enabled,
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
    rt = build_runtime(
        settings, fake_gpu(), inline_source=NoSource, config=config_file, paused=True
    )
    assert [name for name, _ in rt.stop_steps()] == ["drawing", "sync", "save watcher", "worker"]
    assert rt.url == f"http://127.0.0.1:{settings.ui.port}/"
    assert rt.context.browse is None or callable(rt.context.browse)
    # With the real worker's model check, downloads also stop before waiting for the sync.
    rt = build_runtime(
        settings,
        fake_gpu(models_ready=lambda: True),
        inline_source=NoSource,
        config=config_file,
        paused=True,
    )
    assert [name for name, _ in rt.stop_steps()] == [
        "drawing", "model download", "sync", "save watcher", "worker",
    ]  # fmt: skip
    rt = build_runtime(
        settings, fake_gpu(), inline_source=NoSource, worker=False, watch=False, browse=None
    )
    assert [name for name, _ in rt.stop_steps()] == ["sync"] and rt.context.browse is None


def test_the_real_save_reader_syncs_in_a_process_of_its_own(config_file: Path) -> None:
    """Phase 14: with the real reader each sync runs in `kitshot sync-process`; a test's fake
    source syncs in the calling thread, as WorkerService stays beside WorkerProxy."""
    from kitshot.runtime.sync_process import sync_command

    settings = load_settings(defaults_path=config_file)
    real = build_runtime(settings, fake_gpu(), worker=False, browse=None)
    assert real.syncer.process is not None
    assert real.syncer.process.command == sync_command() == [
        sys.executable, "-m", "kitshot.cli", "sync-process",
    ]  # fmt: skip
    fake = build_runtime(settings, fake_gpu(), inline_source=NoSource, worker=False, browse=None)
    assert fake.syncer.process is None


def test_an_address_beyond_this_pc_is_refused(config_file: Path) -> None:
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace('host = "127.0.0.1"', 'host = "0.0.0.0"'),
        encoding="utf-8",
    )
    settings = load_settings(defaults_path=config_file)
    with pytest.raises(CannotStart, match="this PC only"):
        build_runtime(settings, fake_gpu(), inline_source=NoSource)


def test_the_server_thread_answers_and_stops(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    rt = build_runtime(
        settings, fake_gpu(), inline_source=NoSource, worker=False, watch=False, browse=None
    )
    port = free_port()
    assert not port_answers("127.0.0.1", port)
    server = ServerThread(rt.app, "127.0.0.1", port)
    server.start()
    try:
        assert server.running and port_answers("127.0.0.1", port)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            assert response.status == 200 and b"Kitshot" in response.read()
    finally:
        server.stop()
    assert not server.running and not port_answers("127.0.0.1", port)


def test_a_taken_port_is_a_clear_error(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    rt = build_runtime(
        settings, fake_gpu(), inline_source=NoSource, worker=False, watch=False, browse=None
    )
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
    rt = build_runtime(
        settings, fake_gpu(), inline_source=NoSource, worker=False, watch=False, browse=None
    )
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


def test_the_tray_reads_the_new_pictures_switch_as_saved_now(config_file: Path) -> None:
    db_path = load_settings(defaults_path=config_file).paths.db_path
    assert pictures_notice_enabled(db_path, config_file) is True
    conn = db.connect(db_path)
    try:
        set_override(conn, "notifications.new_pictures", False, config_file)
    finally:
        conn.close()
    assert pictures_notice_enabled(db_path, config_file) is False
