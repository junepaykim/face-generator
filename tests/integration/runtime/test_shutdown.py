"""The command line and desktop intentionally have different cleanup error policies."""

from __future__ import annotations

import argparse
import io
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from kitshot import cli
from kitshot.desktop import shell as desktop
from kitshot.persistence.settings import load_settings
from kitshot.runtime import composition as runtime
from kitshot.runtime import logging as service_logging


@pytest.mark.parametrize("server_fails", [False, True])
def test_serve_propagates_the_first_cleanup_failure(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, server_fails: bool
) -> None:
    calls: list[str] = []
    failure = RuntimeError("download cleanup failed")

    def stop_download() -> None:
        calls.append("download")
        raise failure

    steps: list[tuple[str, Callable[[], None]]] = [
        ("model download", stop_download),
        ("save watcher", lambda: calls.append("watcher")),
        ("worker", lambda: calls.append("worker")),
    ]
    rt = SimpleNamespace(
        app=object(),
        host="127.0.0.1",
        port=8765,
        url="http://127.0.0.1:8765/",
        poller=None,
        start=lambda: calls.append("start"),
        stop_steps=lambda: steps,
        stop_updates=lambda: None,  # the update check (Phase 13) is not a quit step
    )
    monkeypatch.setattr(runtime, "build_runtime", lambda *args, **kwargs: rt)
    monkeypatch.setattr(service_logging, "log_to_file", lambda data_dir: data_dir / "kitshot.log")

    def server(*args: object, **kwargs: object) -> None:
        calls.append("server")
        if server_fails:
            raise OSError("server failed")

    args = argparse.Namespace(
        config=config_file, port=None, paused=True, no_worker=True, no_watch=True
    )
    with pytest.raises(RuntimeError) as caught:
        cli.cmd_serve(
            args,
            load_settings(defaults_path=config_file),
            lambda: None,  # type: ignore[arg-type,return-value]
            cli.Gpu(),
            io.StringIO(),
            run_server=server,
        )
    assert caught.value is failure
    assert calls == ["start", "server", "download"]
    if server_fails:
        assert isinstance(caught.value.__context__, OSError)


def test_desktop_startup_cleanup_attempts_every_step_after_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    def fail(name: str) -> None:
        calls.append(name)
        raise RuntimeError(f"{name} cleanup failed")

    steps: list[tuple[str, Callable[[], None]]] = [
        ("model download", lambda: fail("download")),
        ("save watcher", lambda: calls.append("watcher")),
        ("worker", lambda: fail("worker")),
    ]
    rt = SimpleNamespace(stop_steps=lambda: steps)
    server = SimpleNamespace(stop=lambda: fail("server"))
    # The window opens before the runtime starts, so a failed start closes it too.
    window = SimpleNamespace(destroy=lambda: fail("window"))
    desktop._stop_quietly(rt, server, window)
    assert calls == ["download", "watcher", "worker", "server", "window"]
    for name in ("model download", "worker", "web server", "window"):
        assert f"stopping the {name} failed" in caplog.text
