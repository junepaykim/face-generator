"""Shared runtime composition for `serve`, `tray` and `app`.

`build_runtime` constructs without starting; `Runtime.start` starts the watcher, the worker
and the update check. The update check is not a quit step (it only waits between checks):
`Runtime.stop_updates` ends it when facegen stops.
`Runtime.stop_steps` signals the worker immediately so it can abandon a picture being drawn.
Downloads, sync and the watcher then stop, and the worker is reaped. A committed sync gets
time to publish its pack. Each sync runs in a process of its own by default.
Tests can explicitly select in-thread execution by injecting an inline source.
`serve` runs uvicorn in the main thread and cleans up afterward. The desktop shell runs
`ServerThread` and stops it after the runtime steps.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from facegen.core.kit_values import KitError
from facegen.core.settings_values import Settings
from facegen.datasource.kits import load_kits
from facegen.setup import hardware, models

if TYPE_CHECKING:
    from fastapi import FastAPI

    from facegen.datasource.base import DataSource
    from facegen.runtime.dependencies import Gpu
    from facegen.runtime.poller_service import PollerService
    from facegen.runtime.sync_service import SyncService
    from facegen.runtime.update_service import UpdateService
    from facegen.runtime.worker_process import WorkerProxy
    from facegen.web.app import AppContext

LOOPBACK = ("127.0.0.1", "localhost", "::1")
SERVER_START_SECONDS = 30.0
SERVER_STOP_SECONDS = 10.0
GRACEFUL_SHUTDOWN_SECONDS = 2
"""uvicorn waits this long for open responses, then ends them: a page's live event stream
(Phase 12d) never ends by itself."""

Browse = Callable[[str | None], str | None]
DEFAULT_BROWSE: Any = object()
"""`build_runtime`'s browse when none is given: the tkinter dialog where one can be shown."""


class CannotStart(Exception):
    """facegen cannot start as configured (the message says why); exit code 2."""


@dataclass
class Runtime:
    app: FastAPI
    context: AppContext
    worker: WorkerProxy | None
    syncer: SyncService
    poller: PollerService | None
    download: models.ModelDownload | None
    host: str
    port: int
    updates: UpdateService | None = None

    @property
    def url(self) -> str:
        return f"http://{'[::1]' if self.host == '::1' else self.host}:{self.port}/"

    def start(self) -> None:
        """Start the save watcher, the GPU worker process and the update check."""
        if self.poller is not None:
            self.poller.start()
        else:
            self.syncer.recover_if_pending()
        if self.worker is not None:
            self.worker.start()
        if self.updates is not None:
            self.updates.start()

    def stop_updates(self) -> None:
        """End the update check's thread (at once: it only waits between checks)."""
        if self.updates is not None:
            self.updates.stop()

    def stop_steps(self) -> list[tuple[str, Callable[[], None]]]:
        """How the parts stop, in order (module docstring); the web server is not among them."""
        steps: list[tuple[str, Callable[[], None]]] = []
        if self.worker is not None:
            steps.append(("drawing", self.worker.request_stop))
        if self.download is not None:
            steps.append(("model download", self.download.stop))
        # Before the watcher: its thread may be waiting for that sync.
        steps.append(("sync", self.syncer.stop))
        if self.poller is not None:
            poller = self.poller
            steps.append(("save watcher", lambda: poller.stop(timeout=None)))
        if self.worker is not None:
            worker = self.worker
            steps.append(("worker", lambda: worker.stop(timeout=None)))
        return steps


def build_runtime(
    settings: Settings,
    gpu: Gpu,
    *,
    inline_source: Callable[[], DataSource] | None = None,
    config: Path | None = None,
    port: int | None = None,
    paused: bool = False,
    worker: bool = True,
    watch: bool = True,
    browse: Browse | None = DEFAULT_BROWSE,
    echo_stderr: bool = True,
) -> Runtime:
    """The parts of a running facegen, none started. Raises CannotStart when it cannot run
    as configured (an address beyond this PC, a broken kit table)."""
    from facegen.runtime import sync_process
    from facegen.runtime.poller_service import PollerService
    from facegen.runtime.sync_service import SyncService
    from facegen.runtime.update_service import UpdateService
    from facegen.runtime.worker_process import WorkerProxy, worker_command
    from facegen.setup import folder_setup
    from facegen.web.app import AppContext, create_app

    host = settings.ui.host
    if host not in LOOPBACK:
        raise CannotStart(f"ui.host is {host!r}: facegen serves on this PC only (127.0.0.1).")
    try:  # the worker process loads them again; a broken table is reported here, at once
        load_kits(data_dir=settings.paths.data_dir)
    except KitError as exc:
        raise CannotStart(f"kits: {exc}") from exc
    db_path = settings.paths.db_path
    proxy = None
    if worker:
        proxy = WorkerProxy(
            worker_command(gpu.program, config),
            db_path=db_path,
            start_paused=paused,
            echo_stderr=echo_stderr,
        )
    syncer = SyncService(
        db_path,
        inline_source,
        command=sync_process.sync_command() if inline_source is None else None,
        defaults_path=config,
        on_done=proxy.wake if proxy else (lambda: None),
        log=proxy.log if proxy else (lambda line: None),
        echo_stderr=echo_stderr,
    )
    poller = PollerService(db_path, syncer, defaults_path=config) if watch else None
    # The model files matter only to the real GPU worker: not with --no-worker or fakes.
    download = None
    if proxy is not None and gpu.models_check() is not None:
        download = models.ModelDownload(on_done=proxy.wake)
    if browse is DEFAULT_BROWSE:
        browse = folder_setup.browse if folder_setup.dialog_available() else None
    updates = UpdateService(settings.paths.data_dir, lambda: updates_enabled(db_path, config))
    context = AppContext(
        db_path,
        config,
        proxy,
        syncer,
        poller,
        models=models.status if download is not None else None,
        download=download,
        hardware=lambda: hardware.probe(gpu.program, config),
        browse=browse,
        updates=updates,
    )
    return Runtime(
        create_app(context),
        context,
        proxy,
        syncer,
        poller,
        download,
        host,
        port or settings.ui.port,
        updates,
    )


def updates_enabled(db_path: Path, config: Path | None) -> bool:
    """`updates.check` as saved now (it can change while facegen runs)."""
    from facegen.persistence import db
    from facegen.persistence.settings import load_settings

    conn = db.connect(db_path)
    try:
        return load_settings(conn, defaults_path=config).updates.check
    finally:
        conn.close()


def pictures_notice_enabled(db_path: Path, config: Path | None) -> bool:
    """`notifications.new_pictures` as saved now (it can change while facegen runs)."""
    from facegen.persistence import db
    from facegen.persistence.settings import load_settings

    conn = db.connect(db_path)
    try:
        return load_settings(conn, defaults_path=config).notifications.new_pictures
    finally:
        conn.close()


def port_answers(host: str, port: int, *, timeout: float = 0.5) -> bool:
    """Whether something already listens on host:port (another facegen, or another program)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
        except OSError:
            return False
    return True


class ServerStartError(Exception):
    """The web server did not start (the message says why)."""


class ServerThread:
    """uvicorn in a thread of its own, stopped with `should_exit` (the desktop app's main
    thread belongs to the window or the tray)."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        import uvicorn

        self.app, self.host, self.port = app, host, port
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
            )
        )
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="facegen-server", daemon=True)

    def _run(self) -> None:
        try:
            self.server.run()
        except SystemExit as exc:  # uvicorn leaves this way when it cannot start
            self.error = f"the web server stopped (exit code {exc.code})"
        except BaseException as exc:  # noqa: BLE001 - reported by start()
            self.error = f"{type(exc).__name__}: {exc}"

    def start(self, timeout: float = SERVER_START_SECONDS) -> None:
        """Start the server and wait until it answers; raises ServerStartError."""
        self._thread.start()
        end = time.monotonic() + timeout
        while not self.server.started:
            if not self._thread.is_alive():
                why = self.error or "it stopped at once"
                raise ServerStartError(
                    f"the web server could not listen on {self.host}:{self.port}: {why}"
                )
            if time.monotonic() >= end:
                raise ServerStartError(f"the web server did not start within {timeout:.0f} s")
            time.sleep(0.05)

    def stop(self, timeout: float = SERVER_STOP_SECONDS) -> None:
        from facegen.web.app import close_streams

        close_streams(self.app)  # an open page's live stream would hold the server
        self.server.should_exit = True
        self._thread.join(timeout)
        if self._thread.is_alive():  # an open connection holds it: close them
            self.server.force_exit = True
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()
