"""The desktop app (Phase 11) from start to Quit, with a fake tray and window: the real web
server in its thread, the real worker process (with a fake GPU), one instance at a time, Quit's
order, and what a failure while starting leaves behind (nothing) and shows (a message)."""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRenderer, FakeTagger
from support.paths import TESTS
from support.processes import has_exited

from facegen.cli import Gpu, main
from facegen.desktop import instance
from facegen.desktop import shell as desktop
from facegen.desktop.app_controller import AppController, Notice, View
from facegen.desktop.app_window import NO_WEBVIEW2
from facegen.runtime.composition import CannotStart, port_answers


@pytest.fixture(autouse=True)
def quick_and_clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A quicker heartbeat, and the "facegen" logger without the files earlier tests gave it
    (log_to_file adds one), put back as it was afterwards."""
    monkeypatch.setattr(desktop, "REFRESH_SECONDS", 0.05)
    log = logging.getLogger("facegen")
    monkeypatch.setattr(log, "handlers", [])
    monkeypatch.setattr(log, "propagate", log.propagate)
    monkeypatch.setattr(log, "level", log.level)
    yield
    for handler in list(log.handlers):
        handler.close()


def wait_for(check: Callable[[], bool], timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeTray:
    def __init__(self, controller: AppController) -> None:
        self.controller = controller
        self.views: list[View] = []
        self.notices: list[Notice] = []
        self.stopped = threading.Event()
        self.mode: str | None = None

    def run(self) -> None:
        self.mode = "run"
        self.stopped.wait(60)

    def stop(self) -> None:
        self.stopped.set()

    def show(self, view: View) -> None:
        self.views.append(view)

    def notify(self, notice: Notice) -> None:
        self.notices.append(notice)


class FakeWindow:
    """app_window.WindowProcess as desktop.py sees it: started, in a process of its own."""

    def __init__(
        self,
        url: str,
        hidden: bool,
        on_hide: Callable[[], None],
        on_unavailable: Callable[[str], None],
    ) -> None:
        self.url, self.hidden, self.on_hide = url, hidden, on_hide
        self.on_unavailable = on_unavailable
        self.closed = threading.Event()
        self.shown = 0
        self.browsed: list[str | None] = []
        self.answer: str | None = None

    def show(self) -> None:
        self.shown += 1

    def destroy(self) -> None:
        self.closed.set()

    def browse(self, start: str | None) -> str | None:
        self.browsed.append(start)
        return self.answer


class Record:
    """A fake shell and what it was asked."""

    def __init__(self, *, webview2: bool = True, tray_error: Exception | None = None) -> None:
        self.trays: list[FakeTray] = []
        self.windows: list[FakeWindow] = []
        self.icons: list[str | None] = []
        self.browser: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self.webview2 = webview2
        self.tray_error = tray_error

    def tray(self, controller: AppController) -> FakeTray:
        if self.tray_error is not None:
            raise self.tray_error
        tray = FakeTray(controller)
        self.trays.append(tray)
        return tray

    def window(
        self,
        url: str,
        hidden: bool,
        on_hide: Callable[[], None],
        on_unavailable: Callable[[str], None],
        icon: str | None,
    ) -> FakeWindow:
        self.icons.append(icon)
        window = FakeWindow(url, hidden, on_hide, on_unavailable)
        self.windows.append(window)
        if not self.webview2:  # the window process says so and ends
            on_unavailable(NO_WEBVIEW2)
        return window

    def shell(self) -> desktop.Shell:
        return desktop.Shell(
            make_tray=self.tray,  # type: ignore[arg-type]
            make_window=self.window,  # type: ignore[arg-type]
            startup=lambda data_dir, config: None,
            open_browser=self.browser.append,
            show_error=lambda title, text: self.errors.append((title, text)),
        )


def args_for(config: Path, port: int, **kw: Any) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        port=port,
        paused=kw.get("paused", True),
        minimized=kw.get("minimized", False),
    )


class Running:
    """desktop.run in a thread, as the main thread runs it."""

    def __init__(self, mode: str, args: argparse.Namespace, record: Record, gpu: Gpu) -> None:
        self.started = desktop.Started()
        self.code: int | None = None
        self.error: BaseException | None = None

        def target() -> None:
            try:
                self.code = desktop.run(
                    mode, args, gpu=gpu, shell=record.shell(), started=self.started
                )
            except BaseException as exc:  # noqa: BLE001 - looked at by the test
                self.error = exc

        self.thread = threading.Thread(target=target)
        self.thread.start()
        wait_for(lambda: self.started.controller is not None or not self.thread.is_alive())
        if self.error is not None:
            raise self.error

    @property
    def controller(self) -> AppController:
        assert self.started.controller is not None
        return self.started.controller

    def quit(self) -> None:
        self.controller.quit()
        self.thread.join(90)
        assert not self.thread.is_alive()
        if self.error is not None:
            raise self.error


def fake_gpu() -> Gpu:
    return Gpu(renderer=FakeRenderer, tagger=FakeTagger, fm_check=lambda: True)


def page(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


def post(url: str, path: str, fields: dict[str, str]) -> str:
    token = re.search(r'name="csrf_token" value="([^"]+)"', page(url))
    assert token is not None
    data = urllib.parse.urlencode({"csrf_token": token.group(1), **fields}).encode()
    origin = url.rstrip("/")
    request = urllib.request.Request(url.rstrip("/") + path, data=data, headers={"Origin": origin})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def test_the_app_from_start_to_quit(config_file: Path, tmp_path: Path) -> None:
    """The window on the dashboard, Browse through the window's own dialog, a second start
    showing the first one's window, and Quit: the worker process ends, then the server, the
    tray and the window; the lock is given back."""
    marker = tmp_path / "faulted"
    marker.write_text("no fault this time", encoding="utf-8")  # gpu_child.py draws normally
    gpu = Gpu(program=(sys.executable, str(TESTS / "gpu_child.py"), str(marker)))
    record = Record()
    port = free_port()
    app = Running("app", args_for(config_file, port, paused=False), record, gpu)
    controller = app.controller
    (window,) = record.windows
    (tray,) = record.trays
    url = f"http://127.0.0.1:{port}/"
    assert window.url == url and not window.hidden
    (icon,) = record.icons  # facegen's .ico for the taskbar, on Windows
    assert (icon or "").endswith("facegen.ico") if os.name == "nt" else icon is None
    wait_for(lambda: tray.mode == "run")  # the tray has the main thread; the window its process
    assert app.started.url == url and record.browser == []
    wait_for(lambda: len(tray.views) > 2)  # the heartbeat runs
    # The worker runs in a process of its own.
    proxy = controller.worker
    assert proxy is not None
    wait_for(lambda: proxy.child_pid is not None)  # type: ignore[attr-defined]
    child = proxy.child_pid  # type: ignore[attr-defined]
    # The dashboard answers; FM27's folder is not there, so it asks, with Browse.
    text = page(url)
    assert "Browse…" in text
    fm = tmp_path / "Documents" / "Sports Interactive" / "Football Manager 27"
    (fm / "games").mkdir(parents=True)
    window.answer = str(fm)
    text = post(url, "/setup/browse", {"next": "/", "folder": ""})
    assert window.browsed == [None]  # the window's dialog, not tkinter's
    assert "FM27&#39;s folder is" in text or "FM27's folder is" in text
    assert str(fm) in (config_file.parent / "local.toml").read_text(encoding="utf-8")
    # A second start opens the first one's window and ends.
    second = Record()
    code = desktop.run("app", args_for(config_file, free_port()), shell=second.shell())
    assert code == 0 and second.windows == [] and second.trays == []
    wait_for(lambda: window.shown == 1)
    # Quit.
    app.quit()
    assert app.code == 0
    # The real worker's model check is in use, so there is a model download to stop too.
    assert controller.steps_run == [
        "model download", "sync", "save watcher", "worker", "web server",
    ]  # fmt: skip
    assert tray.stopped.is_set() and window.closed.is_set()
    assert not port_answers("127.0.0.1", port)
    assert not instance.held_elsewhere(config_file.parent / "data")
    assert proxy.child_pid is None  # type: ignore[attr-defined]
    assert has_exited(child)  # the worker process has exited: its memory is free


def test_update_now_starts_the_update_and_quits(config_file: Path) -> None:
    """Where an update in place works (faked here: Windows, not a git checkout), the web app
    gets Update now's hook: it starts Start.bat --update and quits facegen a moment later."""
    record = Record()
    updates: list[str] = []
    base = record.shell

    def shell() -> desktop.Shell:
        s = base()
        s.can_update = lambda: True
        s.start_update = lambda: updates.append("Start.bat --update")
        return s

    record.shell = shell  # type: ignore[method-assign]
    app = Running("app", args_for(config_file, free_port()), record, fake_gpu())
    hook = app.started.install_update
    assert hook is not None and app.controller.updates is not None
    hook()
    assert updates == ["Start.bat --update"]
    app.thread.join(60)
    assert not app.thread.is_alive() and app.code == 0 and app.controller.quitting
    # Without it (Linux, or a git checkout) there is no hook.
    without = Record()
    plain_shell = without.shell

    def no_update_here() -> desktop.Shell:
        s = plain_shell()
        s.can_update = lambda: False
        return s

    without.shell = no_update_here  # type: ignore[method-assign]
    plain = Running("app", args_for(config_file, free_port()), without, fake_gpu())
    assert plain.started.install_update is None
    plain.quit()


def test_the_tray_alone_opens_the_dashboard_in_the_browser(config_file: Path) -> None:
    record = Record()
    port = free_port()
    app = Running("tray", args_for(config_file, port), record, fake_gpu())
    assert record.windows == [] and record.browser == []  # nothing opens by itself
    wait_for(lambda: bool(record.trays) and record.trays[0].mode == "run")
    app.controller.open()
    assert record.browser == [f"http://127.0.0.1:{port}/"]
    view = app.controller.view()
    assert view.entry("pause").label == "Resume"  # type: ignore[union-attr]  # --paused
    app.quit()
    assert app.code == 0
    assert app.controller.steps_run == ["sync", "save watcher", "worker", "web server"]


def test_quit_during_a_sync_ends_its_process_at_once(
    config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 14: the app syncs in a process of its own (here `facegen sync-process` with a
    fake reader that takes a minute, sync_child.py); the tray answers meanwhile, and Quit
    ends the sync within seconds with the database as it was."""
    import sqlite3
    from contextlib import closing

    from facegen.persistence.settings import load_settings
    from facegen.runtime import sync_process

    state = tmp_path / "sync-child"
    program = [sys.executable, str(TESTS / "sync_child.py"), str(state), "slow"]
    command = sync_process.sync_command(program)
    monkeypatch.setattr(sync_process, "sync_command", lambda: command)
    settings = load_settings(defaults_path=config_file)
    games = settings.paths.games_dir
    assert games is not None
    games.mkdir(parents=True)
    (games / "career.fm").write_bytes(b"a made-up save")
    record = Record()
    app = Running("tray", args_for(config_file, free_port()), record, fake_gpu())
    controller = app.controller
    syncer = controller.syncer
    assert syncer is not None and syncer.process is not None  # type: ignore[attr-defined]
    with closing(sqlite3.connect(settings.paths.db_path)) as conn:
        before = list(conn.iterdump())
    assert controller.sync_now()
    wait_for(lambda: (state / "reading").is_file())
    pid = int((state / "reading").read_text(encoding="utf-8"))
    started = time.monotonic()
    view = controller.view()  # the tray, while the save is read
    assert time.monotonic() - started < 0.5
    assert view.entry("sync").label == "Syncing the latest save…"  # type: ignore[union-attr]
    started = time.monotonic()
    app.quit()
    assert time.monotonic() - started < 10.0
    assert app.code == 0 and app.controller.steps_run[0] == "sync"
    assert has_exited(pid)
    with closing(sqlite3.connect(settings.paths.db_path)) as conn:
        assert list(conn.iterdump()) == before


def test_without_webview2_the_dashboard_opens_in_the_browser(config_file: Path) -> None:
    record = Record(webview2=False)
    port = free_port()
    app = Running("app", args_for(config_file, port), record, fake_gpu())
    url = f"http://127.0.0.1:{port}/"
    assert record.browser == [url]  # once, at start
    (tray,) = record.trays
    wait_for(lambda: any(n.title == "Dashboard in the browser" for n in tray.notices))
    notice = next(n for n in tray.notices if n.title == "Dashboard in the browser")
    assert "WebView2" in notice.message and desktop.WEBVIEW2_URL in notice.message
    app.controller.open()  # Open Dashboard: the browser, not the window that cannot be
    assert record.browser == [url, url] and record.windows[0].shown == 0
    app.quit()


def test_started_in_the_tray_the_window_waits_hidden(config_file: Path) -> None:
    record = Record()
    app = Running("app", args_for(config_file, free_port(), minimized=True), record, fake_gpu())
    (window,) = record.windows
    assert window.hidden and window.shown == 0
    window.on_hide()  # the close button, later
    window.on_hide()
    (tray,) = record.trays
    wait_for(lambda: [n.kind for n in tray.notices] == ["hidden"])  # said once
    app.quit()


def test_a_port_in_use_stops_before_anything_starts(config_file: Path) -> None:
    record = Record()
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = int(taken.getsockname()[1])
        with pytest.raises(CannotStart, match="Another program already answers"):
            desktop.run("app", args_for(config_file, port), gpu=fake_gpu(), shell=record.shell())
    assert record.trays == [] and record.windows == []
    assert not instance.held_elsewhere(config_file.parent / "data")


def test_a_failure_after_the_server_started_leaves_nothing_running(config_file: Path) -> None:
    record = Record(tray_error=ModuleNotFoundError("No module named 'pystray'", name="pystray"))
    port = free_port()
    with pytest.raises(ModuleNotFoundError):
        desktop.run("tray", args_for(config_file, port), gpu=fake_gpu(), shell=record.shell())
    assert not port_answers("127.0.0.1", port)
    assert not instance.held_elsewhere(config_file.parent / "data")


def test_a_second_start_that_gets_no_answer_says_so(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(instance, "ANSWER_SECONDS", 0.2)
    lock = instance.InstanceLock(config_file.parent / "data")
    assert lock.acquire()
    try:
        with pytest.raises(CannotStart, match="did not answer"):
            desktop.run("app", args_for(config_file, free_port()), shell=Record().shell())
    finally:
        lock.release()


# -- a failure while starting, without a console -------------------------------------------------


def test_a_crash_is_logged_and_shown_with_the_logs_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = tmp_path / "data" / "logs" / "facegen.log"
    monkeypatch.setattr(desktop, "FALLBACK_LOG", fallback)
    shown: list[tuple[str, str]] = []
    try:
        raise ValueError("config/local.toml: not valid TOML")
    except ValueError as exc:
        path = desktop.report_crash(exc, lambda title, text: shown.append((title, text)))
    assert path == fallback and "ValueError: config/local.toml" in fallback.read_text()
    ((title, text),) = shown
    assert title == "facegen could not start"
    assert text.startswith("ValueError: config/local.toml: not valid TOML")
    assert str(fallback) in text


def test_a_crash_goes_to_the_log_file_once_there_is_one(tmp_path: Path) -> None:
    from facegen.service import log_to_file

    log = log_to_file(tmp_path)
    shown: list[str] = []
    path = desktop.report_crash(CannotStart("Port 8765 is in use."), lambda t, x: shown.append(x))
    assert path.resolve() == log.resolve()
    assert "facegen could not start" in log.read_text(encoding="utf-8")
    assert shown[0].startswith("Port 8765 is in use.\n\nThe details are in:")


def test_known_failures_are_said_in_plain_words() -> None:
    assert desktop.explain(CannotStart("Close it and start facegen again.")) == (
        "Close it and start facegen again."
    )
    missing = ModuleNotFoundError("No module named 'pystray'", name="pystray")
    text = desktop.explain(missing)
    if os.name == "nt":
        assert "Start.bat" in text
    else:
        assert "need Windows" in text and "facegen serve" in text
    other = ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
    assert "A part of facegen is missing (fastapi)" in desktop.explain(other)
    assert desktop.explain(OSError("disk full")) == "OSError: disk full"


def test_wrong_arguments_without_a_console_are_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop, "has_console", lambda: False)
    record = Record()
    monkeypatch.setattr(sys, "stderr", io.StringIO())  # argparse's usage goes nowhere
    assert desktop.main(["window"], shell=record.shell()) == 2
    assert record.errors == [("facegen", "Not a facegen command: window")]


def test_pythonw_counts_as_no_console_whatever_handles_it_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert desktop.has_console()  # pytest gives this process stdout and stderr
    monkeypatch.setattr(sys, "executable", r"C:\facegen\.venv\Scripts\pythonw.exe")
    assert not desktop.has_console()  # Start.bat's console closes right after starting it
    monkeypatch.setattr(sys, "executable", r"C:\facegen\.venv\Scripts\facegen-app.exe")
    assert not desktop.has_console()  # pythonw.exe's copy under facegen's name
    monkeypatch.setattr(sys, "executable", r"C:\facegen\.venv\Scripts\python.exe")
    assert desktop.has_console()
    monkeypatch.setattr(sys, "stdout", None)
    assert not desktop.has_console()


def test_without_a_console_the_output_goes_to_the_log(caplog: pytest.LogCaptureFixture) -> None:
    stream = desktop.LogStream(logging.getLogger("facegen.stdout"), logging.INFO)
    with caplog.at_level(logging.INFO, logger="facegen.stdout"):
        stream.write("Log: data/logs/facegen.log\nfacegen UI at http://127.0.0.1:8765/")
        assert [r.getMessage() for r in caplog.records] == ["Log: data/logs/facegen.log"]
        stream.write("\n\n")  # empty lines are dropped
        stream.write("half a line")
        stream.flush()
    assert [r.getMessage() for r in caplog.records] == [
        "Log: data/logs/facegen.log",
        "facegen UI at http://127.0.0.1:8765/",
        "half a line",
    ]
    assert stream.writable() and not stream.isatty() and stream.encoding == "utf-8"


def test_the_log_stream_does_not_loop_through_a_handler_writing_back() -> None:
    log = logging.getLogger("facegen.loop-test")
    log.propagate = False
    stream = desktop.LogStream(log, logging.WARNING)
    seen: list[str] = []

    class Echo(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())
            stream.write("from the handler\n")  # as logging's handleError would, to stderr

    log.addHandler(Echo())
    stream.write("one line\n")
    assert seen == ["one line"]


def test_the_heartbeat_answers_a_second_start_and_passes_notices(tmp_path: Path) -> None:
    opened: list[str] = []
    controller = AppController(worker=None, syncer=None, open_dashboard=lambda: opened.append("x"))
    tray = FakeTray(controller)
    beat = desktop.Refresher(controller, tray, tmp_path, every=60)
    beat.announce(Notice("hidden", "facegen is still running", "In the tray."))
    instance.request_open(tmp_path)
    beat.tick()
    assert opened == ["x"] and not (tmp_path / instance.OPEN_REQUEST_NAME).exists()
    assert [n.kind for n in tray.notices] == ["hidden"] and len(tray.views) == 1
    beat.tick()
    assert opened == ["x"] and len(tray.notices) == 1


def test_facegen_tray_on_a_pc_without_pystray_says_why(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("facegen")
    try:
        import pystray  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pystray is installed here")
    port = free_port()
    code = main(
        ["--config", str(config_file), "tray", "--port", str(port), "--paused"],
        gpu=fake_gpu(),
        out=io.StringIO(),
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "facegen could not start" in err and "pystray" in err
    assert not port_answers("127.0.0.1", port)
    log = (config_file.parent / "data" / "logs" / "facegen.log").read_text(encoding="utf-8")
    assert "ModuleNotFoundError" in log
