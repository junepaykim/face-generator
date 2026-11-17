"""The desktop app (Phase 11): `facegen app`, the dashboard in a window of its own with a tray
icon, and `facegen tray`, the tray icon alone (Open Dashboard opens the browser).

Both run what `facegen serve` runs (runtime.py), with the web server in a thread of its own,
and a tray icon over app_controller.AppController: Open Dashboard, Pause or Resume, Sync
Latest Save Now, Start with Windows and Quit, a state dot, and notifications. Closing the
window hides it; the worker carries on, and Quit is in the tray. Quit stops the model
download, the save watcher, the worker process (the graphics card's memory is free once it
has exited), the web server, then the tray and the window.

Started without a console, `pythonw.exe -m facegen.desktop app` (Start.bat, the shortcuts):

- one at a time: the first holds data/app.lock, and a second start asks it to open its
  window and ends (instance.py);
- what would go to the console (stdout, stderr: uvicorn's lines, the worker process's
  messages) goes to data/logs/facegen.log instead (`LogStream`);
- a failure while starting is shown in a message box with the log's path, not nothing.
  Its top-level dependencies use only the standard library, so even a broken installation
  (a package missing after an update) reaches that message box.

The window and the tray come from a `Shell`; the tests give fakes, so all of this runs on
Linux without pystray or pywebview.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Protocol

from facegen.locations import PROJECT_ROOT

if TYPE_CHECKING:
    from facegen.app_controller import AppController, Notice, StartupSwitch, View
    from facegen.datasource.base import DataSource
    from facegen.runtime_dependencies import Gpu

FALLBACK_LOG = PROJECT_ROOT / "data" / "logs" / "facegen.log"
"""Where a failure is written when it happens before the settings say where the log is."""
REFRESH_SECONDS = 1.0
"""How often the tray reads the state, answers a second start and shows notifications."""
QUIT_WAIT_SECONDS = 120.0
"""How long the end of `run` waits for Quit: the worker's 60 s, then the server's."""
WEBVIEW2_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"

logger = logging.getLogger("facegen")


class TrayLike(Protocol):
    def run(self) -> None: ...
    def stop(self) -> None: ...
    def show(self, view: View) -> None: ...
    def notify(self, notice: Notice) -> None: ...


class WindowLike(Protocol):
    """The window, already started in a process of its own (app_window.WindowProcess)."""

    def show(self) -> None: ...
    def destroy(self) -> None: ...
    def browse(self, start: str | None) -> str | None: ...


MakeWindow = Callable[[str, bool, Callable[[], None], Callable[[str], None]], WindowLike]


def _make_tray(controller: AppController) -> TrayLike:
    from facegen.tray import TrayIcon

    return TrayIcon(controller)


def _make_window(
    url: str, hidden: bool, on_hide: Callable[[], None], on_unavailable: Callable[[str], None]
) -> WindowLike:
    from facegen.app_window import WindowProcess

    window = WindowProcess(url, hidden=hidden, on_hide=on_hide, on_unavailable=on_unavailable)
    window.start()
    return window


def _startup_switch(data_dir: Path, config: Path | None) -> StartupSwitch | None:
    """Start with Windows, on Windows only."""
    if os.name != "nt":
        return None
    from facegen import shortcuts

    folder = shortcuts.windows_folders().get("startup")
    if not folder:
        return None
    return shortcuts.StartupShortcut(folder, shortcuts.this_app(data_dir, config))


def _open_browser(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


def show_error(title: str, text: str) -> None:
    """A message box on Windows (a top-most error box: there is no console to read);
    elsewhere the text on stderr."""
    if os.name == "nt":
        import ctypes

        flags = 0x10 | 0x10000 | 0x40000  # MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST
        try:
            ctypes.windll.user32.MessageBoxW(None, text, title, flags)  # type: ignore[attr-defined]
            return
        except (OSError, AttributeError):
            pass
    if sys.stderr is not None:
        print(f"{title}: {text}", file=sys.stderr, flush=True)


@dataclass
class Shell:
    """The window, the tray and this PC's services that `run` uses; the tests give fakes."""

    make_tray: Callable[[AppController], TrayLike] = _make_tray
    make_window: MakeWindow = _make_window
    startup: Callable[[Path, Path | None], StartupSwitch | None] = _startup_switch
    open_browser: Callable[[str], object] = _open_browser
    show_error: Callable[[str, str], None] = show_error


class LogStream(io.TextIOBase):
    """A text stream whose lines go to a logger: stdout and stderr when there is no console
    (pythonw.exe), so they reach the log file."""

    def __init__(self, log: logging.Logger, level: int) -> None:
        super().__init__()
        self._log = log
        self._level = level
        self._lock = threading.Lock()
        self._buffer = ""
        self._busy = threading.local()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if getattr(self._busy, "on", False):  # a handler writing back into us: drop it
            return len(text)
        self._busy.on = True
        try:
            with self._lock:
                self._buffer += text
                *lines, self._buffer = self._buffer.split("\n")
            for line in lines:
                if line.strip():
                    self._log.log(self._level, "%s", line.rstrip("\r"))
        finally:
            self._busy.on = False
        return len(text)

    def flush(self) -> None:
        with self._lock:
            rest, self._buffer = self._buffer, ""
        if rest.strip():
            self._log.log(self._level, "%s", rest.rstrip("\r"))


def has_console() -> bool:
    """False under pythonw.exe, whatever handles it was given (Start.bat's console closes
    straight after starting it), and whenever there is no stdout or stderr."""
    if PureWindowsPath(sys.executable or "").name.lower() == "pythonw.exe":
        return False
    return sys.stdout is not None and sys.stderr is not None


def output_to_log() -> None:
    """Send stdout and stderr to the "facegen" logger (so to data/logs/facegen.log)."""
    sys.stdout = LogStream(logging.getLogger("facegen.stdout"), logging.INFO)
    sys.stderr = LogStream(logging.getLogger("facegen.stderr"), logging.WARNING)


class Refresher:
    """Every second: the tray shows the controller's view, the new notifications, and a second
    start's request to open the window is answered."""

    def __init__(
        self,
        controller: AppController,
        tray: TrayLike,
        data_dir: Path,
        *,
        every: float | None = None,
    ) -> None:
        self.controller = controller
        self.tray = tray
        self.data_dir = data_dir
        self.every = REFRESH_SECONDS if every is None else every
        self._queued: list[Notice] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def announce(self, notice: Notice) -> None:
        """A notification of our own (no WebView2, the window hidden), shown at the next tick."""
        with self._lock:
            self._queued.append(notice)

    def tick(self) -> None:
        from facegen.instance import take_open_request

        if take_open_request(self.data_dir):
            logger.info("a second start asked for the dashboard")
            self.controller.open()
        self.tray.show(self.controller.view())
        with self._lock:
            queued, self._queued = self._queued, []
        for notice in queued + self.controller.notices():
            logger.info("notice: %s: %s", notice.title, notice.message)
            self.tray.notify(notice)

    def _main(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self.tick()
            except Exception as exc:
                logger.error("tray refresh failed", exc_info=exc)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="facegen-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(5)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of `facegen app` and `facegen tray`."""
    parser.add_argument("--port", type=int, help="port (default: ui.port in the settings)")
    parser.add_argument("--paused", action="store_true", help="start with the worker paused")
    parser.add_argument(
        "--minimized", action="store_true", help="app: start in the tray, without the window"
    )


@dataclass
class Started:
    """What `run` started, for the tests to look at."""

    controller: AppController | None = None
    window: WindowLike | None = None
    tray: TrayLike | None = None
    url: str | None = None
    notices: list[Notice] = field(default_factory=list)


def run(
    mode: str,
    args: argparse.Namespace,
    *,
    source_factory: Callable[[], DataSource] | None = None,
    gpu: Gpu | None = None,
    shell: Shell | None = None,
    started: Started | None = None,
) -> int:
    """`facegen app` (mode "app") or `facegen tray` (mode "tray"), until Quit. Raises on a
    failure while starting (`main` shows it); returns 0 after Quit, and when another facegen
    app was asked to open its window instead."""
    from facegen import db, instance
    from facegen.app_controller import AppController, Notice
    from facegen.app_window import NO_WEBVIEW2
    from facegen.runtime import (
        DEFAULT_BROWSE,
        CannotStart,
        ServerThread,
        build_runtime,
        port_answers,
    )
    from facegen.runtime_dependencies import Gpu, default_source
    from facegen.service_logging import log_to_file
    from facegen.settings import load_settings

    shell = shell or Shell()
    started = started if started is not None else Started()
    config: Path | None = args.config
    paths = load_settings(defaults_path=config).paths
    lock = instance.InstanceLock(paths.data_dir)
    if not lock.acquire():
        # Another facegen app runs: it opens its window. Nothing is written to its log file.
        instance.request_open(paths.data_dir)
        if instance.wait_taken(paths.data_dir):
            print("facegen is running already: its window was asked to open.", flush=True)
            return 0
        raise CannotStart(
            "facegen is running already, but it did not answer within"
            f" {instance.ANSWER_SECONDS:.0f} s. If it is still starting, wait a moment and"
            " try again; if it seems stuck, end it in Task Manager and start facegen again."
        )
    log_path = log_to_file(paths.data_dir)
    console = has_console()
    if not console:
        output_to_log()
    logger.info("facegen %s starting (pid %d)", mode, os.getpid())
    rt = None
    window: WindowLike | None = None
    tray: TrayLike | None = None
    refresher: Refresher | None = None
    server: ServerThread | None = None
    controller: AppController | None = None
    try:
        conn = db.connect(paths.db_path)  # migrated here, once, as every command does
        try:
            settings = load_settings(conn, defaults_path=config)
        finally:
            conn.close()
        want_window = mode == "app"
        notices: list[Notice] = []
        holder: dict[str, WindowLike] = {}
        window_gone = threading.Event()  # no window after all: the browser takes its place

        def browse_in_window(start: str | None) -> str | None:
            from facegen import folder_setup

            win = holder.get("window")
            if win is not None and not window_gone.is_set():
                return win.browse(start)
            return folder_setup.browse(start) if folder_setup.dialog_available() else None

        rt = build_runtime(
            settings,
            source_factory or default_source,
            gpu or Gpu(),
            config=config,
            port=args.port,
            paused=args.paused,
            browse=browse_in_window if want_window else DEFAULT_BROWSE,
            echo_stderr=console,
        )
        if port_answers(rt.host, rt.port):
            raise CannotStart(
                f"Another program already answers at {rt.url} (facegen serve, perhaps). Close"
                " it and start facegen again."
            )
        server = ServerThread(rt.app, rt.host, rt.port)
        server.start()
        rt.start()
        print(f"Log: {log_path}", flush=True)
        print(f"facegen UI at {rt.url}  (Quit is in the tray icon's menu)", flush=True)
        logger.info("facegen %s: UI at %s", mode, rt.url)

        def open_dashboard() -> None:
            if window is not None and not window_gone.is_set():
                window.show()
            else:
                shell.open_browser(rt.url)

        def close_shell() -> None:
            if refresher is not None:
                refresher.stop()
            if window is not None:
                window.destroy()
            if tray is not None:
                tray.stop()

        controller = AppController(
            worker=rt.worker,
            syncer=rt.syncer,
            poller=rt.poller,
            open_dashboard=open_dashboard,
            quit_steps=[*rt.stop_steps(), ("web server", server.stop)],
            on_quit_done=close_shell,
            startup=shell.startup(paths.data_dir, config),
        )
        said_hidden: list[bool] = []

        def on_hide() -> None:
            if refresher is not None and not said_hidden:
                said_hidden.append(True)
                refresher.announce(
                    Notice(
                        "hidden",
                        "facegen is still running",
                        "It goes on drawing in the background. Open the dashboard or Quit from"
                        " the tray icon.",
                    )
                )

        def on_unavailable(reason: str) -> None:
            """No window after all (no WebView2): the browser instead, said once."""
            if window_gone.is_set():
                return
            window_gone.set()
            text = f"{reason}, so facegen has no window of its own: the dashboard opens in your"
            text += " browser."
            if reason == NO_WEBVIEW2:
                text += f" Install WebView2 from {WEBVIEW2_URL}."
            notice = Notice("setup", "Dashboard in the browser", text)
            notices.append(notice)
            if refresher is not None:
                refresher.announce(notice)
            if not args.minimized and not controller_quitting():
                shell.open_browser(rt.url)

        def controller_quitting() -> bool:
            return controller is not None and controller.quitting

        tray = shell.make_tray(controller)
        refresher = Refresher(controller, tray, paths.data_dir)
        if want_window:
            window = shell.make_window(rt.url, bool(args.minimized), on_hide, on_unavailable)
            holder["window"] = window
        started.controller, started.window, started.tray, started.url = (
            controller,
            window,
            tray,
            rt.url,
        )
        started.notices = notices
        refresher.start()
        # The tray has this process's main thread; the window runs in a process of its own.
        tray.run()  # until Quit stops it
        # The loop also ends without Quit (Windows logging off): stop everything the same way.
        controller.quit()
        if not controller.wait_quit(QUIT_WAIT_SECONDS):
            logger.error("quitting did not finish within %.0f s", QUIT_WAIT_SECONDS)
        logger.info("facegen %s quit", mode)
        return 0
    except BaseException:
        # Started parts must not outlive a failure: the worker process holds the card.
        if controller is not None:
            controller.quit()
            controller.wait_quit(QUIT_WAIT_SECONDS)
        else:
            _stop_quietly(rt, server)
        raise
    finally:
        if refresher is not None:
            refresher.stop()
        lock.release()


def _stop_quietly(rt: Any, server: Any) -> None:
    for name, step in rt.stop_steps() if rt is not None else []:
        try:
            step()
        except Exception as exc:
            logger.error("stopping the %s failed", name, exc_info=exc)
    if server is not None:
        try:
            server.stop()
        except Exception as exc:
            logger.error("stopping the web server failed", exc_info=exc)


def _log_file() -> Path | None:
    for handler in logger.handlers:
        name = getattr(handler, "baseFilename", None)
        if name:
            return Path(name)
    return None


def report_crash(exc: BaseException, show: Callable[[str, str], None] = show_error) -> Path:
    """Write the failure to the log (or, before there is one, to FALLBACK_LOG) and show it in
    a message box with the log's path. Returns the path written to."""
    path = _log_file()
    if path is not None:
        logger.error("facegen could not start", exc_info=exc)
    else:
        text = "".join(traceback.format_exception(exc))
        path = FALLBACK_LOG
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR facegen could not start\n")
                f.write(text)
        except OSError:
            import tempfile

            path = Path(tempfile.gettempdir()) / "facegen-error.log"
            with path.open("a", encoding="utf-8") as f:
                f.write(text)
    show("facegen could not start", f"{explain(exc)}\n\nThe details are in:\n{path}")
    return path


def explain(exc: BaseException) -> str:
    """The failure in plain words where it is a known one."""
    from_runtime = type(exc).__name__ in ("CannotStart", "ServerStartError", "SettingsError")
    if from_runtime:
        return str(exc)
    if isinstance(exc, ModuleNotFoundError) and exc.name in ("pystray", "webview", "clr"):
        if os.name != "nt":
            return (
                f"facegen's window and tray icon need Windows ({exc.name} is installed there"
                " only). Here, run `facegen serve` and open the dashboard in a browser."
            )
        return (
            f"{exc.name} is not installed. Start facegen with Start.bat, which installs what is"
            " missing (or run `uv sync --inexact` in facegen's folder)."
        )
    if isinstance(exc, ModuleNotFoundError):
        return (
            f"A part of facegen is missing ({exc.name}). Start facegen with Start.bat, which"
            " installs what is missing."
        )
    return f"{type(exc).__name__}: {exc}"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m facegen.desktop", description="facegen's window and tray icon"
    )
    parser.add_argument("--config", type=Path, help="defaults file (default: config/default.toml)")
    sub = parser.add_subparsers(dest="mode", required=True)
    add_arguments(sub.add_parser("app", help="the dashboard in a window, with a tray icon"))
    add_arguments(sub.add_parser("tray", help="the tray icon; the dashboard in the browser"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, shell: Shell | None = None) -> int:
    """`pythonw.exe -m facegen.desktop app|tray`: no console, and a message box if starting
    fails."""
    shell = shell or Shell()
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        if exc.code and not has_console():
            shell.show_error("facegen", f"Not a facegen command: {' '.join(argv or sys.argv[1:])}")
        return int(exc.code or 0)
    try:
        if has_console():
            from facegen.output import utf8_output

            utf8_output(sys.stdout, sys.stderr)
        return run(args.mode, args, shell=shell)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - a window-less program must say why it ended
        report_crash(exc, shell.show_error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
