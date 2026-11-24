"""The WebView2 dashboard window and its parent proxy.

pywebview handles page requests on its UI thread through Python. A separate process keeps
those requests from waiting for the server or save parser's GIL and keeps .NET out of the
app process. Only the child imports pywebview.

`python -m kitshot.app_window --url URL [--hidden]` runs `AppWindow`. `WindowProcess`
exchanges one JSON object per line over stdin/stdout:

    parent to child   {"cmd": "show"}   {"cmd": "browse", "id": 1, "start": "C:\\..."|null}
                      {"cmd": "close"}   (also stdin EOF)
    child to parent   {"type": "ready"}   {"type": "hidden"}  (user closed the window)
                      {"type": "browsed", "id": 1, "path": "C:\\..."|null}
                      {"type": "unavailable", "reason": "..."}  (then exits)

Closing hides the window while the tray/worker continue; Open Dashboard shows or restarts
it, and Quit destroys it. Browse opens its folder dialog. Without WebView2, the child
reports unavailability so the app uses a browser; pywebview's IE fallback cannot run the UI.
"ready" says the child takes commands; it comes before pywebview starts its window, so it
does not mean the window started. A process that fails within QUICK_FAILURE_SECONDS of its
start counts as a window that could not start, whatever it said first.

The child takes an AppUserModelID of its own (APP_ID) and `--icon`, Kitshot's .ico. The
taskbar then shows Kitshot's icon on the window's button instead of Python's, and does
not group the window with other Python programs. Before Phase 17o it was facegen.app, so a
taskbar pin made by an older version shows as a button of its own until pinned again.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TextIO

from kitshot.platform import procs
from kitshot.platform.pipes import JsonWriter, decode, drain_lines

logger = logging.getLogger("kitshot")

TITLE = "Kitshot"
APP_ID = "kitshot.app"
"""The window process's AppUserModelID (module docstring)."""
SIZE = (1280, 860)
MIN_SIZE = (640, 480)
"""Phase 12b: below 960 px the sidebar keeps only its icons, so pages still fit at 640."""
NO_WEBVIEW2 = "Microsoft Edge WebView2 is not installed"
BROWSE_SECONDS = 600.0
"""How long Browse waits for the folder dialog (it waits for the user)."""
CLOSE_SECONDS = 10.0
"""How long Quit waits for the window process to end before ending it."""
QUICK_FAILURE_SECONDS = 5.0
"""A window process that ends sooner than this after starting failed to start."""
QUICK_FAILURES = 2
"""After this many quick failures in a row the window is given up: the browser is used."""


def use_app_id(os_name: str = os.name) -> None:
    """Give this process APP_ID on Windows (nothing elsewhere). A failure is only logged:
    the taskbar then shows the window as Python's, as before."""
    if os_name != "nt":
        return
    import ctypes

    try:
        result = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)  # type: ignore[attr-defined]
    except (OSError, AttributeError) as exc:
        logger.warning("window: no app ID of its own: %s", exc)
        return
    if result != 0:
        logger.warning("window: no app ID of its own (HRESULT 0x%08X)", result & 0xFFFFFFFF)


def webview2_missing() -> bool:
    """Whether pywebview found no WebView2 runtime on this PC (it would use IE's engine).
    False when it cannot tell: the window is tried."""
    try:
        from webview.platforms import winforms  # loads .NET through pythonnet
    except Exception as exc:  # noqa: BLE001 - the window is tried; it reports its own error
        logger.warning("window: could not check for WebView2: %s", exc)
        return False
    chromium = getattr(winforms, "is_chromium", None)
    return chromium is False


class AppWindow:
    """One pywebview window on `url`. `run` blocks in the main thread until `destroy`."""

    def __init__(
        self,
        url: str,
        *,
        hidden: bool = False,
        on_hide: Callable[[], None] | None = None,
        icon: str | None = None,
        webview: Any = None,
    ) -> None:
        if webview is None:
            import webview

        self._webview = webview
        self._on_hide = on_hide
        self._icon = icon
        self._closing_for_good = False
        self.window = webview.create_window(
            TITLE,
            url,
            width=SIZE[0],
            height=SIZE[1],
            min_size=MIN_SIZE,
            hidden=hidden,
            text_select=True,
            zoomable=True,
        )
        self.window.events.closing += self._on_closing

    def _on_closing(self) -> bool:
        """The window's close button: hide it (False keeps it), unless quitting."""
        if self._closing_for_good:
            return True
        threading.Thread(target=self.hide, name="kitshot-window-hide", daemon=True).start()
        if self._on_hide is not None:
            self._on_hide()
        return False

    def run(self) -> None:
        """The window's loop in this thread (the main one), until `destroy`. pywebview
        gives the window `icon` (title bar and taskbar), else the running program's."""
        self._webview.start(icon=self._icon)

    def show(self) -> None:
        try:
            self.window.show()
            self.window.restore()
        except Exception as exc:  # noqa: BLE001 - the window may be going away
            logger.warning("window: could not show it: %s", exc)

    def hide(self) -> None:
        try:
            self.window.hide()
        except Exception as exc:  # noqa: BLE001 - the window may be going away
            logger.warning("window: could not hide it: %s", exc)

    def destroy(self) -> None:
        """Close the window for good (Quit); `run` returns."""
        self._closing_for_good = True
        try:
            self.window.destroy()
        except Exception as exc:  # noqa: BLE001 - it may be closed already
            logger.warning("window: could not close it: %s", exc)

    def browse(self, start: str | None) -> str | None:
        """The window's folder dialog (Browse on the setup panel): the folder chosen, or None."""
        dialogs = getattr(self._webview, "FileDialog", None)
        kind = dialogs.FOLDER if dialogs is not None else self._webview.FOLDER_DIALOG
        try:
            chosen = self.window.create_file_dialog(kind, directory=start or "")
        except Exception as exc:  # noqa: BLE001 - the page says no folder was chosen
            logger.warning("window: the folder dialog failed: %s", exc)
            return None
        if not chosen:
            return None
        first = chosen if isinstance(chosen, str) else chosen[0]
        return str(first) or None


def child_main(
    url: str,
    hidden: bool,
    commands: Iterable[str],
    out: TextIO,
    *,
    icon: str | None = None,
    missing: Callable[[], bool] = webview2_missing,
    make: Callable[..., AppWindow] = AppWindow,
) -> int:
    """The window process: the window in this (main) thread, commands read in another."""
    writer = JsonWriter()

    def send(message: dict[str, Any]) -> None:
        try:
            writer.send(out, message)
        except (OSError, ValueError):  # the parent has gone; the end of stdin follows
            pass

    if missing():
        send({"type": "unavailable", "reason": NO_WEBVIEW2})
        return 3
    window = make(url, hidden=hidden, on_hide=lambda: send({"type": "hidden"}), icon=icon)

    def browse(message: dict[str, Any]) -> None:
        start = message.get("start")
        path = window.browse(start if isinstance(start, str) else None)
        send({"type": "browsed", "id": message.get("id"), "path": path})

    def read() -> None:
        for line in commands:
            message = decode(line)
            if message is None:
                continue
            match message.get("cmd"):
                case "show":
                    window.show()
                case "browse":  # waits for the user: in a thread of its own
                    threading.Thread(target=browse, args=(message,), daemon=True).start()
                case "close":
                    break
        window.destroy()

    threading.Thread(target=read, name="kitshot-window-commands", daemon=True).start()
    send({"type": "ready"})
    window.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from kitshot.platform.pipes import protocol_streams

    parser = argparse.ArgumentParser(prog="python -m kitshot.app_window")
    parser.add_argument("--url", required=True)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--icon", help="the window's .ico file")
    args = parser.parse_args(argv)
    use_app_id()
    commands, out = protocol_streams(sys.stdin, sys.stdout)
    return child_main(args.url, args.hidden, commands, out, icon=args.icon)


def window_command(url: str, hidden: bool, icon: str | None = None) -> list[str]:
    return [
        sys.executable,
        "-m",
        "kitshot.app_window",
        "--url",
        url,
        *(["--hidden"] if hidden else []),
        *(["--icon", icon] if icon else []),
    ]


class WindowProcess:
    """Window child proxy for the desktop shell: show, browse, destroy.

    `on_hide` is called when the user closes (hides) the window; `on_unavailable` with the
    reason when there can be no window (no WebView2, or it failed to start twice in a row),
    after which `show` does nothing and the app uses the browser."""

    def __init__(
        self,
        url: str,
        *,
        hidden: bool = False,
        on_hide: Callable[[], None] = lambda: None,
        on_unavailable: Callable[[str], None] = lambda reason: None,
        command: Callable[[str, bool], list[str]] = window_command,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self._hidden = hidden
        self._on_hide = on_hide
        self._on_unavailable = on_unavailable
        self._command = command
        self._clock = clock
        self._lock = threading.Lock()
        self._writer = JsonWriter()
        self._child: subprocess.Popen[str] | None = None
        self._started_at = 0.0
        self._quick_failures = 0
        self._closing = False
        self.unavailable: str | None = None
        self._ids = itertools.count(1)
        self._answers: dict[int, tuple[threading.Event, list[str | None]]] = {}
        self.ready = threading.Event()
        self.starts = 0

    @property
    def available(self) -> bool:
        return self.unavailable is None

    def start(self) -> None:
        """Start the window process (hidden or not, as asked)."""
        self._spawn(self._hidden)

    def _spawn(self, hidden: bool) -> None:
        child = procs.popen_json(
            self._command(self.url, hidden),
            detached=True,  # Ctrl+C belongs to the app, which sends "close".
        )
        with self._lock:
            self._child = child
            self._started_at = self._clock()
            self.ready.clear()
            self.starts += 1
        logger.info("window process %d started", child.pid)
        threading.Thread(
            target=self._read, args=(child,), name="kitshot-window-reader", daemon=True
        ).start()
        threading.Thread(
            target=self._drain, args=(child,), name="kitshot-window-stderr", daemon=True
        ).start()

    def _send(self, message: dict[str, Any]) -> bool:
        with self._lock:
            child = self._child
        if child is None or child.stdin is None or child.poll() is not None:
            return False
        try:
            self._writer.send(child.stdin, message)
        except (OSError, ValueError):
            return False
        return True

    def show(self) -> None:
        """Show the window; start its process again if it has ended."""
        if not self.available or self._closing:
            return
        if not self._send({"cmd": "show"}):
            logger.info("window process not running: starting it again")
            self._spawn(False)

    def browse(self, start: str | None, timeout: float = BROWSE_SECONDS) -> str | None:
        """The window's folder dialog: the folder chosen, or None."""
        ident = next(self._ids)
        done, answer = threading.Event(), [None]
        with self._lock:
            self._answers[ident] = (done, answer)
        try:
            if not self._send({"cmd": "browse", "id": ident, "start": start}):
                return None
            done.wait(timeout)
            return answer[0]
        finally:
            with self._lock:
                self._answers.pop(ident, None)

    def destroy(self, timeout: float = CLOSE_SECONDS) -> None:
        """Close the window for good (Quit): ask, wait, then end the process."""
        self._closing = True
        with self._lock:
            child = self._child
        if child is None:
            return
        self._send({"cmd": "close"})
        procs.close_streams(child.stdin)
        procs.wait_or_kill(
            child,
            timeout=timeout,
            on_timeout=lambda: logger.warning(
                "window process %d did not close; ended it", child.pid
            ),
        )

    def _read(self, child: subprocess.Popen[str]) -> None:
        assert child.stdout is not None
        for raw in child.stdout:
            message = decode(raw)
            if message is None:
                logger.info("window: %s", raw.rstrip())
                continue
            try:
                self._receive(message)
            except Exception as exc:  # noqa: BLE001 - a bad message must not end the reader
                logger.warning("window process: bad message %r (%s)", raw[:200], exc)
        code = child.wait()
        logger.info("window process %d ended (exit code %s)", child.pid, code)
        self._ended(child, code)

    def _receive(self, message: dict[str, Any]) -> None:
        match message.get("type"):
            case "ready":  # sent before the window starts: the failures are counted at the end
                self.ready.set()
            case "hidden":
                self._on_hide()
            case "browsed":
                with self._lock:
                    waiting = self._answers.get(int(message["id"]))
                if waiting is not None:
                    path = message.get("path")
                    waiting[1][0] = str(path) if path else None
                    waiting[0].set()
            case "unavailable":
                self._give_up(str(message.get("reason") or "the window could not be shown"))
            case _:
                logger.warning("window process: unknown message %r", message)

    def _ended(self, child: subprocess.Popen[str], code: int) -> None:
        with self._lock:
            if self._child is not child:
                return
            for done, _ in self._answers.values():  # Browse gets no folder
                done.set()
            quick = self._clock() - self._started_at < QUICK_FAILURE_SECONDS
            failed = not self._closing and self.available and code != 0
            if failed and quick:
                self._quick_failures += 1
            elif not quick:  # it ran: a failure later on starts the count again
                self._quick_failures = 0
            give_up = failed and self._quick_failures >= QUICK_FAILURES
        if give_up:
            self._give_up("the window could not be started (the log says why)")

    def _give_up(self, reason: str) -> None:
        with self._lock:
            if self.unavailable is not None:
                return
            self.unavailable = reason
        logger.warning("window: %s; the dashboard opens in the browser", reason)
        self._on_unavailable(reason)

    def _drain(self, child: subprocess.Popen[str]) -> None:
        assert child.stderr is not None
        drain_lines(child.stderr, lambda line: logger.info("window: %s", line))


if __name__ == "__main__":
    raise SystemExit(main())
