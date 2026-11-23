"""The dashboard's window in a process of its own (app_window.WindowProcess and child_main):
the pipe protocol both ways, in real processes, with a stand-in for pywebview
(fake_window_child.py)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from support.paths import TESTS

from facegen.desktop import app_window
from facegen.desktop.app_window import NO_WEBVIEW2, WindowProcess

URL = "http://127.0.0.1:8765/"


def wait_for(check: Callable[[], bool], timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


class Seen:
    def __init__(self) -> None:
        self.hidden = 0
        self.unavailable: list[str] = []


def window(tmp_path: Path, mode: str, *, hidden: bool = False) -> tuple[WindowProcess, Seen, Path]:
    notes = tmp_path / "notes.txt"
    seen = Seen()

    def command(url: str, hide: bool) -> list[str]:
        script = str(TESTS / "fake_window_child.py")
        return [
            sys.executable,
            script,
            str(notes),
            mode,
            "--url",
            url,
            *(["--hidden"] if hide else []),
        ]

    def hid() -> None:
        seen.hidden += 1

    w = WindowProcess(
        URL, hidden=hidden, on_hide=hid, on_unavailable=seen.unavailable.append, command=command
    )
    return w, seen, notes


def lines(notes: Path) -> list[str]:
    return notes.read_text(encoding="utf-8").splitlines() if notes.exists() else []


def test_show_browse_and_quit_through_the_pipe(tmp_path: Path) -> None:
    w, seen, notes = window(tmp_path, "ok")
    w.start()
    assert w.ready.wait(20)
    assert lines(notes) == [f"window {URL} hidden=False"]
    w.show()
    wait_for(lambda: "show" in lines(notes))
    chosen = w.browse(r"C:\Users\me")
    assert chosen == r"C:\Users\me\Documents\Sports Interactive\Football Manager 27"
    assert w.browse("cancel") is None  # the dialog was cancelled
    assert w.browse(None) is not None
    assert [x for x in lines(notes) if x.startswith("browse")] == [
        r"browse C:\Users\me",
        "browse cancel",
        "browse None",
    ]
    w.destroy()
    assert lines(notes)[-1] == "closed"
    assert w.available and seen.unavailable == [] and w.starts == 1


def test_started_in_the_tray_the_window_is_hidden(tmp_path: Path) -> None:
    w, _, notes = window(tmp_path, "ok", hidden=True)
    w.start()
    assert w.ready.wait(20)
    assert lines(notes) == [f"window {URL} hidden=True"]
    w.destroy()


def test_closing_the_window_tells_the_app(tmp_path: Path) -> None:
    w, seen, _ = window(tmp_path, "hide")
    w.start()
    wait_for(lambda: seen.hidden == 1)
    w.destroy()


def test_without_webview2_there_is_no_window(tmp_path: Path) -> None:
    w, seen, notes = window(tmp_path, "nowebview")
    w.start()
    wait_for(lambda: seen.unavailable == [NO_WEBVIEW2])
    assert not w.available
    w.show()  # nothing to show: the app uses the browser
    time.sleep(0.3)
    assert w.starts == 1 and lines(notes) == []
    assert w.browse(None) is None
    w.destroy()


def test_a_window_process_that_ended_is_started_again_by_show(tmp_path: Path) -> None:
    w, seen, notes = window(tmp_path, "ok")
    w.start()
    assert w.ready.wait(20)
    first = w._child
    assert first is not None
    first.kill()  # as Task Manager would
    wait_for(lambda: first.poll() is not None)
    time.sleep(0.2)  # the reader notes the end
    w.show()
    wait_for(lambda: w.ready.is_set() and w.starts == 2)
    assert lines(notes)[-1] == f"window {URL} hidden=False"  # a new one, shown
    assert seen.unavailable == []
    w.destroy()


def test_a_window_that_cannot_start_twice_gives_way_to_the_browser(tmp_path: Path) -> None:
    w, seen, _ = window(tmp_path, "crash")
    w.start()
    wait_for(lambda: w._child is not None and w._child.poll() is not None)
    time.sleep(0.2)
    assert seen.unavailable == []  # once may be bad luck
    w.show()
    wait_for(lambda: seen.unavailable != [])
    assert seen.unavailable == ["the window could not be started (the log says why)"]
    assert w.starts == 2 and not w.available


def test_a_window_that_fails_after_saying_ready_gives_way_to_the_browser(tmp_path: Path) -> None:
    # "ready" comes before pywebview starts the window, so it is no sign the window started.
    w, seen, _ = window(tmp_path, "ready-crash")
    w.start()
    wait_for(lambda: w._child is not None and w._child.poll() is not None)
    time.sleep(0.2)
    assert seen.unavailable == []
    w.show()
    wait_for(lambda: seen.unavailable != [])
    assert seen.unavailable == ["the window could not be started (the log says why)"]
    assert w.starts == 2 and not w.available


def test_a_window_that_ran_for_a_while_starts_its_failure_count_again(tmp_path: Path) -> None:
    clock = [0.0]
    w, seen, _ = window(tmp_path, "crash")
    w._clock = lambda: clock[0]
    w.start()
    wait_for(lambda: w._child is not None and w._child.poll() is not None)
    time.sleep(0.2)
    assert w._quick_failures == 1
    # The next one fails only when told to, once it has run past QUICK_FAILURE_SECONDS.
    fail_when_told = "import sys; sys.stdin.readline(); raise SystemExit(1)"
    w._command = lambda url, hidden: [sys.executable, "-c", fail_when_told]
    w.show()
    second = w._child
    assert second is not None and w.starts == 2
    clock[0] = app_window.QUICK_FAILURE_SECONDS + 1
    assert w._send({"cmd": "show"})
    wait_for(lambda: second.poll() is not None)
    time.sleep(0.2)
    assert w._quick_failures == 0 and seen.unavailable == [] and w.available


def test_browse_gets_no_folder_when_the_window_process_ends(tmp_path: Path) -> None:
    w, _, _ = window(tmp_path, "ok")
    w.start()
    assert w.ready.wait(20)
    child = w._child
    assert child is not None
    answer: list[str | None] = ["unset"]

    def ask() -> None:
        answer[0] = w.browse("slow", timeout=30)

    # Make the child slow to answer, then end it while Browse waits.
    original = w._send

    def send_then_kill(message):  # type: ignore[no-untyped-def]
        sent = original(message)
        if message.get("cmd") == "browse":
            child.kill()
        return sent

    w._send = send_then_kill  # type: ignore[method-assign]
    t = threading.Thread(target=ask)
    t.start()
    t.join(20)
    assert not t.is_alive() and answer[0] in (
        None,
        r"C:\Users\me\Documents\Sports Interactive\Football Manager 27",
    )
    w.destroy()


def test_the_window_command_runs_this_python() -> None:
    assert app_window.window_command(URL, True) == [
        sys.executable,
        "-m",
        "facegen.app_window",
        "--url",
        URL,
        "--hidden",
    ]
    assert "--hidden" not in app_window.window_command(URL, False)
    assert app_window.window_command(URL, False, r"C:\facegen\data\app\facegen.ico")[-2:] == [
        "--icon",
        r"C:\facegen\data\app\facegen.ico",
    ]


@pytest.mark.parametrize("missing", [True, False])
def test_child_main_in_this_process(missing: bool) -> None:
    """child_main itself: no WebView2 is said and ends it; else ready, then close ends it."""
    import io

    out = io.StringIO()
    made: list[object] = []

    class Win:
        def __init__(self, url: str, *, hidden: bool, on_hide, icon: str | None) -> None:  # type: ignore[no-untyped-def]
            self.closed = threading.Event()
            self.icon = icon
            made.append(self)

        def run(self) -> None:
            self.closed.wait(10)

        def show(self) -> None: ...

        def destroy(self) -> None:
            self.closed.set()

        def browse(self, start: str | None) -> str | None:
            return None

    code = app_window.child_main(
        URL,
        False,
        iter(['{"cmd": "close"}\n']),
        out,
        icon="facegen.ico",
        missing=lambda: missing,
        make=Win,
    )
    if missing:
        assert code == 3 and made == []
        assert '"type":"unavailable"' in out.getvalue()
    else:
        assert code == 0 and len(made) == 1
        assert made[0].icon == "facegen.ico"  # type: ignore[attr-defined]
        assert out.getvalue().splitlines() == ['{"type":"ready"}']
