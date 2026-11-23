"""The tray icon and the window (Phase 11) are thin layers over AppController; here with
stand-ins for pystray and pywebview, which only Windows installs."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, Self

from support.desktop import FakeStartup, FakeSyncer, FakeWorker, status

from facegen.desktop import app_window
from facegen.desktop.app_controller import NOTICE_CHARS, NOTICE_TITLE_CHARS, AppController, Notice
from facegen.desktop.app_window import AppWindow
from facegen.desktop.tray import TrayIcon


def wait_for(check, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


# -- a stand-in for pystray ------------------------------------------------------------------


class FakeMenuItem:
    def __init__(self, text, action, checked=None, default=False, enabled=True):  # type: ignore[no-untyped-def]
        self.text, self.action, self.default, self.enabled = text, action, default, enabled
        self.checked = checked(self) if callable(checked) else checked


class FakeMenu:
    SEPARATOR = object()

    def __init__(self, *items: Any) -> None:
        self.items = items

    def __iter__(self):  # type: ignore[no-untyped-def]
        source = self.items[0]() if len(self.items) == 1 and callable(self.items[0]) else self.items
        return iter(list(source))


class FakeIcon:
    HAS_NOTIFICATION = True

    def __init__(self, name: str, icon: Any, title: str, menu: FakeMenu) -> None:
        self.name, self.icon, self.title, self.menu = name, icon, title, menu
        self.visible = False
        self.notified: list[tuple[str, str]] = []
        self.menu_updates = 0
        self.stopped = False
        self.mode: str | None = None

    def update_menu(self) -> None:
        self.menu_updates += 1

    def notify(self, message: str, title: str) -> None:
        self.notified.append((title, message))

    def run(self, setup) -> None:  # type: ignore[no-untyped-def]
        self.mode = "run"
        setup(self)

    def stop(self) -> None:
        self.stopped = True


PYSTRAY = SimpleNamespace(Icon=FakeIcon, Menu=FakeMenu, MenuItem=FakeMenuItem)


def tray_for(worker: FakeWorker, syncer: FakeSyncer, **kw: Any) -> tuple[TrayIcon, AppController]:
    opened: list[str] = []
    controller = AppController(
        worker=worker, syncer=syncer, open_dashboard=lambda: opened.append("open"), **kw
    )
    controller.opened = opened  # type: ignore[attr-defined]
    return TrayIcon(controller, pystray=PYSTRAY), controller


def items(tray: TrayIcon) -> list[FakeMenuItem]:
    return [i for i in tray.icon.menu if i is not FakeMenu.SEPARATOR]


def test_the_menu_is_the_controllers_view() -> None:
    tray, _ = tray_for(FakeWorker(status("idle", detail="the queue is empty")), FakeSyncer())
    tray.run()
    assert tray.icon.mode == "run" and tray.icon.visible
    texts = [i.text for i in items(tray)]
    assert texts == [
        "Idle: the queue is empty",
        "Open Dashboard",
        "Pause",
        "Sync Latest Save Now",
        "Quit",
    ]
    status_item = items(tray)[0]
    assert status_item.action is None and not status_item.enabled
    assert [i.default for i in items(tray)] == [False, True, False, False, False]
    assert tray.icon.title == "Kitshot: idle: the queue is empty"


def test_menu_clicks_call_the_controller_without_holding_the_icon() -> None:
    worker, syncer = FakeWorker(status("idle")), FakeSyncer()
    switch = FakeStartup()
    tray, controller = tray_for(worker, syncer, startup=switch)
    tray.run()
    by_text = {i.text: i for i in items(tray)}
    by_text["Open Dashboard"].action()
    by_text["Pause"].action()
    by_text["Sync Latest Save Now"].action()
    by_text["Start with Windows"].action()
    wait_for(lambda: worker.calls == ["pause"] and syncer.starts == 1 and switch.sets == [True])
    wait_for(lambda: controller.opened == ["open"])  # type: ignore[attr-defined]
    # After a click the menu follows at once: Resume now, Start with Windows ticked.
    wait_for(lambda: "Resume" in [i.text for i in items(tray)])
    ticked = next(i for i in items(tray) if i.text == "Start with Windows")
    assert ticked.checked is True


def test_quit_from_the_menu_runs_the_controllers_quit() -> None:
    done: list[str] = []
    tray, controller = tray_for(
        FakeWorker(status("idle")),
        FakeSyncer(),
        quit_steps=[("worker", lambda: done.append("worker"))],
        on_quit_done=lambda: done.append("closed"),
    )
    tray.run()
    next(i for i in items(tray) if i.text == "Quit").action()
    wait_for(lambda: controller.wait_quit(0) and done == ["worker", "closed"])


def test_show_changes_only_what_changed() -> None:
    worker = FakeWorker(status("idle"))
    tray, controller = tray_for(worker, FakeSyncer())
    tray.run()
    first_icon, updates = tray.icon.icon, tray.icon.menu_updates
    tray.show(controller.view())
    assert tray.icon.icon is first_icon and tray.icon.menu_updates == updates
    worker.s = status("gpu-fault")
    tray.show(controller.view())
    assert tray.icon.icon is not first_icon  # the dot turned red
    assert tray.icon.title == "Kitshot: drawing stopped: the graphics driver failed"
    assert tray.icon.menu_updates == updates + 1


def test_notifications_wait_for_the_icon_to_be_shown() -> None:
    tray, _ = tray_for(FakeWorker(), FakeSyncer())
    early = Notice("setup", "Dashboard in the browser", "No WebView2.")
    tray.notify(early)  # the icon is not up yet: kept, not lost
    assert tray.icon.notified == []
    tray.run()
    assert tray.icon.notified == [("Dashboard in the browser", "No WebView2.")]
    tray.notify(Notice("pictures", "New pictures", "3 new pictures: restart FM27."))
    assert tray.icon.notified[-1] == ("New pictures", "3 new pictures: restart FM27.")
    assert len(tray.icon.notified) == 2
    tray.stop()
    assert tray.icon.stopped and not tray.icon.visible


def test_shell_notifications_fit_windows_buffers_even_before_the_icon_is_ready() -> None:
    tray, _ = tray_for(FakeWorker(), FakeSyncer())
    notice = Notice("setup", "😀" * 80, "Window failed: " + "😀" * 300)
    tray.notify(notice)
    tray.run()
    tray.notify(notice)
    assert len(tray.icon.notified) == 2
    for title, message in tray.icon.notified:
        assert len(title.encode("utf-16-le")) <= NOTICE_TITLE_CHARS * 2
        assert len(message.encode("utf-16-le")) <= NOTICE_CHARS * 2
        assert title.endswith("…") and message.endswith("…")


# -- a stand-in for pywebview ------------------------------------------------------------------


class FakeEvent:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> Self:
        self.handlers.append(handler)
        return self

    def fire(self) -> bool:
        """pywebview's rule: closing is cancelled when a handler returns False."""
        return not any(h() is False for h in self.handlers)


class FakeWindow:
    def __init__(self, **kw: Any) -> None:
        self.kw = kw
        self.events = SimpleNamespace(closing=FakeEvent())
        self.calls: list[str] = []
        self.dialog_answer: Any = None
        self.closed = threading.Event()

    def show(self) -> None:
        self.calls.append("show")

    def restore(self) -> None:
        self.calls.append("restore")

    def hide(self) -> None:
        self.calls.append("hide")

    def destroy(self) -> None:
        self.calls.append("destroy")
        if self.events.closing.fire():
            self.closed.set()

    def create_file_dialog(self, kind: Any, directory: str = "") -> Any:
        self.calls.append(f"dialog {kind} {directory!r}")
        if isinstance(self.dialog_answer, Exception):
            raise self.dialog_answer
        return self.dialog_answer


class FakeWebview:
    class FileDialog:
        FOLDER = 20

    def __init__(self) -> None:
        self.window: FakeWindow | None = None
        self.started: dict[str, Any] | None = None

    def create_window(self, title: str, url: str, **kw: Any) -> FakeWindow:
        self.window = FakeWindow(title=title, url=url, **kw)
        return self.window

    def start(self, **kw: Any) -> None:
        self.started = kw
        assert self.window is not None
        self.window.closed.wait(5)


def test_the_window_shows_the_dashboard() -> None:
    webview = FakeWebview()
    AppWindow("http://127.0.0.1:8765/", webview=webview)
    assert webview.window is not None
    kw = webview.window.kw
    assert (kw["title"], kw["url"]) == ("Kitshot", "http://127.0.0.1:8765/")
    assert kw["text_select"] is True and kw["hidden"] is False
    assert kw["min_size"][0] <= kw["width"]


def test_the_window_has_kitshots_icon_and_app_id() -> None:
    webview = FakeWebview()
    win = AppWindow("http://127.0.0.1:8765/", icon=r"C:\facegen\kitshot.ico", webview=webview)
    assert webview.window is not None
    webview.window.closed.set()
    win.run()
    assert webview.started == {"icon": r"C:\facegen\kitshot.ico"}
    assert app_window.APP_ID == "facegen.app"  # kept until Phase 17o
    app_window.use_app_id(os_name="posix")  # nothing to do off Windows


def test_closing_the_window_hides_it_and_quit_closes_it_for_good() -> None:
    webview = FakeWebview()
    hidden: list[str] = []
    win = AppWindow(
        "http://127.0.0.1:8765/", hidden=True, on_hide=lambda: hidden.append("x"), webview=webview
    )
    fake = webview.window
    assert fake is not None and fake.kw["hidden"] is True  # started in the tray
    assert fake.events.closing.fire() is False  # the close button: kept
    wait_for(lambda: "hide" in fake.calls)
    assert hidden == ["x"]
    win.show()
    assert fake.calls[-2:] == ["show", "restore"]
    runner = threading.Thread(target=win.run)
    runner.start()
    win.destroy()  # Quit
    runner.join(5)
    assert not runner.is_alive() and fake.closed.is_set()
    assert webview.started == {"icon": None}  # no icon given: the program's own


def test_browse_uses_the_windows_folder_dialog() -> None:
    webview = FakeWebview()
    win = AppWindow("http://127.0.0.1:8765/", webview=webview)
    fake = webview.window
    assert fake is not None
    fake.dialog_answer = (r"C:\Users\me\Documents\Sports Interactive\Football Manager 27",)
    assert win.browse(r"C:\Users\me") == fake.dialog_answer[0]
    assert fake.calls[-1] == "dialog 20 'C:\\\\Users\\\\me'"
    fake.dialog_answer = None  # cancelled
    assert win.browse(None) is None and fake.calls[-1] == "dialog 20 ''"
    fake.dialog_answer = RuntimeError("no dialog")
    assert win.browse(None) is None


def test_an_older_pywebview_names_the_folder_dialog_differently() -> None:
    webview = FakeWebview()
    old = SimpleNamespace(
        create_window=webview.create_window, start=webview.start, FOLDER_DIALOG=20
    )
    win = AppWindow("http://127.0.0.1:8765/", webview=old)
    assert webview.window is not None
    webview.window.dialog_answer = "/home/me/fm"
    assert win.browse(None) == "/home/me/fm"
