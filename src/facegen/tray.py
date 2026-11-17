"""The tray icon (Phase 11): pystray over app_controller.AppController.

A thin layer: it shows the controller's view (the icon with its state dot, the tooltip and the
menu) and calls the controller's actions. desktop.Refresher tells it when the view changes
and what to announce. pystray is imported only here, and only on Windows does the `desktop`
dependency group install it; the tests use a fake in its place.

A left click on the icon opens the dashboard (the menu's default item); the right button
shows the menu. Menu actions run in a thread of their own, so the icon's message loop never
waits on them (Start with Windows writes its shortcut through PowerShell).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

from facegen.app_controller import (
    NOTICE_CHARS,
    NOTICE_TITLE_CHARS,
    AppController,
    MenuEntry,
    Notice,
    View,
    clip,
)
from facegen.app_icon import draw_icon

logger = logging.getLogger("facegen")

ICON_PIXELS = 64


class TrayIcon:
    """The pystray icon. `run` blocks in the app's main thread until `stop`."""

    def __init__(self, controller: AppController, *, pystray: Any = None) -> None:
        if pystray is None:
            import pystray  # the desktop dependency group, Windows only

        self._pystray = pystray
        self._controller = controller
        self._lock = threading.Lock()
        self._view = controller.view()
        self._ready = threading.Event()
        self._waiting: list[Notice] = []
        """Notifications that came before the icon was shown; given once it is."""
        self.icon = pystray.Icon(
            "facegen",
            draw_icon(self._view.dot, ICON_PIXELS),
            self._view.tooltip,
            menu=pystray.Menu(self._items),
        )

    def _items(self) -> Iterator[Any]:
        MenuItem = self._pystray.MenuItem
        with self._lock:
            view = self._view
        for entry in view.menu:
            yield MenuItem(
                entry.label,
                self._action(entry),
                checked=(lambda item, on=entry.checked: on) if entry.checked is not None else None,
                default=entry.default,
                enabled=entry.enabled,
            )
            if entry.key == "status":
                yield self._pystray.Menu.SEPARATOR

    def _action(self, entry: MenuEntry) -> Callable[[], None] | None:
        actions: dict[str, Callable[[], object]] = {
            "open": self._controller.open,
            "pause": self._controller.pause_or_resume,
            "sync": self._controller.sync_now,
            "startup": self._controller.toggle_startup,
            "quit": self._controller.quit,
        }
        act = actions.get(entry.key)
        if act is None:
            return None

        def run() -> None:
            def call() -> None:
                try:
                    act()
                except Exception as exc:
                    logger.error("tray: %s failed", entry.key, exc_info=exc)
                self.show(self._controller.view())

            threading.Thread(target=call, name=f"facegen-tray-{entry.key}", daemon=True).start()

        return run

    def show(self, view: View) -> None:
        """Show a new view: the icon when its dot changed, the tooltip, the menu."""
        with self._lock:
            old, self._view = self._view, view
        if not self._ready.is_set():
            return
        try:
            if view.dot != old.dot:
                self.icon.icon = draw_icon(view.dot, ICON_PIXELS)
            if view.tooltip != old.tooltip:
                self.icon.title = view.tooltip
            if view.menu != old.menu:
                self.icon.update_menu()
        except Exception as exc:  # noqa: BLE001 - the shell may be going away
            logger.warning("tray: could not update the icon: %s", exc)

    def notify(self, notice: Notice) -> None:
        with self._lock:
            if not self._ready.is_set():
                self._waiting.append(notice)
                return
        if not getattr(self.icon, "HAS_NOTIFICATION", True):
            return  # the refresher has logged it
        try:
            # Shell notices (for example a failed window start) bypass AppController's
            # clipping. Enforce Windows's limits here for every notification source.
            self.icon.notify(
                clip(notice.message, NOTICE_CHARS), clip(notice.title, NOTICE_TITLE_CHARS)
            )
        except Exception as exc:  # noqa: BLE001 - a notification is not worth stopping for
            logger.warning("tray: could not show a notification: %s", exc)

    def _setup(self, icon: Any) -> None:
        icon.visible = True
        with self._lock:
            self._ready.set()
            view, waiting, self._waiting = self._view, self._waiting, []
        icon.icon = draw_icon(view.dot, ICON_PIXELS)
        icon.title = view.tooltip
        icon.update_menu()
        for notice in waiting:
            self.notify(notice)

    def run(self) -> None:
        """The icon's loop in this thread, until `stop`."""
        self.icon.run(setup=self._setup)

    def stop(self) -> None:
        try:
            self.icon.visible = False
        except Exception as exc:  # noqa: BLE001 - it may never have been shown
            logger.debug("tray: could not hide the icon: %s", exc)
        self.icon.stop()
