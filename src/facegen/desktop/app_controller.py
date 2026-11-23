"""GUI-independent desktop state and actions for the tray and window.

`view()` provides menu availability, the running/paused/fault dot and a tooltip clipped to
Windows's 127-character buffer. Notifications are emitted once for completed drawing runs
(unless `pictures_notice` says no), GPU faults, job errors, unsupported saves and syncs
requested from the tray.
Quit runs supplied cleanup steps in order on a thread, continuing after errors, then calls
`on_quit_done` to close the shell. Dependencies are plain values and callables.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from facegen.runtime.contracts import PollerStatus, SyncStatus, WorkerControl, WorkerStatus

if TYPE_CHECKING:
    from facegen.runtime.update_service import UpdateStatus

logger = logging.getLogger("facegen")

Dot = Literal["running", "paused", "fault"]

PRODUCT = "Kitshot"
"""The name in the tooltip and the update notification."""

TOOLTIP_CHARS = 127
"""Windows keeps 128 characters of a tray icon's tooltip, the last one the terminating zero."""
NOTICE_CHARS = 255
"""And 256 of a notification's text."""
NOTICE_TITLE_CHARS = 63
"""And 64 of its title."""
MENU_STATUS_CHARS = 64
"""The status line at the top of the menu: long enough to read, short enough for a menu."""

PICTURES_QUIET_SECONDS = 10.0
"""New pictures are announced once drawing has stopped (the worker is not "running") and no
picture has been added for this long, so one run of drawing is one notification."""

FAULT_STATES = frozenset({"gpu-fault", "error", "restarting"})
"""Worker states shown with the fault dot."""
PAUSED_STATES = frozenset({"paused", "stopped"})

PICTURES_ADVICE = "restart FM27 or reload the skin to see them."
UPDATE_LOOK_SECONDS = 60.0
"""How often the tray looks at the update check's answer (it is asked once a day)."""


class SyncControl(Protocol):
    """What the controller uses of service.SyncService."""

    def status(self) -> SyncStatus: ...
    def start(self) -> bool: ...


class PollerView(Protocol):
    def status(self) -> PollerStatus: ...


class UpdatesView(Protocol):
    """What the controller uses of update_service.UpdateService."""

    def status(self) -> UpdateStatus: ...


class StartupSwitch(Protocol):
    """Start with Windows: a shortcut in the Startup folder (shortcuts.StartupShortcut)."""

    def is_on(self) -> bool: ...
    def set(self, on: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class MenuEntry:
    key: str
    """status, open, pause, sync, startup or quit."""
    label: str
    enabled: bool = True
    checked: bool | None = None
    """None: not a check box."""
    default: bool = False
    """The item a click on the icon runs (Open Dashboard)."""


@dataclass(frozen=True, slots=True)
class View:
    dot: Dot
    tooltip: str
    menu: tuple[MenuEntry, ...]

    def entry(self, key: str) -> MenuEntry | None:
        return next((e for e in self.menu if e.key == key), None)


@dataclass(frozen=True, slots=True)
class Notice:
    kind: str
    """pictures, fault, error, unsupported, sync or update."""
    title: str
    message: str


def clip(text: str, limit: int) -> str:
    """One line fitting a Windows UTF-16 buffer, with room for an ellipsis if cut.

    Characters outside the BMP take two Windows wchar slots. Do not split their surrogate
    pairs, or let an otherwise short tooltip or notification overflow the shell's buffer.
    """
    flat = " ".join(text.split())
    encoded = flat.encode("utf-16-le")
    if len(encoded) <= limit * 2:
        return flat
    return encoded[: (limit - 1) * 2].decode("utf-16-le", errors="ignore").rstrip() + "…"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def dot_for(worker: WorkerStatus | None) -> Dot:
    """The icon's state dot: fault when drawing failed, paused when paused (or there is no
    worker), else running (drawing, or ready to)."""
    if worker is None:
        return "paused"
    if worker.state in FAULT_STATES:
        return "fault"
    if worker.paused or worker.state in PAUSED_STATES:
        return "paused"
    return "running"


def describe_worker(worker: WorkerStatus | None) -> str:
    """The worker's state in a few plain words, for the tooltip and the menu."""
    if worker is None:
        return "not drawing in this process"
    match worker.state:
        case "running":
            if worker.current is not None:
                return f"drawing {worker.current.name} ({worker.current.group.upper()})"
            return "drawing"
        case "idle":
            return "idle: " + (worker.detail or "the queue is empty")
        case "waiting":
            return "waiting: " + (worker.detail or "FM27 is running")
        case "setup":
            return "set up: the models are not downloaded yet"
        case "paused":
            return "paused"
        case "starting":
            return "starting"
        case "restarting":
            return "restarting after a graphics driver error"
        case "gpu-fault":
            return "drawing stopped: the graphics driver failed"
        case "error":
            return "paused by an error"
        case "stopped":
            return "stopped"
        case other:
            return other


class AppController:
    """The tray's and the window's state and actions; no GUI here (module docstring)."""

    def __init__(
        self,
        *,
        worker: WorkerControl | None,
        syncer: SyncControl | None,
        open_dashboard: Callable[[], None],
        quit_steps: Sequence[tuple[str, Callable[[], None]]] = (),
        on_quit_done: Callable[[], None] = lambda: None,
        poller: PollerView | None = None,
        startup: StartupSwitch | None = None,
        updates: UpdatesView | None = None,
        clock: Callable[[], float] = time.monotonic,
        quiet_seconds: float = PICTURES_QUIET_SECONDS,
        pictures_notice: Callable[[], bool] | None = None,
    ) -> None:
        self.worker = worker
        self.syncer = syncer
        self.poller = poller
        self.startup = startup
        self.updates = updates
        self._update_said: str | None = None
        self._update_looked: float | None = None
        self._open_dashboard = open_dashboard
        self._quit_steps = list(quit_steps)
        self._on_quit_done = on_quit_done
        self._clock = clock
        self._quiet = quiet_seconds
        self._pictures_notice = pictures_notice
        self._lock = threading.Lock()
        self._quitting = False
        self._quit_step: str | None = None
        self._quit_thread: threading.Thread | None = None
        self._quit_done = threading.Event()
        # What has been announced, so each notification is given once.
        self._announced_done: int | None = None
        self._done_seen: int | None = None
        self._done_changed_at = clock()
        self._last_state: str | None = None
        self._unsupported_said: str | None = None
        self._sync_pending = False
        self.steps_run: list[str] = []
        """The quit steps in the order they ran (for the log and the tests)."""

    def open(self) -> None:
        """Open Dashboard: show the window, or the dashboard in the browser."""
        try:
            self._open_dashboard()
        except Exception as exc:
            logger.error("could not open the dashboard", exc_info=exc)

    def pause_or_resume(self) -> bool:
        """Pause, or Resume when paused (after a fault or an error too). False when nothing
        was done: no worker, quitting, or Resume refused (a GPU fault in a worker thread)."""
        if self.worker is None or self.quitting:
            return False
        if self.worker.status().paused:
            return self.worker.resume()
        self.worker.pause()
        return True

    def sync_now(self) -> bool:
        """Sync the newest save; its result is a notification. False if a sync is running."""
        if self.syncer is None or self.quitting:
            return False
        started = self.syncer.start()
        if started:
            with self._lock:
                self._sync_pending = True
        return started

    def toggle_startup(self) -> None:
        """Start with Windows: on or off. The switch changes the PC's setup (a shortcut in
        the Startup folder), so it is only ever done from this visible menu item."""
        if self.startup is None:
            return
        try:
            self.startup.set(not self.startup.is_on())
        except Exception as exc:
            logger.error("could not change Start with Windows", exc_info=exc)

    def quit(self) -> None:
        """Start quitting, once: the steps run in a thread of their own (module docstring)."""
        with self._lock:
            if self._quitting:
                return
            self._quitting = True
        self._quit_thread = threading.Thread(target=self._run_quit, name="facegen-quit")
        self._quit_thread.start()

    def wait_quit(self, timeout: float | None = None) -> bool:
        """Wait until Quit has finished; True when it has."""
        return self._quit_done.wait(timeout)

    @property
    def quitting(self) -> bool:
        with self._lock:
            return self._quitting

    def _run_quit(self) -> None:
        try:
            for name, step in self._quit_steps:
                with self._lock:
                    self._quit_step = name
                logger.info("quitting: stopping the %s", name)
                try:
                    step()
                except Exception as exc:
                    logger.error("quitting: stopping the %s failed", name, exc_info=exc)
                self.steps_run.append(name)
            with self._lock:
                self._quit_step = None
            try:
                self._on_quit_done()
            except Exception as exc:
                logger.error("quitting: closing the tray and the window failed", exc_info=exc)
        finally:
            self._quit_done.set()

    def view(self) -> View:
        worker = self.worker.status() if self.worker is not None else None
        sync = self.syncer.status() if self.syncer is not None else None
        with self._lock:
            quitting, step = self._quitting, self._quit_step
        if quitting:
            status = f"quitting: stopping the {step}" if step else "quitting"
            if step == "drawing":
                status = "quitting: stopping drawing"
        else:
            status = describe_worker(worker)
        working = worker is not None and not quitting
        if worker is not None and worker.paused:
            pause = MenuEntry("pause", "Resume", enabled=working)
        else:
            pause = MenuEntry("pause", "Pause", enabled=working)
        syncing = sync is not None and sync.running
        sync_entry = MenuEntry(
            "sync",
            "Syncing the latest save…" if syncing else "Sync Latest Save Now",
            enabled=self.syncer is not None and not syncing and not quitting,
        )
        entries = [
            MenuEntry("status", clip(status[0].upper() + status[1:], MENU_STATUS_CHARS), False),
            MenuEntry("open", "Open Dashboard", enabled=not quitting, default=True),
            pause,
            sync_entry,
        ]
        if self.startup is not None:
            entries.append(
                MenuEntry("startup", "Start with Windows", not quitting, checked=self._startup_on())
            )
        entries.append(MenuEntry("quit", "Quitting…" if quitting else "Quit", not quitting))
        dot: Dot = "paused" if quitting else dot_for(worker)
        return View(dot, clip(f"{PRODUCT}: {status}", TOOLTIP_CHARS), tuple(entries))

    def _startup_on(self) -> bool:
        assert self.startup is not None
        try:
            return self.startup.is_on()
        except Exception:  # noqa: BLE001 - shown as off
            return False

    def notices(self) -> list[Notice]:
        """What has happened since the last call that the user should be told, each once."""
        worker = self.worker.status() if self.worker is not None else None
        sync = self.syncer.status() if self.syncer is not None else None
        out: list[Notice] = []
        if worker is not None:
            out += self._worker_notices(worker)
        if sync is not None:
            out += self._sync_notices(sync)
        out += self._update_notices()
        return out

    def _update_notices(self) -> list[Notice]:
        """A newer Kitshot, said once per version (and not again after Later)."""
        now = self._clock()
        if self.updates is None or (
            self._update_looked is not None and now - self._update_looked < UPDATE_LOOK_SECONDS
        ):
            return []
        self._update_looked = now
        try:
            status = self.updates.status()
        except Exception as exc:  # noqa: BLE001 - the tray goes on without it
            logger.warning("could not read the update check: %s", exc)
            return []
        if not status.offered or status.latest is None or status.latest == self._update_said:
            return []
        self._update_said = status.latest
        return [
            self._clipped(
                Notice(
                    "update",
                    f"{PRODUCT} {status.latest} is available",
                    f"You have {status.current}. Open the dashboard to update.",
                )
            )
        ]

    def _worker_notices(self, w: WorkerStatus) -> list[Notice]:
        out: list[Notice] = []
        now = self._clock()
        if self._announced_done is None:
            self._announced_done = w.done  # pictures from before the tray are not news
        if w.done != self._done_seen:
            self._done_seen, self._done_changed_at = w.done, now
        new = w.done - self._announced_done
        if new > 0 and w.state != "running" and now - self._done_changed_at >= self._quiet:
            self._announced_done = w.done  # counted as said when switched off, too
            if self._pictures_wanted():
                out.append(
                    Notice(
                        "pictures",
                        "New pictures",
                        f"{plural(new, 'new picture')}: {PICTURES_ADVICE}",
                    )
                )
        elif new < 0:  # a counter that went back (cannot happen with one worker); start afresh
            self._announced_done = w.done
        previous, self._last_state = self._last_state, w.state
        if w.state != previous:
            if w.state == "restarting":
                out.append(
                    Notice(
                        "fault",
                        "Graphics driver error",
                        "The graphics driver reported an error while drawing. Kitshot starts"
                        " its worker again by itself; the queue is kept.",
                    )
                )
            elif w.state == "gpu-fault":
                out.append(
                    Notice(
                        "fault",
                        "Drawing has stopped",
                        "The graphics card or its driver failed several times. The queue is"
                        " kept: open the dashboard and press Resume to try again.",
                    )
                )
            elif w.state == "error":
                why = w.last_error or w.detail or "an error"
                out.append(
                    Notice(
                        "error",
                        "Drawing paused by an error",
                        f"{why}. The queue is kept: Resume on the dashboard tries again.",
                    )
                )
        return [self._clipped(n) for n in out]

    def _pictures_wanted(self) -> bool:
        """`notifications.new_pictures`, read only when the notification is due; on when it
        cannot be read."""
        if self._pictures_notice is None:
            return True
        try:
            return bool(self._pictures_notice())
        except Exception as exc:  # noqa: BLE001 - said rather than lost
            logger.warning("could not read notifications.new_pictures: %s", exc)
            return True

    def _sync_notices(self, s: SyncStatus) -> list[Notice]:
        out: list[Notice] = []
        if s.running:
            return out
        if s.unsupported and s.error and s.error != self._unsupported_said:
            self._unsupported_said = s.error
            out.append(
                Notice(
                    "unsupported",
                    "Save not supported",
                    f"{s.error}. The save reader cannot read this game build yet: existing"
                    " pictures stay, and nothing new is queued from this save.",
                )
            )
        elif not s.unsupported and s.message:
            self._unsupported_said = None  # a later unsupported save is news again
        with self._lock:
            pending, self._sync_pending = self._sync_pending, False
        if pending and not (s.unsupported and out):
            if s.error:
                out.append(Notice("sync", "Sync failed", s.error))
            else:
                text = s.message or "Synced."
                if s.notice:
                    text = f"{text} {s.notice}"
                out.append(Notice("sync", "Synced", text))
        return [self._clipped(n) for n in out]

    @staticmethod
    def _clipped(n: Notice) -> Notice:
        return Notice(n.kind, clip(n.title, NOTICE_TITLE_CHARS), clip(n.message, NOTICE_CHARS))
