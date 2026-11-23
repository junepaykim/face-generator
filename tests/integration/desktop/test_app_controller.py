"""The desktop app's controller (Phase 11): the tray's menu, dot, tooltip and notifications for
every worker and sync state, and Quit's order, all with fakes."""

from __future__ import annotations

import dataclasses
import threading

import pytest
from support.desktop import FakeStartup, FakeSyncer, FakeWorker, status

from facegen.desktop.app_controller import (
    MENU_STATUS_CHARS,
    NOTICE_CHARS,
    NOTICE_TITLE_CHARS,
    TOOLTIP_CHARS,
    AppController,
    clip,
    describe_worker,
    dot_for,
)
from facegen.persistence.jobs import Job
from facegen.service import SyncStatus

STATES = (
    "starting",
    "idle",
    "paused",
    "waiting",
    "setup",
    "running",
    "error",
    "gpu-fault",
    "restarting",
    "stopped",
)
"""Every state the worker reports (service.WorkerStatus, worker_process.WorkerProxy)."""


def job(name: str = "Jane Roe", group: str = "a") -> Job:
    return Job(
        id=7,
        unique_id=2000000001,
        kind="initial",
        group=group,  # type: ignore[arg-type]
        target_club_id=None,
        target_band="W03",
        name=name,
        at_my_club=False,
        goalkeeper=False,
        height_cm=None,
        strength=None,
        age=20,
    )


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def controller(worker: FakeWorker | None = None, syncer: FakeSyncer | None = None, **kw):  # type: ignore[no-untyped-def]
    opened: list[str] = []
    c = AppController(
        worker=worker if worker is not None else FakeWorker(),
        syncer=syncer if syncer is not None else FakeSyncer(),
        open_dashboard=lambda: opened.append("open"),
        **kw,
    )
    c.opened = opened  # type: ignore[attr-defined]
    return c


# -- the menu, the dot and the tooltip for every state ---------------------------------------


@pytest.mark.parametrize("state", STATES)
def test_every_worker_state_has_a_dot_a_status_line_and_a_usable_menu(state: str) -> None:
    w = FakeWorker(status(state, current=job() if state == "running" else None))
    view = controller(w).view()
    expected_dot = {
        "error": "fault",
        "gpu-fault": "fault",
        "restarting": "fault",
        "paused": "paused",
        "stopped": "paused",
    }.get(state, "running")
    assert view.dot == expected_dot
    keys = [e.key for e in view.menu]
    assert keys == ["status", "open", "pause", "sync", "quit"]
    status_line = view.entry("status")
    assert status_line is not None and not status_line.enabled  # a label, not an action
    assert status_line.label == clip(
        describe_worker(w.s)[0].upper() + describe_worker(w.s)[1:], MENU_STATUS_CHARS
    )
    pause = view.entry("pause")
    assert pause is not None and pause.enabled
    assert pause.label == ("Resume" if w.s.paused else "Pause")
    assert view.entry("open").default  # type: ignore[union-attr]
    assert view.entry("sync").label == "Sync Latest Save Now"  # type: ignore[union-attr]
    assert view.tooltip.startswith("facegen: ") and len(view.tooltip) <= TOOLTIP_CHARS


def test_the_status_line_says_who_is_drawn_and_why_it_waits() -> None:
    assert describe_worker(status("running", current=job("Ana Lima", "b"))) == (
        "drawing Ana Lima (B)"
    )
    assert describe_worker(status("waiting", detail="FM27 is running; 3 job(s) wait.")) == (
        "waiting: FM27 is running; 3 job(s) wait."
    )
    assert describe_worker(status("idle", detail="the queue is empty")) == (
        "idle: the queue is empty"
    )
    assert describe_worker(status("setup")).startswith("set up: the models")
    assert describe_worker(None) == "not drawing in this process"
    assert dot_for(None) == "paused"


def test_no_worker_means_no_pause_and_the_paused_dot() -> None:
    c = AppController(worker=None, syncer=FakeSyncer(), open_dashboard=lambda: None)
    view = c.view()
    assert not view.entry("pause").enabled  # type: ignore[union-attr]
    assert view.dot == "paused"
    assert c.pause_or_resume() is False


def test_a_long_detail_is_cut_for_the_tooltip_and_the_menu() -> None:
    w = FakeWorker(status("waiting", detail="x " * 400))
    view = controller(w).view()
    assert TOOLTIP_CHARS - 2 <= len(view.tooltip) <= TOOLTIP_CHARS
    assert view.tooltip.endswith("…")
    assert len(view.entry("status").label) <= MENU_STATUS_CHARS  # type: ignore[union-attr]
    assert "\n" not in clip("two\nlines", 50)


@pytest.mark.parametrize(
    ("sync", "label", "enabled"),
    [
        (SyncStatus(), "Sync Latest Save Now", True),
        (SyncStatus(running=True), "Syncing the latest save…", False),
        (SyncStatus(error="Cannot read x.fm: boom"), "Sync Latest Save Now", True),
        (SyncStatus(error="Cannot read", unsupported=True), "Sync Latest Save Now", True),
        (SyncStatus(message="Synced x.fm"), "Sync Latest Save Now", True),
    ],
)
def test_every_sync_state_sets_the_sync_item(sync: SyncStatus, label: str, enabled: bool) -> None:
    s = FakeSyncer()
    s.s = sync
    entry = controller(syncer=s).view().entry("sync")
    assert entry is not None and (entry.label, entry.enabled) == (label, enabled)


def test_no_syncer_disables_sync() -> None:
    c = AppController(worker=FakeWorker(), syncer=None, open_dashboard=lambda: None)
    assert not c.view().entry("sync").enabled  # type: ignore[union-attr]
    assert c.sync_now() is False


# -- the actions -------------------------------------------------------------------------------


def test_pause_then_resume_from_the_tray() -> None:
    w = FakeWorker(status("running", current=job(), paused=False))
    c = controller(w)
    assert c.pause_or_resume() and w.calls == ["pause"]
    assert c.view().entry("pause").label == "Resume"  # type: ignore[union-attr]
    assert c.pause_or_resume() and w.calls == ["pause", "resume"]
    assert c.view().entry("pause").label == "Pause"  # type: ignore[union-attr]


def test_resume_after_a_gpu_fault_is_offered_and_a_refusal_is_reported() -> None:
    w = FakeWorker(status("gpu-fault", paused=True))
    c = controller(w)
    assert c.view().entry("pause").label == "Resume"  # type: ignore[union-attr]
    w.resume_ok = False  # WorkerService in a thread refuses after a fault
    assert c.pause_or_resume() is False and w.calls == ["resume"]


def test_open_calls_the_shell_and_survives_its_failure() -> None:
    c = controller()
    c.open()
    assert c.opened == ["open"]  # type: ignore[attr-defined]

    def broken() -> None:
        raise OSError("no browser")

    AppController(worker=None, syncer=None, open_dashboard=broken).open()  # no exception


def test_start_with_windows_is_a_check_box_only_when_offered() -> None:
    assert controller().view().entry("startup") is None
    switch = FakeStartup(on=False)
    c = controller(startup=switch)
    entry = c.view().entry("startup")
    assert entry is not None and entry.checked is False and entry.label == "Start with Windows"
    c.toggle_startup()
    assert switch.sets == [True] and c.view().entry("startup").checked is True  # type: ignore[union-attr]
    c.toggle_startup()
    assert switch.sets == [True, False]


def test_a_broken_startup_switch_shows_off_and_does_not_raise() -> None:
    class Broken(FakeStartup):
        def is_on(self) -> bool:
            raise OSError("no Startup folder")

    c = controller(startup=Broken())
    assert c.view().entry("startup").checked is False  # type: ignore[union-attr]
    c.toggle_startup()  # logged, not raised


# -- Quit ----------------------------------------------------------------------------------


def test_quit_runs_its_steps_in_order_then_closes_the_shell_once() -> None:
    order: list[str] = []
    release = threading.Event()

    def worker_stop() -> None:
        release.wait(5)  # the process is being reaped
        order.append("worker")

    c = controller(
        quit_steps=[
            ("model download", lambda: order.append("model download")),
            ("save watcher", lambda: order.append("save watcher")),
            ("worker", worker_stop),
            ("web server", lambda: order.append("web server")),
        ],
        on_quit_done=lambda: order.append("tray and window"),
    )
    c.quit()
    c.quit()  # a second click does nothing more
    # While waiting for the worker, the tray shows the current step and disables actions.
    deadline = threading.Event()
    for _ in range(200):
        if c.view().entry("status").label.startswith("Quitting: stopping the worker"):  # type: ignore[union-attr]
            break
        deadline.wait(0.01)
    view = c.view()
    assert view.entry("status").label == "Quitting: stopping the worker"  # type: ignore[union-attr]
    assert [e.key for e in view.menu if e.enabled] == []
    assert view.entry("quit").label == "Quitting…"  # type: ignore[union-attr]
    assert view.dot == "paused"
    assert c.pause_or_resume() is False and c.sync_now() is False
    release.set()
    assert c.wait_quit(5)
    assert order == [
        "model download",
        "save watcher",
        "worker",
        "web server",
        "tray and window",
    ]
    assert c.steps_run == ["model download", "save watcher", "worker", "web server"]


def test_a_failing_quit_step_does_not_stop_the_next() -> None:
    order: list[str] = []

    def boom() -> None:
        raise RuntimeError("the watcher was gone")

    c = controller(
        quit_steps=[("save watcher", boom), ("worker", lambda: order.append("worker"))],
        on_quit_done=lambda: order.append("closed"),
    )
    c.quit()
    assert c.wait_quit(5) and order == ["worker", "closed"]


# -- notifications ------------------------------------------------------------------------------


def test_new_pictures_are_announced_once_when_a_run_of_drawing_ends() -> None:
    clock = Clock()
    w = FakeWorker(status("running", current=job(), done=5, paused=False))
    c = controller(w, clock=clock, quiet_seconds=10)
    assert c.notices() == []  # the five from before the tray are not news
    for done in (6, 7, 8):
        w.s = dataclasses.replace(w.s, done=done)
        clock.now += 20
        assert c.notices() == []  # still drawing
    w.s = dataclasses.replace(w.s, state="idle", current=None)
    clock.now += 5
    assert c.notices() == []  # not quiet for long enough yet
    clock.now += 6
    (notice,) = c.notices()
    assert notice.kind == "pictures"
    assert notice.message == "3 new pictures: restart FM27 or reload the skin to see them."
    clock.now += 60
    assert c.notices() == []  # said once
    w.s = dataclasses.replace(w.s, done=9, state="waiting")
    assert c.notices() == []
    clock.now += 11
    assert [n.message for n in c.notices()] == [
        "1 new picture: restart FM27 or reload the skin to see them."
    ]


def test_the_new_pictures_notice_can_be_switched_off() -> None:
    clock, wanted, asked = Clock(), [False], []

    def setting() -> bool:
        asked.append(clock.now)
        return wanted[0]

    w = FakeWorker(status("idle", done=0, paused=False))
    c = controller(w, clock=clock, quiet_seconds=10, pictures_notice=setting)
    assert c.notices() == []
    w.s = dataclasses.replace(w.s, done=2)
    assert c.notices() == []  # seen now; said once quiet for 10 s
    for _ in range(3):
        clock.now += 11
        assert c.notices() == []  # off: those two are not said, now or later
    assert len(asked) == 1  # read only when the notice is due, not every tick
    wanted[0] = True
    w.s = dataclasses.replace(w.s, done=3)
    assert c.notices() == []
    clock.now += 11
    assert [n.message for n in c.notices()] == [
        "1 new picture: restart FM27 or reload the skin to see them."
    ]


def test_the_new_pictures_notice_is_shown_when_its_setting_cannot_be_read() -> None:
    def broken() -> bool:
        raise OSError("database is locked")

    clock = Clock()
    w = FakeWorker(status("idle", done=0, paused=False))
    c = controller(w, clock=clock, quiet_seconds=10, pictures_notice=broken)
    c.notices()
    w.s = dataclasses.replace(w.s, done=1)
    assert c.notices() == []
    clock.now += 11
    assert [n.kind for n in c.notices()] == ["pictures"]


@pytest.mark.parametrize(
    ("state", "kind", "title"),
    [
        ("restarting", "fault", "Graphics driver error"),
        ("gpu-fault", "fault", "Drawing has stopped"),
        ("error", "error", "Drawing paused by an error"),
    ],
)
def test_faults_and_errors_are_announced_once_per_occurrence(
    state: str, kind: str, title: str
) -> None:
    w = FakeWorker(status("running", current=job(), paused=False))
    c = controller(w)
    assert c.notices() == []
    w.s = status(state, last_error="Jane Roe: RuntimeError: out of memory")
    (notice,) = c.notices()
    assert (notice.kind, notice.title) == (kind, title)
    assert c.notices() == []  # the same state is not news
    w.s = status("running", current=job(), paused=False)
    assert c.notices() == []
    w.s = status(state)
    assert [n.kind for n in c.notices()] == [kind]  # a second fault is


def test_an_error_names_what_failed() -> None:
    w = FakeWorker(status("idle"))
    c = controller(w)
    c.notices()
    w.s = status("error", last_error="Jane Roe: RuntimeError: CUDA out of memory")
    (notice,) = c.notices()
    assert notice.message.startswith("Jane Roe: RuntimeError: CUDA out of memory.")
    assert "Resume" in notice.message


def test_an_unsupported_save_is_announced_once_until_a_save_reads_again() -> None:
    s = FakeSyncer()
    c = controller(syncer=s)
    s.finish(error="Cannot read Career.fm: build 27.1 is not supported", unsupported=True)
    (notice,) = c.notices()
    assert notice.kind == "unsupported" and notice.title == "Save not supported"
    assert notice.message.startswith("Cannot read Career.fm")
    assert c.notices() == []  # the watcher's next look at the same save: not again
    s.finish(message="Synced Other.fm")
    assert c.notices() == []
    s.finish(error="Cannot read Career.fm: build 27.1 is not supported", unsupported=True)
    assert [n.kind for n in c.notices()] == ["unsupported"]


def test_a_sync_from_the_tray_reports_its_result() -> None:
    s = FakeSyncer()
    c = controller(syncer=s)
    assert c.sync_now() and s.starts == 1
    assert c.sync_now() is False  # one at a time
    assert c.view().entry("sync").label == "Syncing the latest save…"  # type: ignore[union-attr]
    assert c.notices() == []  # still running
    s.finish(message="Synced Career.fm (in-game 2027-08-01): 3 queued.")
    (notice,) = c.notices()
    assert (notice.kind, notice.title) == ("sync", "Synced")
    assert notice.message.startswith("Synced Career.fm")
    assert c.notices() == []
    # A failure, and FM27 not having saved since, are said too.
    c.sync_now()
    s.finish(error="FM27's folder is not set.")
    assert [(n.title, n.message) for n in c.notices()] == [
        ("Sync failed", "FM27's folder is not set.")
    ]
    c.sync_now()
    s.finish(message="Synced Career.fm.", notice="FM27 has not saved since.")
    assert c.notices()[0].message == "Synced Career.fm. FM27 has not saved since."


def test_a_sync_from_the_dashboard_is_not_announced() -> None:
    s = FakeSyncer()
    c = controller(syncer=s)
    s.start()  # the web UI's Sync now, not the tray's
    s.finish(message="Synced Career.fm")
    assert c.notices() == []


def test_clip_counts_windows_utf16_slots_without_splitting_characters() -> None:
    assert clip("한글 😀", 4) == "한글…"
    assert clip("😀😀😀", 5) == "😀😀…"
    assert clip("😀😀", 4) == "😀😀"
    assert len(clip("😀" * 200, TOOLTIP_CHARS).encode("utf-16-le")) <= TOOLTIP_CHARS * 2


def test_notifications_fit_windows_limits() -> None:
    w = FakeWorker(status("idle"))
    c = controller(w)
    c.notices()
    w.s = status("error", last_error="x" * 1000)
    (notice,) = c.notices()
    assert len(notice.message) <= NOTICE_CHARS and len(notice.title) <= NOTICE_TITLE_CHARS
