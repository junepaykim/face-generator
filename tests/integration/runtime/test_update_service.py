"""The update check in the background (runtime/update_service.py, Phase 13)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from facegen.runtime.update_service import CHECK_EVERY, RETRY_EVERY, UpdateService
from facegen.setup import update


def app_folder(tmp_path: Path, version: str = "0.9.0") -> Path:
    root = tmp_path / "app"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "facegen"\nversion = "{version}"\n', encoding="utf-8"
    )
    return root


class Releases:
    """GitHub's releases list, as the test sets it; counts the requests."""

    def __init__(self, *tags: str) -> None:
        self.tags = list(tags)
        self.asked = 0
        self.error: str | None = None

    def __call__(self, url: str) -> Any:
        self.asked += 1
        if self.error:
            raise update.UpdateError(self.error)
        return [
            {
                "tag_name": t,
                "draft": False,
                "prerelease": False,
                "html_url": f"https://github.com/x/{t}",
            }
            for t in self.tags
        ]


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def service(tmp_path: Path, releases: Releases, **kw: Any) -> UpdateService:
    return UpdateService(
        tmp_path / "data",
        kw.pop("enabled", lambda: True),
        root=kw.pop("root", app_folder(tmp_path)),
        fetch=releases,
        **kw,
    )


def test_nothing_is_offered_before_a_check(tmp_path: Path) -> None:
    s = service(tmp_path, Releases("v1.0.0"))
    status = s.status()
    assert status.current == "0.9.0" and status.enabled
    assert not status.available and not status.offered and status.checked_at is None
    assert not status.checkout


def test_a_newer_release_is_offered_until_put_off(tmp_path: Path) -> None:
    releases = Releases("v1.0.0", "native-v2.0.0")
    s = service(tmp_path, releases)
    status = s.check_now()
    assert releases.asked == 1
    assert status.available and status.offered and status.latest == "1.0.0"
    assert status.page == "https://github.com/x/v1.0.0"
    s.later()
    assert s.status().available and not s.status().offered
    # A newer one again is offered.
    releases.tags.append("v1.1.0")
    assert s.check_now().offered and s.status().latest == "1.1.0"


def test_the_same_or_an_older_release_is_not_offered(tmp_path: Path) -> None:
    for tags in (("v0.9.0",), ("v0.8.0",), ()):
        s = service(tmp_path, Releases(*tags))
        assert not s.check_now().available


def test_later_during_a_check_keeps_that_version_dismissed(tmp_path: Path) -> None:
    releases = Releases("v1.0.0")
    fetching, resume = threading.Event(), threading.Event()

    def fetch(url: str) -> Any:
        if releases.asked:
            fetching.set()
            assert resume.wait(5)
        return releases(url)

    s = UpdateService(tmp_path / "data", lambda: True, root=app_folder(tmp_path), fetch=fetch)
    assert s.check_now().offered
    with ThreadPoolExecutor(max_workers=1) as pool:
        check = pool.submit(s.check_now)
        try:
            assert fetching.wait(5)
            s.later("1.0.0")
        finally:
            resume.set()
        assert not check.result(timeout=5).offered
    assert s.status().available and not s.status().offered


def test_a_failed_check_keeps_what_was_known(tmp_path: Path) -> None:
    releases = Releases("v1.0.0")
    s = service(tmp_path, releases)
    s.check_now()
    releases.error = "could not reach GitHub: timed out"
    status = s.check_now()
    assert status.error == "could not reach GitHub: timed out"
    assert status.available and status.latest == "1.0.0"


def test_a_check_is_due_once_a_day_and_sooner_after_a_failure(tmp_path: Path) -> None:
    clock = Clock()
    releases = Releases("v1.0.0")
    s = service(tmp_path, releases, clock=clock)
    assert s.due()
    s.check_now()
    assert not s.due()
    clock.now += CHECK_EVERY - 1
    assert not s.due()
    clock.now += 1
    assert s.due()
    releases.error = "offline"
    s.check_now()
    clock.now += RETRY_EVERY
    assert s.due()


def test_the_switch_is_read_each_time_and_a_failure_reads_as_off(tmp_path: Path) -> None:
    switch = {"on": True}
    s = service(tmp_path, Releases(), enabled=lambda: switch["on"])
    assert s.enabled()
    switch["on"] = False
    assert not s.enabled() and not s.status().enabled

    def broken() -> bool:
        raise RuntimeError("database is locked")

    assert not service(tmp_path, Releases(), enabled=broken).enabled()


def test_the_thread_checks_when_on_and_due_and_stops(tmp_path: Path) -> None:
    releases = Releases("v1.0.0")
    switch = {"on": False}
    s = service(tmp_path, releases, enabled=lambda: switch["on"], first_delay=0.0, look_every=0.05)
    s.start()
    try:
        time.sleep(0.3)
        assert releases.asked == 0  # switched off: nothing is asked
        switch["on"] = True
        deadline = time.monotonic() + 5
        while releases.asked == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert releases.asked == 1
        time.sleep(0.3)
        assert releases.asked == 1  # not due again for a day
    finally:
        s.stop()
    assert s._thread is not None and not s._thread.is_alive()


def test_a_git_checkout_says_so(tmp_path: Path) -> None:
    root = app_folder(tmp_path)
    (root / ".git").mkdir()
    assert service(tmp_path, Releases(), root=root).status().checkout


def test_the_check_runs_once_at_a_time(tmp_path: Path) -> None:
    gate = threading.Event()
    calls: list[int] = []

    def slow(url: str) -> Any:
        calls.append(1)
        gate.wait(5)
        return []

    s = UpdateService(tmp_path / "data", lambda: True, root=app_folder(tmp_path), fetch=slow)
    worker = threading.Thread(target=s.check_now)
    worker.start()
    deadline = time.monotonic() + 5
    while not s.status().checking and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s.status().checking
    gate.set()
    worker.join(5)
    assert not s.status().checking and calls == [1]
