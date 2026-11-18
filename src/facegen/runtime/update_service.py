"""The update check in the background (Phase 13; setup/update.py does the asking).

While `updates.check` is on, the app asks GitHub for the newest release a minute after it
starts and then once a day (six hours after a failed check), keeping the answer in
data/app/update-check.json so a restart does not ask again. "Check now" asks at once.
Later puts off the offer of that version; a newer one is offered again.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from facegen.locations import PROJECT_ROOT
from facegen.setup import update

logger = logging.getLogger("facegen")

FIRST_DELAY = 60.0
"""Seconds after the app starts: its start is not slowed, and a quick restart asks nothing."""
CHECK_EVERY = 24 * 3600.0
RETRY_EVERY = 6 * 3600.0
"""After a failed check (no network, say)."""
LOOK_EVERY = 3600.0
"""How often the thread looks whether a check is due (and whether the switch is on)."""


@dataclass(frozen=True, slots=True)
class UpdateStatus:
    enabled: bool
    current: str
    latest: str | None = None
    page: str | None = None
    """The newest release's page (what is new)."""
    available: bool = False
    """The newest release is newer than this folder's version."""
    offered: bool = False
    """Available and not put off with Later."""
    checked_at: float | None = None
    error: str | None = None
    checking: bool = False
    checkout: bool = False
    """A git checkout: updated with git pull, not in place."""


class UpdateService:
    """See the module docstring. `enabled` reads the setting each time it is asked."""

    def __init__(
        self,
        data_dir: Path,
        enabled: Callable[[], bool],
        *,
        root: Path = PROJECT_ROOT,
        fetch: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.time,
        first_delay: float = FIRST_DELAY,
        look_every: float = LOOK_EVERY,
    ) -> None:
        self.data_dir = data_dir
        self._enabled = enabled
        self.root = root
        self._fetch = fetch
        self._clock = clock
        self._first_delay = first_delay
        self._look_every = look_every
        self._lock = threading.Lock()
        self._check_lock = threading.Lock()
        self._checking = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            self.current = update.installed_version(root)
        except update.UpdateError:
            self.current = "0.0.0"
        self.checkout = update.is_checkout(root)

    def enabled(self) -> bool:
        try:
            return bool(self._enabled())
        except Exception as exc:  # noqa: BLE001 - the database busy, say: ask next time
            logger.warning("could not read updates.check: %s", exc)
            return False

    def status(self) -> UpdateStatus:
        record = update.load_record(self.data_dir)
        latest = record.latest if record else None
        available = update.is_newer(latest, self.current)
        with self._lock:
            checking = self._checking
        return UpdateStatus(
            enabled=self.enabled(),
            current=self.current,
            latest=latest.version if latest else None,
            page=latest.page if latest else None,
            available=available,
            offered=available and (record is None or record.dismissed != latest.version),
            checked_at=record.checked_at if record else None,
            error=record.error if record else None,
            checking=checking,
            checkout=self.checkout,
        )

    def due(self) -> bool:
        record = update.load_record(self.data_dir)
        if record is None:
            return True
        wait = RETRY_EVERY if record.error else CHECK_EVERY
        return self._clock() - record.checked_at >= wait

    def check_now(self) -> UpdateStatus:
        """Ask GitHub now (one request), whatever the switch says: the user asked."""
        with self._check_lock:
            with self._lock:
                self._checking = True
            try:
                old = update.load_record(self.data_dir)
                try:
                    latest = update.latest_release(self._fetch)
                    error = None
                except update.UpdateError as exc:
                    latest, error = (old.latest if old else None), str(exc)
                    logger.info("update check failed: %s", exc)
                record = update.CheckRecord(
                    self._clock(), latest, error, old.dismissed if old else None
                )
                update.save_record(self.data_dir, record)
                if error is None:
                    logger.info(
                        "update check: newest release %s (this is %s)",
                        latest.version if latest else "none",
                        self.current,
                    )
            finally:
                with self._lock:
                    self._checking = False
        return self.status()

    def later(self) -> None:
        """Put off the offer of the newest version."""
        record = update.load_record(self.data_dir)
        if record is not None and record.latest is not None:
            update.save_record(self.data_dir, replace(record, dismissed=record.latest.version))

    def _main(self) -> None:
        if self._stop.wait(self._first_delay):
            return
        while True:
            try:
                if self.enabled() and self.due():
                    self.check_now()
            except Exception as exc:
                logger.error("the update check failed", exc_info=exc)
            if self._stop.wait(self._look_every):
                return

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._main, name="facegen-update", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
