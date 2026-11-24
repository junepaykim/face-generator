"""Enforce one desktop app and forward second-launch activation through local files.

The first app holds data/app.lock using an OS lock, released even after a crash. A second
start writes data/app.open and ends; the running shell checks about once a second, removes
the request and opens its window or browser. This uses our data folder rather than adding
an activation endpoint to the web server, whose mutations require a form token.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

from kitshot.platform.locking import LockBusy, file_lock

LOCK_NAME = "app.lock"
OPEN_REQUEST_NAME = "app.open"
ANSWER_SECONDS = 10.0
"""How long a second start waits for the first to take its request."""


class InstanceLock:
    """data/app.lock, held from `acquire` until `release` or the process ends."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / LOCK_NAME
        self._stack: contextlib.ExitStack | None = None

    def acquire(self) -> bool:
        """Take the lock; False if another Kitshot app holds it."""
        if self._stack is not None:
            return True
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(file_lock(self.path, blocking=False))
        except LockBusy:
            stack.close()
            return False
        self._stack = stack
        return True

    @property
    def held(self) -> bool:
        return self._stack is not None

    def release(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def held_elsewhere(data_dir: Path) -> bool:
    """Whether a Kitshot app is running: its lock cannot be taken (it is let go at once)."""
    probe = InstanceLock(data_dir)
    if probe.acquire():
        probe.release()
        return False
    return True


def request_open(data_dir: Path) -> Path:
    """Ask the running app to open its window: write data/app.open (atomically)."""
    path = data_dir / OPEN_REQUEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(f"{time.time():.3f}\n", encoding="ascii")
    os.replace(tmp, path)
    return path


def take_open_request(data_dir: Path) -> bool:
    """For the running app: whether a second start asked for the window. The request is
    deleted, so it is answered once."""
    path = data_dir / OPEN_REQUEST_NAME
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:  # on Windows, while the other process is still writing it: next time
        return False
    return True


def wait_taken(
    data_dir: Path,
    timeout: float | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """For a second start: whether the running app took the request within `timeout`
    (default ANSWER_SECONDS)."""
    path = data_dir / OPEN_REQUEST_NAME
    end = clock() + (ANSWER_SECONDS if timeout is None else timeout)
    while path.exists():
        if clock() >= end:
            return False
        sleep(0.1)
    return True
