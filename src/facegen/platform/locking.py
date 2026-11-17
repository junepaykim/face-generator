"""Exclusive file locks for the worker and publisher, on Linux and Windows."""

from __future__ import annotations

import errno
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LockBusy(RuntimeError):
    """Another process or thread holds the file lock."""


@contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never truncate a file another process has locked.
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            # Windows can lock a byte beyond EOF; no write is necessary.
            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not blocking:
                        raise LockBusy(str(path)) from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise LockBusy(str(path)) from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
