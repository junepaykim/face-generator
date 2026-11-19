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


@contextmanager
def installation_lock(root: Path, *, app: bool = False) -> Iterator[None]:
    """Installed apps share this lock. An updater needs exclusive use of the source tree.

    The lock belongs to the installation even if its configured data folder changes.
    Never unlink its file: a process may already have opened the same inode.
    """
    path = root / "data" / "install.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        # GENERIC_READ, FILE_SHARE_READ (apps only), OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL.
        handle = create(str(path), 0x80000000, int(app), None, 4, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            error = ctypes.get_last_error()
            if error == 32:  # ERROR_SHARING_VIOLATION
                raise LockBusy(str(path))
            raise ctypes.WinError(error)
        try:
            yield
        finally:
            kernel.CloseHandle(handle)
    else:
        import fcntl

        with path.open("a+b") as stream:
            mode = fcntl.LOCK_SH if app else fcntl.LOCK_EX
            try:
                fcntl.flock(stream, mode | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LockBusy(str(path)) from exc
            yield
