"""Hand freed heap memory back to the operating system where the C library keeps it."""

from __future__ import annotations

import ctypes
import ctypes.util
import sys


def release_free_memory() -> bool:
    """On Linux with glibc, return freed heap memory to the OS (`malloc_trim`); returns
    whether it was asked. Elsewhere this does nothing.

    glibc keeps freed memory for reuse: after Z-Image-Turbo was closed, 7.8 of its 8.6 GiB of
    private memory stayed with the process, 1.4 GiB after a trim; an encode trip that moves
    the transformer through system memory left 4.4 GiB behind the same way
    (docs/research.md section 20). Windows' heap is not measured yet.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        trim = libc.malloc_trim
    except (OSError, AttributeError):  # not glibc (musl has no malloc_trim)
        return False
    trim(0)
    return True
