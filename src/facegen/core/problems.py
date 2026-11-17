"""What went wrong, as one of a few kinds a person can act on (Phase 12e).

Errors arrive as text: an exception's type and message from the worker, a sync or the model
download. The pages say what happened and what to do for the kind, with the text itself folded
away as the technical details. The first kind whose words the text contains wins, so the
narrower kinds come first (a "CUDA error: out of memory" is a memory problem, not a fault).
"""

from __future__ import annotations

UNKNOWN = "unknown"

_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "gpu_memory",
        (
            "out of memory",
            "outofmemoryerror",
            "cuda_error_out_of_memory",
            "cudaerrormemoryallocation",
        ),
    ),
    (
        "no_gpu",
        (
            "torch not compiled with cuda",
            "no cuda gpus are available",
            "cuda driver version is insufficient",
            "no kernel image is available",
            "found no nvidia driver",
            "no cuda graphics card",
        ),
    ),
    (
        "gpu_fault",
        (
            "illegal memory access",
            "cuda error",
            "device-side assert",
            "cudnn_status",
            "cublas_status",
            "unspecified launch failure",
            "gpu fault",
            "graphics driver",
        ),
    ),
    ("disk_full", ("no space left on device", "[errno 28]", "[winerror 112]", "disk is full")),
    ("no_permission", ("permission denied", "[errno 13]", "[winerror 5]", "access is denied")),
    ("file_in_use", ("[winerror 32]", "being used by another process")),
    ("db_busy", ("database is locked", "database is busy")),
    ("folder", ("folder is not set", "fm27's folder is not")),
    ("no_save", ("no save found", "no such save")),
    ("save_busy", ("kept changing while it was copied",)),
    (
        "models_missing",
        (
            "models are not downloaded",
            "localentrynotfounderror",
            "cannot find the requested files in the disk cache",
            "does not appear to have a file named",
        ),
    ),
    (
        "network",
        (
            "connectionerror",
            "connection refused",
            "connection reset",
            "max retries exceeded",
            "getaddrinfo failed",
            "name resolution",
            "timed out",
            "readtimeout",
            "sslerror",
        ),
    ),
)

KINDS = (*(kind for kind, _ in _KINDS), UNKNOWN)


def diagnose(text: str | None) -> str:
    """The kind of problem an error's text describes; UNKNOWN when none fits."""
    low = (text or "").lower()
    for kind, words in _KINDS:
        if any(word in low for word in words):
            return kind
    return UNKNOWN
