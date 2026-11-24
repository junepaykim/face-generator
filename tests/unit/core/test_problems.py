"""Errors sorted into the kinds a person can act on (Phase 12e)."""

from __future__ import annotations

import pytest

from kitshot.core.problems import KINDS, UNKNOWN, diagnose


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        # The line that started the UI plan: a raw CUDA message, and an AttributeError.
        ("Mine: RuntimeError: CUDA error: an illegal memory access was encountered", "gpu_fault"),
        ("AttributeError: 'NoneType' object has no attribute 'unet'", UNKNOWN),
        ("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "gpu_memory"),
        ("RuntimeError: CUDA error: out of memory", "gpu_memory"),  # memory, not a fault
        ("AssertionError: Torch not compiled with CUDA enabled", "no_gpu"),
        ("OSError: [Errno 28] No space left on device: 'data/masters/x.png'", "disk_full"),
        ("PermissionError: [WinError 5] Access is denied: 'C:\\\\FM\\\\graphics'", "no_permission"),
        ("PermissionError: [Errno 13] Permission denied: '/fm/graphics/kitshot'", "no_permission"),
        ("[WinError 32] The process cannot access the file because it is being used by another process", "file_in_use"),
        ("sqlite3.OperationalError: database is locked", "db_busy"),
        ("FM27's folder is not set and was not found on this PC", "folder"),
        ("No save found in /fm/games", "no_save"),
        ("career.fm kept changing while it was copied; is FM27 saving?", "save_busy"),
        ("the models are not downloaded yet: download them once from the dashboard", "models_missing"),
        ("HTTPSConnectionPool(host='huggingface.co'): Max retries exceeded", "network"),
        ("the download stopped (ReadTimeout: timed out)", "network"),
        ("", UNKNOWN),
        (None, UNKNOWN),
    ],
)  # fmt: skip
def test_diagnose(text: str | None, kind: str) -> None:
    assert diagnose(text) == kind


def test_every_kind_is_listed_once() -> None:
    assert len(set(KINDS)) == len(KINDS) and KINDS[-1] == UNKNOWN
