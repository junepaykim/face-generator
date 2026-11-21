"""platform.memory: freed heap memory goes back to the OS on Linux, and nothing breaks elsewhere."""

from __future__ import annotations

import sys

import pytest

from facegen.platform import memory


def test_release_free_memory_trims_on_linux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert memory.release_free_memory() is sys.platform.startswith("linux")
    monkeypatch.setattr(memory.sys, "platform", "win32")
    assert memory.release_free_memory() is False


def test_a_c_library_without_malloc_trim_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoTrim:  # musl's libc has no malloc_trim
        pass

    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda name: NoTrim())
    assert memory.release_free_memory() is False
