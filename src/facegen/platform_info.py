"""Which kind of system facegen runs on: native Windows, WSL, or another Linux.

Pure: `kind` takes the platform name and the kernel's version text as arguments, so the
tests can ask about any system. `current` reads this one's.
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path
from typing import Literal

Kind = Literal["windows", "wsl", "linux"]


def kind(sys_platform: str, proc_version: str | None) -> Kind:
    """`windows` for native Windows, `wsl` for Linux under WSL, else `linux`.

    `proc_version` is the text of /proc/version (None if it could not be read): WSL's
    kernels name Microsoft there ("...-microsoft-standard-WSL2", "...-Microsoft").
    """
    if sys_platform == "win32":
        return "windows"
    if sys_platform.startswith("linux") and proc_version and "microsoft" in proc_version.lower():
        return "wsl"
    return "linux"


@cache
def current() -> Kind:
    """This system's kind; read once."""
    try:
        text = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = None
    return kind(sys.platform, text)
