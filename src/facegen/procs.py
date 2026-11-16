"""Running other programs without a console window flashing up on Windows.

Every subprocess facegen starts goes through `run_quiet` or `popen_quiet`. On Windows they add
CREATE_NO_WINDOW, so a console program (tasklist.exe, the worker's python.exe) started by a
window-less facegen (pythonw.exe, Phase 11) opens no console of its own; elsewhere they are
plain `subprocess.run` and `subprocess.Popen`.

`windows_tool` finds a program in Windows's System32 when it is not on PATH: the folder named
by %SystemRoot% on Windows, and the C: drive's from WSL (whose PATH may leave Windows out).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from facegen.platform_info import Kind

CREATE_NO_WINDOW = 0x08000000
"""Windows's process creation flag for a console program that gets no console window
(`subprocess.CREATE_NO_WINDOW`, which exists only on Windows)."""

WSL_SYSTEM32 = PurePosixPath("/mnt/c/Windows/System32")


def quiet_flags(os_name: str = os.name) -> int:
    """The creation flags that keep a child's console hidden: CREATE_NO_WINDOW on Windows."""
    return CREATE_NO_WINDOW if os_name == "nt" else 0


def _quiet(kwargs: dict[str, Any], os_name: str) -> dict[str, Any]:
    flags = quiet_flags(os_name)
    if flags:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | flags
    return kwargs


def run_quiet(
    args: Sequence[str], *, timeout: float, os_name: str = os.name, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """`subprocess.run` with no console window on Windows. A timeout is required: nothing
    facegen asks of another program may hang it."""
    kwargs.setdefault("check", False)
    return subprocess.run(list(args), timeout=timeout, **_quiet(kwargs, os_name))  # noqa: PLW1510


def popen_quiet(args: Sequence[str], *, os_name: str = os.name, **kwargs: Any) -> subprocess.Popen:
    """`subprocess.Popen` with no console window on Windows."""
    return subprocess.Popen(list(args), **_quiet(kwargs, os_name))


def windows_tool(
    name: str,
    kind: Kind,
    env: Mapping[str, str],
    *,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> str | None:
    """Where a Windows program in System32 is (`tasklist.exe`, `powershell.exe`'s folder is
    given as `WindowsPowerShell\\v1.0\\powershell.exe`): on PATH first, then System32. None on
    Linux, or when it is in neither place."""
    if kind == "linux":
        return None
    found = which(PureWindowsPath(name).name)
    if found:
        return found
    if kind == "windows":
        root = env.get("SystemRoot") or env.get("SYSTEMROOT") or env.get("windir") or r"C:\Windows"
        path = str(PureWindowsPath(root, "System32", name))
    else:
        path = str(WSL_SYSTEM32.joinpath(*PureWindowsPath(name).parts))
    return path if exists(path) else None


FM_PROCESS = "fm.exe"
"""FM27's process (user, 2026-11-13: it stays `fm.exe`)."""


def tasklist_command(tasklist: str, image: str = FM_PROCESS) -> list[str]:
    """tasklist's command line that lists only `image`, as CSV without a header."""
    return [tasklist, "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"]


def tasklist_says_running(stdout: bytes, image: str = FM_PROCESS) -> bool:
    """Whether tasklist's CSV output names `image`. With no match it prints a line of text
    in the system's language instead ("INFO: No tasks are running..."), never a quoted name."""
    return f'"{image}"'.encode() in stdout.lower()


def linux_process_running(comms: Sequence[str], image: str = FM_PROCESS) -> bool:
    """Whether a process named `image` runs, from the processes' names (/proc/<pid>/comm),
    as FM27 does under Wine or Proton. The kernel keeps 15 characters of a name."""
    wanted = image[:15].lower()
    return any(c.strip().lower() == wanted for c in comms)
