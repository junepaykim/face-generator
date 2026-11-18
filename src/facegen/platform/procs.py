"""Run subprocesses without console windows on Windows.

All launches go through `run_quiet` or `popen_quiet`, which add CREATE_NO_WINDOW on Windows
and use ordinary subprocess behavior elsewhere. `windows_tool` finds System32 tools when
PATH omits them, using %SystemRoot% on Windows and the C: drive from WSL.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, TextIO

from facegen.platform.platform_info import Kind

CREATE_NO_WINDOW = 0x08000000
"""Windows's process creation flag for a console program that gets no console window
(`subprocess.CREATE_NO_WINDOW`, which exists only on Windows)."""

CREATE_NEW_PROCESS_GROUP = 0x00000200
"""Isolate a Windows child from the parent's console signals."""

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


def popen_json(
    args: Sequence[str],
    *,
    detached: bool,
    stdin: int = subprocess.PIPE,
    env: Mapping[str, str] | None = None,
    os_name: str = os.name,
) -> subprocess.Popen[str]:
    """Quiet UTF-8 line pipes. Environment and console-signal isolation are caller policy."""
    extra: dict[str, Any] = {}
    if detached:
        if os_name == "nt":
            extra["creationflags"] = CREATE_NEW_PROCESS_GROUP
        else:
            extra["start_new_session"] = True
    return popen_quiet(
        args,
        os_name=os_name,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        **extra,
    )


def start_console(
    program: str,
    arguments: str,
    *,
    cwd: str,
    os_name: str = os.name,
    startfile: Callable[..., None] | None = None,
) -> None:
    """Start `program` in a console window of its own, detached from facegen (Windows's
    ShellExecute): for a step the user should watch and that outlives facegen, such as
    `Start.bat --update`. Raises OSError where it cannot."""
    if os_name != "nt":
        raise OSError("a program in a console of its own is started on Windows only")
    run = startfile if startfile is not None else getattr(os, "startfile", None)
    if run is None:
        raise OSError("os.startfile is not available")
    run(program, "open", arguments, cwd)


def close_streams(*streams: TextIO | None) -> None:
    """Close supplied streams, tolerating OS errors from a pipe whose peer has gone away."""
    for stream in streams:
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


def wait_or_kill(
    child: subprocess.Popen[str],
    *,
    timeout: float,
    on_timeout: Callable[[], None] | None = None,
) -> int:
    """Wait for exit, then kill and reap on timeout. The caller chooses the grace period."""
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if on_timeout is not None:
            on_timeout()
        child.kill()
        return child.wait()


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
"""FM27's executable name for process detection."""


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
