"""Native Windows pipe-handle regression; synthetic child, no GPU or GUI."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.windows_only


@pytest.mark.skipif(os.name != "nt", reason="how a Windows pipe handle behaves")
def test_waiting_for_a_command_leaves_the_stdin_handle_free() -> None:
    """While the worker waits for a command, Windows must still answer questions about its
    stdin handle: loading a DLL asks some, and a read left waiting on the pipe held `import
    torch` up until the next command, which might never come (docs/research.md section 10).
    GetFileInformationByHandle is one of them: it waited behind a plain `os.read`."""
    code = (
        "import ctypes, msvcrt, threading, time\n"
        "from kitshot.runtime.worker_process import fd_lines\n"
        "threading.Thread(target=lambda: list(fd_lines(0)), daemon=True).start()\n"
        "time.sleep(0.5)\n"
        "info = ctypes.create_string_buffer(64)\n"
        "ctypes.windll.kernel32.GetFileInformationByHandle(msvcrt.get_osfhandle(0), info)\n"
        "print('answered', flush=True)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    got: list[str] = []
    assert child.stdout is not None
    reader = threading.Thread(target=lambda: got.append(child.stdout.readline()), daemon=True)
    reader.start()
    reader.join(30)  # stdin stays open: no command ever comes
    child.kill()
    child.wait()
    assert got and got[0].strip() == "answered"
