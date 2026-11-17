"""JSON lines over a child process's stdin and stdout: the worker process (worker_process.py)
and the dashboard's window process (app_window.py) speak this way to their parent.

One JSON object per line, in ASCII, so no console encoding can break it. Standard library
only, so a child that imports this loads nothing else.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TextIO


def encode(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=True, separators=(",", ":")) + "\n"


def decode(line: str) -> dict[str, Any] | None:
    """One message, or None for a line that is not one."""
    try:
        message = json.loads(line)
    except ValueError:
        return None
    return message if isinstance(message, dict) else None


class JsonWriter:
    """Serialize each JSON line and its flush, including across a child's replacement.

    Callers own broken-pipe handling. The reentrant lock also lets a caller read state and
    send its message under the same lock, so concurrent status reports cannot be reordered.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()

    def send(self, out: TextIO, message: dict[str, Any]) -> None:
        self.send_encoded(out, encode(message))

    def send_encoded(self, out: TextIO, line: str) -> None:
        """Send a line from `encode`, letting callers separate encoding and pipe errors."""
        with self.lock:
            out.write(line)
            out.flush()


def drain_lines(stream: Iterable[str], receive: Callable[[str], None]) -> None:
    """Drain a text stream, stripping line endings only; the caller owns logging and errors."""
    for raw in stream:
        receive(raw.rstrip("\r\n"))


def protocol_streams(stdin: TextIO, stdout: TextIO) -> tuple[Iterator[str], TextIO]:
    """The pipe to the parent: commands in, messages out.

    From here on anything else written to stdout (a library's print) goes to stderr, so it
    cannot break a message. Call once, in the child's main thread, before the models load.
    """
    stdout.flush()
    messages = os.fdopen(os.dup(stdout.fileno()), "w", encoding="ascii", newline="\n", buffering=1)
    os.dup2(sys.stderr.fileno(), stdout.fileno())
    return fd_lines(stdin.fileno()), messages


PIPE_POLL_SECONDS = 0.05
"""How often the worker process looks for a command on Windows (`pipe_reader`)."""


def pipe_reader(fd: int, *, poll: float = PIPE_POLL_SECONDS) -> Callable[[], bytes]:
    """A function that returns the next bytes from `fd`, b"" at the end of input.

    On Windows a pipe is not read until something is waiting in it (PeekNamedPipe). A read
    left waiting on a pipe handle holds up every other call on that handle, and loading a
    DLL asks Windows about the standard handles: with a thread waiting on stdin, `import
    torch` stopped inside numpy until the parent next sent a command, which could be never
    (docs/research.md section 10). Elsewhere, and for anything but a pipe, a plain read.
    """
    if os.name != "nt":
        return lambda: os.read(fd, 4096)
    import _winapi
    import msvcrt

    handle = msvcrt.get_osfhandle(fd)
    if _winapi.GetFileType(handle) != _winapi.FILE_TYPE_PIPE:
        return lambda: os.read(fd, 4096)

    def read() -> bytes:
        while True:
            try:
                available, _ = _winapi.PeekNamedPipe(handle, 0)
            except OSError:  # the parent closed its end (ERROR_BROKEN_PIPE): the end
                return b""
            if available:
                return os.read(fd, min(available, 65536))
            time.sleep(poll)

    return read


def fd_lines(fd: int) -> Iterator[str]:
    """The lines read from a file descriptor, without Python's buffered reader.

    The commands are read in a daemon thread that is still waiting when the loop ends. A
    thread waiting inside sys.stdin's buffered reader holds its lock, and the interpreter
    cannot take that lock as it shuts down: it aborts ("could not acquire lock for
    <stdin> at interpreter shutdown", SIGABRT) after a few seconds. A bare `os.read` holds
    no such lock. On Windows the pipe is polled instead of waited on (`pipe_reader`).
    """
    read = pipe_reader(fd)
    pending = b""
    while chunk := read():
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            yield line.decode("utf-8", errors="replace")
    if pending:
        yield pending.decode("utf-8", errors="replace")
