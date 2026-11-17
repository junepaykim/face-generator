"""Console output helpers that need only the standard library."""

from typing import TextIO


def utf8_output(*streams: TextIO) -> None:
    """Write UTF-8 whatever the console's code page: names such as James Rodríguez cannot be
    written in cp949 or cp1252, and a print would raise. Python already writes Unicode to a
    Windows console window; this is for output sent to a pipe or a file."""
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # already written to, or a closed stream
            continue
