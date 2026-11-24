"""Show a folder in Windows Explorer (Phase 17i): the Log page's Open log folder.

Windows only: under WSL or Linux there is no Explorer for the app to open, so the page hides
the button (`available`). `os.startfile` hands the folder to the shell, which opens it in
Explorer without a console window.
"""

from __future__ import annotations

import os
from pathlib import Path

from facegen.platform import platform_info


def available() -> bool:
    """Explorer can be opened from this process: native Windows."""
    return platform_info.current() == "windows"


def open_folder(path: Path) -> None:
    """Show `path` in Explorer. Raises OSError when the folder cannot be opened."""
    startfile = getattr(os, "startfile", None)
    if startfile is None:
        raise OSError("folders open in Explorer on Windows only")
    startfile(str(path))
