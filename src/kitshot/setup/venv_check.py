"""Detect incomplete distributions left by interrupted uv sync.

uv may accept a .dist-info directory without METADATA or RECORD as installed, leaving
imports broken. `python -m kitshot.venv_check` prints `--reinstall-package NAME ...` on
one line and exits 5 for these packages, or prints nothing and exits 0. Start.bat retries
uv sync with those options. Standard library only, so broken packages cannot prevent repair.
"""

from __future__ import annotations

import sys
import sysconfig
from collections.abc import Iterable
from pathlib import Path

BROKEN = 5
NEEDED = ("METADATA", "RECORD")


def broken(site_dirs: Iterable[Path]) -> list[str]:
    """The distributions in these folders whose .dist-info lacks METADATA or RECORD, by name
    (as in the folder's name: `torch-2.15.0+cu132.dist-info` is `torch`)."""
    names: list[str] = []
    for folder in dict.fromkeys(site_dirs):
        try:
            entries = sorted(folder.glob("*.dist-info"))
        except OSError:
            continue
        for info in entries:
            if not info.is_dir():
                continue
            if all((info / name).is_file() for name in NEEDED):
                continue
            name = info.name.removesuffix(".dist-info").split("-", 1)[0]
            if name and name not in names:
                names.append(name)
    return names


def reinstall_options(names: Iterable[str]) -> str:
    return " ".join(f"--reinstall-package {name}" for name in names)


def site_dirs() -> list[Path]:
    paths = sysconfig.get_paths()
    return [Path(paths["purelib"]), Path(paths["platlib"])]


def main() -> int:
    names = broken(site_dirs())
    if not names:
        return 0
    print(reinstall_options(names))
    print(f"half-installed: {', '.join(names)}", file=sys.stderr)
    return BROKEN


if __name__ == "__main__":
    raise SystemExit(main())
