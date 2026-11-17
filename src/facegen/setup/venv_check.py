"""Packages left half-installed by an interrupted `uv sync` (Start.bat, Phase 11).

Found 2026-11-16: the console was closed while the first `uv sync` was unpacking torch. Its
`torch-<version>.dist-info` folder was there with neither METADATA nor RECORD, and some of its
files were missing (`torchgen`). uv counts a package as installed by that folder's name, so
every later `uv sync` said all was well, and torch could not be imported.

`python -m facegen.venv_check` lists such packages: it prints the `uv sync` options that
install them again (`--reinstall-package NAME ...`, one line) and exits 5, or prints nothing
and exits 0. Start.bat then runs `uv sync` once more with those options, which puts back every
file and completes the folder. Standard library only: a half-installed package must not stop
this from running.
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
