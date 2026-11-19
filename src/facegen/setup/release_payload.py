"""Select and unpack the identical source payload for installation and in-place updates."""

from __future__ import annotations

import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

MAX_UNPACKED_BYTES = 400 * 1024 * 1024
MAX_ENTRIES = 20_000

LEFT_OUT = frozenset(
    {".github", "tests", "scripts", "docs", "installer", ".gitignore", ".gitattributes",
     "AGENTS.md", "CLAUDE.md"}
)  # fmt: skip
"""Top-level entries of the source that a release does not ship: developer files."""
NEVER_WRITTEN = (".git", ".venv", ".bin", "data", "config/local.toml")
"""This PC's own: an archive that had them is refused rather than written over them."""
MANAGED_DIRS = ("src", "native")
"""Folders whose files are the release's alone: a file the new release lacks is removed."""
SKIPPED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})
SKIPPED_UNDER = (PurePosixPath("native/target"),)
"""Build output, never part of a payload and never removed by an update."""


class UpdateError(Exception):
    """The check or the update could not be done (the message says why)."""


def _skipped(rel: PurePosixPath) -> bool:
    return any(part in SKIPPED_PARTS for part in rel.parts) or any(
        rel == under or under in rel.parents for under in SKIPPED_UNDER
    )


def never_written(rel: PurePosixPath) -> bool:
    return any(rel == PurePosixPath(p) or PurePosixPath(p) in rel.parents for p in NEVER_WRITTEN)


def in_payload(rel: PurePosixPath) -> bool:
    """Whether a file of the source (relative, '/'-separated) is shipped in a release."""
    if not rel.parts or rel.parts[0] in LEFT_OUT or rel.name == "AGENTS.md" or _skipped(rel):
        return False
    return not never_written(rel)


def safe_relative(name: str) -> PurePosixPath:
    """An archive member's path, refused when it would land outside the folder."""
    if "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise UpdateError(f"the archive has an unsafe path: {name!r}")
    rel = PurePosixPath(name)
    if any(part in ("..", "") for part in rel.parts):
        raise UpdateError(f"the archive has an unsafe path: {name!r}")
    return rel


def unpack_payload(archive: Path, dest: Path) -> list[PurePosixPath]:
    """Unpack the release payload of a GitHub source archive (one top folder) into `dest`.
    Returns the payload's files. Raises UpdateError for anything unexpected."""
    try:
        zf = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdateError(f"the downloaded archive cannot be read: {exc}") from exc
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if not members or len(members) > MAX_ENTRIES:
            raise UpdateError("the archive is empty or has too many files")
        if sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
            raise UpdateError("the archive unpacks to more than facegen's source can be")
        tops = {safe_relative(m.filename).parts[0] for m in members}
        if len(tops) != 1:
            raise UpdateError("the archive is not one folder of facegen's source")
        files: list[PurePosixPath] = []
        for m in members:
            rel = PurePosixPath(*safe_relative(m.filename).parts[1:])
            if stat.S_ISLNK(m.external_attr >> 16):
                raise UpdateError(f"the archive has a link: {m.filename!r}")
            if not rel.parts:
                continue
            if never_written(rel):
                raise UpdateError(f"the archive has {rel}, which is this PC's own")
            if not in_payload(rel):
                continue
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            files.append(rel)
    if PurePosixPath("pyproject.toml") not in files or PurePosixPath("Start.bat") not in files:
        raise UpdateError("the archive is not facegen's source (no pyproject.toml or Start.bat)")
    return sorted(files)


def managed_files(root: Path) -> list[PurePosixPath]:
    """This folder's files that belong to the release alone (MANAGED_DIRS)."""
    out = []
    for top in MANAGED_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            rel = PurePosixPath(path.relative_to(root).as_posix())
            if path.is_file() and not _skipped(rel):
                out.append(rel)
    return out
