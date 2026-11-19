"""Who already has a picture in FM27 other than ours (docs/research.md section 4).

Two sources, both only read:

- **FM27's own photos.** The game loads them through Unity Addressables, and its catalog,
  `fm_Data/StreamingAssets/AddrData/catalog_info.bin` in the install folder, has one key
  `person_<unique ID>` per photo it ships. The photos themselves (`players-*.bundle`) are
  never opened.
- **Facepacks the user installed.** Every `config.xml` under FM27's graphics folder, except
  our own `graphics/facegen/`, maps people to pictures with records whose `to` is
  `graphics/pictures/person/<unique ID>/portrait` or its generated-person `r-<unique ID>` form.

A pre-existing player with neither shows a silhouette in FM27; with `no_photo.enabled` the
group rules pick them like newgens (docs/spec.md section 2).

Each sync scans again, cheaply: the catalog's version is its hash file plus its size and time,
and a folder is listed again only when its modification time changed (an entry was added,
removed or renamed), so a facepack folder of 300,000 pictures is not listed at every sync.
The IDs are re-read only when the catalog or a `config.xml` changed. What was looked at is
stored with the IDs, in the tables `photos` and `photo_scan`.

Nothing here writes under FM27's folders (AGENTS.md hard rule 2).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from facegen.core.settings_values import Paths
from facegen.persistence.photo_index import PhotoStatus, player_counts, status, store, stored_state

__all__ = [
    "CATALOG_DIR",
    "CATALOG_FILE",
    "CATALOG_HASH",
    "OUR_PACKS",
    "PhotoIndex",
    "PhotoScan",
    "PhotoStatus",
    "facepack_ids",
    "fingerprint",
    "game_photo_ids",
    "player_counts",
    "scan",
    "status",
    "store",
    "stored_state",
]

CATALOG_DIR = Path("fm_Data", "StreamingAssets", "AddrData")
CATALOG_FILE = "catalog_info.bin"
CATALOG_HASH = "catalog_info.hash"
OUR_PACKS = "facegen"
"""graphics/facegen/: our own pictures, which never count as a photo."""

# Keys are length-prefixed strings; "person_default" and "person_default_w" (the silhouettes)
# have no digits and do not match.
_PERSON_KEY = re.compile(rb"person_([0-9]{1,10})(?![0-9A-Za-z_])")
_MAPPED_PERSON = re.compile(
    rb"""\bto\s*=\s*["']graphics/pictures/person/(?:r-)?([0-9]{1,10})/portrait["']""", re.IGNORECASE
)


def game_photo_ids(catalog: bytes) -> frozenset[int]:
    """The unique IDs FM27 ships a photo for, from its Addressables catalog."""
    return frozenset(int(m) for m in _PERSON_KEY.findall(catalog))


def facepack_ids(config_xml: bytes) -> frozenset[int]:
    """The unique IDs a facepack's config.xml maps to a portrait."""
    return frozenset(int(m) for m in _MAPPED_PERSON.findall(config_xml))


@dataclass(frozen=True, slots=True)
class PhotoIndex:
    """Who has a photo, as read from FM27's folders."""

    game: frozenset[int] | None
    """IDs with a photo FM27 ships; None when its catalog could not be read."""
    facepack: frozenset[int]
    """IDs a facepack's config.xml maps."""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PhotoScan:
    """One look at FM27's folders."""

    state: dict[str, Any]
    """What was looked at (JSON): the catalog's version, each config.xml's size and time, and
    each folder's time and entries, so the next scan can skip what did not change."""
    index: PhotoIndex | None
    """The IDs, read again because something changed; None when the inputs are the same as
    in `previous` or when a read failed, so the stored IDs stand."""
    error: str | None = None
    """External mapping discovery/read failed; do not replace the last successful index."""


def fingerprint(state: Mapping[str, Any]) -> list[Any]:
    """The inputs that decide the IDs: the catalog's version and every config.xml's."""
    return [state.get("format"), state.get("catalog"), state.get("configs", [])]


def _catalog_version(install_dir: Path) -> str | None:
    folder = install_dir / CATALOG_DIR
    try:
        digest = (folder / CATALOG_HASH).read_text(encoding="ascii", errors="replace").strip()
        st = (folder / CATALOG_FILE).stat()
    except OSError:
        return None
    return f"{digest}:{st.st_size}:{st.st_mtime_ns}"


def _list(folder: Path) -> tuple[list[str], str | None]:
    """A folder's sub-folders, sorted, and the name of its config.xml if it has one."""
    subdirs: list[str] = []
    config = None
    with os.scandir(folder) as entries:
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                subdirs.append(e.name)
            elif e.name.lower() == "config.xml" and e.is_file():
                config = e.name
    return sorted(subdirs), config


def _walk(graphics: Path, previous: Mapping[str, Any]) -> tuple[dict[str, Any], list[list[Any]]]:
    """Every folder under graphics/ but ours, and every config.xml in them.

    A folder whose modification time is the one in `previous` is not listed again: its
    sub-folders and config.xml are taken from there. Returns (folders, configs), each config
    as [path relative to graphics/, size, modification time in ns].
    """
    folders: dict[str, Any] = {}
    configs: list[list[Any]] = []
    stack = [""]
    while stack:
        rel = stack.pop()
        path = graphics / rel if rel else graphics
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        old = previous.get(rel)
        if old is not None and old.get("mtime") == mtime:
            subdirs, config = list(old["subdirs"]), old["config"]
        else:
            subdirs, config = _list(path)
        folders[rel] = {"mtime": mtime, "subdirs": subdirs, "config": config}
        if config:
            try:
                st = (path / config).stat()
            except FileNotFoundError:
                pass
            else:
                configs.append([f"{rel}/{config}" if rel else config, st.st_size, st.st_mtime_ns])
        for name in reversed(subdirs):
            if not rel and name.lower() == OUR_PACKS:
                continue
            stack.append(f"{rel}/{name}" if rel else name)
    return folders, sorted(configs)


def _read_index(paths: Paths, configs: list[list[Any]]) -> PhotoIndex:
    warnings: list[str] = []
    game: frozenset[int] | None
    if paths.fm_install_dir is None:
        game = None
        warnings.append(
            "FM27's install folder is not set and was not found on this PC, so no pre-existing"
            " player counts as without a photo; set paths.fm_install_dir in config/local.toml"
        )
        return PhotoIndex(game, _facepack_ids(paths, configs), tuple(warnings))
    catalog = paths.fm_install_dir / CATALOG_DIR / CATALOG_FILE
    try:
        game = game_photo_ids(catalog.read_bytes())
    except OSError:
        game = None
        warnings.append(
            f"FM27's photo list ({catalog}) could not be read, so no pre-existing player counts "
            "as without a photo; check paths.fm_install_dir"
        )
    else:
        if not game:
            game = None
            warnings.append(
                f"FM27's photo list ({catalog}) names no photos: has a game update changed its "
                "format? No pre-existing player counts as without a photo (docs/research.md)"
            )
    return PhotoIndex(game, _facepack_ids(paths, configs), tuple(warnings))


def _facepack_ids(paths: Paths, configs: list[list[Any]]) -> frozenset[int]:
    facepack: set[int] = set()
    graphics = paths.graphics_dir
    for rel, _size, _mtime in configs if graphics is not None else ():
        facepack.update(facepack_ids((graphics / rel).read_bytes()))
    return frozenset(facepack)


def scan(paths: Paths, previous: Mapping[str, Any] | None = None) -> PhotoScan:
    """Look at FM27's folders; read the IDs again only if the catalog or a config.xml changed.

    `previous` is the state the last scan stored (`stored_state`), or None to read everything.
    """
    previous = previous or {}
    graphics = paths.graphics_dir
    old_folders = previous.get("folders", {}) if previous.get("format") == 2 else {}
    try:
        folders, configs = _walk(graphics, old_folders) if graphics is not None else ({}, [])
        install = paths.fm_install_dir
        state = {
            # Older releases ignored r- mappings and could persist a partial scan as complete.
            "format": 2,
            "catalog": _catalog_version(install) if install is not None else None,
            "configs": configs,
            "folders": folders,
        }
        if not previous.get("retry") and fingerprint(previous) == fingerprint(state):
            return PhotoScan(state, None)
        index = _read_index(paths, configs)
    except OSError as exc:
        return PhotoScan(
            {**previous, "retry": True},
            None,
            f"Could not scan external facepacks; will retry: {exc}",
        )
    if index.game is None and state["catalog"] is not None:
        state["retry"] = True
    return PhotoScan(state, index)
