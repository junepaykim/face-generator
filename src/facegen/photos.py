"""Who already has a picture in FM27 other than ours (docs/research.md section 4).

Two sources, both only read:

- **FM27's own photos.** The game loads them through Unity Addressables, and its catalog,
  `fm_Data/StreamingAssets/AddrData/catalog_info.bin` in the install folder, has one key
  `person_<unique ID>` per photo it ships. The photos themselves (`players-*.bundle`) are
  never opened.
- **Facepacks the user installed.** Every `config.xml` under FM27's graphics folder, except
  our own `graphics/facegen/`, maps people to pictures with records whose `to` is
  `graphics/pictures/person/<unique ID>/portrait`.

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

import json
import os
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from facegen.settings_values import Paths

CATALOG_DIR = Path("fm_Data", "StreamingAssets", "AddrData")
CATALOG_FILE = "catalog_info.bin"
CATALOG_HASH = "catalog_info.hash"
OUR_PACKS = "facegen"
"""graphics/facegen/: our own pictures, which never count as a photo."""

# Keys are length-prefixed strings; "person_default" and "person_default_w" (the silhouettes)
# have no digits and do not match.
_PERSON_KEY = re.compile(rb"person_([0-9]{1,10})(?![0-9A-Za-z_])")
_MAPPED_PERSON = re.compile(
    rb"""\bto\s*=\s*["']graphics/pictures/person/([0-9]{1,10})/portrait["']""", re.IGNORECASE
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
    in `previous`, so the stored IDs stand."""


def fingerprint(state: Mapping[str, Any]) -> list[Any]:
    """The inputs that decide the IDs: the catalog's version and every config.xml's."""
    return [state.get("catalog"), state.get("configs", [])]


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
            try:
                if e.is_dir(follow_symlinks=False):
                    subdirs.append(e.name)
                elif e.name.lower() == "config.xml" and e.is_file():
                    config = e.name
            except OSError:
                continue
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
        except OSError:
            continue
        old = previous.get(rel)
        if old is not None and old.get("mtime") == mtime:
            subdirs, config = list(old["subdirs"]), old["config"]
        else:
            try:
                subdirs, config = _list(path)
            except OSError:
                continue
        folders[rel] = {"mtime": mtime, "subdirs": subdirs, "config": config}
        if config:
            try:
                st = (path / config).stat()
            except OSError:
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
        return PhotoIndex(game, _facepack_ids(paths, configs, warnings), tuple(warnings))
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
    return PhotoIndex(game, _facepack_ids(paths, configs, warnings), tuple(warnings))


def _facepack_ids(paths: Paths, configs: list[list[Any]], warnings: list[str]) -> frozenset[int]:
    facepack: set[int] = set()
    graphics = paths.graphics_dir
    for rel, _size, _mtime in configs if graphics is not None else ():
        try:
            facepack |= facepack_ids((graphics / rel).read_bytes())
        except OSError as exc:
            warnings.append(f"could not read the facepack mapping graphics/{rel}: {exc}")
    return frozenset(facepack)


def scan(paths: Paths, previous: Mapping[str, Any] | None = None) -> PhotoScan:
    """Look at FM27's folders; read the IDs again only if the catalog or a config.xml changed.

    `previous` is the state the last scan stored (`stored_state`), or None to read everything.
    """
    previous = previous or {}
    graphics = paths.graphics_dir
    folders, configs = (
        _walk(graphics, previous.get("folders", {})) if graphics is not None else ({}, [])
    )
    install = paths.fm_install_dir
    state = {
        "catalog": _catalog_version(install) if install is not None else None,
        "configs": configs,
        "folders": folders,
    }
    if previous and fingerprint(previous) == fingerprint(state):
        return PhotoScan(state, None)
    return PhotoScan(state, _read_index(paths, configs))


@dataclass(frozen=True, slots=True)
class PhotoStatus:
    """What the last stored scan found."""

    game_known: bool
    """FM27's photo list was read. If not, no one counts as without a photo."""
    game: int
    facepack: int
    facepack_files: int
    scanned_at: str
    changed_at: str
    """When the IDs last changed: a game update, or a facepack added, changed or removed."""
    warnings: tuple[str, ...]


def stored_state(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT state FROM photo_scan WHERE id = 1").fetchone()
    return json.loads(row[0]) if row else None


def store(conn: sqlite3.Connection, result: PhotoScan) -> bool:
    """Store a scan in the caller's transaction. Returns whether the IDs were rewritten."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    state = json.dumps(result.state)
    idx = result.index
    if idx is None:
        conn.execute("UPDATE photo_scan SET state = ?, scanned_at = ? WHERE id = 1", (state, now))
        return False
    conn.execute("DELETE FROM photos")
    game = idx.game or frozenset()
    conn.executemany(
        "INSERT INTO photos (unique_id, game, facepack) VALUES (?, ?, ?)",
        ((uid, int(uid in game), int(uid in idx.facepack)) for uid in sorted(game | idx.facepack)),
    )
    conn.execute(
        "INSERT INTO photo_scan (id, state, game_known, game, facepack, facepack_files,"
        " warnings, scanned_at, changed_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (id) DO UPDATE SET state = excluded.state,"
        " game_known = excluded.game_known, game = excluded.game,"
        " facepack = excluded.facepack, facepack_files = excluded.facepack_files,"
        " warnings = excluded.warnings, scanned_at = excluded.scanned_at,"
        " changed_at = excluded.changed_at",
        (
            state,
            int(idx.game is not None),
            len(game),
            len(idx.facepack),
            len(result.state["configs"]),
            json.dumps(list(idx.warnings)),
            now,
            now,
        ),
    )
    return True


def status(conn: sqlite3.Connection) -> PhotoStatus | None:
    """The last stored scan, or None before the first sync that scanned."""
    r = conn.execute(
        "SELECT game_known, game, facepack, facepack_files, warnings, scanned_at, changed_at"
        " FROM photo_scan WHERE id = 1"
    ).fetchone()
    if r is None:
        return None
    return PhotoStatus(
        game_known=bool(r["game_known"]),
        game=r["game"],
        facepack=r["facepack"],
        facepack_files=r["facepack_files"],
        scanned_at=r["scanned_at"],
        changed_at=r["changed_at"],
        warnings=tuple(json.loads(r["warnings"])),
    )


def player_counts(conn: sqlite3.Connection, career_id: int) -> dict[str, int]:
    """A career's pre-existing players in scope, staff left out: with a photo, without one,
    and those without one that the rules put in a group outside my club (unpinned). All 0
    until FM27's photo list has been read."""
    s = status(conn)
    if s is None or not s.game_known:
        return {"with_photo": 0, "without_photo": 0, "without_photo_picked": 0}
    row = conn.execute(
        "SELECT SUM(ph.unique_id IS NOT NULL) AS with_photo,"
        " SUM(ph.unique_id IS NULL) AS without_photo,"
        " SUM(ph.unique_id IS NULL AND p.group_name IS NOT NULL AND p.at_my_club = 0"
        "     AND p.pin IS NULL) AS without_photo_picked"
        " FROM players p LEFT JOIN photos ph ON ph.unique_id = p.unique_id"
        " WHERE p.career_id = ? AND p.present = 1 AND p.in_scope = 1 AND p.is_newgen = 0"
        " AND p.is_staff = 0 AND p.media_role IS NULL",
        (career_id,),
    ).fetchone()
    return {k: v or 0 for k, v in dict(row).items()}
