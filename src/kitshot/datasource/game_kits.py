"""FM27's own kits, read from the game's database in its install folder (read only).

The database is `shared/data/database/db/<NNNN>/<NNNN>_fm/client_db.dat` (27.0.0 is `2700`).
It is a 12-byte header, then a container laid out as a save's: Zstandard frames, and at the
end a directory naming each frame (research.md, "FM27's kits"). Two tables are read, both
counted and walked to their exact end:

- `table_3`, the clubs. Each row opens with a kind byte. Kind 0: 26 bytes, the long and short
  names (u32 length, UTF-8), 3 bytes. Kind 12, an empty slot: 41 bytes. Every row closes with
  its own position, then the club's unique ID twice (u32 each). Nine bytes follow the rows.
- `table_extra_3`, one row per club in the same order: its position (u32), 8 bytes, a short
  text (u32 length), a count of kit entries (u8), the entries of 27 bytes each, 8 bytes.

A kit entry: bytes 2-11 five colours (RGB555, little-endian), byte 12 the style, bytes 13-20 a
competition ID twice (-1: any), bytes 21-22 a season (0: any), byte 23 the set (0 home,
1 away, 2 third), the low 4 bits of byte 24 the garment (1 shirt), byte 26 the wearer (1 the
outfield players, 0 the goalkeeper). A club's home shirt is the one entry of the home set,
shirt, outfield, any competition and any season; a club with none, or two, has no kit here.

Anything that does not walk as described makes the whole database unreadable: no club's kit
is taken from it (turn off, don't guess) and Kitshot keeps its other kit sources. What was read
is kept in the data folder (`game-kits.json`, colours as FM27 stores them) and read again only
when the database file or this reader changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kitshot.core.kit_values import RGB, ClubKits, GameShirt, game_kits

logger = logging.getLogger("kitshot")

READER_VERSION = 1
"""Raise when what this module reads changes: a cache from another version is read again."""
CACHE_NAME = "game-kits.json"
"""The cache, in the data folder."""
DB_DIR = Path("shared") / "data" / "database" / "db"
DB_FILE = "client_db.dat"
GAME_MAJOR = "27"
"""Database folders for this game start with it (`2700`); older games' folders are skipped."""
VERSION = re.compile(r'<string id="version" value="([0-9.]{1,20})"')

HEADER = b"\x03\x01tad."
CONTAINER = b"\x02\x01fmf."
CONTAINER_AT = 12
FRAME_BASE = 26
TRAILER_HEADER = 9
ENTRY_TAIL = struct.Struct("<5Q")
ENTRY_NAME = re.compile(rb"[A-Za-z0-9_]+")
MAX_DIRECTORY = 4 * 1024**2
MAX_TABLE = 256 * 1024**2
CLUBS = "table_3"
EXTRA = "table_extra_3"

TABLE_HEAD = 8
"""`03 01 tad.` and two bytes, then the row count (u32)."""
CLUB_ROW_KINDS = {0: 26, 12: 41}
"""A club row's kind byte -> the fixed bytes after it, before its names (kind 0) or footer."""
CLUB_TAIL = 3
CLUBS_AFTER_ROWS = 9
MAX_NAME = 512
KIT_ENTRY = 27
EXTRA_TAIL = 8
ANY = b"\xff" * 8
SHIRT, OUTFIELD, KEEPER, HOME = 1, 1, 0, 0

_U32 = struct.Struct("<I")
_FOOTER = struct.Struct("<III")


class GameKitsError(ValueError):
    """FM27's database is not laid out as this reader expects."""


@dataclass(frozen=True, slots=True)
class Database:
    """FM27's database file: where it is, its version (the folder's name) and identity."""

    path: Path
    version: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class GameKits:
    """What FM27's database gives: each club's home shirt and goalkeeper shirt."""

    database: Database
    sha256: str
    shirts: Mapping[int, tuple[GameShirt, GameShirt | None]]

    def kits(self) -> dict[int, ClubKits]:
        return {club: game_kits(home, keeper) for club, (home, keeper) in self.shirts.items()}


def find_database(install_dir: Path | None) -> Database | None:
    """FM27's newest database for this game in the install folder, or None."""
    if install_dir is None:
        return None
    root = install_dir / DB_DIR
    try:
        names = sorted(
            (p.name for p in root.iterdir() if p.name.isdigit() and p.name.startswith(GAME_MAJOR)),
            key=int,
            reverse=True,
        )
    except OSError:
        return None
    for name in names:
        path = root / name / f"{name}_fm" / DB_FILE
        try:
            st = path.stat()
        except OSError:
            continue
        return Database(path, _version(path.with_name(f"{name}_fm.xml")) or name, st.st_size,
                        st.st_mtime_ns)  # fmt: skip
    return None


def _version(xml: Path) -> str | None:
    """The version its description file gives ("27.0.0"), if it can be read."""
    try:
        found = VERSION.search(xml.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:
        return None
    return found.group(1) if found else None


# --- the container -------------------------------------------------------------------------


def _u32(data: bytes, at: int) -> int:
    if at + 4 > len(data):
        raise GameKitsError(f"ends at byte {len(data)}, inside a number at {at}")
    return _U32.unpack_from(data, at)[0]


def _decompress(compressed: bytes, most: int, what: str) -> bytes:
    """One Zstandard frame, decompressed to at most `most` + 1 bytes. Imported here, not at
    the top: `kitshot.cli --help` runs without installed packages (test_package_layout)."""
    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        from backports import zstd
    try:
        return zstd.ZstdDecompressor().decompress(compressed, max_length=most + 1)
    except zstd.ZstdError as exc:
        raise GameKitsError(f"{what} does not decompress: {exc}") from exc


def _frame(compressed: bytes, size: int, what: str) -> bytes:
    """One Zstandard frame that must decompress to exactly `size` bytes."""
    if size > MAX_TABLE:
        raise GameKitsError(f"{what} claims {size} bytes")
    data = _decompress(compressed, size, what)
    if len(data) != size:
        raise GameKitsError(f"{what} is {len(data)} bytes, not the {size} listed")
    return data


def read_tables(data: bytes, names: tuple[str, ...]) -> dict[str, bytes]:
    """The named tables of a database file's bytes, decompressed."""
    if data[: len(HEADER)] != HEADER:
        raise GameKitsError("not an FM database (no header)")
    base = CONTAINER_AT
    if data[base : base + len(CONTAINER)] != CONTAINER:
        raise GameKitsError("no container after the header")
    trailer = base + 9 + struct.unpack_from("<Q", data, base + 9)[0]
    if not base + FRAME_BASE <= trailer <= len(data) - TRAILER_HEADER:
        raise GameKitsError("the directory is outside the file")
    if data[trailer : trailer + len(CONTAINER)] != CONTAINER:
        raise GameKitsError("the directory is damaged")
    directory = _decompress(data[trailer + TRAILER_HEADER :], MAX_DIRECTORY, "the directory")
    if len(directory) > MAX_DIRECTORY:
        raise GameKitsError("the directory is implausibly large")
    cursor = 4 + _u32(directory, 0)  # the database's name
    count = _u32(directory, cursor)  # then the entries, counted; four zero bytes end them
    cursor += 4
    found: dict[str, tuple[int, int, int]] = {}
    for _ in range(count):
        length = _u32(directory, cursor)
        name = directory[cursor + 4 : cursor + 4 + length]
        cursor += 4 + length
        ext_length = _u32(directory, cursor)
        cursor += 4 + ext_length
        if not ENTRY_NAME.fullmatch(name) or ext_length != 4:
            raise GameKitsError(f"directory entry {name[:40]!r} is not understood")
        if cursor + ENTRY_TAIL.size > len(directory):
            raise GameKitsError("the directory ends inside an entry")
        offset, compressed, size, _, _ = ENTRY_TAIL.unpack_from(directory, cursor)
        cursor += ENTRY_TAIL.size
        key = name.decode("ascii")
        if key in found:
            raise GameKitsError(f"the directory lists {key} twice")
        found[key] = (base + FRAME_BASE + offset, compressed, size)
    tables = {}
    for name in names:
        if name not in found:
            raise GameKitsError(f"no {name}")
        start, compressed, size = found[name]
        if start + compressed > trailer:
            raise GameKitsError(f"{name} runs past the directory")
        tables[name] = _frame(data[start : start + compressed], size, name)
    return tables


# --- the two tables ------------------------------------------------------------------------


def _row_count(table: bytes, what: str) -> int:
    if table[: len(HEADER)] != HEADER:
        raise GameKitsError(f"{what}: no table header")
    return _u32(table, TABLE_HEAD)


def club_ids(table: bytes) -> list[int]:
    """`table_3`: each row's club unique ID, in row order."""
    count = _row_count(table, CLUBS)
    if table[TABLE_HEAD + 4 : TABLE_HEAD + 5] != b"\x00":
        raise GameKitsError(f"{CLUBS}: no zero byte after the count")
    cursor = TABLE_HEAD + 5
    ids: list[int] = []
    seen: set[int] = set()
    for row in range(count):
        if cursor >= len(table):
            raise GameKitsError(f"{CLUBS}: ends at row {row} of {count}")
        kind = table[cursor]
        if kind not in CLUB_ROW_KINDS:
            raise GameKitsError(f"{CLUBS}: row {row} is of an unknown kind ({kind})")
        cursor += 1 + CLUB_ROW_KINDS[kind]
        if kind == 0:
            for _ in range(2):
                length = _u32(table, cursor)
                if length > MAX_NAME:
                    raise GameKitsError(f"{CLUBS}: row {row} has a name of {length} bytes")
                cursor += 4 + length
            cursor += CLUB_TAIL
        if cursor + _FOOTER.size > len(table):
            raise GameKitsError(f"{CLUBS}: row {row} runs past the end")
        position, uid, again = _FOOTER.unpack_from(table, cursor)
        cursor += _FOOTER.size
        if position != row or uid != again or uid in seen:
            raise GameKitsError(f"{CLUBS}: row {row} does not close with its position and ID")
        seen.add(uid)
        ids.append(uid)
    if cursor + CLUBS_AFTER_ROWS != len(table):
        raise GameKitsError(f"{CLUBS}: {len(table) - cursor} bytes after the rows, not 9")
    return ids


def _rgb(entry: bytes, slot: int) -> RGB:
    """Colour slot 1-5 of a kit entry; FM stores 5 bits a channel, red highest."""
    value = struct.unpack_from("<H", entry, 2 * slot)[0]
    r, g, b = (round(((value >> shift) & 31) * 255 / 31) for shift in (10, 5, 0))
    return (r, g, b)


def _shirt(entries: list[bytes], wearer: int) -> GameShirt | None:
    """The one home shirt for any competition and season that `wearer` wears, if exactly one."""
    found = [
        e
        for e in entries
        if e[23] == HOME
        and e[24] & 15 == SHIRT
        and e[26] == wearer
        and e[13:21] == ANY
        and e[21:23] == b"\0\0"
    ]
    if len(found) != 1:
        return None
    e = found[0]
    return GameShirt(style=e[12], colours=(_rgb(e, 1), _rgb(e, 2), _rgb(e, 3)))


def home_shirts(table: bytes, ids: list[int]) -> dict[int, tuple[GameShirt, GameShirt | None]]:
    """`table_extra_3`: each club's home shirt and goalkeeper shirt, by unique ID."""
    count = _row_count(table, EXTRA)
    if count != len(ids):
        raise GameKitsError(f"{EXTRA} has {count} rows, the clubs {len(ids)}")
    cursor = TABLE_HEAD + 4
    shirts: dict[int, tuple[GameShirt, GameShirt | None]] = {}
    for row, uid in enumerate(ids):
        if _u32(table, cursor) != row:
            raise GameKitsError(f"{EXTRA}: row {row} does not open with its position")
        length = _u32(table, cursor + 12)
        if length > MAX_NAME:
            raise GameKitsError(f"{EXTRA}: row {row} has a text of {length} bytes")
        cursor += 16 + length
        if cursor >= len(table):
            raise GameKitsError(f"{EXTRA}: ends at row {row}")
        kits = table[cursor]
        cursor += 1
        entries = [
            table[cursor + i * KIT_ENTRY : cursor + (i + 1) * KIT_ENTRY] for i in range(kits)
        ]
        cursor += kits * KIT_ENTRY + EXTRA_TAIL
        if cursor > len(table):
            raise GameKitsError(f"{EXTRA}: row {row} runs past the end")
        home = _shirt(entries, OUTFIELD)
        if home is not None:
            shirts[uid] = (home, _shirt(entries, KEEPER))
    if cursor != len(table):
        raise GameKitsError(f"{EXTRA}: {len(table) - cursor} bytes after the rows")
    return shirts


def parse_database(data: bytes) -> dict[int, tuple[GameShirt, GameShirt | None]]:
    """Every club's home and goalkeeper shirt in a database file's bytes."""
    tables = read_tables(data, (CLUBS, EXTRA))
    return home_shirts(tables[EXTRA], club_ids(tables[CLUBS]))


# --- the cache -----------------------------------------------------------------------------


def _hex(rgb: RGB) -> str:
    return "".join(f"{c:02X}" for c in rgb)


def _from_hex(text: str) -> RGB:
    if not re.fullmatch(r"[0-9A-F]{6}", text):
        raise ValueError(f"not a colour: {text!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _shirt_json(shirt: GameShirt) -> list[Any]:
    return [shirt.style, *(_hex(c) for c in shirt.colours)]


def _shirt_from(value: Any) -> GameShirt:
    style, a, b, c = value
    if not isinstance(style, int):
        raise TypeError(f"not a style: {style!r}")
    return GameShirt(style, (_from_hex(a), _from_hex(b), _from_hex(c)))


def _db_json(db: Database) -> dict[str, Any]:
    return {"path": str(db.path), "version": db.version, "size": db.size, "mtime_ns": db.mtime_ns}


def read_cache(path: Path, db: Database) -> GameKits | None:
    """The cache, if it was written by this reader from this very file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("reader") != READER_VERSION or data.get("database") != _db_json(db):
            return None
        shirts = {
            int(club): (_shirt_from(home), _shirt_from(keeper) if keeper is not None else None)
            for club, (home, keeper) in data["clubs"].items()
        }
        return GameKits(db, str(data["sha256"]), shirts)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_cache(path: Path, kits: GameKits) -> None:
    """Write the cache whole, so another process never reads half of it."""
    data = {
        "about": "FM27's own home shirts per club, read by Kitshot from the game's database on "
        "this PC. Read again when the database changes; safe to delete.",
        "reader": READER_VERSION,
        "database": _db_json(kits.database),
        "sha256": kits.sha256,
        "clubs": {
            str(club): [_shirt_json(home), _shirt_json(keeper) if keeper else None]
            for club, (home, keeper) in kits.shirts.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(temp, path)


def load_game_kits(
    install_dir: Path | None,
    data_dir: Path | None,
    *,
    read: Callable[[Path], bytes] = Path.read_bytes,
) -> GameKits | None:
    """FM27's kits from its database: from the cache when the file is unchanged, else read and
    cached. None, with a warning in the log, when there is no database or it cannot be read."""
    db = find_database(install_dir)
    if db is None:
        if install_dir is not None:
            logger.info("FM27's database was not found; its kits are not used.")
        return None
    cache = data_dir / CACHE_NAME if data_dir is not None else None
    if cache is not None and (found := read_cache(cache, db)) is not None:
        return found
    try:
        data = read(db.path)
        kits = GameKits(db, hashlib.sha256(data).hexdigest(), parse_database(data))
    except (OSError, GameKitsError, struct.error) as exc:
        logger.warning(
            "FM27's database %s (%s) could not be read, so its kits are not used: %s",
            db.version,
            db.path.name,
            exc,
        )
        return None
    if cache is not None:
        try:
            write_cache(cache, kits)
        except OSError as exc:
            logger.warning("Could not keep FM27's kits in %s: %s", cache, exc)
    logger.info(
        "FM27's kits: %s clubs from the game's database %s.", f"{len(kits.shirts):,}", db.version
    )
    return kits
