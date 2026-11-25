"""A made-up FM27 database (`client_db.dat`) for the game kit reader: the layout research.md
describes ("FM27's kits"), with a handful of clubs and their kit entries."""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

TABLE_HEAD = b"\x03\x01tad.\xa0\x0f"
FILE_HEAD = b"\x03\x01tad.\xa0\x0f\xa9\x0e\x09\x00"
CONTAINER = b"\x02\x01fmf."
CLUBS_AFTER_ROWS = b"\x00\x00\x00\x00\x01\x31\x72\x41\x77"


def rgb555(hex_colour: str) -> int:
    """FM's 5 bits a channel, red highest; exact for colours whose channels are k * 255 / 31."""
    r, g, b = (round(int(hex_colour[i : i + 2], 16) * 31 / 255) for i in (0, 2, 4))
    return (r << 10) | (g << 5) | b


@dataclass(frozen=True)
class Entry:
    """One 27-byte kit entry; the defaults are the home shirt of the outfield players."""

    colours: tuple[str, str, str]
    style: int = 1
    kit_set: int = 0
    garment: int = 1
    wearer: int = 1
    competition: int = -1
    season: int = 0

    def pack(self) -> bytes:
        b = bytearray(27)
        b[0] = b[1] = 1
        for i, colour in enumerate((*self.colours, "FFFFFF", "000000")):
            struct.pack_into("<H", b, 2 + 2 * i, rgb555(colour))
        b[12] = self.style
        struct.pack_into("<iiH", b, 13, self.competition, self.competition, self.season)
        b[23], b[24], b[26] = self.kit_set, self.garment, self.wearer
        return bytes(b)


@dataclass(frozen=True)
class Club:
    uid: int
    name: str = "Made-up Football Club"
    kits: tuple[Entry, ...] = ()
    short: str = ""
    empty: bool = False
    """An empty slot of the club table (kind 12): no names."""


@dataclass
class Tables:
    clubs: bytes
    extra: bytes
    others: dict[str, bytes] = field(default_factory=lambda: {"simple_vars": b"\x00" * 44})


def clubs_table(clubs: list[Club]) -> bytes:
    out = bytearray(TABLE_HEAD + struct.pack("<I", len(clubs)) + b"\x00")
    for row, club in enumerate(clubs):
        if club.empty:
            out += b"\x0c" + b"\xff" * 24 + b"\x00" * 17
        else:
            out += b"\x00" + b"\x7e\x00\x00\x00" + b"\xff" * 4 + bytes(18)
            for name in (club.name, club.name[:10]):
                raw = name.encode()
                out += struct.pack("<I", len(raw)) + raw
            out += bytes(3)
        out += struct.pack("<III", row, club.uid, club.uid)
    return bytes(out + CLUBS_AFTER_ROWS)


def extra_table(clubs: list[Club]) -> bytes:
    out = bytearray(TABLE_HEAD + struct.pack("<I", len(clubs)))
    for row, club in enumerate(clubs):
        short = club.short.encode()
        out += struct.pack("<IH", row, 1899) + b"\xff" * 4 + bytes(2)
        out += struct.pack("<I", len(short)) + short
        out += bytes([len(club.kits)]) + b"".join(k.pack() for k in club.kits)
        out += b"\xff\xfa\xfa\xfa\xff\x2f\x55\xb5"
    return bytes(out)


def tables(clubs: list[Club]) -> Tables:
    return Tables(clubs_table(clubs), extra_table(clubs))


def database(found: Tables | list[Club], *, sizes: dict[str, int] | None = None) -> bytes:
    """The whole file: header, frames, directory. `sizes` overrides a table's listed size."""
    t = found if isinstance(found, Tables) else tables(found)
    named = {**t.others, "table_3": t.clubs, "table_extra_3": t.extra}
    frames = bytearray()
    entries = bytearray()
    for name, data in named.items():
        compressed = zstd.compress(data)
        size = (sizes or {}).get(name, len(data))
        raw = name.encode()
        entries += struct.pack("<I", len(raw)) + raw + struct.pack("<I", 4) + b".dat"
        entries += struct.pack("<5Q", len(frames), len(compressed), size, 0, 0)
        frames += compressed
    directory = struct.pack("<I", 9) + b"client_db" + struct.pack("<I", len(named))
    directory += bytes(entries) + bytes(4)
    trailer_at = 26 + len(frames)  # from the container's start
    head = CONTAINER + b"\x08\x00\x00" + struct.pack("<Q", trailer_at - 9) + bytes(9)
    trailer = CONTAINER + b"\x08\x00\x00" + zstd.compress(directory)
    return FILE_HEAD + head + bytes(frames) + trailer


def install(root: Path, clubs: list[Club] | bytes, *, folder: str = "2700") -> Path:
    """A made-up FM27 install folder holding the database; returns the database's path."""
    where = root / "shared" / "data" / "database" / "db" / folder / f"{folder}_fm"
    where.mkdir(parents=True, exist_ok=True)
    path = where / "client_db.dat"
    path.write_bytes(clubs if isinstance(clubs, bytes) else database(clubs))
    version = f"{folder[:2]}.{folder[2]}.{folder[3]}"
    (where / f"{folder}_fm.xml").write_text(
        f'﻿<record>\n\t<string id="version" value="{version}" translate="false"/>\n</record>\n',
        encoding="utf-8",
    )
    return path


BARCELONA = Club(
    1708,
    "Barcelona",
    kits=(
        Entry(("BD1952", "19316B", "0042BD"), style=24),  # home shirt
        Entry(("FFBD52", "FF4A00", "FF4A00"), style=12, wearer=0),  # goalkeeper
        Entry(("FFC531", "0042BD", "DE105A"), style=0, garment=2),  # shorts
        Entry(("5A3AEF", "313A3A", "5A3AEF"), kit_set=1),  # away
        Entry(("19316B", "BD1952", "BD1952"), style=2, competition=1301394),  # a cup's
        Entry(("19316B", "BD1952", "BD1952"), style=2, season=2026),  # one season's
    ),
)
REAL_MADRID = Club(1736, "R. Madrid", kits=(Entry(("EF5A8C", "FFFFFF", "21423A"), style=25),))
TWO_HOME_SHIRTS = Club(
    5000, kits=(Entry(("FFFFFF", "C50021", "FFFFFF")), Entry(("000000", "FFFFFF", "000000")))
)
NO_KITS = Club(5001, short="AFC")
EMPTY_SLOT = Club(2000000000, empty=True)
CLUBS = [BARCELONA, NO_KITS, REAL_MADRID, EMPTY_SLOT, TWO_HOME_SHIRTS]
