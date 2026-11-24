"""What the save holds about people that fmsave does not read: gender, journalists, press officers.

Experimental readers over the decompressed game database (`game_db`) of an FM27 save, found by
reading copies of three careers' saves (docs/research.md section 9). They take the section's
bytes and name lookups and return plain values, so they are tested on made-up bytes; the
fmsave adapter hands them the real ones. Nothing here imports fmsave.

**Gender.** A person's record holds no gender. Their first name does: the game database keeps
one row per first-name entry, and a flag in it marks the women's names (the same text, say
"Andrea", has separate entries for men and women). The table is `count` rows of 12 bytes,
indexed by first-name ID, right after the count and two 0xFF bytes:

    [count u32][ff ff] then per entry: [ID u32][u16][u16][flags u8][3 bytes]

Bit 0x04 of `flags` is set on every first name of the women's players in a save with women's
football loaded and on none of the men's (28,660 names, docs/research.md).

**Journalists** are person records of type 6, **people attached to a club** (three per club:
the press officer and two others) of type 8:

    [local ID u32][unique ID u32][unique ID u32][type u8][header][name block]

The header is 8 bytes for a journalist; for a club person it is the club's index u32 (its
place in the game database's club table) and a role u16, 52 for the press officer. The name
block is the players' (first, surname and common name IDs, each followed by a zero byte; the
legal name's length and text), then the birth date (day of year and year, as fmsave reads
them), the save-local nation ID as a u16 at 14 bytes past it, and eight personality bytes from
1 to 20 at 22. The repeated unique ID, a known name, a valid birth date and the personality
run together tell a record from chance bytes.
"""

from __future__ import annotations

import struct
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta

U32 = struct.Struct("<I")
U16 = struct.Struct("<H")
MISSING = 0xFFFFFFFF

NAME_ROW_BYTES = 12
FLAGS_OFFSET_IN_ROW = 8
FEMALE_NAME_FLAG = 0x04

JOURNALIST_TYPE = 6
JOURNALIST_HEADER_BYTES = 8
CLUB_PERSON_TYPE = 8
CLUB_PERSON_HEADER_BYTES = 6
PRESS_OFFICER_ROLE = 52

# From the birth date: the save-local nation ID, then the personality run.
NATION_OFFSET = 14
PERSONALITY_OFFSET = 22
PERSONALITY_BYTES = 8
LEGAL_NAME_MAX = 80
EARLIEST_YEAR = 1901

type NameAt = Callable[[int], str | None]
"""A name pool's lookup: the text at an ID, or None for a missing or blank entry."""


def first_name_flags(db: bytes, count: int, *, checks: int = 64) -> bytes | None:
    """The flags byte of every first-name entry, indexed by first-name ID; None if not found.

    `count` is how many first names the save's pool holds. Each match of the table's start is
    checked at `checks` rows spread over the table, which must hold their own IDs.
    """
    if count <= 0:
        return None
    needle = U32.pack(count) + b"\xff\xff" + U32.pack(0)
    sample = sorted({(count - 1) * k // checks for k in range(checks + 1)})
    at = -1
    while (at := db.find(needle, at + 1)) >= 0:
        start = at + 6
        end = start + NAME_ROW_BYTES * count
        if end > len(db):
            return None
        if all(U32.unpack_from(db, start + NAME_ROW_BYTES * i)[0] == i for i in sample):
            return bytes(db[start + FLAGS_OFFSET_IN_ROW : end : NAME_ROW_BYTES])
    return None


def name_is_female(flags: bytes, first_name_id: int) -> bool | None:
    """Whether a first-name entry is a woman's name; None for no name or one out of range."""
    if not 0 <= first_name_id < len(flags):
        return None
    return bool(flags[first_name_id] & FEMALE_NAME_FLAG)


def female_by_text(wanted: Iterable[str], first_name: NameAt, flags: bytes) -> dict[str, bool]:
    """For each wanted first name, as text, whether most of its entries are women's names.

    For people fmsave reads (staff) only the text of the first name is known, and a text can
    have entries of both kinds. The entries are counted, not the people using them: ties count
    as a man's name. A text with no entry is left out.
    """
    want = set(wanted)
    counts: dict[str, Counter[bool]] = {}
    if not want:
        return {}
    for i in range(len(flags)):
        text = first_name(i)
        if text in want:
            counts.setdefault(text, Counter())[bool(flags[i] & FEMALE_NAME_FLAG)] += 1
    return {text: c[True] > c[False] for text, c in counts.items()}


@dataclass(frozen=True, slots=True)
class PersonRecord:
    """A journalist's or club person's record, as stored."""

    unique_id: int
    person_type: int
    first_name_id: int
    surname_id: int
    common_name_id: int
    birth_date: date
    nation_id: int
    club_index: int | None
    """Club people: the club's index in the game database's club table."""
    role: int | None
    """Club people: their role, PRESS_OFFICER_ROLE for the press officer."""


def _birth(db: bytes, at: int, latest_year: int) -> date | None:
    packed, year = struct.unpack_from("<HH", db, at)
    if packed >> 9:
        return None
    day = packed & 0x1FF
    if not (EARLIEST_YEAR <= year <= latest_year and 1 <= day <= 366):
        return None
    born = date(year, 1, 1) + timedelta(days=day - 1)
    return born if born.year == year else None


def scan_people(
    db: bytes,
    person_type: int,
    header_bytes: int,
    *,
    start: int,
    first_names: int,
    surnames: int,
    latest_year: int,
) -> Iterator[PersonRecord]:
    """Every record of one person type from `start` on, in stored order.

    `first_names` and `surnames` are the pools' sizes; `latest_year` the in-game year, the
    latest a birth date may fall in.
    """
    type_byte = bytes([person_type])
    tail = header_bytes + 1 + 19 + PERSONALITY_OFFSET + PERSONALITY_BYTES
    end = len(db) - tail - LEGAL_NAME_MAX
    pos = max(start, 12) - 1
    while (pos := db.find(type_byte, pos + 1)) >= 0 and pos < end:
        if db[pos - 8 : pos - 4] != db[pos - 4 : pos]:
            continue
        unique_id = U32.unpack_from(db, pos - 4)[0]
        if not 1 <= unique_id < 2**31:
            continue
        q = pos + 1 + header_bytes
        if db[q + 4] or db[q + 9] or db[q + 14]:
            continue
        first, surname, common = (U32.unpack_from(db, q + k)[0] for k in (0, 5, 10))
        if not (
            (first < first_names or first == MISSING)
            and (surname < surnames or surname == MISSING)
            and (first != MISSING or surname != MISSING)
        ):
            continue
        legal = U32.unpack_from(db, q + 15)[0]
        if legal > LEGAL_NAME_MAX:
            continue
        p = q + 19 + legal
        born = _birth(db, p, latest_year)
        if born is None:
            continue
        run = db[p + PERSONALITY_OFFSET : p + PERSONALITY_OFFSET + PERSONALITY_BYTES]
        if not all(1 <= b <= 20 for b in run):
            continue
        club_index = role = None
        if person_type == CLUB_PERSON_TYPE:
            club_index = U32.unpack_from(db, pos + 1)[0]
            role = U16.unpack_from(db, pos + 5)[0]
        yield PersonRecord(
            unique_id=unique_id,
            person_type=person_type,
            first_name_id=first,
            surname_id=surname,
            common_name_id=common,
            birth_date=born,
            nation_id=U16.unpack_from(db, p + NATION_OFFSET)[0],
            club_index=club_index,
            role=role,
        )


def journalists(
    db: bytes, *, start: int, first_names: int, surnames: int, latest_year: int
) -> list[PersonRecord]:
    """Every journalist in the save, each unique ID once (the first record wins)."""
    seen: set[int] = set()
    out = []
    for r in scan_people(
        db,
        JOURNALIST_TYPE,
        JOURNALIST_HEADER_BYTES,
        start=start,
        first_names=first_names,
        surnames=surnames,
        latest_year=latest_year,
    ):
        if r.unique_id not in seen:
            seen.add(r.unique_id)
            out.append(r)
    return out


def press_officers(
    db: bytes,
    club_indexes: Iterable[int],
    *,
    start: int,
    first_names: int,
    surnames: int,
    latest_year: int,
) -> list[PersonRecord]:
    """The press officers attached to the clubs at these indexes, each unique ID once."""
    clubs = set(club_indexes)
    seen: set[int] = set()
    out = []
    for r in scan_people(
        db,
        CLUB_PERSON_TYPE,
        CLUB_PERSON_HEADER_BYTES,
        start=start,
        first_names=first_names,
        surnames=surnames,
        latest_year=latest_year,
    ):
        if r.role == PRESS_OFFICER_ROLE and r.club_index in clubs and r.unique_id not in seen:
            seen.add(r.unique_id)
            out.append(r)
    return out


def display_name(
    r: PersonRecord, first_name: NameAt, surname: NameAt, common_name: NameAt
) -> str | None:
    """The name the game shows: the common name, else first name and surname."""
    if r.common_name_id != MISSING and (common := common_name(r.common_name_id)):
        return common
    parts = [
        n
        for n in (
            first_name(r.first_name_id) if r.first_name_id != MISSING else None,
            surname(r.surname_id) if r.surname_id != MISSING else None,
        )
        if n
    ]
    return " ".join(parts) or None
