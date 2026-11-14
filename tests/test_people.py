"""The game database readers fmsave lacks: gender flags, journalists, press officers.

Made-up bytes laid out as docs/research.md section 9 found them in real saves.
"""

from __future__ import annotations

import struct
from datetime import date

from facegen.datasource.people import (
    CLUB_PERSON_TYPE,
    FEMALE_NAME_FLAG,
    JOURNALIST_TYPE,
    MISSING,
    PRESS_OFFICER_ROLE,
    display_name,
    female_by_text,
    first_name_flags,
    journalists,
    name_is_female,
    press_officers,
)

FIRST = ["Jake", "Emma", "Lauren", "Andrea", "Andrea", "Andrea", "Sam"]
FEMALE = {1, 2, 3, 4}  # one "Andrea" is a man's entry, two are women's
SURNAMES = ["Diaz", "Collinson", "Youngs", "Rouse"]


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def name_table() -> bytes:
    rows = b"".join(
        u32(i)
        + struct.pack("<HH", 7, 1)
        + bytes([FEMALE_NAME_FLAG if i in FEMALE else 0])
        + b"\xff" * 3
        for i in range(len(FIRST))
    )
    return u32(len(FIRST)) + b"\xff\xff" + rows


def person(
    uid: int,
    kind: int,
    header: bytes,
    first: int,
    surname: int,
    *,
    born: date = date(1970, 10, 24),
    nation: int = 139,
    personality: bytes = bytes([10, 12, 5, 15, 9, 11, 7, 3]),
    copy: int | None = None,
) -> bytes:
    day = born.timetuple().tm_yday
    birth = struct.pack("<HH", day, born.year)
    after_birth = b"\xff" + bytes(9) + struct.pack("<H", nation) + bytes(6) + personality
    name_block = u32(first) + b"\0" + u32(surname) + b"\0" + u32(MISSING) + b"\0" + u32(0)
    return (
        u32(280_000 + uid % 1000)
        + u32(uid)
        + u32(copy if copy is not None else uid)
        + bytes([kind])
        + header
        + name_block
        + birth
        + after_birth
        + bytes(24)
    )


def journalist(uid: int, first: int, surname: int, **kw: object) -> bytes:
    return person(uid, JOURNALIST_TYPE, b"\x01\x10\x0b\xb5\x00\x00\x00\x03", first, surname, **kw)


def club_person(uid: int, club_index: int, role: int, first: int, surname: int) -> bytes:
    header = u32(club_index) + struct.pack("<H", role)
    return person(uid, CLUB_PERSON_TYPE, header, first, surname)


def db() -> bytes:
    return (
        b"\x00" * 64
        + name_table()
        + b"\x00" * 40
        + journalist(2002097912, 0, 0)
        + b"\x05" * 7
        + journalist(2002097939, 1, 1, nation=140)
        + journalist(2002097949, 2, 2, copy=1)  # the ID's two copies differ: not a record
        + journalist(2002097950, 2, 2, born=date(2030, 1, 1))  # born after the game year
        + journalist(2002097951, 2, 2, personality=bytes([30] * 8))  # not a personality
        + journalist(2002097912, 0, 0)  # the same person again: kept once
        + club_person(2002076347, 12666, PRESS_OFFICER_ROLE, 2, 3)
        + club_person(2002076346, 12666, 56, 6, 3)  # another role at my club
        + club_person(2002076999, 99, PRESS_OFFICER_ROLE, 1, 3)  # another club's
        + b"\x00" * 300
    )


SIZES = {"start": 0, "first_names": len(FIRST), "surnames": len(SURNAMES), "latest_year": 2026}


def first_at(i: int) -> str | None:
    return FIRST[i] if 0 <= i < len(FIRST) else None


def surname_at(i: int) -> str | None:
    return SURNAMES[i] if 0 <= i < len(SURNAMES) else None


def common_at(i: int) -> str | None:
    return None


def test_the_first_name_flags_mark_womens_names() -> None:
    flags = first_name_flags(db(), len(FIRST))
    assert flags is not None and len(flags) == len(FIRST)
    assert [name_is_female(flags, i) for i in range(len(FIRST))] == [
        i in FEMALE for i in range(len(FIRST))
    ]
    assert name_is_female(flags, MISSING) is None


def test_no_table_without_the_count_or_rows() -> None:
    assert first_name_flags(db(), len(FIRST) + 1) is None
    assert first_name_flags(b"\x00" * 100, 7) is None
    broken = bytearray(db())
    start = bytes(broken).index(name_table()) + 6
    broken[start + 12 * 6 : start + 12 * 6 + 4] = u32(99)  # the last row's ID is wrong
    assert first_name_flags(bytes(broken), len(FIRST)) is None


def test_a_text_is_a_womans_name_by_most_of_its_entries() -> None:
    flags = first_name_flags(db(), len(FIRST))
    assert flags is not None
    found = female_by_text({"Jake", "Emma", "Andrea", "Sam", "Nobody"}, first_at, flags)
    assert found == {"Jake": False, "Emma": True, "Andrea": True, "Sam": False}


def test_journalists_are_found_and_chance_bytes_are_not() -> None:
    found = journalists(db(), **SIZES)
    assert [r.unique_id for r in found] == [2002097912, 2002097939]
    jake, emma = found
    assert (jake.first_name_id, jake.surname_id, jake.nation_id) == (0, 0, 139)
    assert jake.birth_date == date(1970, 10, 24)
    assert emma.nation_id == 140 and jake.club_index is None and jake.role is None
    assert display_name(jake, first_at, surname_at, common_at) == "Jake Diaz"


def test_press_officers_of_my_club_only() -> None:
    found = press_officers(db(), [12666], **SIZES)
    assert [(r.unique_id, r.club_index, r.role) for r in found] == [
        (2002076347, 12666, PRESS_OFFICER_ROLE)
    ]
    assert display_name(found[0], first_at, surname_at, common_at) == "Lauren Rouse"
    assert press_officers(db(), [], **SIZES) == []


def test_a_common_name_wins() -> None:
    (jake, _) = journalists(db(), **SIZES)

    def common(i: int) -> str | None:
        return "Jakey" if i == 7 else None

    assert display_name(jake, first_at, surname_at, common) == "Jake Diaz"  # no common name
    from dataclasses import replace

    assert display_name(replace(jake, common_name_id=7), first_at, surname_at, common) == "Jakey"
