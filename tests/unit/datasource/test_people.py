"""The game database readers fmsave lacks: gender flags, journalists, press officers.

Made-up bytes laid out as docs/research.md section 9 found them in real saves.
"""

from __future__ import annotations

from datetime import date

from support.people import (
    FEMALE,
    FIRST,
    SIZES,
    common_at,
    db,
    first_at,
    name_table,
    surname_at,
    u32,
)

from facegen.datasource.people import (
    MISSING,
    PRESS_OFFICER_ROLE,
    display_name,
    female_by_text,
    first_name_flags,
    journalists,
    name_is_female,
    press_officers,
)


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
