"""The save's competition rules reader, on made-up bytes in the save's tagged format."""

from __future__ import annotations

import struct

from facegen.core.records import DivisionList
from facegen.datasource.league_rules import (
    Record,
    competition_id,
    competition_rules,
    division_lists,
)

# --- writing the format ---------------------------------------------------------------------


def field(tag: str, kind: int, payload: bytes) -> bytes:
    """A field: the tag stored backwards, 0x01, the type and the value."""
    return tag[::-1].encode("latin-1") + bytes([0x01, kind]) + payload


def i32(tag: str, value: int) -> bytes:
    return field(tag, 0x01, struct.pack("<i", value))


def u8(tag: str, value: int) -> bytes:
    return field(tag, 0x11, bytes([value]))


def u16(tag: str, value: int) -> bytes:
    return field(tag, 0x12, struct.pack("<H", value))


def flag(tag: str, value: int) -> bytes:
    return field(tag, 0x03, bytes([value]))


def four(tag: str, letters: str) -> bytes:
    return field(tag, 0x02, letters[::-1].encode("latin-1"))


def text(tag: str, value: str) -> bytes:
    data = value.encode("latin-1")
    return field(tag, 0x1A, struct.pack("<I", len(data)) + data)


def body(*fields: bytes) -> bytes:
    return struct.pack("<I", len(fields)) + b"".join(fields)


def record(tag: str, *fields: bytes) -> bytes:
    return field(tag, 0x0A, body(*fields))


def items(tag: str, *values: tuple[int, bytes]) -> bytes:
    """A list: each item 0x01, its type and its value."""
    return field(
        tag, 0x0B, struct.pack("<I", len(values)) + b"".join(bytes([1, k]) + v for k, v in values)
    )


def top(*fields: bytes) -> bytes:
    """A record standing on its own: 0x01 0x0a, then the field count and the fields."""
    return b"\x01\x0a" + body(*fields)


def comp_item(cid: int) -> tuple[int, bytes]:
    """A list item `{comp: N}`, N in the smallest type that holds it, as the save stores it."""
    if cid < 256:
        return 0x0A, body(u8("comp", cid))
    return 0x0A, body(i32("comp", cid))


def nested_item(cid: int, dbid: int | None = None) -> tuple[int, bytes]:
    """A list item `{comp: {id: comp, comp: (N, N), DBID: N}}` (Germany, Italy, Spain)."""
    inner = (four("id  ", "comp"), field("comp", 0x0F, struct.pack("<ii", cid, cid)))
    inner += (i32("DBID", cid if dbid is None else dbid),)
    return 0x0A, body(record("comp", *inner))


def nation_rules(nation: int, women: int, divisions: list[tuple[int, bytes]]) -> bytes:
    nati = record("nati", four("id  ", "nati"), u16("Nnat", nation))
    return top(
        u8("ftye", 10),
        u8("vers", 1),
        nati,
        items("dvlv", *divisions),
        u8("PLgn", women),
        text("file", "wmn_eng_comps" if women else "eng_comps"),
    )


def competition(cid: int, level: int, children: tuple[int, ...] = ()) -> bytes:
    fields = [u8("ftye", 3), u8("vers", 1), i32("comp", cid), u8("type", 1), u8("levl", level)]
    fields.append(u16("year", 2000))
    if children:
        fields.append(items("chdc", *((0x01, struct.pack("<i", c)) for c in children)))
    fields.append(flag("inac", 0))
    return top(*fields)


ENGLAND = nation_rules(
    765, 0, [comp_item(c) for c in (11, 12, 13, 14, 109201, 150990)]
)  # fmt: skip
NOISE = b"\x00\x01\x0a\xff\xff\xff\xff vlvd\x01\x0b junk \x01\x0a" + bytes(range(40))


def save_bytes(*parts: bytes) -> bytes:
    return NOISE + NOISE.join(parts) + NOISE


# --- reading it ---------------------------------------------------------------------------


def test_a_nations_divisions_from_the_top() -> None:
    db = save_bytes(ENGLAND, ENGLAND)  # each list is in a save twice
    assert division_lists(db) == (
        DivisionList(nation=765, women=False, divisions=(11, 12, 13, 14, 109201, 150990)),
    )


def test_nested_items_and_a_nation_as_a_number() -> None:
    spain = top(
        u8("ftye", 10),
        u16("nati", 796),
        items("dvlv", comp_item(67), comp_item(68), nested_item(2000048840)),
    )
    womens = nation_rules(765, 1, [nested_item(2000101095)])
    found = division_lists(save_bytes(spain, womens))
    assert DivisionList(796, None, (67, 68, 2000048840)) in found
    assert DivisionList(765, True, (2000101095,)) in found


def test_a_list_with_an_item_it_cannot_read_is_left_out() -> None:
    bad_dbid = nation_rules(771, 0, [comp_item(22), nested_item(23, dbid=99)])
    unknown_type = ENGLAND.replace(b"elif\x01\x1a", b"elif\x01\x7f")  # the `file` field
    assert division_lists(save_bytes(bad_dbid, unknown_type)) == ()


def test_levels_and_a_regional_parents_groups() -> None:
    db = save_bytes(
        ENGLAND,
        competition(11, 0),
        competition(109201, 4),
        competition(150990, 5, (5123054, 5123055)),
        competition(150990, 5, (5123054, 5123055)),  # the second copy agrees
    )
    rules = competition_rules(db)
    assert rules.division_lists == division_lists(db)
    assert rules.levels == {11: 0, 109201: 4, 150990: 5}
    assert rules.children == {150990: (5123054, 5123055)}
    assert rules.unknown == frozenset()


def test_copies_that_disagree_make_a_competition_unknown() -> None:
    db = save_bytes(
        competition(109201, 4),
        competition(109201, 3),
        competition(150990, 5, (5123054, 5123055)),
        competition(150990, 5, (5123054,)),
    )
    rules = competition_rules(db)
    assert rules.unknown == {109201, 150990}
    assert 109201 not in rules.levels and 150990 not in rules.children


def test_nothing_to_read() -> None:
    rules = competition_rules(NOISE * 3)
    assert (rules.division_lists, dict(rules.levels), dict(rules.children)) == ((), {}, {})


def test_competition_ids_in_every_stored_form() -> None:
    R = Record
    assert competition_id(14) == 14
    assert competition_id(R([("comp", 109201)])) == 109201
    assert competition_id(R([("comp", (5, 5))])) == 5
    assert competition_id(R([("comp", (5, 6))])) is None
    assert competition_id(R([("comp", R([("id  ", "comp"), ("comp", 7), ("DBID", 7)]))])) == 7
    assert competition_id(R([("comp", R([("id  ", "comp"), ("comp", 7), ("DBID", 8)]))])) is None
    assert competition_id(R([("type", 1)])) is None
    assert competition_id([5, 6]) is None  # a list is not a record, whatever it holds
    assert competition_id([("comp", 14)]) is None


def test_an_odd_list_in_one_nations_rules_leaves_the_others_readable() -> None:
    """A division item that is a list (of numbers) rather than a record makes that nation's
    list unreadable; England's list beside it is still read."""
    numbers = struct.pack("<I", 2) + b"".join(bytes([1, 1]) + struct.pack("<i", n) for n in (5, 6))
    odd = nation_rules(771, 0, [comp_item(22), (0x0B, numbers)])
    rules = competition_rules(save_bytes(odd, ENGLAND))
    assert rules.division_lists == (
        DivisionList(nation=765, women=False, divisions=(11, 12, 13, 14, 109201, 150990)),
    )


def test_malformed_text_ids_leave_other_nations_readable() -> None:
    for invalid in ("\xb2", "9" * 4500, str(1 << 63)):
        odd = nation_rules(771, 0, [(0x0A, body(text("comp", invalid)))])
        assert division_lists(save_bytes(odd, ENGLAND)) == division_lists(ENGLAND)
