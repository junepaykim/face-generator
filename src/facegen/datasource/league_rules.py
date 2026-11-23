"""The save's own competition rules: each nation's ordered divisions (docs/research.md 23-25).

An experimental reader over the decompressed game database (`game_db`) of an FM27 save, like
datasource/people.py: it takes the section's bytes and returns plain values, so it is tested on
made-up bytes; the fmsave adapter hands it the real ones. Nothing here imports fmsave.

**The format.** The rules are tagged values. A field is a four-letter tag stored backwards
(`vlvd` on disk is `dvlv`), the byte 0x01, a type byte and the value:

    0x00 nothing      0x01 int32       0x02 four letters   0x03 one byte (a flag)
    0x0a record: u32 field count, then that many fields
    0x0b list: u32 item count, then items (0x01, a type byte and the value, without a tag)
    0x0f two int32    0x11 u8          0x12 u16            0x18 int64
    0x19 float32      0x1a text: u32 length and the bytes  0x20 four bytes (a time stamp)

An ID is stored in the smallest type that holds it (League Two, 14, as a u8). A field's record
is found by looking back from the field for a record header whose fields include it; an
unknown type makes a record unreadable.

**Division lists.** A nation's rules record holds `nati` (the nation's editor ID, a number or a
small `{id, Nnat}` record), `PLgn` (1 for women's football) and `dvlv`, the league divisions from
the top: each item a record whose `comp` is the competition's editor ID, a number or a
`{id, comp, DBID}` record whose IDs agree. England's men's list is Premier
League, Championship, League One, League Two, National League and the National League
North/South parent (11, 12, 13, 14, 109201, 150990).

**Competition rules.** A competition's record holds `comp`, then `type` and `levl` (its
zero-based level) one after the other, and a regional parent's `chdc` lists its groups: 150990's
are National League North and South (5123054, 5123055). Each list and rule appears twice in a
save; a copy whose record cannot be read is skipped. On four saves of FM27 27.0.0 to 27.1.0 this
read 70 lists in 0.2-0.7 s, each list's levels agreeing with `levl` (docs/research.md 23).
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterator

from facegen.core.records import CompetitionRules, DivisionList

_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_U16 = struct.Struct("<H")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")

_FIXED = {0x00: 0, 0x01: 4, 0x02: 4, 0x03: 1, 0x0F: 8, 0x11: 1, 0x12: 2, 0x18: 8, 0x19: 4, 0x20: 4}
_MAX_FIELDS = 1024
_MAX_ITEMS = 100_000
_MAX_TEXT = 1 << 20
_MAX_DEPTH = 32
_RECORD_START = b"\x01\x0a"
_DIVISIONS = b"vlvd\x01\x0b"
_CHILDREN = b"cdhc\x01\x0b"
_LEVEL = re.compile(rb"epyt\x01\x11.lvel\x01\x11.", re.DOTALL)
"""A competition's record holds `type` and then `levl`."""
_MAX_TRIES = 1024
_LOOK_BACK = 16384
"""How far before a field its record may start: a competition's groups come after some 3 KB of
its other fields."""


class _Unreadable(Exception):
    """Bytes that do not decode as the format above."""


class Record(list[tuple[str, object]]):
    """A decoded record: its (tag, value) fields in order. A type of its own, so a list (whose
    items are values without tags) is never read as a record: an odd list in one nation's rules
    makes that list unreadable, not every nation's."""


def _is_list(value: object) -> bool:
    return isinstance(value, list) and not isinstance(value, Record)


def _value(db: bytes, pos: int, kind: int, depth: int = 0) -> tuple[object, int]:
    """The value of type `kind` at `pos`, and the position after it."""
    if depth > _MAX_DEPTH:
        raise _Unreadable("nested too deep")
    size = _FIXED.get(kind)
    if size is not None:
        if pos + size > len(db):
            raise _Unreadable("past the end")
        if kind == 0x00:
            return None, pos
        if kind == 0x01:
            return _I32.unpack_from(db, pos)[0], pos + 4
        if kind == 0x02:
            return db[pos : pos + 4][::-1].decode("latin-1"), pos + 4
        if kind in (0x03, 0x11):
            return db[pos], pos + 1
        if kind == 0x0F:
            return (_I32.unpack_from(db, pos)[0], _I32.unpack_from(db, pos + 4)[0]), pos + 8
        if kind == 0x12:
            return _U16.unpack_from(db, pos)[0], pos + 2
        if kind == 0x18:
            return _I64.unpack_from(db, pos)[0], pos + 8
        if kind == 0x19:
            return _F32.unpack_from(db, pos)[0], pos + 4
        return _U32.unpack_from(db, pos)[0], pos + 4
    if pos + 4 > len(db):
        raise _Unreadable("past the end")
    count = _U32.unpack_from(db, pos)[0]
    pos += 4
    if kind == 0x1A:
        if count > _MAX_TEXT or pos + count > len(db):
            raise _Unreadable("text too long")
        return db[pos : pos + count].decode("latin-1"), pos + count
    if kind == 0x0A:
        if count > _MAX_FIELDS:
            raise _Unreadable("too many fields")
        record = Record()
        for _ in range(count):
            if pos + 6 > len(db) or db[pos + 4] != 0x01:
                raise _Unreadable(f"no field at {pos}")
            tag = db[pos : pos + 4][::-1].decode("latin-1")
            value, pos = _value(db, pos + 6, db[pos + 5], depth + 1)
            record.append((tag, value))
        return record, pos
    if kind == 0x0B:
        if count > _MAX_ITEMS:
            raise _Unreadable("too many items")
        items = []
        for _ in range(count):
            if pos + 2 > len(db) or db[pos] != 0x01:
                raise _Unreadable(f"no item at {pos}")
            value, pos = _value(db, pos + 2, db[pos + 1], depth + 1)
            items.append(value)
        return items, pos
    raise _Unreadable(f"unknown type {kind:#04x} at {pos}")


def _record_at(db: bytes, start: int) -> tuple[Record, list[int], int] | None:
    """The record whose header (0x01 0x0a) is at `start`: its fields, where each of them starts,
    and the position after it; None where the bytes there are not a record."""
    if db[start : start + 2] != _RECORD_START or start + 6 > len(db):
        return None
    count = _U32.unpack_from(db, start + 2)[0]
    if count > _MAX_FIELDS:
        return None
    pos = start + 6
    record = Record()
    positions: list[int] = []
    try:
        for _ in range(count):
            if pos + 6 > len(db) or db[pos + 4] != 0x01:
                return None
            positions.append(pos)
            value, end = _value(db, pos + 6, db[pos + 5], 1)
            record.append((db[pos : pos + 4][::-1].decode("latin-1"), value))
            pos = end
    except (_Unreadable, struct.error):
        return None
    return record, positions, pos


def _record_holding(db: bytes, at: int) -> Record | None:
    """The record that has a field of its own starting at `at`, or None where none is found
    within _LOOK_BACK bytes before it."""
    floor = max(0, at - _LOOK_BACK)
    start = db.rfind(_RECORD_START, floor, at)
    tries = 0
    while start >= 0 and tries < _MAX_TRIES:
        tries += 1
        found = _record_at(db, start)
        if found is not None and at in found[1]:
            return found[0]
        start = db.rfind(_RECORD_START, floor, start)
    return None


def _field(record: object, tag: str) -> object:
    if not isinstance(record, Record):
        return None
    return next((v for k, v in record if k == tag), None)


def _id(value: object) -> int | None:
    """An ID stored as a number, two equal numbers, digits, or a `{id, comp, DBID}` record whose
    IDs agree; None for anything else."""
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and len(value) == 2 and value[0] == value[1]:
        return int(value[0])
    if isinstance(value, str) and len(value) <= 19 and value.isascii() and value.isdigit():
        number = int(value)
        return number if number < 1 << 63 else None  # the largest numeric field is a signed int64
    if isinstance(value, Record):
        inner = _id(_field(value, "comp"))
        dbid = _field(value, "DBID")
        if inner is not None and not isinstance(_field(value, "comp"), Record):
            return inner if dbid is None or _id(dbid) == inner else None
    return None


def competition_id(item: object) -> int | None:
    """A list item's competition: a number, or a record whose `comp` is one (or a
    `{id, comp, DBID}` record whose IDs agree). None when it is neither."""
    bare = _id(item) if not isinstance(item, Record) else None
    return bare if bare is not None else _id(_field(item, "comp"))


def _nation(record: Record) -> int | None:
    nati = _field(record, "nati")
    if isinstance(nati, Record):
        return _id(_field(nati, "Nnat"))
    return _id(nati)


def _hits(db: bytes, needle: bytes) -> Iterator[int]:
    at = db.find(needle)
    while at >= 0:
        yield at
        at = db.find(needle, at + 1)


def division_lists(db: bytes) -> tuple[DivisionList, ...]:
    """Every nation's division list the save's rules hold, once each (a repeat is dropped);
    a list whose record or items cannot be read is left out."""
    found: dict[DivisionList, None] = {}
    for at in _hits(db, _DIVISIONS):
        record = _record_holding(db, at)
        items = _field(record, "dvlv")
        if record is None or not _is_list(items):
            continue
        ids = [competition_id(i) for i in items]
        if not ids or any(i is None for i in ids):
            continue
        gender = _field(record, "PLgn")
        found[
            DivisionList(
                nation=_nation(record),
                women=gender == 1 if isinstance(gender, int) else None,
                divisions=tuple(i for i in ids if i is not None),
            )
        ] = None
    return tuple(found)


def competition_rules(db: bytes) -> CompetitionRules:
    """The division lists, each competition's level and each regional parent's groups.

    A competition whose copies give different levels or groups is `unknown`."""
    levels: dict[int, int] = {}
    children: dict[int, tuple[int, ...]] = {}
    unknown: set[int] = set()
    for m in _LEVEL.finditer(db):
        record = _record_holding(db, m.start())
        comp = _id(_field(record, "comp"))
        level = _field(record, "levl")
        if comp is None or not isinstance(level, int):
            continue
        if levels.setdefault(comp, level) != level:
            unknown.add(comp)
    for at in _hits(db, _CHILDREN):
        record = _record_holding(db, at)
        comp = _id(_field(record, "comp"))
        items = _field(record, "chdc")
        if comp is None or not _is_list(items):
            continue
        read = [competition_id(i) for i in items]
        kids = tuple(k for k in read if k is not None)
        if len(kids) != len(read) or children.setdefault(comp, kids) != kids:
            unknown.add(comp)
    return CompetitionRules(
        division_lists=division_lists(db),
        levels={c: lv for c, lv in levels.items() if c not in unknown},
        children={c: k for c, k in children.items() if c not in unknown and k},
        unknown=frozenset(unknown),
    )
