"""Map real age in whole years to fixed prompt wordings W01 (youngest) to W10.

My-club players use the finest steps regardless of group. Other players use group tables;
staff use working-age bands from W02, and media bands start at W04. Settings choose each
wording's first age; ages below the first row use that row. Tracked players are redrawn
when their wording changes.

The wording key is everyone's age band: a woman's prompt uses its words, a man's states his
age in years (`style.gender`, core/style.py). Historical M1-M4 pictures show men and get a new
face while everyone is drawn as a woman, detected by drawn_as_man. Prompt text is fixed,
independent of table settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from facegen.core.settings_values import AgeSettings


@dataclass(frozen=True, slots=True)
class Wording:
    text: str
    young: bool
    """Drawn as under 18. The prompt then adds fixed negative terms against adult
    features; that is part of the content rule (spec section 3), not a setting."""


WORDINGS: dict[str, Wording] = {
    "W01": Wording("child, young girl, youthful face, round face", young=True),
    "W02": Wording("aged down, young girl, youthful face, round face", young=True),
    "W03": Wording("young girl, youthful face, round face", young=True),
    "W04": Wording("young teenage girl, teenage, youthful face, round face", young=True),
    "W05": Wording("young teenage girl, teenage, youthful face", young=True),
    "W06": Wording("teenage girl, teenage", young=True),
    "W07": Wording("young adult woman", young=False),
    "W08": Wording("young woman, adult", young=False),
    "W09": Wording("adult woman, mature face", young=False),
    "W10": Wording("mature female, adult woman, mature face", young=False),
}

MEN_BANDS = frozenset({"M1", "M2", "M3", "M4"})
"""Legacy men's wordings that stored pictures may still carry."""

TABLES = ("a_my_club", "a_other_club", "b", "c", "staff", "media")


def table_name(
    group: str | None,
    at_my_club: bool,
    staff: bool = False,
    *,
    media: bool = False,
) -> str | None:
    """Which age table applies. `group` is a Group or its value ("a", "b", "c")."""
    if group is None:
        return None
    if media:
        return "media"
    if staff:
        return "staff"
    if at_my_club:
        # Every player at my club, whatever the group (a youth team's B, a pin): the finest
        # steps, and moving between groups there changes no wording.
        return "a_my_club"
    if group == "a":
        return "a_other_club"
    return str(group)


def age_wording(
    real_age: int | None,
    group: str | None,
    at_my_club: bool,
    s: AgeSettings,
    staff: bool = False,
    *,
    media: bool = False,
) -> str | None:
    """The wording key, which is also the age band, or None if age or group is unknown.

    `staff` and `media` say who the person is: my club's staff, a journalist or press officer.
    """
    name = table_name(group, at_my_club, staff, media=media)
    if real_age is None or name is None:
        return None
    rows = s.table(name)
    key = rows[0][1]
    for first_age, wording in rows:
        if real_age < first_age:
            break
        key = wording
    return key


def drawn_as_man(band: str | None) -> bool:
    """Whether a stored picture's age band is one of the dropped men's wordings."""
    return band in MEN_BANDS


def audience(row: Any) -> dict[str, bool]:
    """`age_wording`'s and `table_name`'s staff and media flags from a players row."""
    return {"staff": bool(row["is_staff"]), "media": row["media_role"] is not None}


def spread_table(first: str, last: str, start: int, end: int) -> tuple[tuple[int, str], ...]:
    """An age table spreading the looks `first` to `last` over the ages `start` to `end`
    (my club's staff in auto mode, Phase 16f): up to `start` the first look, from `end` the last,
    and the years between split evenly over the looks between, a leftover year going to the
    younger looks. ValueError when the looks or ages are out of order, or the span has fewer
    years than looks between.

    W02-W10 over 28-64: (0, W02), (29, W03), (34, W04), ..., (59, W09), (64, W10).
    """
    keys = list(WORDINGS)
    if first not in WORDINGS or last not in WORDINGS:
        raise ValueError(f"unknown look {first if first not in WORDINGS else last}")
    i, j = keys.index(first), keys.index(last)
    if i > j:
        raise ValueError("the first look must be younger than the last")
    if start >= end:
        raise ValueError("the first age must be below the last")
    if i == j:
        return ((0, first),)
    middle = keys[i + 1 : j]
    years = end - start - 1
    if years < len(middle):
        raise ValueError(f"{years} years between the ages cannot hold {len(middle)} looks")
    rows = [(0, first)]
    age = start + 1
    if middle:
        base, extra = divmod(years, len(middle))
        for n, key in enumerate(middle):
            rows.append((age, key))
            age += base + (1 if n < extra else 0)
    rows.append((end, last))
    return tuple(rows)


def band_ages(rows: tuple[tuple[int, str], ...]) -> list[tuple[str, int | None, int | None]]:
    """Each look of a table with its ages: (look, first age or None for "and under", last age or
    None for "and over")."""
    out = []
    for n, (first, key) in enumerate(rows):
        last = rows[n + 1][0] - 1 if n + 1 < len(rows) else None
        out.append((key, None if n == 0 else first, last))
    return out
