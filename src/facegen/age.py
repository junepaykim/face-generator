"""Real age to age wording (docs/spec.md section 3, "Apparent age").

A player's real age, in whole years, picks one of ten fixed prompt wordings: W01, the
youngest look, to W10, the oldest. Which wording depends on the player's group and, in
group A, on whether they are at my club, because the most visible players get finer steps.
My club's staff have a table of their own, whatever their group: wide bands over working
ages, W02 the youngest (user, 2026-11-12).

The tables are settings (`age.<table>.<wording> = first real age`); the wordings' text is
fixed here. A wording applies from its first age up to the next wording's first age; ages
below a table's first row use that row.

The wording key is also the player's age band: a tracked player whose wording changes is
re-drawn (spec section 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from facegen.settings import AgeSettings


@dataclass(frozen=True, slots=True)
class Wording:
    text: str
    young: bool
    """Drawn as under 18. The prompt then adds fixed negative terms against adult
    features; that is part of the content rule (spec section 3), not a setting."""


# Picked by the user from the Phase 0.3 bake-off, round 3 (docs/research.md section 5).
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

# One table per audience. Group A is split by whether the player is at my club; staff have
# their own.
TABLES = ("a_my_club", "a_other_club", "b", "c", "staff")


def table_name(group: str | None, at_my_club: bool, staff: bool = False) -> str | None:
    """Which age table applies. `group` is a Group or its value ("a", "b", "c")."""
    if group is None:
        return None
    if staff:
        return "staff"
    if group == "a":
        return "a_my_club" if at_my_club else "a_other_club"
    return str(group)


def age_wording(
    real_age: int | None,
    group: str | None,
    at_my_club: bool,
    s: AgeSettings,
    staff: bool = False,
) -> str | None:
    """The wording key, which is also the age band, or None if age or group is unknown."""
    name = table_name(group, at_my_club, staff)
    if real_age is None or name is None:
        return None
    rows = s.table(name)
    key = rows[0][1]
    for first_age, wording in rows:
        if real_age < first_age:
            break
        key = wording
    return key
