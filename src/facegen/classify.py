"""Group and tracking rules (docs/spec.md section 2).

Pure: takes a PlayerFacts record and the group thresholds, returns a Classification.
Thresholds are strict ("above"), and CA and PA conditions are joined with OR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from facegen.settings import Groups


class Group(StrEnum):
    A = "a"
    B = "b"
    C = "c"

    @property
    def label(self) -> str:
        return self.value.upper()


# Lower is better.
GROUP_RANK = {Group.A: 0, Group.B: 1, Group.C: 2}

Pin = Literal["a", "b", "c", "never"]


@dataclass(frozen=True, slots=True)
class PlayerFacts:
    """What the rules need to know about one player, already resolved from the save."""

    unique_id: int
    is_newgen: bool
    ca: int
    pa: int | None
    in_scope: bool
    """In men's football (or whatever the eligibility settings allow)."""
    at_my_club: bool
    in_my_league: bool
    in_continental: bool
    academy: bool
    """A newgen who came through my academy (spec section 2)."""
    pin: Pin | None = None


@dataclass(frozen=True, slots=True)
class Classification:
    group: Group | None
    tracked: bool
    reasons: tuple[str, ...]


def _above(name: str, value: int | None, bar: int) -> str | None:
    if value is not None and value > bar:
        return f"{name} {value} > {bar}"
    return None


def _either(ca: int, pa: int | None, ca_bar: int, pa_bar: int) -> str | None:
    """The first of the CA and PA conditions that holds, as a reason, or None."""
    return _above("CA", ca, ca_bar) or _above("PA", pa, pa_bar)


def rule_matches(f: PlayerFacts, g: Groups) -> list[tuple[Group, str]]:
    """Every group rule the player meets, with its reason, ignoring pins."""
    matches: list[tuple[Group, str]] = []
    if f.at_my_club:
        matches.append((Group.A, "at my club"))
    if not f.is_newgen:
        return matches

    if why := _either(f.ca, f.pa, g.a.ca_above, g.a.pa_above):
        matches.append((Group.A, f"newgen, {why}"))

    if g.b.my_academy and f.academy:
        matches.append((Group.B, "came through my academy"))
    if (f.in_my_league or f.in_continental) and (
        why := _either(f.ca, f.pa, g.b.league_ca_above, g.b.league_pa_above)
    ):
        where = "in my league" if f.in_my_league else "in a continental competition"
        matches.append((Group.B, f"newgen {where}, {why}"))
    if why := _either(f.ca, f.pa, g.b.any_ca_above, g.b.any_pa_above):
        matches.append((Group.B, f"newgen, {why}"))

    if f.in_my_league and (why := _either(f.ca, f.pa, g.c.league_ca_above, g.c.league_pa_above)):
        matches.append((Group.C, f"newgen in my league, {why}"))
    return matches


def classify(f: PlayerFacts, g: Groups) -> Classification:
    """The player's group, whether they are tracked, and why.

    - Players outside the eligible football (women's clubs) are ignored, pins included.
    - A pin overrides the rules: "never" excludes the player, a group sets it exactly.
    - Otherwise the player takes the highest group any rule gives.
    - Group A players are tracked; so is an academy newgen, for good.
    """
    if not f.in_scope:
        return Classification(None, False, ("not in men's football",))

    academy_tracked = g.b.my_academy and f.academy and f.is_newgen

    if f.pin == "never":
        return Classification(None, False, ("pinned to never",))

    matches = rule_matches(f, g)
    if f.pin is not None:
        group = Group(f.pin)
        reasons = [f"pinned to {group.label}"]
    elif matches:
        group = min((m[0] for m in matches), key=GROUP_RANK.__getitem__)
        reasons = [why for grp, why in matches if grp is group]
    else:
        return Classification(None, False, ())

    tracked = group is Group.A or academy_tracked
    if academy_tracked and group is not Group.A:
        reasons.append("tracked: came through my academy")
    return Classification(group, tracked, tuple(reasons))
