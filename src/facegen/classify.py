"""Group and tracking rules (docs/spec.md section 2).

Pure: takes a PlayerFacts record and the group thresholds, returns a Classification.
Thresholds are strict ("above"), and CA and PA conditions are joined with OR.

My club's staff are people, not players: they are in group A while on my club's staff
(user, 2026-11-12), and the newgen rules never apply to them, whatever their ID or ability.

A pre-existing player without a photo in FM27 (neither the game's own nor a facepack's) is
picked by the newgen thresholds too, when `no_photo` is on (user, 2026-11-12). The academy
rules stay newgens' own.

Players in my club's teams other than the first team (reserves, under-23s, under-18s) are in
group B, or A when their CA or PA reaches group A's thresholds, while `youth` is on (user,
2026-11-13); moving to the first team puts them in A. They are tracked while at my club.

Journalists from my club's country and my club's press officers are in group B and not
tracked (user, 2026-11-13); other journalists only by a pin. One pinned to A is a regular at
my club: group A quality and tracked like any group A person (user, 2026-11-14). The player
rules never apply to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from facegen.datasource.records import is_first_team_manager
from facegen.settings import Groups

if TYPE_CHECKING:
    from facegen.settings import Settings


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
    is_staff: bool = False
    """A staff member rather than a player, including other clubs' first-team managers."""
    photo: bool | None = None
    """FM27 shows a photo for them other than ours: its own or a facepack's. None: unknown
    (FM27's photo list was not read), which never counts as without a photo."""
    youth: bool = False
    """A player at my club in one of its teams other than the first team."""
    media: str | None = None
    """"journalist" or "press officer" for those people; None for players and staff."""
    local_media: bool = False
    """A journalist from my club's country, or a press officer of my club."""
    staff_job: str | None = None


@dataclass(frozen=True, slots=True)
class Rules:
    """The switches the rules follow, from the settings."""

    staff: bool = True
    """Settings `staff.enabled`: my club's staff and eligible league managers get pictures."""
    no_photo: bool = True
    """Settings `no_photo.enabled`: pre-existing players without a photo follow the newgen
    thresholds."""
    youth: bool = True
    """Settings `youth.enabled`: my club's other teams are in group B unless they reach A."""
    journalists: bool = True
    """Settings `media.journalists`: journalists from my club's country are in group B."""
    press_officers: bool = True
    """Settings `media.press_officers`: my club's press officers are in group B."""

    @classmethod
    def of(cls, settings: Settings) -> Rules:
        return cls(
            staff=settings.staff.enabled,
            no_photo=settings.no_photo.enabled,
            youth=settings.youth.enabled,
            journalists=settings.media.journalists,
            press_officers=settings.media.press_officers,
        )


ALL_RULES = Rules()
"""Every rule on: the defaults."""


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


def without_photo(f: PlayerFacts) -> bool:
    """A pre-existing player FM27 shows no photo for: neither its own nor a facepack's."""
    return not f.is_newgen and not f.is_staff and not f.media and f.photo is False


def my_youth(f: PlayerFacts, rules: Rules) -> bool:
    """A player in one of my club's other teams, under the youth rule."""
    return rules.youth and f.at_my_club and f.youth and not f.is_staff and not f.media


def rule_matches(f: PlayerFacts, g: Groups, rules: Rules = ALL_RULES) -> list[tuple[Group, str]]:
    """Every group rule the person meets, with its reason, ignoring pins."""
    matches: list[tuple[Group, str]] = []
    if f.is_staff:
        if f.at_my_club and rules.staff:
            matches.append((Group.A, "on my club's staff"))
        elif (
            not f.at_my_club
            and rules.staff
            and f.in_my_league
            and is_first_team_manager(f.staff_job)
            and f.photo is False
        ):
            matches.append((Group.C, "manager in my league without a photo"))
        return matches
    if f.media == "journalist":
        if f.local_media and rules.journalists:
            matches.append((Group.B, "journalist from my club's country"))
        return matches
    if f.media == "press officer":
        if f.local_media and rules.press_officers:
            matches.append((Group.B, "my club's press officer"))
        return matches
    if my_youth(f, rules):
        matches.append((Group.B, "in my club's youth or reserve team"))
        if why := _either(f.ca, f.pa, g.a.ca_above, g.a.pa_above):
            matches.append((Group.A, f"youth team, {why}"))
    elif f.at_my_club:
        matches.append((Group.A, "at my club"))
    no_photo = rules.no_photo
    if f.is_newgen:
        who, where_sep = "newgen", " "
    elif no_photo and without_photo(f):
        who, where_sep = "no photo", ", "
    else:
        return matches

    if why := _either(f.ca, f.pa, g.a.ca_above, g.a.pa_above):
        matches.append((Group.A, f"{who}, {why}"))

    if f.is_newgen and g.b.my_academy and f.academy:
        matches.append((Group.B, "came through my academy"))
    if (f.in_my_league or f.in_continental) and (
        why := _either(f.ca, f.pa, g.b.league_ca_above, g.b.league_pa_above)
    ):
        where = "in my league" if f.in_my_league else "in a continental competition"
        matches.append((Group.B, f"{who}{where_sep}{where}, {why}"))
    if why := _either(f.ca, f.pa, g.b.any_ca_above, g.b.any_pa_above):
        matches.append((Group.B, f"{who}, {why}"))

    if f.in_my_league and (why := _either(f.ca, f.pa, g.c.league_ca_above, g.c.league_pa_above)):
        matches.append((Group.C, f"{who}{where_sep}in my league, {why}"))
    return matches


def classify(f: PlayerFacts, g: Groups, rules: Rules = ALL_RULES) -> Classification:
    """The person's group, whether they are tracked, and why.

    - Players outside the eligible football (women's clubs) are ignored, pins included.
    - A pin overrides the rules: "never" excludes the person, a group sets it exactly.
    - Otherwise the person takes the highest group any rule gives.
    - Group A players are tracked; so is an academy newgen, for good, and a player in my
      club's other teams while there.
    - My club's staff are in group A while `rules.staff` is on.
    - Pre-existing players without a photo follow the newgen thresholds while
      `rules.no_photo` is on.
    - Journalists and press officers are in group B (`rules.journalists`,
      `rules.press_officers`) and not tracked: their picture is drawn once. One pinned to A
      is tracked, as a regular at my club.
    """
    if not f.in_scope:
        return Classification(None, False, ("not in men's football",))

    player = not f.is_staff and not f.media
    academy_tracked = g.b.my_academy and f.academy and f.is_newgen and player

    if f.pin == "never":
        return Classification(None, False, ("pinned to never",))

    matches = rule_matches(f, g, rules)
    if f.pin is not None:
        group = Group(f.pin)
        reasons = [f"pinned to {group.label}"]
    elif matches:
        group = min((m[0] for m in matches), key=GROUP_RANK.__getitem__)
        reasons = [why for grp, why in matches if grp is group]
    elif f.is_staff and f.at_my_club:
        return Classification(None, False, ("staff pictures are off (settings: staff)",))
    else:
        return Classification(None, False, ())

    if f.media:
        regular = group is Group.A
        if regular:
            reasons.append("tracked: a regular at my club")
        return Classification(group, regular, tuple(reasons))
    youth_tracked = my_youth(f, rules)
    tracked = group is Group.A or academy_tracked or youth_tracked
    if academy_tracked and group is not Group.A:
        reasons.append("tracked: came through my academy")
    elif youth_tracked and group is not Group.A:
        reasons.append("tracked while at my club")
    return Classification(group, tracked, tuple(reasons))
