"""Pure group and tracking rules (spec section 2).

Every number means "at least" and CA/PA conditions use OR (Phase 16). Who each tier picks comes
from the effective cutoffs (core/cutoffs.py): my club's tiers, shares of my league, the league
above and below and the continental competitions my club plays in, and global numbers. The
share rules pick only candidates: newgens and, when enabled, known photo-less players; academy
rules stay newgens' own. My-club staff are group A; the player rules never apply to staff or
media. Eligible media are group B and untracked unless pinned to A.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from facegen.core.cutoffs import Cut, Cutoffs
from facegen.core.records import is_first_team_manager

if TYPE_CHECKING:
    from facegen.core.settings_values import Settings


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
    """At a club in a continental competition my club plays in."""
    academy: bool
    """A newgen who came through my academy (spec section 2)."""
    pin: Pin | None = None
    is_staff: bool = False
    """A staff member rather than a player, including other clubs' first-team managers."""
    photo: bool | None = None
    """FM27 shows a photo for them other than ours: its own or a facegen's. None: unknown
    (FM27's photo list was not read), which never counts as without a photo."""
    youth: bool = False
    """A player at my club in one of its teams other than the first team."""
    media: str | None = None
    """"journalist" or "press officer" for those people; None for players and staff."""
    local_media: bool = False
    """A journalist from my club's country, or a press officer of my club."""
    staff_job: str | None = None
    league_offset: int | None = None
    """How many levels their club's league is above mine (1: the league above, -1: below);
    None when it is not on my nation's ladder (core/leagues.py)."""
    loaned_out: bool = False
    """A player my club has loaned out to another club."""


@dataclass(frozen=True, slots=True)
class Rules:
    """The switches the rules follow, from the settings."""

    staff: bool = True
    """Settings `staff.enabled`: my club's staff and eligible league managers get pictures."""
    no_photo: bool = True
    """Settings `no_photo.enabled`: pre-existing players without a photo are candidates."""
    journalists: bool = True
    """Settings `media.journalists`: journalists from my club's country are in group B."""
    press_officers: bool = True
    """Settings `media.press_officers`: my club's press officers are in group B."""

    @classmethod
    def of(cls, settings: Settings) -> Rules:
        return cls(
            staff=settings.staff.enabled,
            no_photo=settings.no_photo.enabled,
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


def _either(f: PlayerFacts, ca: Cut, pa: Cut) -> str | None:
    """The first of the CA and PA conditions that holds, as a reason, or None."""
    return ca.why("CA", f.ca) or pa.why("PA", f.pa)


def without_photo(f: PlayerFacts) -> bool:
    """A pre-existing player FM27 shows no photo for: neither its own nor a facepack's."""
    return not f.is_newgen and not f.is_staff and not f.media and f.photo is False


def is_candidate(f: PlayerFacts, rules: Rules = ALL_RULES) -> bool:
    """Someone the share and global rules can pick: a newgen, or (with `rules.no_photo`) a
    pre-existing player without a photo. Staff and media never are."""
    if f.is_staff or f.media:
        return False
    return f.is_newgen or (rules.no_photo and without_photo(f))


def my_club_part(f: PlayerFacts) -> str | None:
    """Which part of my club a player is in: "loans", "youth" or "first_team"; None for
    someone not at my club, staff and media."""
    if not f.at_my_club or f.is_staff or f.media:
        return None
    if f.loaned_out:
        return "loans"
    return "youth" if f.youth else "first_team"


_PART_REASON = {
    "first_team": "in my club's first team",
    "youth": "in my club's youth or reserve team",
    "loans": "loaned out by my club",
}
_PART_SHORT = {"first_team": "first team", "youth": "youth team", "loans": "on loan"}


def rule_matches(
    f: PlayerFacts, cuts: Cutoffs, rules: Rules = ALL_RULES
) -> list[tuple[Group, str]]:
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

    part = my_club_part(f)
    if part is not None:
        tier = Group(getattr(cuts, part))
        matches.append((tier, _PART_REASON[part]))
        if tier is not Group.A:
            # Youth, loanees (and a first team set lower) reach A by my league's or the global
            # A cut, whoever they are.
            why = _either(f, cuts.league_cut("a_ca"), cuts.league_cut("a_pa")) or _either(
                f, cuts.everywhere_cut("a_ca"), cuts.everywhere_cut("a_pa")
            )
            if why:
                matches.append((Group.A, f"{_PART_SHORT[part]}, {why}"))

    if f.is_newgen:
        who, sep = "newgen", " "
    elif rules.no_photo and without_photo(f):
        who, sep = "no photo", ", "
    else:
        return matches

    def league(tier: Group, where: str, ca: Cut, pa: Cut) -> None:
        if why := _either(f, ca, pa):
            matches.append((tier, f"{who}{sep}{where}, {why}"))

    def everywhere(tier: Group) -> None:
        g = tier.value
        if why := _either(f, cuts.everywhere_cut(f"{g}_ca"), cuts.everywhere_cut(f"{g}_pa")):
            matches.append((tier, f"{who}, {why}"))

    everywhere(Group.A)
    if f.in_my_league:
        league(Group.A, "in my league", cuts.league_cut("a_ca"), cuts.league_cut("a_pa"))
    if f.is_newgen and cuts.academy_alumni and f.academy:
        matches.append((Group.B, "came through my academy"))
    if f.in_my_league:
        league(Group.B, "in my league", cuts.league_cut("b_ca"), cuts.league_cut("b_pa"))
    if f.league_offset == 1:
        league(Group.B, "in the league above", cuts.other_cut("above_b_ca"), Cut(None))
    if f.in_continental:
        where = "in a continental competition"
        league(Group.B, where, cuts.other_cut("continental_b_ca"), Cut(None))
    everywhere(Group.B)
    if f.in_my_league:
        league(Group.C, "in my league", cuts.league_cut("c_ca"), cuts.league_cut("c_pa"))
    if f.league_offset == 1:
        league(Group.C, "in the league above", cuts.other_cut("above_c_ca"), Cut(None))
    if f.league_offset == -1:
        league(Group.C, "in the league below", cuts.other_cut("below_c_ca"), Cut(None))
    everywhere(Group.C)
    return matches


def classify(f: PlayerFacts, cuts: Cutoffs, rules: Rules = ALL_RULES) -> Classification:
    """The person's group, whether they are tracked, and why.

    - Players outside the eligible football (women's clubs) are ignored, pins included.
    - A pin overrides the rules: "never" excludes the person, a group sets it exactly.
    - Otherwise the person takes the highest group any rule gives.
    - Group A players are tracked; so is a player at my club while there (a loanee while I
      hold the contract), and an academy newgen, for good.
    - My club's staff are in group A while `rules.staff` is on.
    - Journalists and press officers are in group B (`rules.journalists`,
      `rules.press_officers`) and not tracked: their picture is drawn once. One pinned to A
      is tracked, as a regular at my club.
    """
    if not f.in_scope:
        return Classification(None, False, ("not in men's football",))

    player = not f.is_staff and not f.media
    academy_tracked = cuts.academy_alumni and f.academy and f.is_newgen and player

    if f.pin == "never":
        return Classification(None, False, ("pinned to never",))

    matches = rule_matches(f, cuts, rules)
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
    part = my_club_part(f)
    tracked = group is Group.A or academy_tracked or part is not None
    if group is not Group.A:
        if academy_tracked:
            reasons.append("tracked: came through my academy")
        elif part == "loans":
            reasons.append("tracked while I hold the contract")
        elif part is not None:
            reasons.append("tracked while at my club")
    return Classification(group, tracked, tuple(reasons))
