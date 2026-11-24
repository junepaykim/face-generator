"""Pure group and tracking rules (spec section 2).

Every number means "at least" and CA/PA conditions use OR (Phase 16). Who each tier picks comes
from the effective cutoffs (core/cutoffs.py): my club's tiers, shares of my league, the league
above and below and the continental competitions my club plays in, and global numbers. The
share rules pick only candidates: newgens and, when enabled, known photo-less players; academy
rules stay newgens' own. My-club staff are group A; the player rules never apply to staff or
media. Eligible media are group B and untracked unless pinned to A.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from kitshot.core.cutoffs import Cut, Cutoffs, Met
from kitshot.core.records import is_first_team_manager

if TYPE_CHECKING:
    from kitshot.core.settings_values import Settings


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
    """FM27 shows a photo for them other than ours: its own or a Kitshot's. None: unknown
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


WHO_TEXT = {"newgen": "newgen", "no_photo": "no photo"}
WHERE_TEXT = {
    "league": "in my league",
    "above": "in the league above",
    "below": "in the league below",
    "continental": "in a continental competition",
}
_PART_REASON = {
    "first_team": "in my club's first team",
    "youth": "in my club's youth or reserve team",
    "loans": "loaned out by my club",
}
_PART_SHORT = {"first_team": "first team", "youth": "youth team", "loans": "on loan"}
MY_CLUB_PARTS = tuple(_PART_REASON)
_TEXT = {
    "staff": "on my club's staff",
    "manager_no_photo": "manager in my league without a photo",
    "journalist": "journalist from my club's country",
    "press_officer": "my club's press officer",
    "academy": "came through my academy",
    "not_mens": "not in men's football",
    "pin_never": "pinned to never",
    "staff_off": "staff pictures are off (settings: staff)",
    "tracked_regular": "tracked: a regular at my club",
    "tracked_academy": "tracked: came through my academy",
    "tracked_loan": "tracked while I hold the contract",
    "tracked_at_club": "tracked while at my club",
    "left_staff": "no longer on my club's staff",
    "gone": "not in the latest save",
}
WHY_CODES = (*_TEXT, "part", "part_cut", "cut", "pin")
"""Every reason's code: the pages word each as `why.<code>` (Phase 17h)."""


@dataclass(frozen=True, slots=True)
class Why:
    """One reason as a code and the values it names (Phase 17h). The pages word it from the
    catalogs; `text` is the English sentence the command line and exports print."""

    code: str
    met: Met | None = None
    """The CA or PA condition that holds ("cut", "part_cut")."""
    who: str | None = None
    """Whom a cut picked ("cut"): "newgen" or "no_photo"."""
    where: str | None = None
    """Where the cut applies ("cut"): a WHERE_TEXT key, None for everywhere."""
    part: str | None = None
    """The part of my club ("part", "part_cut"): "first_team", "youth" or "loans"."""
    group: str | None = None
    """The group a person is pinned to ("pin")."""

    @property
    def text(self) -> str:
        match self.code:
            case "part":
                return _PART_REASON[self.part or ""]
            case "part_cut":
                return f"{_PART_SHORT[self.part or '']}, {self.met.text if self.met else ''}"
            case "cut":
                who = WHO_TEXT[self.who or ""]
                met = self.met.text if self.met else ""
                if self.where is None:
                    return f"{who}, {met}"
                sep = " " if self.who == "newgen" else ", "
                return f"{who}{sep}{WHERE_TEXT[self.where]}, {met}"
            case "pin":
                return f"pinned to {(self.group or '').upper()}"
        return _TEXT[self.code]

    @property
    def complete(self) -> bool:
        """Whether its code is known and has the values it names (a stored one may not)."""
        if self.code in _TEXT:
            return True
        part = self.part in _PART_REASON
        match self.code:
            case "part":
                return part
            case "part_cut":
                return part and self.met is not None
            case "cut":
                where = self.where is None or self.where in WHERE_TEXT
                return self.who in WHO_TEXT and where and self.met is not None
            case "pin":
                return self.group in ("a", "b", "c")
        return False

    @property
    def tracking(self) -> bool:
        """Says why the person is tracked rather than why they are in their group."""
        return self.code.startswith("tracked_")

    def to_json(self) -> dict[str, Any]:
        """The code and the values it has, flat: what `players.why` stores."""
        out: dict[str, Any] = {"code": self.code}
        if self.met is not None:
            out.update(
                attr=self.met.attr, value=self.met.value, at=self.met.at, share=self.met.share
            )
        for name in ("who", "where", "part", "group"):
            if (value := getattr(self, name)) is not None:
                out[name] = value
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Why:
        met = None
        if "attr" in data:
            met = Met(data["attr"], data["value"], data["at"], data.get("share"))
        return cls(
            data["code"],
            met,
            data.get("who"),
            data.get("where"),
            data.get("part"),
            data.get("group"),
        )


def whys_from_json(text: str | None) -> tuple[Why, ...] | None:
    """The stored `players.why` list; None for a row not classified since it was added (its
    English `reasons` stand until then) or one this code cannot read."""
    if not text:
        return None
    try:
        whys = tuple(Why.from_json(item) for item in json.loads(text))
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return whys if all(w.complete for w in whys) else None


def whys_to_json(whys: tuple[Why, ...]) -> str:
    return json.dumps([w.to_json() for w in whys])


@dataclass(frozen=True, slots=True)
class Classification:
    group: Group | None
    tracked: bool
    reasons: tuple[str, ...]
    """In English, for the command line and exports: each `why`'s text."""
    why: tuple[Why, ...] = ()


def classified(
    group: Group | None, tracked: bool, whys: list[Why] | tuple[Why, ...]
) -> Classification:
    """A classification with its reasons as records and as English text."""
    return Classification(group, tracked, tuple(w.text for w in whys), tuple(whys))


def _either(f: PlayerFacts, ca: Cut, pa: Cut) -> Met | None:
    """The first of the CA and PA conditions that holds, or None."""
    return ca.met("CA", f.ca) or pa.met("PA", f.pa)


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


def rule_matches(
    f: PlayerFacts, cuts: Cutoffs, rules: Rules = ALL_RULES
) -> list[tuple[Group, Why]]:
    """Every group rule the person meets, with its reason, ignoring pins."""
    matches: list[tuple[Group, Why]] = []
    if f.is_staff:
        if f.at_my_club and rules.staff:
            matches.append((Group.A, Why("staff")))
        elif (
            not f.at_my_club
            and rules.staff
            and f.in_my_league
            and is_first_team_manager(f.staff_job)
            and f.photo is False
        ):
            matches.append((Group.C, Why("manager_no_photo")))
        return matches
    if f.media == "journalist":
        if f.local_media and rules.journalists:
            matches.append((Group.B, Why("journalist")))
        return matches
    if f.media == "press officer":
        if f.local_media and rules.press_officers:
            matches.append((Group.B, Why("press_officer")))
        return matches

    part = my_club_part(f)
    if part is not None:
        tier = Group(getattr(cuts, part))
        matches.append((tier, Why("part", part=part)))
        if tier is not Group.A:
            # Youth, loanees (and a first team set lower) reach A by my league's or the global
            # A cut, whoever they are.
            met = _either(f, cuts.league_cut("a_ca"), cuts.league_cut("a_pa")) or _either(
                f, cuts.everywhere_cut("a_ca"), cuts.everywhere_cut("a_pa")
            )
            if met:
                matches.append((Group.A, Why("part_cut", met, part=part)))

    if f.is_newgen:
        who = "newgen"
    elif rules.no_photo and without_photo(f):
        who = "no_photo"
    else:
        return matches

    def league(tier: Group, where: str, ca: Cut, pa: Cut) -> None:
        if met := _either(f, ca, pa):
            matches.append((tier, Why("cut", met, who=who, where=where)))

    def everywhere(tier: Group) -> None:
        g = tier.value
        if met := _either(f, cuts.everywhere_cut(f"{g}_ca"), cuts.everywhere_cut(f"{g}_pa")):
            matches.append((tier, Why("cut", met, who=who)))

    everywhere(Group.A)
    if f.in_my_league:
        league(Group.A, "league", cuts.league_cut("a_ca"), cuts.league_cut("a_pa"))
    if f.is_newgen and cuts.academy_alumni and f.academy and part is None:
        # Once they leave (spec section 2): while at my club, my club's tier is theirs.
        matches.append((Group.B, Why("academy")))
    if f.in_my_league:
        league(Group.B, "league", cuts.league_cut("b_ca"), cuts.league_cut("b_pa"))
    if f.league_offset == 1:
        league(Group.B, "above", cuts.other_cut("above_b_ca"), Cut(None))
    if f.in_continental:
        league(Group.B, "continental", cuts.other_cut("continental_b_ca"), Cut(None))
    everywhere(Group.B)
    if f.in_my_league:
        league(Group.C, "league", cuts.league_cut("c_ca"), cuts.league_cut("c_pa"))
    if f.league_offset == 1:
        league(Group.C, "above", cuts.other_cut("above_c_ca"), Cut(None))
    if f.league_offset == -1:
        league(Group.C, "below", cuts.other_cut("below_c_ca"), Cut(None))
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
        return classified(None, False, [Why("not_mens")])

    player = not f.is_staff and not f.media
    academy_tracked = cuts.academy_alumni and f.academy and f.is_newgen and player

    if f.pin == "never":
        return classified(None, False, [Why("pin_never")])

    matches = rule_matches(f, cuts, rules)
    if f.pin is not None:
        group = Group(f.pin)
        whys = [Why("pin", group=group.value)]
    elif matches:
        group = min((m[0] for m in matches), key=GROUP_RANK.__getitem__)
        whys = [why for grp, why in matches if grp is group]
    elif f.is_staff and f.at_my_club:
        return classified(None, False, [Why("staff_off")])
    else:
        return classified(None, False, [])

    if f.media:
        regular = group is Group.A
        if regular:
            whys.append(Why("tracked_regular"))
        return classified(group, regular, whys)
    part = my_club_part(f)
    tracked = group is Group.A or academy_tracked or part is not None
    if group is not Group.A:
        if academy_tracked:
            whys.append(Why("tracked_academy"))
        elif part == "loans":
            whys.append(Why("tracked_loan"))
        elif part is not None:
            whys.append(Why("tracked_at_club"))
    return classified(group, tracked, whys)
