"""Kit colours per club (docs/spec.md section 3), from config/kits.toml. Pure.

A kit is described with a fixed vocabulary of colours and patterns, never free text, so the
table cannot put arbitrary words into a prompt: the wardrobe stays football kit (content
rule, spec section 3). Goalkeepers wear their club's goalkeeper kit. My club's staff wear a
track jacket in the home shirt's colours, long-sleeved club sportswear rather than the playing
kit (decisions.md, 2026-11-12). A club missing from the table gets the neutral fallback kit.

Stripes and hoops are drawn as well as the model manages (decisions.md); the quality check
does not look at the kit. Its shirt colours only help rank group A's candidates.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

KITS_PATH = Path(__file__).resolve().parents[2] / "config" / "kits.toml"


@dataclass(frozen=True, slots=True)
class Colour:
    words: str
    """Prompt word."""
    shirt_tags: frozenset[str]
    """Tagger tags for a shirt in this colour."""


def _c(words: str, *tags: str) -> Colour:
    return Colour(words, frozenset(tags))


COLOURS: dict[str, Colour] = {
    "red": _c("red", "red shirt"),
    "claret": _c("dark red", "red shirt"),
    "white": _c("white", "white shirt"),
    "black": _c("black", "black shirt"),
    "blue": _c("blue", "blue shirt"),
    "sky": _c("light blue", "blue shirt"),
    "navy": _c("navy blue", "blue shirt"),
    "yellow": _c("yellow", "yellow shirt"),
    "amber": _c("orange", "orange shirt", "yellow shirt"),
    "orange": _c("orange", "orange shirt"),
    "green": _c("green", "green shirt"),
    "purple": _c("purple", "purple shirt"),
    "grey": _c("grey", "grey shirt"),
    "pink": _c("pink", "pink shirt"),
}

# pattern -> needs a second shirt colour
PATTERNS: dict[str, bool] = {
    "plain": False,
    "stripes": True,  # vertical
    "hoops": True,  # horizontal
    "sleeves": True,  # body in the first colour, sleeves in the second
    "two-tone": True,  # halves or quarters
}


class KitError(ValueError):
    """config/kits.toml has an entry the vocabulary does not cover."""


@dataclass(frozen=True, slots=True)
class Kit:
    shirt: tuple[str, ...]
    pattern: str = "plain"
    shorts: str | None = None
    trim: str | None = None
    """Collar colour."""
    gloves: str | None = None
    goalkeeper: bool = False
    staff: bool = False
    """A staff member's track jacket in this kit's shirt colours, instead of the shirt."""

    def words(self) -> str:
        """Prompt words for this kit."""
        c = [COLOURS[x].words for x in self.shirt]
        if self.staff:
            return self._jacket_words(c)
        match self.pattern:
            case "plain":
                parts = [f"{c[0]} shirt"]
            case "stripes":
                parts = [f"{c[0]} and {c[1]} vertical-striped shirt"]
            case "hoops":
                parts = [f"{c[0]} and {c[1]} horizontal-striped shirt"]
            case "sleeves":
                parts = [f"{c[0]} shirt", f"{c[1]} sleeves"]
            case "two-tone":
                parts = [f"{c[0]} and {c[1]} two-tone shirt"]
            case _:
                raise KitError(f"unknown pattern {self.pattern!r}")
        if self.trim:
            parts.append(f"{COLOURS[self.trim].words} collar")
        if self.goalkeeper:
            # As tested in the bake-off: long sleeves and gloves, no shorts in the prompt.
            parts = ["goalkeeper", *parts, "long sleeves"]
            parts.append(f"{COLOURS[self.gloves or 'black'].words} gloves")
        else:
            parts.append("short sleeves")
            if self.shorts:
                parts.append(f"{COLOURS[self.shorts].words} shorts")
        return ", ".join(parts)

    def _jacket_words(self, c: list[str]) -> str:
        if self.pattern == "sleeves":
            parts = [f"{c[0]} track jacket", f"{c[1]} sleeves"]
        else:  # stripes and hoops are not asked of a jacket: its two colours are enough
            parts = [f"{' and '.join(c)} track jacket"]
        if self.trim:
            parts.append(f"{COLOURS[self.trim].words} collar")
        parts.append("long sleeves")
        return ", ".join(parts)

    def shirt_tag_groups(self) -> tuple[frozenset[str], ...]:
        """One group per shirt colour; any tag in a group shows that colour was drawn."""
        groups = tuple(COLOURS[x].shirt_tags for x in dict.fromkeys(self.shirt))
        if self.staff:  # the tagger names a jacket's colour as "blue jacket"
            return tuple(frozenset(t.replace(" shirt", " jacket") for t in g) for g in groups)
        return groups

    def describe(self) -> str:
        shirt = " and ".join(self.shirt)
        text = f"{shirt} {self.pattern}" if self.pattern != "plain" else shirt
        if self.staff:
            return f"track jacket, {' and '.join(self.shirt)}"
        if self.goalkeeper:
            return f"goalkeeper {text}"
        return f"{text}, {self.shorts} shorts" if self.shorts else text


@dataclass(frozen=True, slots=True)
class ClubKits:
    home: Kit
    goalkeeper: Kit


@dataclass(frozen=True, slots=True)
class KitTable:
    clubs: Mapping[int, ClubKits]
    fallback: ClubKits

    def kit_for(
        self, club_id: int | None, goalkeeper: bool, *, staff: bool = False
    ) -> tuple[Kit, bool]:
        """The kit to draw and whether the club is in the table (False: the fallback).

        Staff get a track jacket in the home kit's colours, goalkeeping coaches included.
        """
        kits = self.clubs.get(club_id) if club_id is not None else None
        chosen = kits or self.fallback
        if staff:
            return replace(chosen.home, staff=True), kits is not None
        return (chosen.goalkeeper if goalkeeper else chosen.home), kits is not None


def _colour(where: str, value: Any) -> str:
    if value not in COLOURS:
        raise KitError(f"{where}: unknown colour {value!r}; use one of {', '.join(COLOURS)}")
    return value


def parse_kit(where: str, data: Mapping[str, Any], *, goalkeeper: bool) -> Kit:
    allowed = {"shirt", "pattern", "trim"} | ({"gloves"} if goalkeeper else {"shorts"})
    unknown = set(data) - allowed
    if unknown:
        raise KitError(f"{where}: unknown keys {sorted(unknown)}")
    shirt = data.get("shirt")
    if not isinstance(shirt, list) or not 1 <= len(shirt) <= 2:
        raise KitError(f"{where}: 'shirt' must list one or two colours")
    shirt_t = tuple(_colour(f"{where}.shirt", x) for x in shirt)
    pattern = data.get("pattern", "plain")
    if pattern not in PATTERNS:
        raise KitError(f"{where}: unknown pattern {pattern!r}; use one of {', '.join(PATTERNS)}")
    if PATTERNS[pattern] != (len(shirt_t) == 2):
        need = "two shirt colours" if PATTERNS[pattern] else "one shirt colour"
        raise KitError(f"{where}: pattern {pattern!r} needs {need}")
    shorts = data.get("shorts")
    if not goalkeeper and shorts is None:
        raise KitError(f"{where}: 'shorts' is required")
    return Kit(
        shirt=shirt_t,
        pattern=pattern,
        shorts=_colour(f"{where}.shorts", shorts) if shorts is not None else None,
        trim=_colour(f"{where}.trim", data["trim"]) if "trim" in data else None,
        gloves=_colour(f"{where}.gloves", data["gloves"]) if "gloves" in data else None,
        goalkeeper=goalkeeper,
    )


def _club_kits(where: str, data: Mapping[str, Any]) -> ClubKits:
    unknown = set(data) - {"home", "goalkeeper"}
    if unknown or "home" not in data or "goalkeeper" not in data:
        raise KitError(f"{where}: needs exactly 'home' and 'goalkeeper'")
    return ClubKits(
        home=parse_kit(f"{where}.home", data["home"], goalkeeper=False),
        goalkeeper=parse_kit(f"{where}.goalkeeper", data["goalkeeper"], goalkeeper=True),
    )


def parse_kits(data: Mapping[str, Any]) -> KitTable:
    if "fallback" not in data:
        raise KitError("kits: a [fallback] kit is required")
    clubs: dict[int, ClubKits] = {}
    for key, value in data.get("clubs", {}).items():
        if not str(key).isdigit():
            raise KitError(f"clubs.{key}: club keys are the game's unique club IDs")
        clubs[int(key)] = _club_kits(f"clubs.{key}", value)
    return KitTable(clubs=clubs, fallback=_club_kits("fallback", data["fallback"]))


def load_kits(path: Path | None = None) -> KitTable:
    with (path or KITS_PATH).open("rb") as f:
        return parse_kits(tomllib.load(f))
