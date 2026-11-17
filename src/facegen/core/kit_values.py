"""Fixed kit vocabulary and selection from plain club-colour records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

type KitSource = Literal["table", "catalog", "bundled", "fallback"]
"""Where a kit came from: config/kits.toml, the kit catalog (a probe report on this PC), the
bundled catalog, or the fallback kit."""


@dataclass(frozen=True, slots=True)
class Colour:
    words: str
    """Prompt word."""
    shirt_tags: frozenset[str]
    """Tagger tags for a shirt in this colour."""
    families: frozenset[str]
    """Broad colour families, to tell which backdrops would blend with it (shoot.py)."""


def _c(words: str, tags: tuple[str, ...], families: tuple[str, ...]) -> Colour:
    return Colour(words, frozenset(tags), frozenset(families))


COLOURS: dict[str, Colour] = {
    "red": _c("red", ("red shirt",), ("red",)),
    "claret": _c("dark red", ("red shirt",), ("red",)),
    "white": _c("white", ("white shirt",), ("white",)),
    "black": _c("black", ("black shirt",), ("black",)),
    "blue": _c("blue", ("blue shirt",), ("blue",)),
    "sky": _c("light blue", ("blue shirt",), ("blue",)),
    "navy": _c("navy blue", ("blue shirt",), ("blue",)),
    "yellow": _c("yellow", ("yellow shirt",), ("yellow",)),
    "amber": _c("orange", ("orange shirt", "yellow shirt"), ("orange", "yellow")),
    "orange": _c("orange", ("orange shirt",), ("orange",)),
    "green": _c("green", ("green shirt",), ("green",)),
    "purple": _c("purple", ("purple shirt",), ("purple",)),
    "grey": _c("grey", ("grey shirt",), ("grey",)),
    "pink": _c("pink", ("pink shirt",), ("pink",)),
}

# pattern -> needs a second shirt colour
PATTERNS: dict[str, bool] = {
    "plain": False,
    "stripes": True,  # vertical
    "hoops": True,  # horizontal
    "sleeves": True,  # body in the first colour, sleeves in the second
    "two-tone": True,  # halves or quarters
    "mixed": True,  # two colours, layout not known (the kit catalog's estimates)
}

KEEPER_COLOURS = ("green", "yellow", "orange", "purple")
"""A catalog club's goalkeeper wears the first of these unlike every home shirt colour."""


def families(colours: tuple[str, ...]) -> frozenset[str]:
    """The colour families of some kit colours."""
    return frozenset(f for c in colours for f in COLOURS[c].families)


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

    def words(self) -> str:
        """Match-kit words: short sleeves and shorts, or goalkeeper long sleeves and gloves
        without shorts."""
        if self.goalkeeper:
            return f"goalkeeper, {self.shirt_words()}, long sleeves, {self.gloves_words()}"
        return ", ".join(p for p in (self.shirt_words(), "short sleeves", self.shorts_words()) if p)

    def shirt_words(self) -> str:
        """The shirt alone: colours, pattern and collar."""
        c = [COLOURS[x].words for x in self.shirt]
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
            case "mixed":
                parts = [f"{c[0]} and {c[1]} shirt"]
            case _:
                raise KitError(f"unknown pattern {self.pattern!r}")
        if self.trim:
            parts.append(f"{COLOURS[self.trim].words} collar")
        return ", ".join(parts)

    def jacket_words(self) -> str:
        """The club track jacket in this kit's shirt colours (and collar)."""
        c = [COLOURS[x].words for x in self.shirt]
        if self.pattern == "sleeves":
            parts = [f"{c[0]} track jacket", f"{c[1]} sleeves"]
        else:  # stripes and hoops are not asked of a jacket: its two colours are enough
            parts = [f"{' and '.join(c)} track jacket"]
        if self.trim:
            parts.append(f"{COLOURS[self.trim].words} collar")
        return ", ".join(parts)

    def shorts_words(self) -> str:
        """The shorts, or nothing where they are not known."""
        return f"{COLOURS[self.shorts].words} shorts" if self.shorts else ""

    def gloves_words(self) -> str:
        return f"{COLOURS[self.gloves or 'black'].words} gloves"

    def colours(self) -> tuple[str, ...]:
        """Every colour of the shirt, collar and shorts, without repeats."""
        return tuple(dict.fromkeys(c for c in (*self.shirt, self.trim, self.shorts) if c))

    def shirt_tag_groups(self, garment: str = "shirt") -> tuple[frozenset[str], ...]:
        """One group per shirt colour; any tag in a group shows that colour was drawn.

        `garment` names what the colour is on: "jacket" gives "blue jacket" for "blue shirt".
        """
        groups = tuple(COLOURS[x].shirt_tags for x in dict.fromkeys(self.shirt))
        if garment == "shirt":
            return groups
        return tuple(frozenset(t.replace(" shirt", f" {garment}") for t in g) for g in groups)

    def rank_tags(self) -> dict[str, frozenset[str]]:
        """Ranking only: each shirt colour to the tags that show it was drawn."""
        return {f"shirt colour {i}": g for i, g in enumerate(self.shirt_tag_groups(), 1)}

    def describe(self) -> str:
        shirt = " and ".join(self.shirt)
        text = f"{shirt} {self.pattern}" if self.pattern not in ("plain", "mixed") else shirt
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
    catalog: Mapping[int, ClubKits] = field(default_factory=dict)
    """Clubs whose colours a probe report on this PC estimated, by unique club ID."""
    bundled: Mapping[int, ClubKits] = field(default_factory=dict)
    """Clubs whose colours the bundled catalog has, by unique club ID."""
    report_first: bool = False
    """The probe report is newer than the bundle, so its estimates come before the bundle's."""

    def kit_for(self, club_id: int | None, goalkeeper: bool) -> tuple[Kit, KitSource]:
        """The kit to draw and where it came from: the table, the catalog, the bundle or the
        fallback.

        Staff, goalkeeping coaches included, take the home kit: its colours dress them.
        """
        kits: ClubKits = self.fallback
        source: KitSource = "fallback"
        estimates: list[tuple[Mapping[int, ClubKits], KitSource]] = [
            (self.catalog, "catalog"),
            (self.bundled, "bundled"),
        ]
        if not self.report_first:
            estimates.reverse()
        if club_id is not None:
            for found, where in ((self.clubs, "table"), *estimates):
                if club_id in found:
                    kits, source = found[club_id], where
                    break
        return (kits.goalkeeper if goalkeeper else kits.home), source


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


def keeper_colour(home: Kit) -> str:
    """A goalkeeper colour unlike the home shirt's, for a club whose keeper kit is unknown."""
    taken = families((*home.shirt, *((home.trim,) if home.trim else ())))
    for colour in KEEPER_COLOURS:
        if not COLOURS[colour].families & taken:
            return colour
    return KEEPER_COLOURS[0]


def estimated_kits(primary: Any, secondary: Any = None) -> ClubKits | None:
    """Kits from estimated home-shirt colour names; None if the first is not a known colour.

    Only the broad colour names are used, never measured RGB values. A second colour makes a
    "mixed" shirt: an estimate does not say how the colours are laid out.
    """
    if primary not in COLOURS:
        return None
    if secondary in COLOURS and secondary != primary:
        home = Kit(shirt=(primary, secondary), pattern="mixed")
    else:
        home = Kit(shirt=(primary,))
    return ClubKits(home=home, goalkeeper=Kit(shirt=(keeper_colour(home),), goalkeeper=True))
