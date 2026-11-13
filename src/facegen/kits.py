"""Kit colours per club (docs/spec.md section 3), from config/kits.toml. Pure, apart from the
loaders at the end.

A kit is described with a fixed vocabulary of colours and patterns, never free text, so the
table cannot put arbitrary words into a prompt: the wardrobe stays football kit (content
rule, spec section 3). Goalkeepers wear their club's goalkeeper kit. My club's staff do not
wear the kit: wardrobe.py dresses them, with the home kit's colours on one piece. A player's
picture wears the kit in one of shoot.py's looks.

Where a kit comes from, first match wins (`KitTable.kit_for`):

1. config/kits.toml, written by hand: colours, pattern, shorts and goalkeeper kit.
2. The kit catalog: colours estimated from FM27's own 2D home-shirt image of the club
   (user, 2026-11-12). `scripts/spikes/kit_colour_probe.py` reads the images from the game
   install and writes its estimates to data/kit-colour-probe/report.json, which stays on
   this PC. An estimate gives one or two shirt colours, never a pattern: two colours are
   drawn as a "mixed" shirt, with no layout asked for, because two colours alone do not say
   whether they are stripes, hoops or sleeves. Its shorts are unknown and left out, and its
   goalkeeper wears the first of KEEPER_COLOURS that is not like the home shirt (FM27's
   goalkeeper colours are not known). A club FM27 has no image for is not in the catalog.
3. The neutral fallback kit.

Stripes and hoops are drawn as well as the model manages (decisions.md); the quality check
does not look at the kit. Its shirt colours only help rank group A's candidates.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

KITS_PATH = Path(__file__).resolve().parents[2] / "config" / "kits.toml"
CATALOG_REPORT = Path("kit-colour-probe") / "report.json"
"""The kit catalog, relative to the data folder: the colour probe's output."""

type KitSource = Literal["table", "catalog", "fallback"]
"""Where a kit came from: config/kits.toml, the kit catalog, or the fallback kit."""


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
        """Prompt words for this kit as a plain match kit: short sleeves and shorts, or for a
        goalkeeper long sleeves and gloves (as tested in the bake-off: no shorts)."""
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
    """Clubs whose colours are estimated from FM27's kit images, by unique club ID."""

    def kit_for(self, club_id: int | None, goalkeeper: bool) -> tuple[Kit, KitSource]:
        """The kit to draw and where it came from: the table, the catalog or the fallback.

        Staff, goalkeeping coaches included, take the home kit: its colours dress them.
        """
        kits: ClubKits
        source: KitSource
        if club_id is not None and club_id in self.clubs:
            kits, source = self.clubs[club_id], "table"
        elif club_id is not None and club_id in self.catalog:
            kits, source = self.catalog[club_id], "catalog"
        else:
            kits, source = self.fallback, "fallback"
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


# --- the kit catalog ---------------------------------------------------------------------


def keeper_colour(home: Kit) -> str:
    """A goalkeeper colour unlike the home shirt's, for a club whose keeper kit is unknown."""
    taken = families((*home.shirt, *((home.trim,) if home.trim else ())))
    for colour in KEEPER_COLOURS:
        if not COLOURS[colour].families & taken:
            return colour
    return KEEPER_COLOURS[0]


def catalog_kit(row: Mapping[str, Any]) -> Kit | None:
    """The home kit for one row of the colour probe's report; None if it gives no colour.

    Only the broad colour names are used, never the measured RGB values. A second colour
    makes a "mixed" shirt: the probe does not say how the colours are laid out.
    """
    if row.get("status") != "estimated":
        return None
    primary = (row.get("primary") or {}).get("name")
    if primary not in COLOURS:
        return None
    secondary = (row.get("secondary") or {}).get("name")
    if secondary in COLOURS and secondary != primary:
        return Kit(shirt=(primary, secondary), pattern="mixed")
    return Kit(shirt=(primary,))


def parse_catalog(report: Mapping[str, Any]) -> dict[int, ClubKits]:
    """Club ID to kits, from the colour probe's report: the home rows with an estimate."""
    results = report.get("results")
    if not isinstance(results, list):
        raise KitError("kit catalog: no 'results' list")
    clubs: dict[int, ClubKits] = {}
    for row in results:
        if not isinstance(row, Mapping) or row.get("kind") != "home":
            continue
        club_id = row.get("club_id")
        home = catalog_kit(row)
        if not isinstance(club_id, int) or home is None:
            continue
        keeper = Kit(shirt=(keeper_colour(home),), goalkeeper=True)
        clubs[club_id] = ClubKits(home=home, goalkeeper=keeper)
    return clubs


def catalog_path(data_dir: Path) -> Path:
    return data_dir / CATALOG_REPORT


def load_kits(path: Path | None = None, *, catalog: Path | None = None) -> KitTable:
    """config/kits.toml (or `path`), with the kit catalog at `catalog` if that file exists.

    No catalog file means no catalog: clubs outside the table get the fallback kit, as before.
    """
    with (path or KITS_PATH).open("rb") as f:
        table = parse_kits(tomllib.load(f))
    if catalog is None or not catalog.is_file():
        return table
    try:
        report = json.loads(catalog.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KitError(f"kit catalog {catalog}: {exc}") from exc
    if not isinstance(report, Mapping):
        raise KitError(f"kit catalog {catalog}: not a JSON object")
    return replace(table, catalog=parse_catalog(report))
