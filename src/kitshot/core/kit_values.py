"""Fixed kit vocabulary and selection from plain club-colour records."""

from __future__ import annotations

import colorsys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

type KitSource = Literal["table", "game", "catalog", "bundled", "fallback"]
"""Where a kit came from: config/kits.toml, FM27's own database (`game`), the kit catalog (a
probe report on this PC), the bundled catalog, or the fallback kit."""

type RGB = tuple[int, int, int]


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

# pattern -> how many shirt colours it takes (fewest, most). The first is always the body.
PATTERNS: dict[str, tuple[int, int]] = {
    "plain": (1, 1),
    "stripes": (2, 3),  # vertical; a third colour for three-colour stripes
    "hoops": (2, 2),  # horizontal
    "sleeves": (2, 2),  # body in the first colour, sleeves in the second
    "two-tone": (2, 2),  # halves or quarters
    "mixed": (2, 2),  # two colours, layout not known (the kit catalog's estimates)
    "pinstripes": (2, 2),  # thin vertical lines on the body
    "thin-hoops": (2, 2),  # thin horizontal lines on the body
    "checks": (2, 2),  # a checkerboard
    "centre-stripe": (2, 2),  # one wide vertical stripe down the middle
    "sash": (2, 3),  # a diagonal sash, in one or two colours
    "band": (2, 3),  # a band across the chest, in one or two colours
    "chevron": (2, 2),  # a V across the chest
    "cross": (2, 2),  # a cross on the front
}


def join_and(words: list[str]) -> str:
    """ "a", "a and b", "a, b and c"."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


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
                parts = [f"{join_and(c)} vertical-striped shirt"]
            case "hoops":
                parts = [f"{c[0]} and {c[1]} horizontal-striped shirt"]
            case "sleeves":
                parts = [f"{c[0]} shirt", f"{c[1]} sleeves"]
            case "two-tone":
                parts = [f"{c[0]} and {c[1]} two-tone shirt"]
            case "mixed":
                parts = [f"{c[0]} and {c[1]} shirt"]
            case "pinstripes":
                parts = [f"{c[0]} and {c[1]} pinstripe shirt"]
            case "thin-hoops":
                parts = [f"{c[0]} shirt", f"thin {c[1]} horizontal stripes"]
            case "checks":
                parts = [f"{c[0]} and {c[1]} checkered shirt"]
            case "centre-stripe":
                parts = [f"{c[0]} shirt", f"wide {c[1]} vertical stripe down the middle"]
            case "sash":
                parts = [f"{c[0]} shirt", f"{join_and(c[1:])} diagonal sash"]
            case "band":
                parts = [f"{c[0]} shirt", f"{join_and(c[1:])} band across the chest"]
            case "chevron":
                parts = [f"{c[0]} shirt", f"{c[1]} V-shaped band across the chest"]
            case "cross":
                parts = [f"{c[0]} shirt", f"{c[1]} cross on the chest"]
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
        else:  # stripes, hoops and the rest are not asked of a jacket: its colours are enough
            parts = [f"{join_and(c)} track jacket"]
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
        shirt = join_and(list(self.shirt))
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
    game: Mapping[int, ClubKits] = field(default_factory=dict)
    """Clubs whose home shirt FM27's own database gives, by unique club ID (game_kits)."""
    catalog: Mapping[int, ClubKits] = field(default_factory=dict)
    """Clubs whose colours a probe report on this PC estimated, by unique club ID."""
    bundled: Mapping[int, ClubKits] = field(default_factory=dict)
    """Clubs whose colours the bundled catalog has, by unique club ID."""
    report_first: bool = False
    """The probe report is newer than the bundle, so its estimates come before the bundle's."""

    def kit_for(self, club_id: int | None, goalkeeper: bool) -> tuple[Kit, KitSource]:
        """The kit to draw and where it came from: the table, FM27's database, the catalog, the
        bundle or the fallback.

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
            for found, where in ((self.clubs, "table"), (self.game, "game"), *estimates):
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
    if not isinstance(shirt, list) or not 1 <= len(shirt) <= 3:
        raise KitError(f"{where}: 'shirt' must list one to three colours")
    shirt_t = tuple(_colour(f"{where}.shirt", x) for x in shirt)
    pattern = data.get("pattern", "plain")
    if pattern not in PATTERNS:
        raise KitError(f"{where}: unknown pattern {pattern!r}; use one of {', '.join(PATTERNS)}")
    fewest, most = PATTERNS[pattern]
    if not fewest <= len(shirt_t) <= most:
        counts = {1: "one", 2: "two", 3: "three"}
        need = counts[fewest] if fewest == most else f"{counts[fewest]} or {counts[most]}"
        raise KitError(
            f"{where}: pattern {pattern!r} needs {need} shirt colour{'s' if most > 1 else ''}"
        )
    if len(set(shirt_t)) < len(shirt_t):
        raise KitError(f"{where}: a shirt colour is listed twice")
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


# --- FM27's own kits (datasource/game_kits.py reads them from the game's database) ----------


def colour_name(rgb: RGB) -> str:
    """The vocabulary colour closest to one of FM27's kit colours (0-255 per channel).

    Rules on hue, saturation and brightness, tuned on the database's flat colours (research.md,
    "FM27's kits"): near-greys by brightness; then by hue, with dark reds as claret, light
    reds as pink, dark blues as navy and light or cyan blues as sky; cream counts as white.
    """
    r, g, b = (c / 255 for c in rgb)
    hue, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = hue * 360
    chroma = max(r, g, b) - min(r, g, b)
    if v < 0.17:
        return "black"
    if s < 0.15 or (chroma < 0.16 and s < 0.3):
        return "white" if v >= 0.80 else "black" if v < 0.30 else "grey"
    if h >= 330 or h < 15:
        if v > 0.75 and (s < 0.65 or (330 <= h <= 345 and s < 0.8)):
            return "pink"  # light, or a bright magenta red
        return "claret" if v < 0.5 else "red"
    if h < 70 and s < 0.3 and v > 0.85:
        return "white"  # cream
    if h < 40:
        if v < 0.5:
            return "claret" if h < 25 else "grey"
        return "orange" if h < 32 else "amber"
    if h < 70:
        if s < 0.5:
            return "amber" if v > 0.55 else "grey"
        return "amber" if h < 47 else "yellow"
    if h < 170:
        return "green"
    if h < 255:
        if v < 0.47:
            return "navy"
        if (h < 200 and v > 0.6) or (h < 215 and s < 0.7 and v > 0.85) or (s < 0.55 and v > 0.75):
            return "sky"
        return "blue"
    if h < 290:
        return "purple" if s >= 0.35 else "grey"
    return "pink" if v > 0.75 else "purple"


@dataclass(frozen=True, slots=True)
class GameShirt:
    """One shirt of FM27's database: its style (the game's shirt template) and its three
    colour slots. Slot 2 is the body; slot 1 the template's pattern or trim; slot 3 a second
    pattern colour or trim (research.md, "FM27's kits")."""

    style: int
    colours: tuple[RGB, RGB, RGB]


SHIRT_STYLES: dict[int, tuple[str, tuple[int, ...]]] = {
    # style: (pattern, the colour slots after the body). For "plain" the slot is the collar.
    # Checked against the clubs that wear each one (research.md, "FM27's kits").
    1: ("plain", (1,)),
    10: ("plain", (1,)),  # side panels: the colour on the edges only
    25: ("plain", (1,)),  # thin edging
    35: ("plain", (1,)),  # a V below the neck
    36: ("plain", (1,)),  # shoulder and sleeve seams
    2: ("sleeves", (1,)),
    3: ("stripes", (1,)),
    11: ("stripes", (1,)),  # two wide panels
    37: ("stripes", (1,)),
    48: ("stripes", (1,)),
    53: ("stripes", (1,)),  # two panels apart
    56: ("stripes", (1,)),
    24: ("stripes", (1, 3)),  # two bands side by side in the middle
    41: ("stripes", (1, 3)),
    4: ("hoops", (1,)),
    13: ("hoops", (1,)),
    49: ("hoops", (1,)),
    5: ("two-tone", (1,)),  # halves
    6: ("two-tone", (1,)),  # quarters
    7: ("checks", (1,)),
    8: ("sash", (1,)),
    38: ("sash", (1, 3)),
    9: ("centre-stripe", (1,)),
    12: ("pinstripes", (1,)),
    14: ("thin-hoops", (1,)),
    15: ("band", (1,)),
    21: ("band", (1, 3)),
    30: ("band", (1, 3)),
    29: ("chevron", (1,)),
    45: ("cross", (1,)),
}
"""FM27's shirt styles whose look is known. Any other style is drawn as its body colour and
slot 1 without a layout ("mixed"), as the kit catalog's estimates are."""


def game_home_kit(shirt: GameShirt) -> Kit:
    """The home kit for a shirt of FM27's database. Colours that read as the body's (the same
    prompt word: amber and orange, say) are left out, so a pattern in the body's own shade
    becomes a plain shirt; shorts are unknown."""
    body = colour_name(shirt.colours[1])
    pattern, slots = SHIRT_STYLES.get(shirt.style, ("mixed", (1,)))
    words = {COLOURS[body].words}
    others = []
    for slot in slots:
        name = colour_name(shirt.colours[slot - 1])
        if COLOURS[name].words not in words:
            words.add(COLOURS[name].words)
            others.append(name)
    if pattern == "plain":
        return Kit(shirt=(body,), trim=others[0] if others else None)
    if not others:
        return Kit(shirt=(body,))
    most = PATTERNS[pattern][1]
    return Kit(shirt=(body, *others[: most - 1]), pattern=pattern)


def game_kits(home: GameShirt, keeper: GameShirt | None) -> ClubKits:
    """A club's kits from FM27's database: the home shirt, and the goalkeeper's shirt colour
    (its body) where the database gives one, else a colour unlike the home shirt's."""
    kit = game_home_kit(home)
    gk = colour_name(keeper.colours[1]) if keeper is not None else keeper_colour(kit)
    return ClubKits(home=kit, goalkeeper=Kit(shirt=(gk,), goalkeeper=True))
