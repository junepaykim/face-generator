"""Deterministic kit looks, poses and backdrops for player portraits (spec section 3).

Each shoot uses the sheet's base seed and image version; favourite style uses the base
seed alone. Staff use the wardrobe and traits.POSE on the grey backdrop instead.
Seasonal weights use the in-game date. Winter accessories are excluded so portraits fit
any season; armbands favour older players because captaincy is unavailable in the save.

Poses that obscure the face or lower the head are rare to protect framing and QC. Backdrops
avoid hair and clothing colour families so the figure scan can separate the subject, and
favour unworn home-kit colours. Kit colour and look tags rank candidates only.

This fixed vocabulary enforces fully clothed, non-sexual sports portraits: full football
tops, never cropped, sleeveless or open over skin. Open jackets always cover a kit shirt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from facegen.core.kit_values import Kit, families
from facegen.core.traits import Option, pick


def _o(words: str, weight: int = 1, *tags: str) -> Option:
    return Option(words, weight, frozenset(tags))


@dataclass(frozen=True, slots=True)
class Look:
    words: str
    """Prompt words. `{shirt}` is the kit's shirt, `{jacket}` the track jacket in the kit's
    colours, `{sleeves}` the kit's usual sleeves (short; long for a goalkeeper)."""
    tags: frozenset[str]
    """Tagger tags that show the look was drawn; any one counts. Ranking only."""
    jacket: bool
    """A training or warm-up top: hands may go in its pockets; no extra, no shorts."""
    outfield: int
    """Weight for an outfield player."""
    keeper: int
    """Weight for a goalkeeper."""


def _look(words: str, tag: str, *, jacket: bool = False, outfield: int, keeper: int) -> Look:
    return Look(words, frozenset({tag}), jacket, outfield, keeper)


LOOKS: dict[str, Look] = {
    "short sleeves": _look("{shirt}, short sleeves", "short sleeves", outfield=6, keeper=2),
    "long sleeves": _look("{shirt}, long sleeves", "long sleeves", outfield=2, keeper=6),
    "layered sleeves": _look("{shirt}, layered sleeves", "layered sleeves", outfield=1, keeper=1),
    "collared shirt": _look(
        "{shirt}, collared shirt, {sleeves}", "collared shirt", outfield=1, keeper=1
    ),
    "track jacket": _look(
        "{jacket}, long sleeves", "track jacket", jacket=True, outfield=1, keeper=1
    ),
    "open jacket": _look(
        "{jacket}, open jacket, {shirt}", "open jacket", jacket=True, outfield=1, keeper=1
    ),
    "high collar": _look(
        "{jacket}, high collar, zipper", "high collar", jacket=True, outfield=1, keeper=1
    ),
}

EXTRAS: dict[str, Option] = {
    "none": _o("", 20),
    "wristband": _o("wristband", 2, "wristband"),
    "armband": _o("armband", 1, "armband"),
}
SENIOR_FROM_AGE = 30
SENIOR_ARMBAND = 3
"""The armband's weight for a player of SENIOR_FROM_AGE or over (1 for younger players)."""


@dataclass(frozen=True, slots=True)
class KitLook:
    """What a player wears in one picture: the kit, worn in one look, and an extra."""

    kit: Kit
    look: str
    extra: str = "none"

    def words(self) -> str:
        """Prompt words for the clothes."""
        kit, look = self.kit, LOOKS[self.look]
        body = look.words.format(
            shirt=kit.shirt_words(),
            jacket=kit.jacket_words(),
            sleeves="long sleeves" if kit.goalkeeper else "short sleeves",
        )
        parts = ["goalkeeper", body, kit.gloves_words()] if kit.goalkeeper else [body]
        if not kit.goalkeeper and not look.jacket:
            parts.append(kit.shorts_words())
        parts.append(EXTRAS[self.extra].words)
        return ", ".join(p for p in parts if p)

    def rank_tags(self) -> dict[str, frozenset[str]]:
        """Ranking only: the kit's colours on the garment that shows them, the look and the
        extra, each to the tags that show it was drawn."""
        look = LOOKS[self.look]
        garment = "jacket" if look.jacket and self.look != "open jacket" else "shirt"
        rank = {
            f"{garment} colour {i}": g for i, g in enumerate(self.kit.shirt_tag_groups(garment), 1)
        }
        rank["look"] = look.tags
        if EXTRAS[self.extra].tags:
            rank["extra"] = EXTRAS[self.extra].tags
        return rank

    def describe(self) -> str:
        extra = "" if self.extra == "none" else f", {self.extra}"
        return f"{self.kit.describe()}; {self.look}{extra}"


ARMS: dict[str, Option] = {
    "crossed": _o("crossed arms", 3, "crossed arms"),
    "hips": _o("hands on own hips", 3, "hands on own hips"),
    "behind": _o("arms behind back", 3, "arms behind back"),
    "ball": _o("holding ball, soccer ball", 2, "holding ball"),
    "thumb": _o("thumbs up", 1, "thumbs up"),
    "v": _o("v", 1, "v"),
    "pockets": _o("hands in pockets", 3, "hands in pockets"),
    "gloves": _o("adjusting gloves", 3, "adjusting gloves"),
}
JACKET_ARMS = frozenset({"pockets"})
"""Only with a training or warm-up top."""
KEEPER_ARMS = frozenset({"gloves"})
"""Only for a goalkeeper."""

ANGLES: dict[str, Option] = {
    "front": _o("", 8),
    "side": _o("from side", 3, "from side"),  # the framing asks for "looking at viewer"
    "tilt": _o("head tilt", 3, "head tilt"),
    "contrapposto": _o("contrapposto", 3, "contrapposto"),
    "below": _o("from below", 1, "from below"),
}


@dataclass(frozen=True, slots=True)
class Pose:
    arms: str
    angle: str = "front"

    def words(self) -> str:
        return ", ".join(p for p in (ARMS[self.arms].words, ANGLES[self.angle].words) if p)

    def describe(self) -> str:
        return f"{self.arms}, {self.angle}"


@dataclass(frozen=True, slots=True)
class BackdropColour:
    words: str
    weight: int
    families: frozenset[str]
    """Colour families (kits.COLOURS) this backdrop would blend with."""


def _bg(name: str, weight: int = 1, family: str | None = None) -> BackdropColour:
    return BackdropColour(f"{name} background", weight, frozenset({family or name}))


BACKDROP_COLOURS: dict[str, BackdropColour] = {
    "grey": _bg("grey", 4),
    "blue": _bg("blue"),
    "red": _bg("red"),
    "yellow": _bg("yellow"),
    "green": _bg("green"),
    "purple": _bg("purple"),
    "orange": _bg("orange"),
    "pink": _bg("pink"),
}
BACKDROP_KINDS: dict[str, Option] = {
    "plain": _o("", 6),
    "gradient": _o("gradient background", 3, "gradient background"),
    "two-tone": _o("two-tone background", 1, "two-tone background"),
}
CLUB_FACTOR = 3
"""A backdrop colour of the club's home kit, not worn in the picture, is this much likelier."""

HAIR_FAMILIES: dict[str, frozenset[str]] = {
    "black": frozenset({"black"}),
    "brown": frozenset({"brown"}),
    "blonde": frozenset({"yellow"}),
    "red": frozenset({"red", "orange"}),
    "silver": frozenset({"grey", "white"}),
    "pink": frozenset({"pink"}),
    "blue": frozenset({"blue"}),
}
"""Hair colour (traits.HAIR_COLOUR) to the backdrop families it would blend with."""


@dataclass(frozen=True, slots=True)
class Backdrop:
    kind: str
    """plain, gradient or two-tone."""
    colours: tuple[str, ...]
    """One colour of BACKDROP_COLOURS; two for two-tone."""

    def words(self) -> str:
        parts = ["simple background", BACKDROP_KINDS[self.kind].words]
        parts += [BACKDROP_COLOURS[c].words for c in self.colours]
        return ", ".join(p for p in parts if p)

    def describe(self) -> str:
        colours = " and ".join(self.colours)
        return colours if self.kind == "plain" else f"{self.kind} {colours}"


DEFAULT_BACKDROP = Backdrop("plain", ("grey",))
"""The fixed grey studio backdrop for staff, also used as the player fallback."""


def pick_backdrop(
    base_seed: int,
    image_version: int,
    avoid: frozenset[str],
    favour: frozenset[str] = frozenset(),
) -> Backdrop:
    """A backdrop with no colour in the `avoid` families; colours in `favour` are likelier."""
    allowed = {
        name: Option("", c.weight * (CLUB_FACTOR if c.families & favour else 1))
        for name, c in BACKDROP_COLOURS.items()
        if not c.families & avoid
    }
    if not allowed:  # cannot happen with this palette: a kit and hair cover at most five
        return DEFAULT_BACKDROP
    kinds = BACKDROP_KINDS if len(allowed) > 1 else {"plain": BACKDROP_KINDS["plain"]}
    kind = pick(kinds, "shoot", base_seed, image_version, "backdrop")
    first = pick(allowed, "shoot", base_seed, image_version, "backdrop colour")
    if kind != "two-tone":
        return Backdrop(kind, (first,))
    rest = {name: o for name, o in allowed.items() if name != first}
    second = pick(rest, "shoot", base_seed, image_version, "backdrop colour 2")
    return Backdrop(kind, (first, second))


SEASON_LOOKS: dict[str, frozenset[str]] = {
    "winter": frozenset({"long sleeves", "layered sleeves", "track jacket"}),
    "summer": frozenset({"short sleeves"}),
}
SEASON_FACTOR = 2


def season_of(game_date: date | None) -> str:
    """winter (November to February), spring, summer (June to August), autumn, or unknown.

    By the in-game date, northern hemisphere: the save names no country for a club.
    """
    if game_date is None:
        return "unknown"
    month = game_date.month
    if month in (11, 12, 1, 2):
        return "winter"
    if month in (6, 7, 8):
        return "summer"
    return "spring" if month < 6 else "autumn"


@dataclass(frozen=True, slots=True)
class Style:
    looks: frozenset[str]
    weight: int


STYLES: dict[str, Style] = {
    "none": Style(frozenset(), 4),
    "short sleeves": Style(frozenset({"short sleeves"}), 1),
    "long sleeves": Style(frozenset({"long sleeves"}), 2),
    "layered": Style(frozenset({"layered sleeves"}), 1),
    "collared": Style(frozenset({"collared shirt"}), 1),
    "jacket": Style(frozenset({"track jacket", "open jacket"}), 2),
}
FAVOURITE_FACTOR = 3
"""A look of the player's favourite style is this many times as likely."""


def favourite_style(base_seed: int) -> str:
    """A player's favourite style: the same in every picture of one character sheet."""
    return pick({n: Option("", s.weight) for n, s in STYLES.items()}, "style", base_seed)


def look_weights(goalkeeper: bool, style: str, season: str) -> dict[str, Option]:
    favoured = STYLES[style].looks
    seasonal = SEASON_LOOKS.get(season, frozenset())
    weights: dict[str, Option] = {}
    for name, look in LOOKS.items():
        weight = look.keeper if goalkeeper else look.outfield
        weight *= FAVOURITE_FACTOR if name in favoured else 1
        weight *= SEASON_FACTOR if name in seasonal else 1
        if weight:
            weights[name] = Option("", weight)
    return weights


def extra_weights(goalkeeper: bool, look: str, age: int | None) -> dict[str, Option]:
    if LOOKS[look].jacket:
        return {"none": EXTRAS["none"]}
    senior = age is not None and age >= SENIOR_FROM_AGE
    weights = dict(EXTRAS)
    if senior:
        weights["armband"] = replace(EXTRAS["armband"], weight=SENIOR_ARMBAND)
    if goalkeeper:
        del weights["wristband"]
    return weights


def arm_weights(goalkeeper: bool, look: str) -> dict[str, Option]:
    return {
        name: option
        for name, option in ARMS.items()
        if (name not in JACKET_ARMS or LOOKS[look].jacket)
        and (name not in KEEPER_ARMS or goalkeeper)
    }


def clothes_families(kit: Kit) -> frozenset[str]:
    """The colour families a picture of the kit shows near the head: shirt and collar."""
    return families((*kit.shirt, *((kit.trim,) if kit.trim else ())))


@dataclass(frozen=True, slots=True)
class Shoot:
    """Everything one picture of a player shows besides the character sheet."""

    wear: KitLook
    pose: Pose
    backdrop: Backdrop
    style: str
    """The player's favourite style."""
    season: str
    """The in-game season the picture was drawn in."""

    def context(self) -> dict[str, Any]:
        """What the picture's record keeps about the shoot."""
        return {
            "look": self.wear.look,
            "extra": self.wear.extra,
            "arms": self.pose.arms,
            "angle": self.pose.angle,
            "backdrop": self.backdrop.describe(),
            "style": self.style,
            "season": self.season,
        }


def shoot_for(
    kit: Kit,
    hair_colour: str,
    base_seed: int,
    image_version: int,
    *,
    home: Kit | None = None,
    age: int | None = None,
    game_date: date | None = None,
) -> Shoot:
    """The shoot for one picture of a player. Same inputs, same shoot.

    `kit` is what the player wears (the goalkeeper kit for a goalkeeper); `home` the club's
    home kit, whose colours the backdrop favours where the picture does not wear them;
    `hair_colour` the sheet's; `age` the real age; `game_date` the in-game date.
    """
    style = favourite_style(base_seed)
    season = season_of(game_date)
    parts = ("shoot", base_seed, image_version)
    look = pick(look_weights(kit.goalkeeper, style, season), *parts, "look")
    extra = pick(extra_weights(kit.goalkeeper, look, age), *parts, "extra")
    arms = pick(arm_weights(kit.goalkeeper, look), *parts, "arms")
    angle = pick(ANGLES, *parts, "angle")
    avoid = clothes_families(kit) | HAIR_FAMILIES.get(hair_colour, frozenset())
    favour = families((home or kit).colours()) - avoid
    backdrop = pick_backdrop(base_seed, image_version, avoid, favour)
    return Shoot(KitLook(kit, look, extra), Pose(arms, angle), backdrop, style, season)
