"""Deterministic role-based outfits for staff and media (spec section 3).

Each outfit uses the sheet's base seed and image version. Role and seniority weight the
fixed looks and props; unknown jobs use the general staff mix. Home-kit colours accent
staff and press-officer outfits; journalists wear neutral smart casual clothes. Media
holding a prop take no arm pose, otherwise a calm one.

This fixed content-rule library contains long-sleeved, layered clothing, never cropped,
sleeveless or open over skin. Headwear would hide hair checked by QC; glasses belong to
the character sheet, so neither is added here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kitshot.core.kit_values import COLOURS, Kit
from kitshot.core.records import is_first_team_manager
from kitshot.core.traits import POSE, Option, pick


@dataclass(frozen=True, slots=True)
class Look:
    words: str
    """Prompt words. `{tone}` is a neutral colour from `tones`; `{club}` the first home shirt
    colour; `{accent}` the first home colour that is not white; `{jacket}` the club track
    jacket in the home kit's colours."""
    tags: frozenset[str]
    """Tagger tags that show the look was drawn; any one of them counts. Ranking only."""
    sporty: bool
    """Club sportswear (True) or smart wear (False); decides the fixed prompt terms."""
    colour_on: str | None = None
    """The garment the club colour is on, for its tag ("necktie" -> "red necktie")."""
    tones: tuple[str, ...] = ()
    lean: str | None = None
    """"senior" or "young": favoured for those people."""


def _look(
    words: str,
    tags: tuple[str, ...],
    *,
    sporty: bool,
    colour_on: str | None = None,
    tones: tuple[str, ...] = (),
    lean: str | None = None,
) -> Look:
    return Look(words, frozenset(tags), sporty, colour_on, tones, lean)


SUIT_TONES = ("black", "grey", "navy blue")

LOOKS: dict[str, Look] = {
    "track jacket": _look("{jacket}", ("track jacket",), sporty=True),
    "half-zip": _look(
        "{club} sweater, high collar, zipper",
        ("high collar", "sweater"),
        sporty=True,
        colour_on="sweater",
        lean="young",
    ),
    "hoodie and vest": _look(
        "{tone} vest, {club} hoodie, hood down",
        ("hoodie", "vest"),
        sporty=True,
        colour_on="hoodie",
        tones=("black", "navy blue", "grey"),
        lean="young",
    ),
    "polo shirt": _look("{club} polo shirt", ("polo shirt",), sporty=True, colour_on="shirt"),
    "polo and jacket": _look(
        "{club} polo shirt, {tone} track jacket, open jacket",
        ("polo shirt", "track jacket"),
        sporty=True,
        colour_on="shirt",
        tones=("black", "navy blue"),
    ),
    "training vest": _look(
        "{tone} vest, {club} shirt",
        ("vest",),
        sporty=True,
        colour_on="shirt",
        tones=("black", "navy blue", "grey"),
        lean="young",
    ),
    "bench coat": _look(
        "{tone} winter coat, long coat, {club} track jacket",
        ("winter coat", "long coat", "coat"),
        sporty=True,
        colour_on="jacket",
        tones=("black", "navy blue", "grey"),
    ),
    "suit and tie": _look(
        "{tone} suit, white collared shirt, {accent} necktie",
        ("suit", "business suit", "formal"),
        sporty=False,
        colour_on="necktie",
        tones=SUIT_TONES,
        lean="senior",
    ),
    "three-piece suit": _look(
        "{tone} suit, waistcoat, white collared shirt, {accent} necktie",
        ("suit", "waistcoat"),
        sporty=False,
        colour_on="necktie",
        tones=SUIT_TONES,
        lean="senior",
    ),
    "open-collar blazer": _look(
        "{tone} blazer, white dress shirt, open collar",
        ("blazer", "dress shirt"),
        sporty=False,
        tones=("navy blue", "grey", "brown"),
    ),
    "turtleneck and blazer": _look(
        "{tone} blazer, black turtleneck sweater",
        ("blazer", "turtleneck"),
        sporty=False,
        tones=("grey", "navy blue", "brown"),
        lean="senior",
    ),
    "overcoat and suit": _look(
        "{tone} overcoat, suit, white collared shirt, {accent} scarf",
        ("overcoat", "coat"),
        sporty=False,
        colour_on="scarf",
        tones=("black", "grey", "brown"),
        lean="senior",
    ),
    "shirt and knit": _look(
        "white collared shirt, {tone} sweater",
        ("sweater",),
        sporty=False,
        tones=("navy blue", "grey", "brown", "green"),
    ),
    "cardigan": _look(
        "{tone} cardigan, white collared shirt, {accent} necktie",
        ("cardigan",),
        sporty=False,
        colour_on="necktie",
        tones=("grey", "brown", "navy blue"),
        lean="senior",
    ),
    "blazer and shirt": _look(
        "{tone} blazer, open jacket, {club} shirt",
        ("blazer",),
        sporty=False,
        colour_on="shirt",
        tones=("black", "navy blue", "grey"),
        lean="young",
    ),
    "trench coat": _look(
        "trench coat, turtleneck sweater, {accent} scarf",
        ("trench coat",),
        sporty=False,
        colour_on="scarf",
        lean="senior",
    ),
    "field jacket": _look(
        "{tone} jacket, high collar, plaid shirt",
        ("plaid shirt", "jacket"),
        sporty=False,
        tones=("green", "brown", "black"),
    ),
    "lab coat and tie": _look(
        "lab coat, white collared shirt, {accent} necktie",
        ("lab coat",),
        sporty=False,
        colour_on="necktie",
    ),
    "lab coat and knit": _look(
        "lab coat, {tone} sweater, collared shirt",
        ("lab coat",),
        sporty=False,
        tones=("navy blue", "grey", "black"),
    ),
    "press trench coat": _look(
        "trench coat, {tone} turtleneck sweater",
        ("trench coat", "turtleneck"),
        sporty=False,
        tones=("black", "grey", "brown"),
        lean="senior",
    ),
    "press cardigan": _look(
        "{tone} cardigan, white collared shirt",
        ("cardigan",),
        sporty=False,
        tones=("grey", "brown", "navy blue", "beige"),
    ),
}

PROPS: dict[str, Option] = {
    "none": Option(""),
    "whistle": Option("whistle around neck", 1, frozenset({"whistle", "whistle around neck"})),
    "lanyard": Option("lanyard, id card", 1, frozenset({"lanyard", "id card"})),
    "stethoscope": Option("stethoscope", 1, frozenset({"stethoscope"})),
    "wristwatch": Option("wristwatch", 1, frozenset({"wristwatch", "watch"})),
    "notebook": Option(
        "holding notebook, holding pen", 1, frozenset({"holding notebook", "notebook"})
    ),
    "microphone": Option("holding microphone", 1, frozenset({"holding microphone", "microphone"})),
    "clipboard": Option("holding clipboard", 1, frozenset({"holding clipboard", "clipboard"})),
}
HELD_PROPS = frozenset({"notebook", "microphone", "clipboard"})
"""Props held in both hands: the picture takes no arm pose."""
MEDIA_POSES = {"crossed": 2, "behind": 2, "hips": 1}
"""Arm poses for journalists and press officers without a prop in hand (keys of traits.POSE)."""


@dataclass(frozen=True, slots=True)
class Role:
    looks: Mapping[str, int]
    """Look name to weight."""
    props: Mapping[str, int]
    """Prop name to weight."""


ROLES: dict[str, Role] = {
    "manager": Role(
        {
            "suit and tie": 3,
            "turtleneck and blazer": 2,
            "overcoat and suit": 2,
            "bench coat": 2,
            "track jacket": 2,
            "half-zip": 1,
            "open-collar blazer": 1,
        },
        {"none": 4, "wristwatch": 1},
    ),
    "coach": Role(
        {
            "track jacket": 3,
            "half-zip": 2,
            "hoodie and vest": 2,
            "polo and jacket": 2,
            "training vest": 1,
            "bench coat": 1,
        },
        {"none": 3, "whistle": 2, "lanyard": 1},
    ),
    "analyst": Role(
        {
            "shirt and knit": 3,
            "blazer and shirt": 2,
            "half-zip": 2,
            "cardigan": 1,
            "open-collar blazer": 1,
            "track jacket": 1,
        },
        {"none": 2, "lanyard": 2},
    ),
    "director": Role(
        {
            "suit and tie": 4,
            "three-piece suit": 2,
            "overcoat and suit": 2,
            "open-collar blazer": 2,
            "turtleneck and blazer": 1,
        },
        {"none": 3, "wristwatch": 1},
    ),
    "scout": Role(
        {
            "trench coat": 2,
            "field jacket": 2,
            "cardigan": 2,
            "shirt and knit": 1,
            "overcoat and suit": 1,
            "half-zip": 1,
        },
        {"none": 3, "lanyard": 1},
    ),
    "doctor": Role(
        {"lab coat and tie": 3, "lab coat and knit": 2, "shirt and knit": 1},
        {"stethoscope": 4, "lanyard": 1, "none": 1},
    ),
    "medical": Role(
        {
            "track jacket": 3,
            "polo shirt": 2,
            "training vest": 2,
            "half-zip": 2,
            "polo and jacket": 1,
        },
        {"none": 2, "lanyard": 2},
    ),
    "journalist": Role(
        {
            "open-collar blazer": 3,
            "shirt and knit": 2,
            "turtleneck and blazer": 2,
            "press cardigan": 2,
            "field jacket": 1,
            "press trench coat": 1,
        },
        {"notebook": 3, "lanyard": 2, "microphone": 1, "none": 2},
    ),
    "press officer": Role(
        {
            "suit and tie": 2,
            "open-collar blazer": 2,
            "blazer and shirt": 2,
            "polo shirt": 1,
            "cardigan": 1,
        },
        {"lanyard": 3, "none": 2, "clipboard": 1},
    ),
    "staff": Role(  # a job the save does not name
        {
            "track jacket": 3,
            "half-zip": 1,
            "polo shirt": 1,
            "shirt and knit": 1,
            "suit and tie": 1,
        },
        {"none": 3, "lanyard": 1},
    ),
}

# First matching word in a lower-case job label wins: specific jobs must precede broad
# titles ("loan manager" is a director, "head of youth development" a coach).
_JOB_WORDS = (
    ("journalist", "journalist"),
    ("press", "press officer"),
    ("doctor", "doctor"),
    ("physio", "medical"),
    ("scien", "medical"),
    ("scout", "scout"),
    ("analys", "analyst"),
    ("recruit", "analyst"),
    ("loan", "director"),
    ("coach", "coach"),
    ("youth", "coach"),
    ("director", "director"),
    ("chair", "director"),
    ("manager", "manager"),
)

SENIOR_TITLE_WORDS = ("head", "chief", "director", "chair")
SENIOR_FROM_AGE = 50
YOUNG_BELOW_AGE = 35
LEAN_FACTOR = 3
"""A look that leans to a person's seniority is this many times as likely."""


def role_for(job: str | None) -> str:
    """The wardrobe role for a staff job in words; "staff" when the save names none."""
    if is_first_team_manager(job):
        return "manager"
    if job:
        for word, role in _JOB_WORDS:
            if word in job:
                return role
    return "staff"


def seniority(job: str | None, age: int | None) -> str | None:
    """ "senior", "young" or None, from the job title and the real age."""
    if job and any(word in job for word in SENIOR_TITLE_WORDS):
        return "senior"
    if age is not None and age >= SENIOR_FROM_AGE:
        return "senior"
    if age is not None and age < YOUNG_BELOW_AGE:
        return "young"
    return None


@dataclass(frozen=True, slots=True)
class Outfit:
    """One picture's clothes for a member of my club's staff."""

    role: str
    look: str
    prop: str
    text: str
    sporty: bool
    rank: tuple[tuple[str, frozenset[str]], ...] = ()
    tone: str | None = None
    """The look's neutral colour (`{tone}`), None for a look without one."""
    home: Kit | None = None
    """The home kit whose colours dress it; the photo prompt words the same clothes."""

    def words(self) -> str:
        """Prompt words for the clothes and the prop."""
        return self.text

    def describe(self) -> str:
        return f"{self.role}, {self.look}: {self.text}"

    def rank_tags(self) -> dict[str, frozenset[str]]:
        """Ranking only: what was asked for, to the tags that show it was drawn."""
        return dict(self.rank)


def track_jacket_words(kit: Kit) -> str:
    """The club track jacket in the home kit's shirt colours (and collar), long-sleeved."""
    return kit.jacket_words()


def colour_tags(colour: str, garment: str) -> frozenset[str]:
    """Tagger tags for a garment in one of the kit colours ("blue" -> "blue necktie")."""
    return frozenset(t.replace(" shirt", f" {garment}") for t in COLOURS[colour].shirt_tags)


def accent_colour(kit: Kit) -> str:
    """The first home colour that is not white, for a necktie or scarf; navy for all-white."""
    for colour in (*kit.shirt, *((kit.trim,) if kit.trim else ())):
        if colour != "white":
            return colour
    return "navy"


def _weighted(
    names: Mapping[str, int], favoured: frozenset[str] = frozenset()
) -> dict[str, Option]:
    return {
        name: Option("", weight * (LEAN_FACTOR if name in favoured else 1))
        for name, weight in names.items()
    }


def outfit_for(
    home: Kit, job: str | None, age: int | None, base_seed: int, image_version: int
) -> Outfit:
    """The clothes for one picture of a member of my club's staff. Same inputs, same outfit.

    `home` is the club's home kit (its colours); `job` the staff job in words, None where
    the save names none; `age` the real age.
    """
    role_name = role_for(job)
    role = ROLES[role_name]
    lean = seniority(job, age)
    favoured = frozenset(n for n in role.looks if lean and LOOKS[n].lean == lean)
    look_name = pick(_weighted(role.looks, favoured), "outfit", base_seed, image_version, "look")
    prop_name = pick(_weighted(role.props), "outfit", base_seed, image_version, "prop")
    tones = {t: Option(t) for t in LOOKS[look_name].tones}
    tone = pick(tones, "outfit", base_seed, image_version, "tone") if tones else None
    return dress(home, role_name, look_name, prop_name, tone)


def media_pose(outfit: Outfit, base_seed: int, image_version: int) -> str:
    """A journalist's or press officer's pose: none with a prop in hand, else a calm one.

    The key of traits.POSE, or "" for no arm pose.
    """
    if outfit.prop in HELD_PROPS:
        return ""
    poses = {name: Option(POSE[name].words, weight) for name, weight in MEDIA_POSES.items()}
    return pick(poses, "media pose", base_seed, image_version)


def dress(
    home: Kit, role: str, look_name: str, prop_name: str = "none", tone: str | None = None
) -> Outfit:
    """One look and prop in the home kit's colours; `tone` defaults to the look's first."""
    look, prop = LOOKS[look_name], PROPS[prop_name]
    if look.tones and tone not in look.tones:
        if tone is not None:
            raise ValueError(f"{look_name!r} has no tone {tone!r}")
        tone = look.tones[0]
    club, accent = home.shirt[0], accent_colour(home)
    jacket = "{jacket}" in look.words
    text = look.words.format(
        tone=tone or "",
        club=COLOURS[club].words,
        accent=COLOURS[accent].words,
        jacket=track_jacket_words(home) if jacket else "",
    )

    rank: list[tuple[str, frozenset[str]]] = [("outfit", look.tags)]
    if jacket:
        for i, colour in enumerate(dict.fromkeys(home.shirt), 1):
            rank.append((f"jacket colour {i}", colour_tags(colour, "jacket")))
    elif look.colour_on:
        colour = accent if "{accent}" in look.words else club
        rank.append(("club colour", colour_tags(colour, look.colour_on)))
    if prop.tags:
        rank.append(("prop", prop.tags))
    return Outfit(
        role=role,
        look=look_name,
        prop=prop_name,
        text=", ".join(p for p in (text, prop.words) if p),
        sporty=look.sporty,
        rank=tuple(rank),
        tone=tone if look.tones else None,
        home=home,
    )
