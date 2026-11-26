"""Deterministic men's sheets: a seeded look with a skin, hair and eyes that suit it (Phase 15f).

Everyone is drawn as a man when `style.gender` is "man" (user, 2026-11-20). A sheet's look comes
from the region of the person's nation (core/nations.py) when the save gives one, else from the
world's mix; skin, hair colour, hair texture and eyes are then drawn from that look's own tables,
so they suit each other. Facial hair, expression, glasses and the age the hair starts to grey
are drawn too. Each field is hashed on its own (career, unique ID, sheet version, field) under
"man sheet", so it is independent of the women's sheet (traits.py), and adding a field leaves
the others unchanged. The base seed is the women's: picture seeds are derived the same way.

What a picture shows also depends on the age it is drawn at (`drawn`): no facial hair under
18, stubble at most until 21, and hair greying from the sheet's `greying_from`. Every option
has photo words for the photo prompt's sentences and tags for Animagine; core/man_prompt.py
builds both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

from kitshot.core.nations import region_of
from kitshot.core.records import Personality
from kitshot.core.traits import (
    DEFAULT_PERSONALITY_WEIGHT,
    Option,
    derive_seed,
    personality_leans,
    pick,
    weighted,
)


@dataclass(frozen=True, slots=True)
class Word:
    """One option's words: for the photo prompt's sentences, and Animagine's tags."""

    photo: str
    anime: str
    weight: int = 1
    tags: frozenset[str] = frozenset()
    """Tagger tags that show it was drawn; ranking only (the men's check needs none)."""


def _w(photo: str, anime: str, weight: int = 1, *tags: str) -> Word:
    return Word(photo, anime, weight, frozenset(tags))


SKIN: dict[str, Word] = {
    "pale": _w("pale", "pale skin"),
    "fair": _w("fair", ""),
    "light": _w("light", ""),
    "olive": _w("olive", ""),
    "light brown": _w("light brown", "tan"),
    "brown": _w("brown", "dark skin, dark-skinned male"),
    "dark brown": _w("dark brown", "dark skin, dark-skinned male"),
    "deep": _w("very dark brown", "dark skin, very dark skin, dark-skinned male"),
}
HAIR_COLOUR: dict[str, Word] = {
    "black": _w("black", "black hair", 1, "black hair"),
    "dark brown": _w("dark brown", "brown hair", 1, "brown hair"),
    "brown": _w("brown", "brown hair", 1, "brown hair", "light brown hair"),
    "light brown": _w("light brown", "light brown hair", 1, "light brown hair", "brown hair"),
    "blond": _w("blond", "blonde hair", 1, "blonde hair"),
    "ginger": _w("ginger", "orange hair", 1, "orange hair", "red hair"),
    "grey": _w("grey", "grey hair", 1, "grey hair", "white hair"),
}
""""grey" is never drawn into a sheet: hair turns grey with age (`drawn`)."""
EYE_COLOUR: dict[str, Word] = {
    "brown": _w("brown", "brown eyes", 1, "brown eyes"),
    "dark brown": _w("dark brown", "brown eyes", 1, "brown eyes", "black eyes"),
    "hazel": _w("hazel", "brown eyes", 1, "brown eyes", "green eyes"),
    "green": _w("green", "green eyes", 1, "green eyes"),
    "blue": _w("blue", "blue eyes", 1, "blue eyes", "aqua eyes"),
    "grey": _w("grey", "grey eyes", 1, "grey eyes"),
}


@dataclass(frozen=True, slots=True)
class Style:
    """A hairstyle. `{c}` in the photo words is the hair colour."""

    photo: str
    anime: str
    coloured: bool = True
    """False for a shaved head: no colour, and no greying, to name."""


HAIR_STYLE: dict[str, Style] = {
    "side parting": Style("short {c} hair with a side parting", "short hair"),
    "crew cut": Style("a short {c} crew cut", "very short hair"),
    "swept back": Style("{c} hair swept back", "short hair, slicked back hair"),
    "textured crop": Style("a short textured {c} crop with a short fringe", "short hair, bangs"),
    "undercut": Style("{c} hair in an undercut, longer on top", "undercut, short hair"),
    "messy": Style("messy short {c} hair", "short hair, messy hair"),
    "medium": Style("medium-length straight {c} hair", "medium hair"),
    "medium wavy": Style("medium-length wavy {c} hair", "medium hair, wavy hair"),
    "buzz cut": Style("a {c} buzz cut", "buzz cut, very short hair"),
    "shaved": Style("a shaved head", "very short hair, buzz cut", coloured=False),
    "curly crop": Style("short curly {c} hair", "short hair, curly hair"),
    "curly top": Style("curly {c} hair on top, short on the sides", "short hair, curly hair"),
    "medium curly": Style("medium-length curly {c} hair", "medium hair, curly hair"),
    "high-top fade": Style("a short {c} high-top fade", "short hair, afro"),
    "close fade": Style("a close-cropped {c} fade", "very short hair"),
    "twists": Style("short {c} twists", "short hair, curly hair"),
    "dreadlocks": Style("short {c} dreadlocks tied back", "dreadlocks"),
    "cornrows": Style("{c} cornrows", "cornrows"),
    "short afro": Style("a short {c} afro", "afro, short hair"),
}
TEXTURES: dict[str, dict[str, int]] = {
    "straight": {
        "side parting": 3, "crew cut": 3, "swept back": 2, "textured crop": 3, "undercut": 2,
        "messy": 2, "medium": 1, "buzz cut": 1, "shaved": 1,
    },
    "wavy": {
        "side parting": 2, "crew cut": 2, "swept back": 3, "textured crop": 2, "undercut": 2,
        "messy": 2, "medium wavy": 2, "buzz cut": 1, "shaved": 1,
    },
    "curly": {
        "curly crop": 4, "curly top": 3, "medium curly": 2, "crew cut": 1, "buzz cut": 1,
        "shaved": 1,
    },
    "afro": {
        "high-top fade": 2, "close fade": 4, "twists": 2, "dreadlocks": 2, "cornrows": 1,
        "short afro": 2, "buzz cut": 1, "shaved": 1,
    },
}  # fmt: skip
"""Hair texture to the hairstyles it takes, with their weights."""

FACIAL_HAIR: dict[str, Word] = {
    "none": _w("no facial hair", "", 5),
    "stubble": _w("light stubble", "stubble", 4, "stubble"),
    "short beard": _w("a short trimmed beard", "beard", 3, "beard", "facial hair"),
    "full beard": _w("a full beard", "beard", 2, "beard", "facial hair"),
    "moustache": _w("a moustache", "mustache", 1, "mustache", "facial hair"),
    "goatee": _w("a goatee", "goatee", 1, "goatee", "facial hair"),
    "moustache and goatee": _w("a moustache and goatee", "mustache, goatee", 1, "goatee"),
}
SHAVEN_BELOW_AGE = 18
"""Under this age a man is drawn with no facial hair, whatever his sheet says."""
STUBBLE_BELOW_AGE = 21
"""Under this age facial hair heavier than stubble is drawn as stubble."""

EXPRESSION: dict[str, Word] = {
    "cheerful": _w("a warm smile", "smile", 1, "smile"),
    "serious": _w("a serious expression", "serious, closed mouth", 1, "serious"),
    "shy": _w("a calm, reserved expression", "light smile, closed mouth", 1, "light smile"),
    "confident": _w("a confident smile", "confident, smirk", 1, "smirk"),
}
"""The women's expression keys (traits.EXPRESSION), so personality leans the same way."""

GLASSES: dict[str, Option] = {"no": Option("", 6), "yes": Option("", 1)}
"""Drawn on staff and journalists only (man_prompt.py)."""
GREYING_FROM: dict[str, Option] = {
    str(age): Option("", 1 if age < 38 else 3 if age < 50 else 2 if age < 57 else 1)
    for age in range(33, 66)
}
"""The age a man's hair starts to grey: from 33 to 65, most often in his forties."""
GREY_AFTER_YEARS = 12
"""Years after `greying_from` the hair is grey all over."""


@dataclass(frozen=True, slots=True)
class Look:
    """A heritage a face is drawn with, and the skin, hair and eyes that go with it."""

    words: str
    """For the photo prompt ("a 24-year-old West African male football player")."""
    skin: Mapping[str, int]
    hair: Mapping[str, int]
    eyes: Mapping[str, int]
    texture: Mapping[str, int]


_DARK_EYES = {"dark brown": 7, "brown": 3}

LOOKS: dict[str, Look] = {
    "northern european": Look(
        "Northern European",
        {"pale": 2, "fair": 5, "light": 3},
        {"blond": 4, "light brown": 4, "brown": 5, "dark brown": 4, "black": 1, "ginger": 2},
        {"blue": 5, "grey": 2, "green": 3, "hazel": 3, "brown": 4},
        {"straight": 6, "wavy": 4, "curly": 1},
    ),
    "southern european": Look(
        "Southern European",
        {"fair": 2, "light": 3, "olive": 5},
        {"black": 4, "dark brown": 6, "brown": 3, "light brown": 1},
        {"brown": 6, "dark brown": 3, "hazel": 3, "green": 2, "blue": 1},
        {"straight": 4, "wavy": 5, "curly": 2},
    ),
    "eastern european": Look(
        "Eastern European",
        {"pale": 2, "fair": 5, "light": 3},
        {"blond": 3, "light brown": 4, "brown": 5, "dark brown": 4, "black": 1},
        {"blue": 4, "grey": 3, "green": 3, "hazel": 3, "brown": 4},
        {"straight": 7, "wavy": 3},
    ),
    "west african": Look(
        "West African",
        {"brown": 2, "dark brown": 5, "deep": 3},
        {"black": 1},
        _DARK_EYES,
        {"afro": 1},
    ),
    "central african": Look(
        "Central African",
        {"brown": 2, "dark brown": 5, "deep": 3},
        {"black": 1},
        _DARK_EYES,
        {"afro": 1},
    ),
    "east african": Look(
        "East African",
        {"brown": 4, "dark brown": 5, "deep": 1},
        {"black": 1},
        _DARK_EYES,
        {"afro": 8, "curly": 2},
    ),
    "southern african": Look(
        "Southern African",
        {"brown": 3, "dark brown": 5, "deep": 2},
        {"black": 1},
        _DARK_EYES,
        {"afro": 1},
    ),
    "north african": Look(
        "North African",
        {"light": 1, "olive": 4, "light brown": 5},
        {"black": 6, "dark brown": 4},
        {"brown": 6, "dark brown": 3, "hazel": 2, "green": 1},
        {"straight": 1, "wavy": 4, "curly": 5},
    ),
    "middle eastern": Look(
        "Middle Eastern",
        {"light": 1, "olive": 5, "light brown": 4},
        {"black": 7, "dark brown": 3},
        {"brown": 6, "dark brown": 4, "hazel": 1},
        {"straight": 3, "wavy": 5, "curly": 2},
    ),
    "caribbean": Look(
        "Afro-Caribbean",
        {"light brown": 1, "brown": 4, "dark brown": 4, "deep": 1},
        {"black": 1},
        _DARK_EYES,
        {"afro": 1},
    ),
    "african american": Look(
        "African-American",
        {"light brown": 1, "brown": 4, "dark brown": 4, "deep": 1},
        {"black": 1},
        _DARK_EYES,
        {"afro": 1},
    ),
    "afro-latin": Look(
        "Afro-Latin American",
        {"light brown": 2, "brown": 5, "dark brown": 3},
        {"black": 9, "dark brown": 1},
        {"dark brown": 6, "brown": 4},
        {"afro": 6, "curly": 4},
    ),
    "latin american": Look(
        "Latin American",
        {"light": 2, "olive": 3, "light brown": 5},
        {"black": 6, "dark brown": 4},
        {"brown": 6, "dark brown": 4, "hazel": 1},
        {"straight": 6, "wavy": 3, "curly": 1},
    ),
    "mixed": Look(
        "mixed-race",
        {"olive": 2, "light brown": 5, "brown": 3},
        {"black": 4, "dark brown": 4, "brown": 2},
        {"brown": 5, "hazel": 3, "green": 1, "dark brown": 2},
        {"curly": 6, "afro": 3, "wavy": 2},
    ),
    "south asian": Look(
        "South Asian",
        {"light brown": 4, "brown": 5, "dark brown": 1},
        {"black": 7, "dark brown": 3},
        {"dark brown": 6, "brown": 4},
        {"straight": 5, "wavy": 5},
    ),
    "east asian": Look(
        "East Asian",
        {"fair": 3, "light": 6},
        {"black": 8, "dark brown": 2},
        _DARK_EYES,
        {"straight": 9, "wavy": 1},
    ),
    "southeast asian": Look(
        "Southeast Asian",
        {"light": 3, "light brown": 5, "brown": 2},
        {"black": 9, "dark brown": 1},
        _DARK_EYES,
        {"straight": 6, "wavy": 4},
    ),
    "central asian": Look(
        "Central Asian",
        {"fair": 2, "light": 5, "olive": 3},
        {"black": 6, "dark brown": 4},
        {"brown": 6, "dark brown": 3, "hazel": 1},
        {"straight": 8, "wavy": 2},
    ),
    "pacific islander": Look(
        "Pacific Islander",
        {"light brown": 3, "brown": 6, "dark brown": 1},
        {"black": 8, "dark brown": 2},
        _DARK_EYES,
        {"wavy": 5, "curly": 5},
    ),
}

WORLD: dict[str, int] = {
    "northern european": 22, "southern european": 14, "eastern european": 10,
    "west african": 9, "central african": 3, "east african": 2, "southern african": 1,
    "north african": 5, "middle eastern": 4, "caribbean": 4, "african american": 2,
    "afro-latin": 4, "latin american": 9, "mixed": 5, "east asian": 3, "southeast asian": 1,
    "south asian": 1, "central asian": 1,
}  # fmt: skip
"""The mix for someone whose nation is not known: roughly the world's professional players."""

REGIONS: dict[str, dict[str, int]] = {
    "england": {
        "northern european": 62, "caribbean": 9, "west african": 9, "mixed": 10,
        "southern european": 3, "south asian": 2, "north african": 2, "east african": 2,
        "central african": 1,
    },
    "celtic": {
        "northern european": 88, "mixed": 4, "west african": 3, "caribbean": 2,
        "southern european": 2, "eastern european": 1,
    },
    "nordic": {
        "northern european": 80, "east african": 5, "west african": 4, "middle eastern": 4,
        "mixed": 4, "eastern european": 2, "southern european": 1,
    },
    "western_europe": {
        "northern european": 58, "southern european": 5, "eastern european": 5,
        "middle eastern": 7, "north african": 7, "west african": 5, "central african": 5,
        "mixed": 8,
    },
    "france": {
        "northern european": 26, "southern european": 14, "west african": 18,
        "central african": 10, "north african": 14, "caribbean": 8, "mixed": 10,
    },
    "southern_europe": {
        "southern european": 82, "northern european": 5, "west african": 4, "north african": 3,
        "latin american": 2, "mixed": 4,
    },
    "portugal": {
        "southern european": 70, "west african": 8, "central african": 6, "mixed": 10,
        "afro-latin": 3, "northern european": 3,
    },
    "eastern_europe": {
        "eastern european": 86, "northern european": 8, "southern european": 2,
        "central asian": 2, "mixed": 2,
    },
    "balkans": {"eastern european": 50, "southern european": 46, "mixed": 4},
    "turkey": {
        "southern european": 45, "middle eastern": 45, "eastern european": 6, "mixed": 4,
    },
    "caucasus": {"southern european": 40, "middle eastern": 35, "eastern european": 25},
    "israel": {
        "southern european": 40, "middle eastern": 35, "eastern european": 15,
        "northern european": 5, "east african": 5,
    },
    "north_africa": {"north african": 86, "west african": 6, "mixed": 5, "middle eastern": 3},
    "middle_east": {
        "middle eastern": 86, "north african": 5, "east african": 5, "south asian": 4,
    },
    "west_africa": {"west african": 94, "mixed": 3, "north african": 3},
    "sahel": {"west african": 40, "north african": 35, "east african": 25},
    "cape_verde": {"mixed": 50, "west african": 45, "southern european": 5},
    "central_africa": {"central african": 94, "west african": 3, "mixed": 3},
    "east_africa": {"east african": 94, "middle eastern": 3, "mixed": 3},
    "southern_africa": {"southern african": 95, "mixed": 5},
    "south_africa": {
        "southern african": 72, "northern european": 14, "mixed": 10, "south asian": 4,
    },
    "indian_ocean": {
        "mixed": 40, "east african": 25, "south asian": 20, "southeast asian": 15,
    },
    "south_asia": {"south asian": 96, "central asian": 2, "mixed": 2},
    "central_asia": {
        "central asian": 74, "eastern european": 14, "east asian": 8, "middle eastern": 4,
    },
    "east_asia": {"east asian": 96, "mixed": 2, "southeast asian": 2},
    "southeast_asia": {
        "southeast asian": 86, "east asian": 8, "south asian": 3, "mixed": 3,
    },
    "united_states": {
        "northern european": 42, "african american": 22, "latin american": 20, "mixed": 8,
        "southern european": 3, "east asian": 3, "west african": 2,
    },
    "canada": {
        "northern european": 55, "caribbean": 9, "west african": 8, "mixed": 10,
        "south asian": 5, "east asian": 5, "latin american": 4, "southern european": 4,
    },
    "central_america": {
        "latin american": 82, "afro-latin": 10, "southern european": 5, "mixed": 3,
    },
    "hispanic_caribbean": {
        "afro-latin": 40, "mixed": 35, "southern european": 20, "latin american": 5,
    },
    "caribbean": {"caribbean": 86, "mixed": 10, "south asian": 2, "southern european": 2},
    "indo_caribbean": {"caribbean": 50, "south asian": 35, "mixed": 15},
    "brazil": {
        "mixed": 34, "southern european": 32, "afro-latin": 30, "east asian": 2,
        "northern european": 2,
    },
    "rio_de_la_plata": {
        "southern european": 72, "latin american": 22, "afro-latin": 3, "mixed": 3,
    },
    "andes": {"latin american": 75, "southern european": 20, "mixed": 5},
    "northern_south_america": {
        "latin american": 62, "afro-latin": 20, "southern european": 12, "mixed": 6,
    },
    "australia": {
        "northern european": 70, "southern european": 8, "mixed": 6, "east african": 4,
        "east asian": 4, "pacific islander": 3, "middle eastern": 3, "south asian": 2,
    },
    "new_zealand": {
        "northern european": 70, "pacific islander": 18, "mixed": 8, "east asian": 4,
    },
    "pacific": {"pacific islander": 88, "mixed": 7, "southeast asian": 5},
    "fiji": {"pacific islander": 60, "south asian": 35, "mixed": 5},
}  # fmt: skip
"""Each region's mix of looks, in percent (core/nations.py names each nation's region).
Proposed 2026-11-20 for the user to adjust: the home look first, with a share for players of
other heritage where a country's football has many."""


def _table(weights: Mapping[str, int]) -> dict[str, Option]:
    return {name: Option("", weight) for name, weight in weights.items()}


def _words(table: Mapping[str, Word]) -> dict[str, Option]:
    return {name: Option(word.photo, word.weight) for name, word in table.items()}


@dataclass(frozen=True, slots=True)
class ManSheet:
    look: str
    """A key of LOOKS."""
    skin: str
    hair_colour: str
    hair_style: str
    facial_hair: str
    eye_colour: str
    expression: str
    greying_from: int
    """The age his hair starts to grey (GREYING_FROM)."""
    glasses: bool
    base_seed: int

    def to_json(self) -> str:
        data = asdict(self)
        del data["base_seed"]  # its own column, as for the women's sheet
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, base_seed: int) -> ManSheet:
        data = json.loads(text)
        known = {f.name for f in fields(cls)} - {"base_seed"}
        return cls(**{k: data[k] for k in known}, base_seed=base_seed)

    def describe(self) -> str:
        return (
            f"{self.look}, {self.skin} skin, {self.hair_colour} {self.hair_style} hair, "
            f"facial hair {self.facial_hair}, {self.eye_colour} eyes, {self.expression}, "
            f"greying from {self.greying_from}{', glasses' if self.glasses else ''}"
        )


def sample_man_sheet(
    career_key: str,
    unique_id: int,
    version: int,
    *,
    nation: int | None = None,
    personality: Personality | None = None,
    personality_weight: int = DEFAULT_PERSONALITY_WEIGHT,
) -> ManSheet:
    """The men's sheet for a person. Same inputs, same sheet.

    `nation` is the game's database ID of the person's nation (core/nations.py); without one,
    or for a nation not listed, the look comes from the world's mix. Personality re-weights
    the expression only, as for the women's sheet.
    """

    def choose(table: Mapping[str, Option], name: str) -> str:
        return pick(table, "man sheet", career_key, unique_id, version, name)

    region = region_of(nation)
    look_name = choose(_table(REGIONS[region] if region else WORLD), "look")
    look = LOOKS[look_name]
    texture = choose(_table(look.texture), "texture")
    favoured = (lean.expression for lean in personality_leans(personality))
    expressions = weighted(_words(EXPRESSION), favoured, personality_weight)
    return ManSheet(
        look=look_name,
        skin=choose(_table(look.skin), "skin"),
        hair_colour=choose(_table(look.hair), "hair colour"),
        hair_style=choose(_table(TEXTURES[texture]), "hair style"),
        facial_hair=choose(_words(FACIAL_HAIR), "facial hair"),
        eye_colour=choose(_table(look.eyes), "eye colour"),
        expression=choose(expressions, "expression"),
        greying_from=int(choose(GREYING_FROM, "greying from")),
        glasses=choose(GLASSES, "glasses") == "yes",
        base_seed=derive_seed("seed", career_key, unique_id, version),
    )


@dataclass(frozen=True, slots=True)
class Drawn:
    """What a picture of a man shows at one age: keys of the tables above."""

    hair_colour: str
    """The sheet's colour, or "grey" once his hair has fully greyed."""
    greying: bool
    """Going grey at the temples (before it is grey all over)."""
    facial_hair: str


def drawn(sheet: ManSheet, age: int) -> Drawn:
    """The sheet as drawn at `age` (real age in years at drawing time)."""
    facial = sheet.facial_hair
    if age < SHAVEN_BELOW_AGE:
        facial = "none"
    elif age < STUBBLE_BELOW_AGE and facial not in ("none", "stubble"):
        facial = "stubble"
    coloured = HAIR_STYLE[sheet.hair_style].coloured
    grey = coloured and age >= sheet.greying_from + GREY_AFTER_YEARS
    greying = coloured and not grey and age >= sheet.greying_from
    return Drawn("grey" if grey else sheet.hair_colour, greying, facial)
