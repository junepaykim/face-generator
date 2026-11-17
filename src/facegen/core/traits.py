"""Deterministic character sheets and fixed trait/pose vocabulary (spec section 4).

Choices are hashed separately by career, player ID, sheet version and field, so adding a
field leaves existing choices unchanged. Hair and eye colour tags are QC pass conditions;
other tags only rank group A candidates.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace

from facegen.core.records import Personality


@dataclass(frozen=True, slots=True)
class Option:
    words: str
    """Prompt words; empty adds nothing to the prompt."""
    weight: int = 1
    tags: frozenset[str] = frozenset()
    """Tagger tags that show the option was drawn; any one of them counts."""


def _o(words: str, weight: int = 1, *tags: str) -> Option:
    return Option(words, weight, frozenset(tags))


# Natural colours are weighted above anime colours; tag aliases match observed tagger output.
HAIR_COLOUR = {
    "black": _o("black hair", 3, "black hair"),
    "brown": _o("brown hair", 3, "brown hair", "light brown hair"),
    "blonde": _o("blonde hair", 2, "blonde hair"),
    "red": _o("red hair", 1, "red hair"),
    "silver": _o("silver hair", 1, "grey hair", "white hair"),
    "pink": _o("pink hair", 1, "pink hair"),
    "blue": _o("blue hair", 1, "blue hair", "light blue hair"),
}
MANAGER_HAIR_COLOUR = {
    name: replace(HAIR_COLOUR[name], weight=weight)
    for name, weight in {
        "black": 35,
        "brown": 30,
        "blonde": 15,
        "red": 7,
        "silver": 11,
        "pink": 1,
        "blue": 1,
    }.items()
}
EYE_COLOUR = {
    "brown": _o("brown eyes", 3, "brown eyes"),
    "blue": _o("blue eyes", 2, "blue eyes", "aqua eyes"),
    "green": _o("green eyes", 2, "green eyes"),
    "amber": _o("amber eyes, yellow eyes", 1, "yellow eyes", "orange eyes"),
    "violet": _o("purple eyes", 1, "purple eyes"),
}
HAIR_STYLE = {
    "long": _o("long hair, straight hair", 1, "long hair"),
    "ponytail": _o("ponytail", 1, "ponytail"),
    "twintails": _o("twintails", 1, "twintails"),
    "bob": _o("bob cut", 1, "bob cut"),
    "short": _o("short hair", 1, "short hair"),
    "braid": _o("braid", 1, "braid", "single braid"),
}
FRINGE = {
    "blunt": _o("blunt bangs", 1, "blunt bangs"),
    "parted": _o("parted bangs", 1, "parted bangs"),
    "swept": _o("swept bangs", 1, "swept bangs"),
}
EYE_SHAPE = {
    "gentle": _o("tareme", 1, "tareme"),
    "sharp": _o("tsurime", 1, "tsurime"),
    "round": _o("", 1),
}
SKIN = {
    "fair": _o("pale skin", 1),
    "light": _o("", 1),
    "tan": _o("tan", 1),
    "dark": _o("dark skin, dark-skinned female", 1),
}
ACCESSORY = {
    "none": _o("", 4),
    "hairband": _o("hairband", 1, "hairband"),
    "ribbon": _o("hair ribbon", 1, "hair ribbon"),
    "clip": _o("hairclip", 1, "hairclip"),
    "glasses": _o("glasses", 1, "glasses"),
}
MARK = {
    "none": _o("", 4),
    "freckles": _o("freckles", 1, "freckles"),
    "mole": _o("mole under eye", 1, "mole under eye"),
}
EXPRESSION = {
    "cheerful": _o("smile, open mouth", 1, "smile"),
    "serious": _o("serious, closed mouth", 1, "serious"),
    "shy": _o("shy, light blush, closed mouth", 1, "blush"),
    "confident": _o("confident, smirk", 1, "smirk"),
}

# Poses are drawn per picture, not stored in the sheet. Neutral media-day poses only: this
# library is part of the content rule (spec section 3) and has no setting.
POSE = {
    "crossed": _o("crossed arms", 1, "crossed arms"),
    "hips": _o("hands on own hips", 1, "hands on own hips"),
    "ball": _o("holding ball, soccer ball", 1, "holding ball"),
    "behind": _o("arms behind back", 1, "arms behind back"),
    "thumb": _o("thumbs up", 1, "thumbs up"),
    "v": _o("v", 1, "v"),
}

# Sampled sheet fields, in prompt order.
FIELDS: dict[str, Mapping[str, Option]] = {
    "hair_colour": HAIR_COLOUR,
    "hair_style": HAIR_STYLE,
    "fringe": FRINGE,
    "eye_colour": EYE_COLOUR,
    "eye_shape": EYE_SHAPE,
    "skin": SKIN,
    "accessory": ACCESSORY,
    "mark": MARK,
    "expression": EXPRESSION,
}


# Skin, lightest to darkest. "slightly tanned skin" sits between light and tan, so a softened
# tan or dark skin keeps some of its colour (spec section 3, players at my club).
SKIN_SCALE = ("pale skin", "", "slightly tanned skin", "tan", "dark skin, dark-skinned female")
_SKIN_STEP = {"fair": 0, "light": 1, "tan": 3, "dark": 4}


def skin_words(skin: str, soften: int = 0) -> str:
    """Prompt words for a skin tone; tan and dark go `soften` steps lighter, never past light."""
    step = _SKIN_STEP[skin]
    if step >= 3:
        step = max(step - soften, 1)
    return SKIN_SCALE[step]


# Height and build are stored from player facts. Defaults describe only the outer tenths
# of men's players; Strength substitutes for weight, which fmsave does not expose.

HEIGHT = {
    "short": _o("petite", 1, "petite"),
    "average": _o("", 1),
    "tall": _o("tall female", 1, "tall female"),
}
BUILD = {
    "slim": _o("skinny", 1, "skinny"),
    "average": _o("", 1),
    "muscular": _o("muscular female", 1, "muscular female", "muscular"),
}


@dataclass(frozen=True, slots=True)
class BodyThresholds:
    """Settings `body`: where the height and build classes start."""

    short_below_cm: int = 174
    tall_from_cm: int = 191
    slim_strength_at_most: int = 4
    muscular_strength_from: int = 14


DEFAULT_BODY = BodyThresholds()


def height_class(height_cm: int | None, t: BodyThresholds = DEFAULT_BODY) -> str:
    if height_cm is None:
        return "average"
    if height_cm < t.short_below_cm:
        return "short"
    return "tall" if height_cm >= t.tall_from_cm else "average"


def build_class(strength: int | None, t: BodyThresholds = DEFAULT_BODY) -> str:
    """From Strength on the 1 to 20 scale."""
    if strength is None:
        return "average"
    if strength <= t.slim_strength_at_most:
        return "slim"
    return "muscular" if strength >= t.muscular_strength_from else "average"


# Personality weights expression only. Cut-offs approximate the outer tenths of men's
# players; attributes with no matching expression intentionally have no lean.


@dataclass(frozen=True, slots=True)
class Lean:
    attribute: str
    """A field of Personality."""
    high: bool
    """True: notable at `cut` or above; False: at `cut` or below."""
    cut: int
    expression: str
    """The expression this end of the attribute makes more likely."""

    def applies(self, personality: Personality) -> bool:
        value = getattr(personality, self.attribute)
        return value >= self.cut if self.high else value <= self.cut

    def describe(self) -> str:
        return f"{'high' if self.high else 'low'} {self.attribute}"


LEANS = (
    Lean("pressure", True, 14, "confident"),
    Lean("pressure", False, 8, "shy"),
    Lean("ambition", True, 15, "confident"),
    Lean("controversy", True, 10, "confident"),
    Lean("professionalism", True, 16, "serious"),
    Lean("professionalism", False, 9, "cheerful"),
    Lean("temperament", True, 19, "serious"),
    Lean("adaptability", True, 19, "cheerful"),
    Lean("adaptability", False, 7, "shy"),
    Lean("sportsmanship", True, 14, "cheerful"),
)

DEFAULT_PERSONALITY_WEIGHT = 2
"""Settings `personality.weight`."""


def personality_leans(personality: Personality | None) -> tuple[Lean, ...]:
    """The leans a player's personality reaches, in LEANS order; none without one."""
    if personality is None:
        return ()
    return tuple(lean for lean in LEANS if lean.applies(personality))


def weighted(
    table: Mapping[str, Option], favoured: Iterable[str], weight: int
) -> Mapping[str, Option]:
    """`table` with each option's weight times (1 + weight x the times it is favoured).

    With nothing favoured, or a weight of 0, the table itself comes back, so the draw is
    exactly what it would be without a personality.
    """
    hits = Counter(favoured)
    if not hits or weight == 0:
        return table
    return {
        name: replace(option, weight=option.weight * (1 + weight * hits[name]))
        for name, option in table.items()
    }


# `my_club.round_eyes` adjusts prompts only; the stored eye shape remains unchanged.


def eyes_drawn_round(eye_shape: str, base_seed: int, share: float) -> bool:
    """Whether a sheet's sharp eyes are drawn round at my club.

    Seeded by the sheet's base seed alone, so all of a player's pictures agree. Raising
    `share` only adds players: whoever is drawn round at 0.2 is also drawn round at 0.3.
    """
    if eye_shape != "sharp" or share <= 0:
        return False
    return int.from_bytes(_digest("round eyes", base_seed)[:8], "big") < share * 2**64


TRAIT_TABLES: dict[str, Mapping[str, Option]] = {**FIELDS, "height": HEIGHT, "build": BUILD}

# Traits that `my_club.banned_traits` may name: accessories, marks, and the height and
# build classes that add words (e.g. "muscular", banned at my club by default).
BANNABLE: dict[str, Option] = {
    name: option
    for table in (ACCESSORY, MARK, HEIGHT, BUILD)
    for name, option in table.items()
    if option.words
}


def _digest(*parts: object) -> bytes:
    text = "\x1f".join(str(p) for p in ("facegen", *parts))
    return hashlib.sha256(text.encode()).digest()


def derive_seed(*parts: object) -> int:
    """A 32-bit seed from the given parts. Same parts, same seed, on any machine."""
    return int.from_bytes(_digest(*parts)[:4], "big")


def pick(options: Mapping[str, Option], *parts: object) -> str:
    """A weighted choice seeded by `parts`."""
    total = sum(o.weight for o in options.values())
    point = int.from_bytes(_digest(*parts)[:8], "big") % total
    for name, option in options.items():
        point -= option.weight
        if point < 0:
            return name
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class CharacterSheet:
    hair_colour: str
    hair_style: str
    fringe: str
    eye_colour: str
    eye_shape: str
    skin: str
    accessory: str
    mark: str
    expression: str
    height: str
    build: str
    base_seed: int

    def to_json(self) -> str:
        data = asdict(self)
        del data["base_seed"]  # its own column
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, base_seed: int) -> CharacterSheet:
        data = json.loads(text)
        known = {f.name for f in fields(cls)} - {"base_seed"}
        return cls(**{k: data[k] for k in known}, base_seed=base_seed)

    def describe(self) -> str:
        return (
            f"{self.hair_colour} {self.hair_style} hair, {self.fringe} fringe, "
            f"{self.eye_colour} {self.eye_shape} eyes, {self.skin} skin, "
            f"accessory {self.accessory}, mark {self.mark}, {self.expression}, "
            f"{self.height}, {self.build}"
        )


def sample_sheet(
    career_key: str,
    unique_id: int,
    version: int,
    *,
    height_cm: int | None = None,
    strength: int | None = None,
    body: BodyThresholds = DEFAULT_BODY,
    personality: Personality | None = None,
    personality_weight: int = DEFAULT_PERSONALITY_WEIGHT,
    manager: bool = False,
) -> CharacterSheet:
    """The character sheet for a player. Same inputs, same sheet.

    Personality only re-weights the expression; each field still uses its own hash, so the
    other fields are the same with or without one.
    """
    favoured = (lean.expression for lean in personality_leans(personality))
    tables = {**FIELDS, "expression": weighted(EXPRESSION, favoured, personality_weight)}
    if manager:
        tables["hair_colour"] = MANAGER_HAIR_COLOUR
    chosen = {
        name: pick(table, "sheet", career_key, unique_id, version, name)
        for name, table in tables.items()
    }
    return CharacterSheet(
        **chosen,
        height=height_class(height_cm, body),
        build=build_class(strength, body),
        base_seed=derive_seed("seed", career_key, unique_id, version),
    )


def pick_pose(base_seed: int, image_version: int) -> str:
    """The pose for one picture of a player; a new picture may get a new pose."""
    return pick(POSE, "pose", base_seed, image_version)


# Seeds within one picture version: attempt * _ATTEMPT_STEP + candidate.
_VERSION_STEP = 100
_ATTEMPT_STEP = 10


def picture_seed(base_seed: int, image_version: int, attempt: int, candidate: int) -> int:
    """The seed for one render: the sheet's base seed plus a version offset (architecture §6).

    `attempt` counts quality-check retries from 0; `candidate` counts group A's candidates
    from 0. Each (version, attempt, candidate) gets its own seed.
    """
    if not 0 <= candidate < _ATTEMPT_STEP or not 0 <= attempt < _VERSION_STEP // _ATTEMPT_STEP:
        raise ValueError(f"attempt {attempt} or candidate {candidate} out of range")
    offset = (image_version - 1) * _VERSION_STEP + attempt * _ATTEMPT_STEP + candidate
    return (base_seed + offset) % 2**32


def personality_to_json(personality: Personality | None) -> str | None:
    """The players table's `personality` column."""
    return None if personality is None else json.dumps(asdict(personality), sort_keys=True)


def personality_from_json(text: str | None) -> Personality | None:
    if not text:
        return None
    data = json.loads(text)
    return Personality(**{f.name: int(data[f.name]) for f in fields(Personality)})
