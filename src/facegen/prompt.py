"""Prompt assembly for Animagine XL 4.0 (docs/research.md section 5, Generation recipe). Pure.

The content rule (docs/spec.md section 3; AGENTS.md hard rule 3) is fixed here, in code:
every prompt carries the rating `safe` and the sportswear terms; every negative prompt
carries CONTENT_NEGATIVE; a wording flagged as under 18 adds YOUNG_NEGATIVE. Nothing in the
settings can remove or weaken these terms.

Players at my club get two adjustments (spec section 3), applied here so the stored
character sheet is unchanged: banned traits (by default freckles and the muscular build) are
left out of the prompt and added to its negative, and tan and dark skin are drawn
`soften_skin` steps lighter.

Height and build words (petite, tall female, skinny, muscular female) follow the skin, for
the tenth of players at each end only (traits.py, settings `body`).

Logos, brand marks, lettering and numbers on kits are accepted (decisions.md, 2026-11-10),
so the negative prompt deliberately has no terms for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from facegen.age import WORDINGS
from facegen.traits import (
    ACCESSORY,
    BANNABLE,
    BUILD,
    EXPRESSION,
    EYE_COLOUR,
    EYE_SHAPE,
    FRINGE,
    HAIR_COLOUR,
    HAIR_STYLE,
    HEIGHT,
    MARK,
    POSE,
    CharacterSheet,
    skin_words,
)

if TYPE_CHECKING:
    from facegen.kits import Kit
    from facegen.settings import MyClubSettings

SUBJECT = "1girl, solo"
RATING = "safe"
FRAMING = "upper body, looking at viewer, standing, simple background, grey background"
CONTENT_POSITIVE = "sportswear, soccer uniform, soccer"
QUALITY = "masterpiece, high score, great score, absurdres"

CONTENT_NEGATIVE = (
    "nsfw, explicit, questionable, sensitive, nude, nipples, underwear, lingerie, swimsuit, "
    "bikini, cleavage, midriff, navel, see-through, suggestive, revealing clothes, large breasts"
)
YOUNG_NEGATIVE = "mature female, adult, mature face, breasts"
COMPOSITION_NEGATIVE = (
    "multiple girls, 2girls, multiple views, text, watermark, signature, username, artist name"
)
QUALITY_NEGATIVE = (
    "lowres, bad anatomy, bad hands, error, missing finger, extra digits, fewer digits, "
    "cropped, worst quality, low quality, low score, bad score, average score, blurry"
)


@dataclass(frozen=True, slots=True)
class Prompt:
    positive: str
    negative: str


def _join(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


def build_prompt(
    sheet: CharacterSheet,
    wording: str,
    kit: Kit,
    pose: str,
    my_club: MyClubSettings | None = None,
) -> Prompt:
    """The prompt for one picture.

    `wording` is the age wording key (W01 to W10, age.py). Pass `my_club` only for a player
    at my club when the picture is drawn; it applies that club's adjustments.
    """
    age = WORDINGS[wording]  # KeyError on an unknown wording: never guess an age
    banned = frozenset(my_club.banned_traits) & BANNABLE.keys() if my_club else frozenset()
    soften = my_club.soften_skin if my_club else 0

    look = _join(
        HAIR_COLOUR[sheet.hair_colour].words,
        HAIR_STYLE[sheet.hair_style].words,
        FRINGE[sheet.fringe].words,
        EYE_COLOUR[sheet.eye_colour].words,
        EYE_SHAPE[sheet.eye_shape].words,
        skin_words(sheet.skin, soften),
        "" if sheet.height in banned else HEIGHT[sheet.height].words,
        "" if sheet.build in banned else BUILD[sheet.build].words,
        "" if sheet.accessory in banned else ACCESSORY[sheet.accessory].words,
        "" if sheet.mark in banned else MARK[sheet.mark].words,
    )
    positive = _join(
        SUBJECT,
        RATING,
        age.text,
        look,
        kit.words(),
        POSE[pose].words,
        EXPRESSION[sheet.expression].words,
        FRAMING,
        CONTENT_POSITIVE,
        QUALITY,
    )
    negative = _join(
        CONTENT_NEGATIVE,
        YOUNG_NEGATIVE if age.young else "",
        _join(*(BANNABLE[t].words for t in sorted(banned))),
        COMPOSITION_NEGATIVE,
        QUALITY_NEGATIVE,
    )
    return Prompt(positive, negative)


def drawn_traits(sheet: CharacterSheet, my_club: MyClubSettings | None = None) -> dict[str, str]:
    """The sheet fields the prompt actually asks for, after my-club bans. Used by qc.py."""
    banned = frozenset(my_club.banned_traits) if my_club else frozenset()
    drawn = {
        "hair_colour": sheet.hair_colour,
        "hair_style": sheet.hair_style,
        "fringe": sheet.fringe,
        "eye_colour": sheet.eye_colour,
        "eye_shape": sheet.eye_shape,
        "expression": sheet.expression,
    }
    for name in ("height", "build"):
        value = getattr(sheet, name)
        if value != "average" and value not in banned:
            drawn[name] = value
    if sheet.accessory != "none" and sheet.accessory not in banned:
        drawn["accessory"] = sheet.accessory
    if sheet.mark != "none" and sheet.mark not in banned:
        drawn["mark"] = sheet.mark
    return drawn
