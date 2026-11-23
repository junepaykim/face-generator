"""Women's prompts, in Animagine XL tags, with fixed content safeguards (spec section 3).

Every prompt includes the safe rating; negatives include CONTENT_NEGATIVE and, for under-18
wordings, YOUNG_NEGATIVE. Players receive sportswear terms. Staff and media receive long
sleeves and negatives excluding playing kit; sportswear outfits add sportswear to the
positive prompt, smart outfits to the negative. Settings cannot weaken these terms.
Every model draws women from these prompts for now; men's prompts are in man_prompt.py, and
core/style.py picks one by `style.gender`.

My-club adjustments omit banned traits and negate them, soften tan/dark skin, and draw a
seeded share of sharp eyes round without changing the stored sheet. Players use shoots;
staff use wardrobe outfits, traits.POSE and the grey backdrop. Height/build words come
from the sheet's derived classes. Kits may carry logos, brands, lettering and numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from facegen.core.age import WORDINGS
from facegen.core.kit_values import Kit
from facegen.core.shoot import DEFAULT_BACKDROP, Backdrop, KitLook, Pose
from facegen.core.traits import (
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
    eyes_drawn_round,
    skin_words,
)
from facegen.core.wardrobe import Outfit

if TYPE_CHECKING:
    from facegen.core.settings_values import MyClubSettings

SUBJECT = "1girl, solo"
RATING = "safe"
FRAMING = "upper body, looking at viewer, standing"
"""Followed by the backdrop's words (shoot.Backdrop; "simple background, grey background")."""
CONTENT_POSITIVE = "sportswear, soccer uniform, soccer"
STAFF_POSITIVE = "long sleeves"
STAFF_SPORTSWEAR = "sportswear"
QUALITY = "masterpiece, high score, great score, absurdres"

CONTENT_NEGATIVE = (
    "nsfw, explicit, questionable, sensitive, nude, nipples, underwear, lingerie, swimsuit, "
    "bikini, cleavage, midriff, navel, see-through, suggestive, revealing clothes, large breasts"
)
YOUNG_NEGATIVE = "mature female, adult, mature face, breasts"
STAFF_NEGATIVE = "soccer uniform, jersey, shorts"
STAFF_SMART_NEGATIVE = "sportswear"
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


def drawn_eye_shape(sheet: CharacterSheet, my_club: MyClubSettings | None = None) -> str:
    """The sheet's eye shape, or "round" where my club's `round_eyes` share picks it."""
    if my_club and eyes_drawn_round(sheet.eye_shape, sheet.base_seed, my_club.round_eyes):
        return "round"
    return sheet.eye_shape


type Wear = Kit | KitLook | Outfit
"""A player's kit (as a plain match kit, or in a shoot's look), or a staff outfit."""


def content_terms(wear: Wear) -> tuple[str, str]:
    """The fixed clothing terms for a kit or a staff outfit: (positive, negative)."""
    if isinstance(wear, Kit | KitLook):
        return CONTENT_POSITIVE, ""
    if wear.sporty:
        return _join(STAFF_POSITIVE, STAFF_SPORTSWEAR), STAFF_NEGATIVE
    return STAFF_POSITIVE, _join(STAFF_NEGATIVE, STAFF_SMART_NEGATIVE)


def build_prompt(
    sheet: CharacterSheet,
    wording: str,
    wear: Wear,
    pose: str | Pose,
    my_club: MyClubSettings | None = None,
    *,
    backdrop: Backdrop = DEFAULT_BACKDROP,
) -> Prompt:
    """The prompt for one picture.

    `wording` is the age wording key (W01 to W10, age.py). `wear` is the player's kit, or a
    staff outfit (wardrobe.py). `pose` is a key of traits.POSE or a shoot's arms and angle.
    Pass `my_club` only for someone at my club when the picture is drawn; it applies that
    club's adjustments.
    """
    age = WORDINGS[wording]  # KeyError on an unknown wording: never guess an age
    banned = frozenset(my_club.banned_traits) & BANNABLE.keys() if my_club else frozenset()
    soften = my_club.soften_skin if my_club else 0

    look = _join(
        HAIR_COLOUR[sheet.hair_colour].words,
        HAIR_STYLE[sheet.hair_style].words,
        FRINGE[sheet.fringe].words,
        EYE_COLOUR[sheet.eye_colour].words,
        EYE_SHAPE[drawn_eye_shape(sheet, my_club)].words,
        skin_words(sheet.skin, soften),
        "" if sheet.height in banned else HEIGHT[sheet.height].words,
        "" if sheet.build in banned else BUILD[sheet.build].words,
        "" if sheet.accessory in banned else ACCESSORY[sheet.accessory].words,
        "" if sheet.mark in banned else MARK[sheet.mark].words,
    )
    content, wear_negative = content_terms(wear)
    positive = _join(
        SUBJECT,
        RATING,
        age.text,
        look,
        wear.words(),
        pose_words(pose),
        EXPRESSION[sheet.expression].words,
        FRAMING,
        backdrop.words(),
        content,
        QUALITY,
    )
    negative = _join(
        CONTENT_NEGATIVE,
        YOUNG_NEGATIVE if age.young else "",
        _join(*(BANNABLE[t].words for t in sorted(banned))),
        wear_negative,
        COMPOSITION_NEGATIVE,
        QUALITY_NEGATIVE,
    )
    return Prompt(positive, negative)


def pose_words(pose: str | Pose) -> str:
    """A shoot's arms and angle, a key of traits.POSE, or "" for no pose (a held prop)."""
    if isinstance(pose, Pose):
        return pose.words()
    return POSE[pose].words if pose else ""


def drawn_traits(sheet: CharacterSheet, my_club: MyClubSettings | None = None) -> dict[str, str]:
    """The sheet fields the prompt actually asks for, after my-club rules, by TRAIT_TABLES
    name. Used by qc.py."""
    banned = frozenset(my_club.banned_traits) if my_club else frozenset()
    drawn = {
        "hair_colour": sheet.hair_colour,
        "hair_style": sheet.hair_style,
        "fringe": sheet.fringe,
        "eye_colour": sheet.eye_colour,
        "eye_shape": drawn_eye_shape(sheet, my_club),
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
