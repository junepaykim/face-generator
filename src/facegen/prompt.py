"""Prompt assembly for Animagine XL 4.0 (docs/research.md section 5, Generation recipe). Pure.

The content rule (docs/spec.md section 3; AGENTS.md hard rule 3) is fixed here, in code:
every prompt carries the rating `safe`; every negative prompt carries CONTENT_NEGATIVE; a
wording flagged as under 18 adds YOUNG_NEGATIVE. A player's prompt carries the sportswear
terms (CONTENT_POSITIVE), whichever kit look shoot.py draws. My club's staff are drawn in the wardrobe's club wear instead of the
kit (wardrobe.py): their prompt carries STAFF_POSITIVE (long sleeves), plus `sportswear` for
club sportswear, and their negative adds STAFF_NEGATIVE so the playing kit is not drawn, plus
`sportswear` for smart wear (user, 2026-11-12). Nothing in the settings can remove or weaken
these terms.

Players at my club get three adjustments (spec section 3), applied here so the stored
character sheet is unchanged: banned traits (by default freckles and the muscular build) are
left out of the prompt and added to its negative, tan and dark skin are drawn `soften_skin`
steps lighter, and a seeded share (`round_eyes`) of sharp eyes are drawn round.

A player's picture takes its clothes, pose and backdrop from shoot.py (the kit in one of its
looks, arms and an angle, a plain, gradient or two-tone backdrop). My club's staff keep a pose
from traits.POSE on the grey backdrop, DEFAULT_BACKDROP.

Height and build words (petite, tall female, skinny, muscular female) follow the skin, for
the tenth of players at each end only (traits.py, settings `body`).

Logos, brand marks, lettering and numbers on kits are accepted (decisions.md, 2026-11-10),
so the negative prompt deliberately has no terms for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from facegen.age import WORDINGS
from facegen.kits import Kit
from facegen.shoot import DEFAULT_BACKDROP, Backdrop, KitLook, Pose
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
    eyes_drawn_round,
    skin_words,
)
from facegen.wardrobe import Outfit

if TYPE_CHECKING:
    from facegen.settings import MyClubSettings

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
        pose.words() if isinstance(pose, Pose) else POSE[pose].words,
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


def drawn_traits(sheet: CharacterSheet, my_club: MyClubSettings | None = None) -> dict[str, str]:
    """The sheet fields the prompt actually asks for, after my-club rules. Used by qc.py."""
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
