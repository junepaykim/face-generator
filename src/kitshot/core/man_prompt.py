"""Prompts for a man: photo prompts in sentences (RealVisXL, Z-Image-Turbo) and Animagine tags
(Phase 15f; user, 2026-11-20).

The photo prompt follows run `realistic`'s measured template (scripts/spikes/realistic_test.py,
docs/research.md section 20): a studio headshot, head and shoulders, his age in years, his
look, then his clothes "fully clothed", the backdrop, and "He is the only person in the
picture". Z-Image-Turbo draws without a negative prompt, so the positive states the clothing;
RealVisXL takes PHOTO_NEGATIVE, the content negative the user approved for photos of men.

The anime prompt is the women's (prompt.py) with a man's subject and age words: the same
rating, framing, quality and fixed clothing terms, and a negative of every term of the women's
content negative plus MAN_CONTENT_NEGATIVE, with BOY_NEGATIVE under 18 (user, 2026-11-20).

Players wear their kit in the shoot's look; staff and media the wardrobe's work clothes
(wardrobe.py), worded here as sentences for photos. My club's banned-trait and round-eye
rules are the women's and do not apply to men (user, 2026-11-20).
"""

from __future__ import annotations

from kitshot.core.kit_values import COLOURS, Kit, join_and
from kitshot.core.man_traits import (
    EXPRESSION,
    EYE_COLOUR,
    FACIAL_HAIR,
    HAIR_COLOUR,
    HAIR_STYLE,
    LOOKS,
    SKIN,
    ManSheet,
    drawn,
)
from kitshot.core.prompt import (
    CONTENT_NEGATIVE,
    FRAMING,
    QUALITY,
    QUALITY_NEGATIVE,
    RATING,
    Prompt,
    Wear,
    content_terms,
    pose_words,
)
from kitshot.core.shoot import DEFAULT_BACKDROP, Backdrop, KitLook, Pose
from kitshot.core.wardrobe import HELD_PROPS, Outfit, accent_colour

# --- Animagine ---------------------------------------------------------------------------------

MAN_SUBJECT = "1boy, solo, male focus"
MAN_CONTENT_NEGATIVE = "shirtless, topless male, bare chest, pectorals, abs, bara"
"""Added to every term of prompt.CONTENT_NEGATIVE for a man (user, 2026-11-20)."""
BOY_NEGATIVE = "adult, mature male, mature face"
"""Under ADULT_FROM_AGE: part of the content rule, as prompt.YOUNG_NEGATIVE is for girls."""
MAN_COMPOSITION_NEGATIVE = (
    "1girl, multiple girls, multiple boys, 2boys, multiple views, text, watermark, signature, "
    "username, artist name"
)
ADULT_FROM_AGE = 18

ANIME_AGES: tuple[tuple[int, str], ...] = (
    (0, "teenage boy, teenage, youthful face"),
    (ADULT_FROM_AGE, "young man"),
    (30, "mature male, adult"),
    (45, "mature male, middle-aged man"),
    (60, "old man, mature male, wrinkled skin"),
)
"""First age and words: the 2026-11-13 men's wordings by real age, and a teenager's below."""


def anime_age_words(age: int) -> str:
    words = ANIME_AGES[0][1]
    for first, text in ANIME_AGES:
        if age >= first:
            words = text
    return words


def _join(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


def anime_prompt(
    sheet: ManSheet,
    age: int,
    wear: Wear,
    pose: str | Pose,
    *,
    backdrop: Backdrop = DEFAULT_BACKDROP,
) -> Prompt:
    """Animagine's prompt for a picture of a man at `age` (real age in years).

    `wear` is a player's kit, or a staff or media outfit (wardrobe.py); `pose` a key of
    traits.POSE or a shoot's arms and angle, as for prompt.build_prompt.
    """
    now = drawn(sheet, age)
    hair = HAIR_COLOUR[now.hair_colour].anime
    look = _join(
        hair,
        HAIR_STYLE[sheet.hair_style].anime,
        FACIAL_HAIR[now.facial_hair].anime,
        EYE_COLOUR[sheet.eye_colour].anime,
        SKIN[sheet.skin].anime,
        "glasses" if _glasses(sheet, wear) else "",
    )
    content, wear_negative = content_terms(wear)
    positive = _join(
        MAN_SUBJECT,
        RATING,
        anime_age_words(age),
        look,
        wear.words(),
        pose_words(pose),
        EXPRESSION[sheet.expression].anime,
        FRAMING,
        backdrop.words(),
        content,
        QUALITY,
    )
    negative = _join(
        CONTENT_NEGATIVE,
        MAN_CONTENT_NEGATIVE,
        BOY_NEGATIVE if age < ADULT_FROM_AGE else "",
        wear_negative,
        MAN_COMPOSITION_NEGATIVE,
        QUALITY_NEGATIVE,
    )
    return Prompt(positive, negative)


# --- Photo -------------------------------------------------------------------------------------

PHOTO_CONTENT_NEGATIVE = (
    "nsfw, nude, naked, shirtless, bare chest, topless, underwear, swimwear, revealing clothes, "
    "suggestive"
)
"""RealVisXL's content negative from run `realistic`, approved for men (user, 2026-11-20)."""
PHOTO_COMPOSITION_NEGATIVE = "multiple people, two people, text, watermark, signature, frame"
"""Run `realistic`'s, without "logo": kit logos and lettering are accepted (2026-11-10)."""
PHOTO_STYLE_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, 3d render, cgi, plastic skin, deformed, "
    "bad anatomy, asymmetric eyes, blurry, lowres"
)
PHOTO_NEGATIVE = _join(PHOTO_CONTENT_NEGATIVE, PHOTO_COMPOSITION_NEGATIVE, PHOTO_STYLE_NEGATIVE)
"""Z-Image-Turbo draws without guidance and ignores it; RealVisXL uses it."""

FULLY_CLOTHED = "fully clothed"
ONLY_PERSON = "He is the only person in the picture."
LIGHTING = "soft even lighting, sharp focus, natural skin texture, realistic photo"

ROLE_WORDS: dict[str, str] = {
    "manager": "male football manager",
    "coach": "male football coach",
    "analyst": "male football analyst",
    "director": "male football club director",
    "scout": "male football scout",
    "doctor": "male club doctor",
    "medical": "male sports physiotherapist",
    "journalist": "male sports journalist",
    "press officer": "male football club press officer",
    "staff": "male football club staff member",
}
"""What a man on the staff or in the media is, by wardrobe role."""
PLAYER_WORDS = "male football player"
KEEPER_WORDS = "male football goalkeeper"


def _a(phrase: str) -> str:
    """The phrase with "a" or "an"."""
    return f"{'an' if phrase[:1].lower() in 'aeiou' else 'a'} {phrase}"


def _age_phrase(age: int) -> str:
    """ "an 18-year-old", "a 24-year-old": "an" for 8, 11, 18 and 80 to 89."""
    article = "an" if str(age).startswith(("8", "11", "18")) else "a"
    return f"{article} {age}-year-old"


def _colour(name: str) -> str:
    return COLOURS[name].words


def shirt_sentence(kit: Kit, sleeves: str = "") -> str:
    """The shirt in words: "a football shirt with blue and white vertical stripes and a red
    collar". `sleeves` ("long-sleeved", "short-sleeved", "collared") goes before the garment."""
    c = [_colour(x) for x in kit.shirt]
    garment = _join_words(sleeves, "goalkeeper jersey" if kit.goalkeeper else "football shirt")
    body = _a(f"{c[0]} {garment}")
    match kit.pattern:
        case "plain":
            text, extras = body, []
        case "stripes":
            text, extras = _a(garment), [f"{join_and(c)} vertical stripes"]
        case "hoops":
            text, extras = _a(garment), [f"{c[0]} and {c[1]} horizontal hoops"]
        case "sleeves":
            text, extras = body, [f"{c[1]} sleeves"]
        case "two-tone":
            text, extras = _a(f"{c[0]} and {c[1]} two-tone {garment}"), []
        case "pinstripes":
            text, extras = body, [f"thin {c[1]} pinstripes"]
        case "thin-hoops":
            text, extras = body, [f"thin {c[1]} horizontal stripes"]
        case "checks":
            text, extras = _a(garment), [_a(f"{c[0]} and {c[1]} checkered pattern")]
        case "centre-stripe":
            text, extras = body, [_a(f"wide {c[1]} vertical stripe down the middle")]
        case "sash":
            text, extras = body, [_a(f"{join_and(c[1:])} diagonal sash")]
        case "band":
            text, extras = body, [_a(f"{join_and(c[1:])} band across the chest")]
        case "chevron":
            text, extras = body, [_a(f"{c[1]} V-shaped band across the chest")]
        case "cross":
            text, extras = body, [_a(f"{c[1]} cross on the chest")]
        case _:  # "mixed": two colours, layout not known
            text, extras = _a(f"{c[0]} and {c[1]} {garment}"), []
    if kit.trim:
        extras.append(_a(f"{_colour(kit.trim)} collar"))
    return text + (" with " + " and ".join(extras) if extras else "")


def jacket_sentence(kit: Kit) -> str:
    """The club track jacket in the kit's shirt colours: "a red and white club track jacket"."""
    c = [_colour(x) for x in kit.shirt]
    if kit.pattern == "sleeves":
        text = _a(f"{c[0]} club track jacket with {c[1]} sleeves")
    else:
        text = _a(f"{join_and(c)} club track jacket")
    if kit.trim:
        text += f"{' and' if ' with ' in text else ' with'} {_a(_colour(kit.trim) + ' collar')}"
    return text


def _join_words(*words: str) -> str:
    return " ".join(w for w in words if w)


def _without_article(phrase: str) -> str:
    return phrase.split(" ", 1)[1]


def kit_sentence(wear: Kit | KitLook) -> str:
    """A player's clothes in words, in the shoot's look (shoot.LOOKS) or as a match kit."""
    if isinstance(wear, Kit):
        return shirt_sentence(wear, "long-sleeved" if wear.goalkeeper else "")
    kit = wear.kit
    keeper_sleeves = "short-sleeved" if kit.goalkeeper else ""
    match wear.look:
        case "short sleeves":
            return shirt_sentence(kit, keeper_sleeves)
        case "long sleeves":
            return shirt_sentence(kit, "long-sleeved")
        case "layered sleeves":
            return f"{shirt_sentence(kit)} over a long-sleeved base layer"
        case "collared shirt":
            return shirt_sentence(kit, "collared")
        case "track jacket":
            return f"{jacket_sentence(kit)}, zipped up"
        case "open jacket":
            return f"an open {_without_article(jacket_sentence(kit))} over {shirt_sentence(kit)}"
        case "high collar":
            return f"{jacket_sentence(kit)}, zipped up to its high collar"
    raise ValueError(f"no words for the look {wear.look!r}")


OUTFIT_SENTENCES: dict[str, str] = {
    "track jacket": "{jacket}, zipped up",
    "half-zip": "a long-sleeved {club} half-zip training top with a high collar",
    "hoodie and vest": "a {tone} padded gilet over a {club} hoodie with the hood down",
    "polo shirt": "a long-sleeved {club} polo shirt",
    "polo and jacket": "an open {tone} track jacket over a {club} polo shirt",
    "training vest": "a {tone} padded gilet over a long-sleeved {club} training top",
    "bench coat": "a long {tone} padded coat over a {club} track jacket",
    "suit and tie": "a {tone} suit, a white collared shirt and a {accent} tie",
    "three-piece suit": "a {tone} three-piece suit, a white collared shirt and a {accent} tie",
    "open-collar blazer": "a {tone} blazer over a white dress shirt with an open collar",
    "turtleneck and blazer": "a {tone} blazer over a black turtleneck sweater",
    "overcoat and suit": "a {tone} overcoat over a suit, a white collared shirt and a {accent} scarf",
    "shirt and knit": "a {tone} sweater over a white collared shirt",
    "cardigan": "a {tone} cardigan over a white collared shirt and a {accent} tie",
    "blazer and shirt": "an open {tone} blazer over a {club} collared shirt",
    "trench coat": "a beige trench coat over a turtleneck sweater and a {accent} scarf",
    "field jacket": "a {tone} field jacket with a high collar over a plaid shirt",
    "lab coat and tie": "a white lab coat over a white collared shirt and a {accent} tie",
    "lab coat and knit": "a white lab coat over a {tone} sweater and a collared shirt",
    "press trench coat": "a beige trench coat over a {tone} turtleneck sweater",
    "press cardigan": "a {tone} cardigan over a white collared shirt",
}
"""Each wardrobe look (wardrobe.LOOKS) in words: long-sleeved and layered, as the anime words."""
PROP_SENTENCES: dict[str, str] = {
    "none": "",
    "whistle": "a whistle on a cord around his neck",
    "lanyard": "a lanyard with an ID card",
    "stethoscope": "a stethoscope around his neck",
    "wristwatch": "",  # out of a head-and-shoulders picture
    "notebook": "He is holding a notebook and a pen.",
    "microphone": "He is holding a microphone.",
    "clipboard": "He is holding a clipboard.",
}
"""Worn props join the clothes; held props (wardrobe.HELD_PROPS) are a sentence of their own."""


def outfit_sentence(outfit: Outfit) -> str:
    """A staff or media outfit's clothes in words, with a worn prop."""
    if outfit.home is None:
        raise ValueError("the outfit does not say which kit's colours dress it")
    home = outfit.home
    text = OUTFIT_SENTENCES[outfit.look].format(
        tone=outfit.tone or "",
        club=_colour(home.shirt[0]),
        accent=_colour(accent_colour(home)),
        jacket=jacket_sentence(home),
    )
    worn = "" if outfit.prop in HELD_PROPS else PROP_SENTENCES[outfit.prop]
    return f"{text}, with {worn}" if worn else text


def backdrop_sentence(backdrop: Backdrop) -> str:
    """ "A plain grey studio background", "A blue gradient ...", "A two-tone blue and red ..."."""
    colours = " and ".join(backdrop.colours)
    match backdrop.kind:
        case "plain":
            return f"A plain {colours} studio background"
        case "gradient":
            return f"A {colours} gradient studio background"
    return f"A two-tone {colours} studio background"


def _glasses(sheet: ManSheet, wear: Wear) -> bool:
    """Glasses are drawn on staff and media, never on a player in his kit."""
    return sheet.glasses and isinstance(wear, Outfit)


def _who(wear: Wear) -> str:
    if isinstance(wear, Outfit):
        return ROLE_WORDS[wear.role]
    kit = wear if isinstance(wear, Kit) else wear.kit
    return KEEPER_WORDS if kit.goalkeeper else PLAYER_WORDS


def photo_prompt(
    sheet: ManSheet, age: int, wear: Wear, *, backdrop: Backdrop = DEFAULT_BACKDROP
) -> Prompt:
    """The photo prompt for a picture of a man at `age` (real age in years). The negative is
    RealVisXL's; Z-Image-Turbo uses the positive alone."""
    now = drawn(sheet, age)
    style = HAIR_STYLE[sheet.hair_style]
    hair = style.photo.format(c=HAIR_COLOUR[now.hair_colour].photo)
    if now.greying:
        hair += ", going grey at the temples"
    face = (
        f"He has {SKIN[sheet.skin].photo} skin, {hair}, {FACIAL_HAIR[now.facial_hair].photo}, "
        f"{EYE_COLOUR[sheet.eye_colour].photo} eyes and {EXPRESSION[sheet.expression].photo}."
    )
    if isinstance(wear, Outfit):
        clothes = outfit_sentence(wear)
        held = PROP_SENTENCES[wear.prop] if wear.prop in HELD_PROPS else ""
    else:
        clothes, held = kit_sentence(wear), ""
    subject = (
        f"Professional studio headshot photograph of {_age_phrase(age)} "
        f"{LOOKS[sheet.look].words} {_who(wear)}, head and shoulders, centred, facing the "
        "camera and looking into the lens."
    )
    positive = " ".join(
        p
        for p in (
            subject,
            face,
            "He wears glasses." if _glasses(sheet, wear) else "",
            f"He wears {clothes}, {FULLY_CLOTHED}.",
            held,
            f"{backdrop_sentence(backdrop)}, {LIGHTING}.",
            ONLY_PERSON,
        )
        if p
    )
    return Prompt(positive, PHOTO_NEGATIVE)
