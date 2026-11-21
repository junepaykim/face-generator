"""Men's prompts (Phase 15f). The content-rule tests here guard AGENTS.md hard rule 3 and the
negatives the user approved on 2026-11-20."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from facegen.core.kit_values import Kit
from facegen.core.man_prompt import (
    ADULT_FROM_AGE,
    BOY_NEGATIVE,
    MAN_COMPOSITION_NEGATIVE,
    MAN_CONTENT_NEGATIVE,
    MAN_SUBJECT,
    ONLY_PERSON,
    OUTFIT_SENTENCES,
    PHOTO_NEGATIVE,
    PROP_SENTENCES,
    ROLE_WORDS,
    anime_age_words,
    anime_prompt,
    backdrop_sentence,
    jacket_sentence,
    kit_sentence,
    outfit_sentence,
    photo_prompt,
    shirt_sentence,
)
from facegen.core.man_traits import ManSheet, sample_man_sheet
from facegen.core.prompt import CONTENT_NEGATIVE, CONTENT_POSITIVE, QUALITY, RATING
from facegen.core.shoot import LOOKS as KIT_LOOKS
from facegen.core.shoot import Backdrop, KitLook, Pose
from facegen.core.wardrobe import HELD_PROPS, LOOKS, PROPS, ROLES, dress, outfit_for

SHEET = ManSheet(
    look="west african",
    skin="dark brown",
    hair_colour="black",
    hair_style="close fade",
    facial_hair="short beard",
    eye_colour="dark brown",
    expression="confident",
    greying_from=45,
    glasses=True,
    base_seed=1,
)
HOME = Kit(shirt=("red", "white"), pattern="stripes", shorts="black", trim="navy")
KEEPER = Kit(shirt=("green",), goalkeeper=True)
APPROVED_PHOTO_CONTENT = (
    "nsfw", "nude", "naked", "shirtless", "bare chest", "topless", "underwear", "swimwear",
    "revealing clothes", "suggestive",
)  # fmt: skip
SHOWS_SKIN = (
    "short sleeves", "sleeveless", "unzipped", "unbuttoned", "open shirt", "crop top",
    "tank top", "shorts", "skirt", "vest top", "singlet", "shirtless", "bare", "topless",
)  # fmt: skip


def terms(text: str) -> list[str]:
    return [t.strip() for t in text.split(",")]


# --- photo ----------------------------------------------------------------------------------


def test_photo_prompt_follows_the_measured_template() -> None:
    p = photo_prompt(SHEET, 27, KitLook(HOME, "short sleeves"))
    assert p.positive == (
        "Professional studio headshot photograph of a 27-year-old West African male football "
        "player, head and shoulders, centred, facing the camera and looking into the lens. "
        "He has dark brown skin, a close-cropped black fade, a short trimmed beard, dark brown "
        "eyes and a confident smile. He wears a football shirt with red and white vertical "
        "stripes and a navy blue collar, fully clothed. A plain grey studio background, soft "
        "even lighting, sharp focus, natural skin texture, realistic photo. He is the only "
        "person in the picture."
    )
    assert p.negative == PHOTO_NEGATIVE


def test_photo_negative_is_the_approved_one_without_logos() -> None:
    assert terms(PHOTO_NEGATIVE)[: len(APPROVED_PHOTO_CONTENT)] == list(APPROVED_PHOTO_CONTENT)
    assert "logo" not in PHOTO_NEGATIVE  # kit logos are accepted (decisions, 2026-11-10)


@pytest.mark.parametrize("age", [15, 16, 17, 18, 24, 35, 52, 70])
def test_every_photo_states_the_clothing_and_one_person(age: int) -> None:
    for uid in range(40):
        s = sample_man_sheet("c", uid, 1, nation=765)
        for wear in (KitLook(HOME, "track jacket"), outfit_for(HOME, "coach", age, uid, 1)):
            p = photo_prompt(s, age, wear, backdrop=Backdrop("gradient", ("blue",)))
            assert ", fully clothed." in p.positive and p.positive.endswith(ONLY_PERSON)
            assert f" {age}-year-old " in p.positive and p.negative == PHOTO_NEGATIVE
            assert not re.search(r"\b(woman|girl|female|she|her)\b", p.positive, re.IGNORECASE)
            assert len(p.positive.split()) < 130  # well under Z-Image-Turbo's 512 tokens


def test_age_articles() -> None:
    for age, text in ((8, "an 8-"), (11, "an 11-"), (18, "an 18-"), (24, "a 24-"), (83, "an 83-")):
        assert f"photograph of {text}year-old" in photo_prompt(SHEET, age, HOME).positive


def test_no_facial_hair_under_18_and_greying_with_age() -> None:
    young = photo_prompt(SHEET, 17, HOME).positive
    assert "no facial hair" in young and "beard" not in young
    assert "a short trimmed beard" in photo_prompt(SHEET, 25, HOME).positive
    assert "going grey at the temples" in photo_prompt(SHEET, 45, HOME).positive
    assert "a close-cropped grey fade" in photo_prompt(SHEET, 57, HOME).positive


def test_glasses_on_staff_and_media_only() -> None:
    assert "He wears glasses." not in photo_prompt(SHEET, 30, HOME).positive
    outfit = dress(HOME, "coach", "track jacket")
    assert "He wears glasses." in photo_prompt(SHEET, 30, outfit).positive
    assert "glasses" in anime_prompt(SHEET, 30, outfit, "crossed").positive
    assert "glasses" not in anime_prompt(SHEET, 30, HOME, "crossed").positive
    plain = replace(SHEET, glasses=False)
    assert "glasses" not in photo_prompt(plain, 30, outfit).positive


def test_who_he_is_follows_the_wear() -> None:
    assert "male football goalkeeper," in photo_prompt(SHEET, 30, KEEPER).positive
    for role in ROLES:
        look = next(iter(ROLES[role].looks))
        p = photo_prompt(SHEET, 40, dress(HOME, role, look))
        assert f"{ROLE_WORDS[role]}, head and shoulders" in p.positive


def test_shirts_in_words() -> None:
    plain = Kit(shirt=("orange",), shorts="black")
    assert shirt_sentence(plain) == "an orange football shirt"
    assert (
        shirt_sentence(replace(plain, trim="white"))
        == "an orange football shirt with a white collar"
    )
    assert shirt_sentence(Kit(shirt=("blue", "white"), pattern="hoops")) == (
        "a football shirt with blue and white horizontal hoops"
    )
    assert shirt_sentence(Kit(shirt=("red", "white"), pattern="sleeves", trim="black")) == (
        "a red football shirt with white sleeves and a black collar"
    )
    assert shirt_sentence(Kit(shirt=("sky", "navy"), pattern="two-tone"), "long-sleeved") == (
        "a light blue and navy blue two-tone long-sleeved football shirt"
    )
    assert shirt_sentence(Kit(shirt=("yellow", "black"), pattern="mixed")) == (
        "a yellow and black football shirt"
    )
    assert kit_sentence(KEEPER) == "a green long-sleeved goalkeeper jersey"
    assert jacket_sentence(HOME) == "a red and white club track jacket with a navy blue collar"


def test_every_kit_look_has_words_that_cover_him() -> None:
    for kit in (HOME, KEEPER, Kit(shirt=("blue", "white"), pattern="sleeves", shorts="blue")):
        for look in KIT_LOOKS:
            text = kit_sentence(KitLook(kit, look, "armband"))
            assert "shirt" in text or "jersey" in text or "jacket" in text, (look, text)
            for word in ("sleeveless", "shirtless", "bare", "unzipped", "crop"):
                assert word not in text
    assert kit_sentence(KitLook(HOME, "open jacket")) == (
        "an open red and white club track jacket with a navy blue collar over a football shirt "
        "with red and white vertical stripes and a navy blue collar"
    )


def test_every_outfit_and_prop_has_words_that_show_no_skin() -> None:
    assert set(OUTFIT_SENTENCES) == set(LOOKS) and set(PROP_SENTENCES) == set(PROPS)
    for name, look in LOOKS.items():
        fields = set(re.findall(r"{(\w+)}", OUTFIT_SENTENCES[name]))
        assert ("tone" in fields) == bool(look.tones), name
        assert fields <= {"tone", "club", "accent", "jacket"}, name
        for tone in look.tones or (None,):
            for prop in PROPS:
                text = outfit_sentence(dress(HOME, "staff", name, prop, tone))
                assert "{" not in text and "  " not in text, text
                for word in SHOWS_SKIN:
                    assert not re.search(rf"\b{word}", text), (word, text)
    for prop in HELD_PROPS:
        assert PROP_SENTENCES[prop].startswith("He is holding")


def test_a_held_prop_is_its_own_sentence() -> None:
    p = photo_prompt(SHEET, 40, dress(HOME, "journalist", "press cardigan", "notebook", "grey"))
    assert "He wears a grey cardigan over a white collared shirt, fully clothed. He is holding" in (
        p.positive
    )
    worn = photo_prompt(SHEET, 40, dress(HOME, "coach", "track jacket", "whistle"))
    assert "zipped up, with a whistle on a cord around his neck, fully clothed." in worn.positive


def test_backdrops_in_words() -> None:
    assert backdrop_sentence(Backdrop("plain", ("grey",))) == "A plain grey studio background"
    assert backdrop_sentence(Backdrop("gradient", ("blue",))) == "A blue gradient studio background"
    assert backdrop_sentence(Backdrop("two-tone", ("blue", "red"))) == (
        "A two-tone blue and red studio background"
    )


# --- anime ----------------------------------------------------------------------------------


@pytest.mark.parametrize("age", [15, 17, 18, 29, 30, 44, 45, 59, 60, 75])
def test_anime_negative_keeps_every_content_term_and_adds_the_mens(age: int) -> None:
    for wear in (KitLook(HOME, "short sleeves"), KEEPER, dress(HOME, "doctor", "lab coat and tie")):
        p = anime_prompt(SHEET, age, wear, Pose("crossed"))
        negative = terms(p.negative)
        for term in (*terms(CONTENT_NEGATIVE), *terms(MAN_CONTENT_NEGATIVE)):
            assert term in negative, term
        for term in terms(MAN_COMPOSITION_NEGATIVE):
            assert term in negative, term
        young = all(t in negative for t in terms(BOY_NEGATIVE))
        assert young is (age < ADULT_FROM_AGE)
        assert p.positive.startswith(f"{MAN_SUBJECT}, {RATING}, {anime_age_words(age)}, ")
        assert p.positive.endswith(QUALITY)
        assert not re.search(r"\b(1girl|girl|woman|female)\b", p.positive)


def test_anime_clothing_terms_are_the_womens() -> None:
    assert f"{CONTENT_POSITIVE}, " in anime_prompt(SHEET, 25, HOME, "crossed").positive
    staff = anime_prompt(SHEET, 50, dress(HOME, "director", "suit and tie"), "crossed")
    assert "long sleeves" in staff.positive and "soccer uniform" in staff.negative


def test_anime_age_words() -> None:
    assert anime_age_words(16) == "teenage boy, teenage, youthful face"
    assert anime_age_words(18) == "young man"
    assert anime_age_words(30) == "mature male, adult"
    assert anime_age_words(45) == "mature male, middle-aged man"
    assert anime_age_words(60) == "old man, mature male, wrinkled skin"


def test_anime_look_words() -> None:
    p = anime_prompt(SHEET, 25, HOME, "crossed").positive
    assert "black hair, very short hair, beard, brown eyes, dark skin, dark-skinned male" in p
    assert "stubble" in anime_prompt(SHEET, 19, HOME, "crossed").positive


def test_same_inputs_same_prompts() -> None:
    s = sample_man_sheet("c", 5, 1, nation=1651)
    assert photo_prompt(s, 22, HOME) == photo_prompt(s, 22, HOME)
    assert anime_prompt(s, 22, HOME, "v") == anime_prompt(s, 22, HOME, "v")
