"""Prompt assembly. The content-rule tests here guard AGENTS.md hard rule 3."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kitshot.core.age import WORDINGS
from kitshot.core.prompt import (
    COMPOSITION_NEGATIVE,
    CONTENT_NEGATIVE,
    CONTENT_POSITIVE,
    QUALITY,
    QUALITY_NEGATIVE,
    YOUNG_NEGATIVE,
    build_prompt,
    drawn_traits,
)
from kitshot.core.traits import POSE, CharacterSheet, eyes_drawn_round, sample_sheet
from kitshot.datasource.kits import Kit
from kitshot.persistence.settings import MyClubSettings

SHEET = CharacterSheet(
    hair_colour="black",
    hair_style="long",
    fringe="blunt",
    eye_colour="brown",
    eye_shape="gentle",
    skin="dark",
    accessory="hairband",
    mark="freckles",
    expression="cheerful",
    height="average",
    build="average",
    base_seed=1,
)
KIT = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
GK = Kit(shirt=("yellow",), goalkeeper=True)
MY_CLUB = MyClubSettings(banned_traits=frozenset({"freckles"}), soften_skin=2)


def test_prompt_order_matches_the_recipe() -> None:
    p = build_prompt(SHEET, "W07", KIT, "crossed")
    assert p.positive == (
        "1girl, solo, safe, young adult woman, black hair, long hair, straight hair, "
        "blunt bangs, brown eyes, tareme, dark skin, dark-skinned female, hairband, freckles, "
        "blue and white vertical-striped shirt, short sleeves, blue shorts, crossed arms, "
        "smile, open mouth, upper body, looking at viewer, standing, simple background, "
        "grey background, sportswear, soccer uniform, soccer, "
        "masterpiece, high score, great score, absurdres"
    )
    assert p.negative == f"{CONTENT_NEGATIVE}, {COMPOSITION_NEGATIVE}, {QUALITY_NEGATIVE}"


@pytest.mark.parametrize("wording", sorted(WORDINGS))
@pytest.mark.parametrize("pose", sorted(POSE))
def test_content_terms_are_always_there(wording: str, pose: str) -> None:
    for my_club in (None, MY_CLUB):
        for kit in (KIT, GK):
            p = build_prompt(SHEET, wording, kit, pose, my_club)
            assert p.positive.startswith("1girl, solo, safe, ")
            assert CONTENT_POSITIVE in p.positive
            assert p.positive.endswith(QUALITY)
            assert p.negative.startswith(CONTENT_NEGATIVE)
            young = YOUNG_NEGATIVE in p.negative
            assert young == WORDINGS[wording].young


def test_young_wordings_are_w01_to_w06() -> None:
    young = {k for k, w in WORDINGS.items() if w.young}
    assert young == {"W01", "W02", "W03", "W04", "W05", "W06"}
    p = build_prompt(SHEET, "W01", KIT, "v")
    assert p.negative.startswith(f"{CONTENT_NEGATIVE}, {YOUNG_NEGATIVE}, ")


def test_negative_has_no_logo_or_lettering_terms() -> None:
    """Kit logos and lettering are accepted (decisions.md, 2026-11-10)."""
    p = build_prompt(SHEET, "W01", KIT, "v", MY_CLUB)
    for term in ("logo", "brand", "lettering", "clothes writing", "number", "adidas", "nike"):
        assert term not in p.negative


def test_unknown_wording_is_an_error() -> None:
    with pytest.raises(KeyError):
        build_prompt(SHEET, "W11", KIT, "v")


def test_my_club_bans_freckles_and_softens_skin() -> None:
    elsewhere = build_prompt(SHEET, "W07", KIT, "v")
    mine = build_prompt(SHEET, "W07", KIT, "v", MY_CLUB)
    assert "freckles" in elsewhere.positive and "freckles" not in elsewhere.negative
    assert "freckles" not in mine.positive
    # Banned traits sit after the content and young terms, before composition.
    assert (
        mine.negative == f"{CONTENT_NEGATIVE}, freckles, {COMPOSITION_NEGATIVE}, {QUALITY_NEGATIVE}"
    )
    assert "dark skin" in elsewhere.positive
    assert "dark skin" not in mine.positive and "slightly tanned skin" in mine.positive
    # Other traits are unaffected.
    assert "hairband" in mine.positive


def test_banned_trait_order_with_a_young_wording() -> None:
    mc = MyClubSettings(banned_traits=frozenset({"freckles", "glasses"}), soften_skin=0)
    p = build_prompt(SHEET, "W02", KIT, "v", mc)
    assert p.negative == f"{CONTENT_NEGATIVE}, {YOUNG_NEGATIVE}, freckles, glasses, {COMPOSITION_NEGATIVE}, {QUALITY_NEGATIVE}"  # fmt: skip
    assert "dark skin, dark-skinned female" in p.positive  # soften 0


def test_banned_trait_the_sheet_lacks_still_goes_to_the_negative() -> None:
    sheet = replace(SHEET, mark="none", accessory="none")
    p = build_prompt(sheet, "W08", KIT, "v", MY_CLUB)
    assert "freckles" in p.negative and "freckles" not in p.positive


def test_tan_skin_at_my_club_has_no_skin_term() -> None:
    sheet = replace(SHEET, skin="tan")
    p = build_prompt(sheet, "W08", KIT, "v", MY_CLUB)
    assert "tan," not in p.positive and "skin" not in p.positive


def test_goalkeeper_prompt() -> None:
    p = build_prompt(SHEET, "W04", GK, "ball")
    assert "goalkeeper, yellow shirt, long sleeves, black gloves, holding ball, soccer ball" in (
        p.positive
    )


def test_same_inputs_same_prompt() -> None:
    s = sample_sheet("k", 1, 1)
    assert build_prompt(s, "W03", KIT, "v", MY_CLUB) == build_prompt(s, "W03", KIT, "v", MY_CLUB)


def test_drawn_traits_drop_bans_and_empty_values() -> None:
    assert drawn_traits(SHEET)["mark"] == "freckles"
    mine = drawn_traits(SHEET, MY_CLUB)
    assert "mark" not in mine and mine["accessory"] == "hairband"
    plain = drawn_traits(replace(SHEET, accessory="none", mark="none"))
    assert "accessory" not in plain and "mark" not in plain


def test_some_sharp_eyes_are_drawn_round_at_my_club_only() -> None:
    mc = replace(MY_CLUB, round_eyes=0.2)
    sharp = [replace(SHEET, eye_shape="sharp", base_seed=seed) for seed in range(400)]
    rounded = [s for s in sharp if eyes_drawn_round("sharp", s.base_seed, 0.2)]
    kept = [s for s in sharp if not eyes_drawn_round("sharp", s.base_seed, 0.2)]
    assert rounded and kept
    for s in rounded:
        assert "tsurime" not in build_prompt(s, "W07", KIT, "v", mc).positive
        assert "tsurime" in build_prompt(s, "W07", KIT, "v").positive  # elsewhere: as sheet
        assert drawn_traits(s, mc)["eye_shape"] == "round"
        assert drawn_traits(s)["eye_shape"] == "sharp"
    for s in kept:
        assert "tsurime" in build_prompt(s, "W07", KIT, "v", mc).positive
    # Gentle eyes are never changed, whatever the share.
    gentle = replace(SHEET, eye_shape="gentle")
    assert "tareme" in build_prompt(gentle, "W07", KIT, "v", replace(mc, round_eyes=1.0)).positive


def test_height_and_build_words_follow_the_skin() -> None:
    sheet = replace(SHEET, height="tall", build="slim", skin="light")
    p = build_prompt(sheet, "W07", KIT, "v")
    assert "brown eyes, tareme, tall female, skinny, hairband, freckles, blue and white" in (
        p.positive
    )
    short = build_prompt(replace(SHEET, height="short", build="average"), "W02", KIT, "v")
    assert "petite" in short.positive and "petite" not in short.negative


def test_muscular_is_banned_at_my_club_only() -> None:
    sheet = replace(SHEET, build="muscular")
    mc = MyClubSettings(banned_traits=frozenset({"freckles", "muscular"}), soften_skin=2)
    elsewhere = build_prompt(sheet, "W08", KIT, "v")
    mine = build_prompt(sheet, "W08", KIT, "v", mc)
    assert "muscular female" in elsewhere.positive and "muscular" not in elsewhere.negative
    assert "muscular" not in mine.positive
    assert mine.negative == (
        f"{CONTENT_NEGATIVE}, freckles, muscular female, {COMPOSITION_NEGATIVE}, {QUALITY_NEGATIVE}"
    )
    # Every player at my club carries the ban, whatever their build.
    assert "muscular female" in build_prompt(SHEET, "W08", KIT, "v", mc).negative
