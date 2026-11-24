"""Which prompt a picture gets from the model and gender settings (core/style.py, Phase 15g).

Women keep their prompt on every model, bit for bit; a man gets the photo prompt on a photo
model and Animagine's tags on Animagine; a model without guidance gets no negative.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from kitshot.core.kit_values import Kit
from kitshot.core.man_prompt import PHOTO_NEGATIVE, anime_prompt, photo_prompt
from kitshot.core.man_traits import HAIR_COLOUR as MAN_HAIR_COLOUR
from kitshot.core.man_traits import ManSheet, drawn, sample_man_sheet
from kitshot.core.prompt import build_prompt
from kitshot.core.settings_values import IMAGE_MODELS, MODEL_CHOICES, MyClubSettings
from kitshot.core.shoot import DEFAULT_BACKDROP, HAIR_FAMILIES, Backdrop, Pose
from kitshot.core.style import portrait_prompt, shoot_hair
from kitshot.core.traits import CharacterSheet, sample_sheet

SHEET = sample_sheet("career", 7, 1, height_cm=180, strength=12)
MAN = sample_man_sheet("career", 7, 1, nation=765)
KIT = Kit(shirt=("red", "white"), pattern="stripes", shorts="black")
POSE = Pose("crossed")
BACKDROP = Backdrop("gradient", ("blue",))
MY_CLUB = MyClubSettings(frozenset({"freckles"}), soften_skin=2, round_eyes=1.0)


def prompt(model: str, gender: str, **kw: object):
    args: dict[str, object] = {
        "sheet": SHEET,
        "man": MAN,
        "wording": "W07",
        "age": 27,
        "wear": KIT,
        "pose": POSE,
        "backdrop": BACKDROP,
        "my_club": None,
    }
    args.update(kw)
    return portrait_prompt(model, gender, **args)  # type: ignore[arg-type]


def test_the_choices_are_the_models() -> None:
    assert MODEL_CHOICES == ("animagine", "realvis", "zimage_turbo")
    assert [m for m in MODEL_CHOICES if IMAGE_MODELS[m].photo] == ["realvis", "zimage_turbo"]
    assert [m for m in MODEL_CHOICES if not IMAGE_MODELS[m].reference] == ["zimage_turbo"]
    assert [m for m in MODEL_CHOICES if not IMAGE_MODELS[m].negative] == ["zimage_turbo"]


@pytest.mark.parametrize("my_club", [None, MY_CLUB])
def test_a_woman_keeps_her_prompt_on_every_model(my_club: MyClubSettings | None) -> None:
    """Animagine's pictures of women stay bit-identical (Phase 15 exit test); the photo models
    draw the same prompt for now (user, 2026-11-20)."""
    women = build_prompt(SHEET, "W07", KIT, POSE, my_club, backdrop=BACKDROP)
    assert prompt("animagine", "woman", my_club=my_club) == women
    assert prompt("realvis", "woman", my_club=my_club) == women
    turbo = prompt("zimage_turbo", "woman", my_club=my_club)
    assert turbo is not None and turbo.positive == women.positive and turbo.negative == ""
    # A woman needs no men's sheet, and her age in years is not asked for.
    assert prompt("animagine", "woman", man=None, age=None) == build_prompt(
        SHEET, "W07", KIT, POSE, backdrop=BACKDROP
    )


def test_a_man_is_drawn_from_his_sheet_at_his_age() -> None:
    assert prompt("animagine", "man") == anime_prompt(MAN, 27, KIT, POSE, backdrop=BACKDROP)
    photo = photo_prompt(MAN, 27, KIT, backdrop=BACKDROP)
    assert prompt("realvis", "man") == photo and photo.negative == PHOTO_NEGATIVE
    turbo = prompt("zimage_turbo", "man")
    assert turbo is not None and turbo.positive == photo.positive and turbo.negative == ""
    assert "27-year-old" in photo.positive
    assert "28-year-old" in prompt("realvis", "man", age=28).positive  # type: ignore[union-attr]
    # The band does not change a man's prompt: only his age does.
    assert prompt("realvis", "man", wording="W09") == photo
    # My club's rules are the women's (user, 2026-11-20).
    for model in MODEL_CHOICES:
        assert prompt(model, "man", my_club=MY_CLUB) == prompt(model, "man")


def test_a_man_without_an_age_has_no_prompt() -> None:
    for model in MODEL_CHOICES:
        assert prompt(model, "man", age=None) is None
    with pytest.raises(ValueError, match="needs his sheet"):
        prompt("realvis", "man", man=None)
    with pytest.raises(ValueError, match="no gender"):
        prompt("realvis", "other")
    with pytest.raises(KeyError):
        prompt("flux", "woman")


def test_the_backdrop_keeps_clear_of_the_hair_drawn() -> None:
    assert shoot_hair(SHEET, None, 30) == SHEET.hair_colour
    man = ManSheet(
        look="northern european",
        skin="fair",
        hair_colour="blond",
        hair_style="side parting",
        facial_hair="none",
        eye_colour="blue",
        expression="cheerful",
        greying_from=40,
        glasses=False,
        base_seed=1,
    )
    assert shoot_hair(SHEET, man, 30) == shoot_hair(SHEET, man, None) == "blond"
    assert shoot_hair(SHEET, man, 52) == drawn(man, 52).hair_colour == "grey"
    shaved = replace(man, hair_style="shaved")
    assert shoot_hair(SHEET, shaved, 30) == ""
    assert set(MAN_HAIR_COLOUR) <= set(HAIR_FAMILIES)
    assert DEFAULT_BACKDROP.kind == "plain"


def test_the_same_inputs_give_the_same_prompt() -> None:
    again = sample_man_sheet("career", 7, 1, nation=765)
    assert again == MAN
    for model in MODEL_CHOICES:
        for gender in ("woman", "man"):
            assert prompt(model, gender) == prompt(model, gender, man=again)
    assert isinstance(SHEET, CharacterSheet)
