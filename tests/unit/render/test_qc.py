"""Quality check. The content-rating tests guard AGENTS.md hard rule 3."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fakes import BACKDROPS, backdrop
from fakes import figure as figure_on
from PIL import Image, ImageDraw

from kitshot import qc
from kitshot.core.man_traits import ManSheet
from kitshot.core.traits import CharacterSheet
from kitshot.datasource.kits import Kit
from kitshot.persistence.settings import MyClubSettings
from kitshot.qc import (
    CheckResult,
    TagResult,
    best,
    check,
    expectation,
    head_top,
    man_expectation,
)

SHEET = CharacterSheet(
    hair_colour="silver",
    hair_style="ponytail",
    fringe="swept",
    eye_colour="blue",
    eye_shape="round",
    skin="light",
    accessory="glasses",
    mark="freckles",
    expression="serious",
    height="tall",
    build="muscular",
    base_seed=5,
)
KIT = Kit(shirt=("red", "white"), pattern="stripes", shorts="black")
EXPECT = expectation(SHEET, KIT)


def tags(rating: str = "general", **extra: float) -> TagResult:
    ratings = {"general": 0.1, "sensitive": 0.05, "questionable": 0.01, "explicit": 0.0}
    ratings[rating] = 0.9
    base = {"solo": 0.99, "1girl": 0.99, "grey hair": 0.8, "blue eyes": 0.9}
    base.update({k.replace("_", " "): v for k, v in extra.items()})
    return TagResult(ratings, base)


def test_expectation_from_sheet_and_kit() -> None:
    assert EXPECT.hair == {"grey hair", "white hair"}
    assert EXPECT.eyes == {"blue eyes", "aqua eyes"}
    assert set(EXPECT.rank) == {
        "hair_colour", "hair_style", "fringe", "eye_colour", "expression", "accessory", "mark",
        "height", "build", "shirt colour 1", "shirt colour 2",
    }  # fmt: skip  # eye shape "round" has no tag
    assert EXPECT.rank["height"] == {"tall female"}
    mine = expectation(SHEET, KIT, MyClubSettings(frozenset({"freckles", "muscular"}), 2))
    assert "build" not in mine.rank
    assert "mark" not in mine.rank


def test_a_clean_picture_passes() -> None:
    r = check(tags(), EXPECT, head_top=0.03)
    assert r.passed and r.failures == () and r.rating == "general"


@pytest.mark.parametrize("rating", ["sensitive", "questionable", "explicit"])
def test_anything_but_general_fails(rating: str) -> None:
    r = check(tags(rating), EXPECT, head_top=0.03)
    assert not r.passed
    assert r.failures[0].startswith(f"rated {rating}")


def test_general_must_be_the_top_rating() -> None:
    t = TagResult(
        {"general": 0.49, "sensitive": 0.51, "questionable": 0, "explicit": 0}, tags().tags
    )
    assert not check(t, EXPECT, 0.03).passed
    t = TagResult(
        {"general": 0.51, "sensitive": 0.49, "questionable": 0, "explicit": 0}, tags().tags
    )
    assert check(t, EXPECT, 0.03).passed


def test_hair_and_eye_colour_must_be_seen() -> None:
    t = TagResult(tags().ratings, {**tags().tags, "grey hair": 0.2})
    assert check(t, EXPECT, 0.03).failures == ("hair colour not seen",)
    t = TagResult(tags().ratings, {**tags().tags, "grey hair": 0.0, "white hair": 0.36})
    assert check(t, EXPECT, 0.03).passed  # any accepted tag, at the 0.35 threshold
    t = TagResult(tags().ratings, {**tags().tags, "blue eyes": 0.34})
    assert check(t, EXPECT, 0.03).failures == ("eye colour not seen",)


@pytest.mark.parametrize(
    "change",
    [{"solo": 0.1}, {"1girl": 0.0}, {"multiple_views": 0.5}, {"multiple_girls": 0.4},
     {"2girls": 0.9}, {"no_humans": 0.6}],
)  # fmt: skip
def test_exactly_one_girl(change: dict[str, float]) -> None:
    r = check(tags(**change), EXPECT, 0.03)
    assert not r.passed and r.failures[0].startswith("not one girl")


def test_head_must_start_near_the_top() -> None:
    assert check(tags(), EXPECT, 0.20).passed
    r = check(tags(), EXPECT, 0.30)
    assert not r.passed and "head starts 30% down" in r.failures[0]
    assert not check(tags(), EXPECT, None).passed


def test_kit_pattern_and_logos_are_not_checked() -> None:
    t = tags(logo=0.9, english_text=0.9, clothes_writing=0.9)  # no shirt colour either
    assert check(t, EXPECT, 0.03).passed


def test_score_prefers_traits_and_tight_framing() -> None:
    plain = check(tags(), EXPECT, 0.03)
    traits = check(tags(ponytail=0.9, glasses=0.9, red_shirt=0.9, white_shirt=0.8), EXPECT, 0.03)
    wide = check(tags(cowboy_shot=0.9), EXPECT, 0.03)
    assert traits.score > plain.score > wide.score
    assert traits.traits["accessory"] == pytest.approx(0.9)
    assert traits.traits["shirt colour 2"] == pytest.approx(0.8)


def _result(passed: bool, score: float) -> CheckResult:
    return CheckResult(passed, () if passed else ("x",), "general", 0.9, score, {}, 0.0)


def test_best_never_picks_a_failed_candidate() -> None:
    assert best([_result(False, 9.0), _result(True, 1.0), _result(True, 2.0)]) == 2
    assert best([_result(False, 9.0), _result(False, 1.0)]) is None
    assert best([_result(True, 2.0), _result(True, 2.0)]) == 0  # ties: first
    assert best([]) is None


def figure(top: float, size: int = 1024, bg: tuple[int, int, int] = (200, 200, 205)) -> Image.Image:
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    y = int(top * size)
    draw.ellipse((size * 0.38, y, size * 0.62, y + size * 0.24), fill=(240, 205, 180))
    draw.rectangle((size * 0.2, y + size * 0.25, size * 0.8, size), fill=(30, 60, 160))
    return img


@pytest.mark.parametrize("top", [0.0, 0.05, 0.12, 0.3, 0.5])
def test_head_top_finds_the_figure(top: float) -> None:
    found = head_top(figure(top))
    assert found is not None and abs(found - top) < 0.02


def test_head_top_on_an_empty_picture() -> None:
    assert head_top(Image.new("RGB", (512, 512), (128, 128, 128))) is None


def test_check_result_json_keeps_tags_and_probes() -> None:
    t = tags(child=0.2)
    data = check(t, EXPECT, 0.031).to_json(t)
    assert data["passed"] and data["rating"] == "general"
    assert data["probes"]["child"] == 0.2 and data["head_top"] == 0.031
    assert "solo" in data["tags"]


def test_threshold_is_the_recipe_value() -> None:
    assert qc.TAG_THRESHOLD == 0.35
    assert set(qc.FORBIDDEN_TAGS) == {"multiple views", "multiple girls", "2girls", "no humans"}
    assert replace(EXPECT, rank={}).rank == {}


@pytest.mark.parametrize("tag", ["midriff", "navel", "crop top", "cleavage", "see-through"])
def test_content_tags_fail_even_when_rated_general(tag: str) -> None:
    t = tags(**{tag.replace(" ", "_"): 0.5})
    assert t.rating == "general"
    r = check(t, EXPECT, 0.03)
    assert not r.passed and r.failures == (f"content: {tag}",)
    assert check(tags(**{tag.replace(" ", "_"): 0.3}), EXPECT, 0.03).passed  # below 0.35


@pytest.mark.parametrize("kind", BACKDROPS)
@pytest.mark.parametrize("top", [0.0, 0.05, 0.15, 0.3])
def test_head_top_on_every_backdrop(kind: str, top: float) -> None:
    """A gradient or two-tone backdrop (shoot.py) is backdrop: the head is found where it is."""
    found = head_top(figure_on(224, 288, top=top, kind=kind))
    assert found is not None and abs(found - top) <= 0.02, (kind, top, found)
    assert head_top(backdrop(224, 288, kind)) is None


# --- the men's check (user, 2026-11-20; Phase 15h) ----------------------------------------------

MAN = ManSheet(
    look="northern european",
    skin="fair",
    hair_colour="blond",
    hair_style="side parting",
    facial_hair="full beard",
    eye_colour="blue",
    expression="serious",
    greying_from=40,
    glasses=False,
    base_seed=5,
)
MAN_EXPECT = man_expectation(MAN, 30, KIT)


def man_tags(rating: str = "general", **extra: float) -> TagResult:
    ratings = {"general": 0.1, "sensitive": 0.05, "questionable": 0.01, "explicit": 0.0}
    ratings[rating] = 0.9
    base = {"solo": 0.98, "1boy": 0.99}
    base.update({k.replace("_", " "): v for k, v in extra.items()})
    return TagResult(ratings, base)


def test_the_mens_check_is_the_one_approved() -> None:
    assert qc.MAN_CONTENT_TAGS == ("topless male", "shirtless", "male underwear")
    assert qc.MAN_REQUIRED_TAGS == ("solo", "1boy")
    assert set(qc.MAN_FORBIDDEN_TAGS) == {
        "1girl", "multiple boys", "2boys", "multiple views", "no humans",
    }  # fmt: skip
    # The women's check is unchanged.
    assert qc.REQUIRED_TAGS == ("solo", "1girl")
    assert not set(qc.MAN_CONTENT_TAGS) & set(qc.CONTENT_TAGS)
    assert not EXPECT.man and MAN_EXPECT.man


def test_a_man_needs_no_hair_or_eye_colour_and_ranks_by_them() -> None:
    assert MAN_EXPECT.hair == MAN_EXPECT.eyes == frozenset()
    assert MAN_EXPECT.rank["hair_colour"] == {"blonde hair"}
    assert MAN_EXPECT.rank["facial_hair"] == {"beard", "facial hair"}
    assert {"shirt colour 1", "shirt colour 2"} <= set(MAN_EXPECT.rank)
    r = check(man_tags(), MAN_EXPECT, 0.03)
    assert r.passed and r.failures == ()
    better = check(man_tags(blonde_hair=0.9, beard=0.8), MAN_EXPECT, 0.03)
    assert better.passed and better.score > r.score
    # As drawn at his age: grey later, no beard on a boy; a shaved head ranks no colour.
    old = man_expectation(MAN, 60, KIT)
    assert old.rank["hair_colour"] == {"grey hair", "white hair"}
    assert "facial_hair" not in man_expectation(MAN, 17, KIT).rank
    shaved = man_expectation(replace(MAN, hair_style="shaved"), 30, KIT)
    assert "hair_colour" not in shaved.rank


@pytest.mark.parametrize(
    "extra,failure",
    [
        ({"1boy": 0.1}, "not one man: no 1boy"),
        ({"solo": 0.1}, "not one man: no solo"),
        ({"1girl": 0.6}, "not one man: 1girl"),
        ({"multiple_boys": 0.5}, "not one man: multiple boys"),
        ({"2boys": 0.5}, "not one man: 2boys"),
        ({"multiple_views": 0.5}, "not one man: multiple views"),
        ({"no_humans": 0.5}, "not one man: no humans"),
    ],
)
def test_one_man_alone(extra: dict[str, float], failure: str) -> None:
    r = check(man_tags(**extra), MAN_EXPECT, 0.03)
    assert not r.passed and r.failures == (failure,)


@pytest.mark.parametrize("tag", [*qc.CONTENT_TAGS, *qc.MAN_CONTENT_TAGS])
def test_a_mans_content_tags_fail_even_when_rated_general(tag: str) -> None:
    t = man_tags(**{tag.replace(" ", "_"): 0.5})
    assert t.rating == "general"
    r = check(t, MAN_EXPECT, 0.03)
    assert not r.passed and r.failures == (f"content: {tag}",)
    assert check(man_tags(**{tag.replace(" ", "_"): 0.3}), MAN_EXPECT, 0.03).passed


@pytest.mark.parametrize("rating", ["sensitive", "questionable", "explicit"])
def test_a_man_must_be_rated_general(rating: str) -> None:
    r = check(man_tags(rating), MAN_EXPECT, 0.03)
    assert not r.passed and r.failures[0].startswith(f"rated {rating}")


def test_a_mans_head_position_is_checked_as_a_womans() -> None:
    assert check(man_tags(), MAN_EXPECT, qc.HEAD_TOP_MAX).passed
    low = check(man_tags(), MAN_EXPECT, 0.25)
    assert not low.passed and low.failures == ("head starts 25% down the picture",)
    assert check(man_tags(), MAN_EXPECT, None).failures == (
        "no figure found against the background",
    )


def test_the_womens_check_is_unchanged_for_shirtless_or_a_boy() -> None:
    """The men's content tags are not added to the women's check (user, 2026-11-20)."""
    assert check(tags(shirtless=0.5), EXPECT, 0.03).passed
    boy = check(man_tags(**{"grey hair": 0.8, "blue eyes": 0.9}), EXPECT, 0.03)
    assert not boy.passed and boy.failures == ("not one girl: no 1girl",)
