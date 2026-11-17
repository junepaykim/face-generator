"""Quality check. The content-rating tests guard AGENTS.md hard rule 3."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fakes import BACKDROPS, backdrop
from fakes import figure as figure_on
from PIL import Image, ImageDraw

from facegen import qc
from facegen.core.traits import CharacterSheet
from facegen.datasource.kits import Kit
from facegen.persistence.settings import MyClubSettings
from facegen.qc import CheckResult, TagResult, best, check, expectation, head_top

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
