"""The men's sheet (Phase 15f): seeded, a look from the nation's region, features that suit it."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest
from factories import PLAIN_PERSONALITY

from kitshot.core.man_traits import (
    EXPRESSION,
    FACIAL_HAIR,
    GREY_AFTER_YEARS,
    HAIR_COLOUR,
    HAIR_STYLE,
    LOOKS,
    SHAVEN_BELOW_AGE,
    SKIN,
    STUBBLE_BELOW_AGE,
    TEXTURES,
    ManSheet,
    drawn,
    sample_man_sheet,
)
from kitshot.core.man_traits import EYE_COLOUR as MAN_EYES
from kitshot.core.traits import EXPRESSION as WOMEN_EXPRESSION
from kitshot.core.traits import sample_sheet

ENGLAND, JAPAN, NIGERIA = 765, 116, 38


def sheets(nation: int | None, n: int = 2000, career: str = "c") -> list[ManSheet]:
    return [sample_man_sheet(career, uid, 1, nation=nation) for uid in range(1, n + 1)]


def test_same_inputs_same_sheet_and_the_womens_base_seed() -> None:
    a = sample_man_sheet("career-1", 29141735, 1, nation=ENGLAND)
    assert a == sample_man_sheet("career-1", 29141735, 1, nation=ENGLAND)
    # Picture seeds come from the base seed as for the women's sheet.
    assert a.base_seed == sample_sheet("career-1", 29141735, 1).base_seed
    others = [
        sample_man_sheet("career-2", 29141735, 1, nation=ENGLAND),
        sample_man_sheet("career-1", 29141736, 1, nation=ENGLAND),
        sample_man_sheet("career-1", 29141735, 2, nation=ENGLAND),
    ]
    assert all(o != a for o in others)


def test_the_tables_name_what_exists() -> None:
    for name, look in LOOKS.items():
        assert set(look.skin) <= set(SKIN), name
        assert set(look.hair) <= set(HAIR_COLOUR) - {"grey"}, name  # grey comes with age
        assert set(look.eyes) <= set(MAN_EYES), name
        assert set(look.texture) <= set(TEXTURES), name
    for texture, styles in TEXTURES.items():
        assert set(styles) <= set(HAIR_STYLE), texture
    assert {s for styles in TEXTURES.values() for s in styles} == set(HAIR_STYLE)
    for style in HAIR_STYLE.values():
        assert ("{c}" in style.photo) == style.coloured
    assert set(EXPRESSION) == set(WOMEN_EXPRESSION)  # personality leans the same way


def test_skin_hair_and_eyes_suit_the_look() -> None:
    for s in sheets(None, 3000):
        look = LOOKS[s.look]
        assert s.skin in look.skin and s.hair_colour in look.hair and s.eye_colour in look.eyes
        assert any(s.hair_style in TEXTURES[t] for t in look.texture), s


def test_the_look_follows_the_nations_region() -> None:
    england = Counter(s.look for s in sheets(ENGLAND))
    assert 0.56 < england["northern european"] / 2000 < 0.68  # 62 in REGIONS
    assert england["caribbean"] and england["west african"] and england["mixed"]
    assert Counter(s.look for s in sheets(JAPAN))["east asian"] / 2000 > 0.92  # 96
    assert Counter(s.look for s in sheets(NIGERIA))["west african"] / 2000 > 0.90  # 94


def test_without_a_known_nation_the_world_mix_is_drawn() -> None:
    world = sheets(None, 500)
    assert world == sheets(8_162_661, 500)  # a nation not in the table
    assert len({s.look for s in world}) > 10


def test_only_the_look_and_what_suits_it_depend_on_the_nation() -> None:
    for uid in range(1, 200):
        a = sample_man_sheet("c", uid, 1, nation=ENGLAND)
        b = sample_man_sheet("c", uid, 1, nation=JAPAN)
        assert (a.facial_hair, a.expression, a.greying_from, a.glasses, a.base_seed) == (
            b.facial_hair, b.expression, b.greying_from, b.glasses, b.base_seed,
        )  # fmt: skip


def test_personality_leans_the_expression() -> None:
    bold = replace(PLAIN_PERSONALITY, pressure=20, ambition=20)  # two leans to "confident"
    plain = Counter(sample_man_sheet("c", u, 1).expression for u in range(2000))
    leaning = Counter(sample_man_sheet("c", u, 1, personality=bold).expression for u in range(2000))
    assert leaning["confident"] > 1.5 * plain["confident"]
    # Weight 0 leaves the draw as without a personality.
    assert [
        sample_man_sheet("c", u, 1, personality=bold, personality_weight=0) for u in (1, 2)
    ] == [sample_man_sheet("c", u, 1) for u in (1, 2)]


def test_every_option_is_drawn_and_the_rare_ones_stay_rare() -> None:
    many = sheets(None, 4000)
    assert {s.facial_hair for s in many} == set(FACIAL_HAIR)
    assert 0.08 < sum(s.glasses for s in many) / 4000 < 0.22  # 1 in 7
    assert {s.greying_from for s in many} == set(range(33, 66))


def test_facial_hair_follows_the_age() -> None:
    bearded = next(s for s in sheets(None) if s.facial_hair == "full beard")
    assert drawn(bearded, SHAVEN_BELOW_AGE - 1).facial_hair == "none"
    assert drawn(bearded, SHAVEN_BELOW_AGE).facial_hair == "stubble"
    assert drawn(bearded, STUBBLE_BELOW_AGE - 1).facial_hair == "stubble"
    assert drawn(bearded, STUBBLE_BELOW_AGE).facial_hair == "full beard"
    stubble = next(s for s in sheets(None) if s.facial_hair == "stubble")
    assert drawn(stubble, 18).facial_hair == "stubble"


def test_hair_greys_from_the_sheets_age() -> None:
    s = next(s for s in sheets(None) if HAIR_STYLE[s.hair_style].coloured)
    g = s.greying_from
    before = drawn(s, g - 1)
    assert (before.hair_colour, before.greying) == (drawn(s, 16).hair_colour, False)
    assert not drawn(s, g - 1).greying and drawn(s, g - 1).hair_colour == s.hair_colour
    assert drawn(s, g).greying and drawn(s, g).hair_colour == s.hair_colour
    assert drawn(s, g + GREY_AFTER_YEARS).hair_colour == "grey"
    assert not drawn(s, g + GREY_AFTER_YEARS).greying
    shaved = replace(s, hair_style="shaved")
    assert drawn(shaved, 90).hair_colour == s.hair_colour and not drawn(shaved, 90).greying


def test_json_round_trip_keeps_the_base_seed_apart() -> None:
    s = sample_man_sheet("c", 7, 3, nation=ENGLAND)
    data = json.loads(s.to_json())
    assert "base_seed" not in data and data["look"] == s.look
    assert ManSheet.from_json(s.to_json(), s.base_seed) == s
    # Fields a later version adds are ignored by this one.
    assert ManSheet.from_json(json.dumps({**data, "later": 1}), s.base_seed) == s


@pytest.mark.parametrize("nation", [ENGLAND, JAPAN, NIGERIA, None])
def test_describe_names_the_look(nation: int | None) -> None:
    s = sample_man_sheet("c", 1, 1, nation=nation)
    assert s.describe().startswith(f"{s.look}, {s.skin} skin")
