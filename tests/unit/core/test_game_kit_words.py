"""FM27's kits in Kitshot's vocabulary: colour names for the database's colours, the shirt
styles as patterns, and the words each new pattern puts in both models' prompts. Kit words may
only add colours and garments to a prompt (content rule): every pattern is checked for it."""

from __future__ import annotations

import pytest

from kitshot.core.kit_values import (
    PATTERNS,
    SHIRT_STYLES,
    GameShirt,
    Kit,
    KitError,
    colour_name,
    game_home_kit,
    game_kits,
    parse_kit,
)
from kitshot.core.man_prompt import jacket_sentence, shirt_sentence
from kitshot.core.prompt import CONTENT_NEGATIVE


def rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def shirt(style: int, slot1: str, body: str, slot3: str = "FFFFFF") -> GameShirt:
    return GameShirt(style, (rgb(slot1), rgb(body), rgb(slot3)))


# FM27's own colours for the clubs research.md lists, with the names they must get.
NAMES = {
    "C50021": "red", "A52142": "red", "CE3131": "red", "E62131": "red", "941929": "red", "BD1952": "red", "EF0021": "red",
    "5A2131": "claret", "632142": "claret", "732129": "claret",
    "EF5A8C": "pink", "D6ADB5": "pink", "E64294": "pink",
    "FFFFFF": "white", "F7EFE6": "white", "F7FFEF": "white", "FFFFB5": "white",
    "080808": "black", "211919": "black", "101010": "black", "313131": "black",
    "52525A": "grey", "9CA5AD": "grey", "7B7B7B": "grey",
    "84A5E6": "sky", "6BADDE": "sky", "84B5F7": "sky", "009CE6": "sky", "2184A5": "sky",
    "529CEF": "sky", "21A5C5": "sky",
    "2142A5": "blue", "0063A5": "blue", "29427B": "blue", "0042BD": "blue", "216BEF": "blue",
    "006BEF": "blue", "426BE6": "blue",
    "19316B": "navy", "29314A": "navy", "313A5A": "navy", "101931": "navy", "314273": "navy",
    "FFE600": "yellow", "FFEF00": "yellow", "E6C521": "yellow",
    "E6AD29": "amber", "E6A500": "amber", "FFB510": "amber",
    "DE7B29": "orange", "FF4A00": "orange", "FF7300": "orange",
    "00633A": "green", "21423A": "green", "29EF00": "green", "194231": "green",
    "63318C": "purple", "944A9C": "purple",
}  # fmt: skip


@pytest.mark.parametrize(("hex_colour", "name"), NAMES.items())
def test_fm27s_colours_get_their_names(hex_colour: str, name: str) -> None:
    assert colour_name(rgb(hex_colour)) == name


def test_the_body_is_slot_2_and_the_pattern_slot_1() -> None:
    inter = game_home_kit(shirt(3, "216BEF", "080808", "080808"))
    assert inter == Kit(shirt=("black", "blue"), pattern="stripes")
    arsenal = game_home_kit(shirt(2, "FFFFFF", "C50021"))
    assert arsenal == Kit(shirt=("red", "white"), pattern="sleeves")


def test_plain_styles_keep_slot_1_as_the_collar() -> None:
    liverpool = game_home_kit(shirt(1, "FFFFFF", "A52142"))
    assert liverpool == Kit(shirt=("red",), trim="white")
    real = game_home_kit(shirt(25, "EF5A8C", "FFFFFF", "21423A"))
    assert real == Kit(shirt=("white",), trim="pink")
    plain = game_home_kit(shirt(25, "C50021", "C50021"))
    assert plain == Kit(shirt=("red",))
    wolves = game_home_kit(shirt(1, "E6AD29", "DE7B29", "080808"))  # amber on orange
    assert wolves == Kit(shirt=("orange",))


def test_a_pattern_in_the_bodys_own_colour_is_a_plain_shirt() -> None:
    """Wolfsburg's thin lime lines on dark green: both are "green", so nothing shows."""
    wolfsburg = game_home_kit(shirt(12, "29EF00", "194231", "29EF00"))
    assert wolfsburg == Kit(shirt=("green",))


def test_two_pattern_colours_where_the_style_has_them() -> None:
    barcelona = game_home_kit(shirt(24, "BD1952", "19316B", "0042BD"))
    assert barcelona == Kit(shirt=("navy", "red", "blue"), pattern="stripes")
    palace = game_home_kit(shirt(38, "00319C", "FFFFFF", "E61908"))
    assert palace == Kit(shirt=("white", "blue", "red"), pattern="sash")
    sao_paulo = game_home_kit(shirt(30, "E60000", "FFFFFF", "080808"))
    assert sao_paulo == Kit(shirt=("white", "red", "black"), pattern="band")
    one_left = game_home_kit(shirt(38, "FFFFFF", "C50021", "C50021"))  # slot 3 = the body
    assert one_left == Kit(shirt=("red", "white"), pattern="sash")


def test_an_unknown_style_is_two_colours_without_a_layout() -> None:
    assert 34 not in SHIRT_STYLES
    assert game_home_kit(shirt(34, "E60000", "FFFFFF")) == Kit(
        shirt=("white", "red"), pattern="mixed"
    )
    assert game_home_kit(shirt(34, "FFFFFF", "FFFFFF", "E60000")) == Kit(shirt=("white",))


def test_every_known_style_is_a_pattern_of_the_vocabulary() -> None:
    for style, (pattern, slots) in SHIRT_STYLES.items():
        assert pattern in PATTERNS, style
        assert 1 + len(slots) <= PATTERNS[pattern][1] or pattern == "plain", style
        assert set(slots) <= {1, 3}, style


def test_the_goalkeeper_wears_the_body_of_fm27s_keeper_shirt() -> None:
    kits = game_kits(shirt(2, "FFFFFF", "C50021"), shirt(25, "FFFFFF", "FF7300", "FF7300"))
    assert kits.goalkeeper == Kit(shirt=("orange",), goalkeeper=True)
    no_keeper = game_kits(shirt(2, "FFFFFF", "C50021"), None)
    assert no_keeper.goalkeeper == Kit(shirt=("green",), goalkeeper=True)  # unlike red


NEW = {
    "stripes": (("navy", "claret", "blue"),
                "navy blue, dark red and blue vertical-striped shirt",
                "a football shirt with navy blue, dark red and blue vertical stripes"),
    "pinstripes": (("blue", "white"), "blue and white pinstripe shirt",
                   "a blue football shirt with thin white pinstripes"),
    "thin-hoops": (("white", "blue"), "white shirt, thin blue horizontal stripes",
                   "a white football shirt with thin blue horizontal stripes"),
    "checks": (("red", "white"), "red and white checkered shirt",
               "a football shirt with a red and white checkered pattern"),
    "centre-stripe": (("white", "red"), "white shirt, wide red vertical stripe down the middle",
                      "a white football shirt with a wide red vertical stripe down the middle"),
    "sash": (("white", "blue", "red"), "white shirt, blue and red diagonal sash",
             "a white football shirt with a blue and red diagonal sash"),
    "band": (("white", "red"), "white shirt, red band across the chest",
             "a white football shirt with a red band across the chest"),
    "chevron": (("navy", "white"), "navy blue shirt, white V-shaped band across the chest",
                "a navy blue football shirt with a white V-shaped band across the chest"),
    "cross": (("white", "black"), "white shirt, black cross on the chest",
              "a white football shirt with a black cross on the chest"),
}  # fmt: skip


@pytest.mark.parametrize("pattern", NEW)
def test_new_pattern_words(pattern: str) -> None:
    colours, tags, sentence = NEW[pattern]
    kit = Kit(shirt=colours, pattern=pattern)
    assert kit.shirt_words() == tags
    assert shirt_sentence(kit) == sentence
    for word in tags.replace(",", " ").split():
        assert word not in CONTENT_NEGATIVE.split(", ")


def test_three_colour_jackets_and_descriptions() -> None:
    kit = Kit(shirt=("navy", "claret", "blue"), pattern="stripes", trim="white")
    assert kit.jacket_words() == "navy blue, dark red and blue track jacket, white collar"
    assert jacket_sentence(kit) == (
        "a navy blue, dark red and blue club track jacket with a white collar"
    )
    assert kit.describe() == "navy, claret and blue stripes"


def test_the_table_may_use_the_new_patterns() -> None:
    sash = parse_kit(
        "t", {"shirt": ["white", "red"], "pattern": "sash", "shorts": "white"}, goalkeeper=False
    )
    assert sash.pattern == "sash"
    three = parse_kit(
        "t", {"shirt": ["green", "claret", "white"], "pattern": "stripes", "shorts": "white"},
        goalkeeper=False,
    )  # fmt: skip
    assert three.shirt == ("green", "claret", "white")
    for bad in (
        {"shirt": ["white", "red", "black"], "pattern": "hoops", "shorts": "white"},
        {"shirt": ["white", "white"], "pattern": "stripes", "shorts": "white"},
        {"shirt": ["white"], "pattern": "cross", "shorts": "white"},
    ):
        with pytest.raises(KitError):
            parse_kit("t", bad, goalkeeper=False)
