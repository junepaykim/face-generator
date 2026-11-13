from __future__ import annotations

import json
from pathlib import Path

import pytest

from facegen.kits import (
    COLOURS,
    KEEPER_COLOURS,
    PATTERNS,
    Kit,
    KitError,
    catalog_kit,
    keeper_colour,
    load_kits,
    parse_catalog,
    parse_kits,
)
from facegen.prompt import CONTENT_NEGATIVE

ROOT = Path(__file__).resolve().parents[1]


def table(**clubs: dict) -> dict:
    return {
        "fallback": {"home": {"shirt": ["white"], "shorts": "black"}, "goalkeeper": {"shirt": ["green"]}},
        "clubs": clubs,
    }  # fmt: skip


def test_kit_words_per_pattern() -> None:
    def home(**kw: object) -> str:
        return Kit(**kw).words()  # type: ignore[arg-type]

    assert home(shirt=("red",), shorts="white") == "red shirt, short sleeves, white shorts"
    assert (
        home(shirt=("blue", "white"), pattern="stripes", shorts="blue")
        == "blue and white vertical-striped shirt, short sleeves, blue shorts"
    )
    assert (
        home(shirt=("yellow", "green"), pattern="hoops", shorts="green")
        == "yellow and green horizontal-striped shirt, short sleeves, green shorts"
    )
    assert (
        home(shirt=("claret", "sky"), pattern="sleeves", shorts="white")
        == "dark red shirt, light blue sleeves, short sleeves, white shorts"
    )
    assert (
        home(shirt=("white",), trim="navy", shorts="navy")
        == "white shirt, navy blue collar, short sleeves, navy blue shorts"
    )


def test_goalkeeper_kit_words_match_the_bake_off() -> None:
    gk = Kit(shirt=("green",), goalkeeper=True)
    assert gk.words() == "goalkeeper, green shirt, long sleeves, black gloves"
    gk = Kit(shirt=("purple",), gloves="grey", goalkeeper=True)
    assert gk.words() == "goalkeeper, purple shirt, long sleeves, grey gloves"


def test_unknown_club_gets_the_fallback() -> None:
    kits = parse_kits(
        table(**{"636": {"home": {"shirt": ["blue"], "shorts": "blue"},
                         "goalkeeper": {"shirt": ["yellow"]}}})
    )  # fmt: skip
    kit, source = kits.kit_for(636, goalkeeper=False)
    assert source == "table" and kit.shirt == ("blue",) and not kit.goalkeeper
    kit, source = kits.kit_for(636, goalkeeper=True)
    assert source == "table" and kit.shirt == ("yellow",) and kit.goalkeeper
    kit, source = kits.kit_for(999, goalkeeper=False)
    assert source == "fallback" and kit.shirt == ("white",)
    kit, source = kits.kit_for(None, goalkeeper=True)
    assert source == "fallback" and kit.goalkeeper


@pytest.mark.parametrize(
    "bad",
    [
        {"home": {"shirt": ["gold"], "shorts": "black"}, "goalkeeper": {"shirt": ["green"]}},
        {"home": {"shirt": ["red"]}, "goalkeeper": {"shirt": ["green"]}},  # no shorts
        {"home": {"shirt": ["red"], "pattern": "stripes", "shorts": "red"},
         "goalkeeper": {"shirt": ["green"]}},  # stripes need two colours
        {"home": {"shirt": ["red", "white"], "shorts": "red"},
         "goalkeeper": {"shirt": ["green"]}},  # plain takes one
        {"home": {"shirt": ["red"], "shorts": "red", "words": "anything"},
         "goalkeeper": {"shirt": ["green"]}},  # no free text
        {"home": {"shirt": ["red"], "shorts": "red"}},  # no goalkeeper kit
    ],
)  # fmt: skip
def test_bad_entries_are_refused(bad: dict) -> None:
    with pytest.raises(KitError):
        parse_kits(table(**{"636": bad}))


def test_club_keys_are_ids() -> None:
    entry = {"home": {"shirt": ["red"], "shorts": "red"}, "goalkeeper": {"shirt": ["green"]}}
    with pytest.raises(KitError):
        parse_kits(table(colchester=entry))
    with pytest.raises(KitError):
        parse_kits({"clubs": {}})  # no fallback


def test_colour_words_are_plain_colours() -> None:
    """Kits can only add colour and garment words to a prompt (content rule)."""
    for colour in COLOURS.values():
        assert all(part.isalpha() for part in colour.words.split()), colour
        assert colour.words not in CONTENT_NEGATIVE


def test_the_shipped_table_loads_and_covers_my_league() -> None:
    kits = load_kits(ROOT / "config" / "kits.toml")
    league_two = {636, 109210, 605, 620, 631, 634, 640, 641, 651, 5103834, 656, 658, 581, 689,
                  694, 698, 703, 704, 5103842, 710, 725, 729, 731, 744}  # fmt: skip
    assert league_two <= set(kits.clubs)
    colchester, source = kits.kit_for(636, goalkeeper=False)
    assert source == "table"
    assert colchester.pattern == "stripes" and set(colchester.shirt) == {"blue", "white"}
    for club in kits.clubs.values():
        assert club.goalkeeper.goalkeeper and not club.home.goalkeeper
        assert club.home.words() and club.goalkeeper.words()


def test_rank_tags_name_each_shirt_colour() -> None:
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    assert kit.rank_tags() == {
        "shirt colour 1": frozenset({"blue shirt"}),
        "shirt colour 2": frozenset({"white shirt"}),
    }


# --- the kit catalog (colours estimated from FM27's kit images) --------------------------------


def estimate(club_id: int, primary: str | None, secondary: str | None = None, **row: object):
    def colour(name: str | None) -> dict | None:
        return None if name is None else {"name": name, "hex": "#000000", "rgb": [0, 0, 0]}

    return {
        "club_id": club_id,
        "kind": "home",
        "status": "estimated",
        "primary": colour(primary),
        "secondary": colour(secondary),
        **row,
    }


REPORT = {
    "summary": {},
    "results": [
        estimate(602, "red", "white"),
        estimate(679, "sky"),
        estimate(1569, "white", "green"),
        estimate(688, "black", "white", kind="away"),  # only home kits are used
        {"key": "kit2d_1708_home", "club_id": 1708, "kind": "home", "status": "not_in_catalog",
         "primary": None, "secondary": None},
        estimate(7, "gold"),  # a name outside the vocabulary: unusable, never guessed
        estimate(8, "blue", "blue"),  # the same name twice is one colour
    ],
}  # fmt: skip


def test_catalog_rows_become_kits_without_a_pattern() -> None:
    clubs = parse_catalog(REPORT)
    assert set(clubs) == {602, 679, 1569, 8}
    arsenal = clubs[602].home
    # Two colours, no layout: the probe does not say whether they are stripes or sleeves.
    assert arsenal.shirt == ("red", "white") and arsenal.pattern == "mixed"
    assert arsenal.words() == "red and white shirt, short sleeves"  # shorts unknown
    assert arsenal.describe() == "red and white"
    assert clubs[679].home == Kit(shirt=("sky",))
    assert clubs[8].home == Kit(shirt=("blue",))
    for club in clubs.values():
        assert club.goalkeeper.goalkeeper and len(club.goalkeeper.shirt) == 1


def test_catalog_skips_what_it_cannot_use() -> None:
    assert catalog_kit({"status": "not_in_catalog", "primary": None}) is None
    assert catalog_kit({"status": "insufficient_shirt_pixels", "primary": None}) is None
    assert catalog_kit(estimate(1, None)) is None
    with pytest.raises(KitError):
        parse_catalog({"summary": {}})


def test_catalog_goalkeepers_contrast_with_the_home_shirt() -> None:
    assert keeper_colour(Kit(shirt=("red",))) == "green"
    assert keeper_colour(Kit(shirt=("white", "green"), pattern="mixed")) == "yellow"
    assert keeper_colour(Kit(shirt=("green", "yellow"), pattern="mixed")) == "orange"
    # Amber is both orange and yellow.
    assert keeper_colour(Kit(shirt=("green", "amber"), pattern="mixed")) == "purple"
    assert set(KEEPER_COLOURS) <= set(COLOURS)


def test_the_table_comes_before_the_catalog(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"results": [estimate(636, "red"), estimate(602, "red")]}))
    kits = load_kits(ROOT / "config" / "kits.toml", catalog=report)
    colchester, source = kits.kit_for(636, goalkeeper=False)
    assert source == "table" and colchester.pattern == "stripes"
    _, source = kits.kit_for(602, goalkeeper=False)
    assert source == "table"  # hand-written: verified against FM27's image
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"results": [estimate(679, "sky")]}))
    city, source = load_kits(catalog=other).kit_for(679, goalkeeper=False)
    assert source == "catalog" and city.shirt == ("sky",)
    keeper, source = load_kits(catalog=other).kit_for(679, goalkeeper=True)
    assert source == "catalog" and keeper.goalkeeper and keeper.shirt == ("green",)
    _, source = load_kits(catalog=other).kit_for(1708, goalkeeper=False)
    assert source == "fallback"


def test_no_catalog_file_means_no_catalog(tmp_path: Path) -> None:
    kits = load_kits(catalog=tmp_path / "missing.json")
    assert kits.catalog == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(KitError):
        load_kits(catalog=broken)


def test_mixed_is_allowed_in_the_table_too() -> None:
    assert PATTERNS["mixed"]
    kits = parse_kits(
        table(**{"992": {"home": {"shirt": ["red", "white"], "pattern": "mixed",
                                  "shorts": "white"},
                         "goalkeeper": {"shirt": ["green"]}}})
    )  # fmt: skip
    assert kits.clubs[992].home.words() == "red and white shirt, short sleeves, white shorts"


def test_shirt_and_jacket_words() -> None:
    kit = Kit(shirt=("red", "white"), pattern="sleeves", shorts="white", trim="black")
    assert kit.shirt_words() == "red shirt, white sleeves, black collar"
    assert kit.jacket_words() == "red track jacket, white sleeves, black collar"
    assert kit.shirt_tag_groups("jacket") == (
        frozenset({"red jacket"}),
        frozenset({"white jacket"}),
    )
    assert kit.colours() == ("red", "white", "black")
