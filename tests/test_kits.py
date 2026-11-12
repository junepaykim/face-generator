from __future__ import annotations

from pathlib import Path

import pytest

from facegen.kits import COLOURS, Kit, KitError, load_kits, parse_kits
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
    kit, known = kits.kit_for(636, goalkeeper=False)
    assert known and kit.shirt == ("blue",) and not kit.goalkeeper
    kit, known = kits.kit_for(636, goalkeeper=True)
    assert known and kit.shirt == ("yellow",) and kit.goalkeeper
    kit, known = kits.kit_for(999, goalkeeper=False)
    assert not known and kit.shirt == ("white",)
    kit, known = kits.kit_for(None, goalkeeper=True)
    assert not known and kit.goalkeeper


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
    assert set(kits.clubs) == league_two
    colchester, known = kits.kit_for(636, goalkeeper=False)
    assert known and colchester.pattern == "stripes" and set(colchester.shirt) == {"blue", "white"}
    for club in kits.clubs.values():
        assert club.goalkeeper.goalkeeper and not club.home.goalkeeper
        assert club.home.words() and club.goalkeeper.words()
