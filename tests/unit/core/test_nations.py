"""The nation table: the game's database IDs to countries and look regions (Phase 15f)."""

from __future__ import annotations

from collections import Counter

from kitshot.core.man_traits import LOOKS, REGIONS, WORLD
from kitshot.core.nations import NATIONS, nation_name, region_of


def test_every_nation_has_a_region_with_a_mix_of_known_looks() -> None:
    assert set(REGIONS) == {n.region for n in NATIONS.values()}  # none unused, none missing
    for name, mix in {**REGIONS, "world": WORLD}.items():
        assert set(mix) <= set(LOOKS), name
        assert sum(mix.values()) == 100, name  # percentages
        assert all(weight > 0 for weight in mix.values()), name


def test_every_look_can_be_drawn() -> None:
    assert {look for mix in REGIONS.values() for look in mix} == set(LOOKS)


def test_one_name_per_country() -> None:
    names = Counter(n.name for n in NATIONS.values())
    assert [name for name, count in names.items() if count > 1] == []
    assert len(NATIONS) == 228  # every nation in the three saves read, but one ID with no clubs


def test_nations_checked_against_known_players() -> None:
    """Some of the 38 players whose nationality was checked in a save (research section 21)."""
    checked = {
        765: "England", 769: "France", 771: "Germany", 796: "Spain", 776: "Italy",
        1651: "Brazil", 1649: "Argentina", 116: "Japan", 135: "South Korea", 38: "Nigeria",
        16: "Egypt", 390: "United States", 1435: "Australia", 377: "Jamaica",
    }  # fmt: skip
    assert {nation: nation_name(nation) for nation in checked} == checked


def test_a_nation_not_known_has_no_region() -> None:
    assert region_of(None) is None and nation_name(None) is None
    assert region_of(8_162_661) is None  # one player, no clubs: the world's mix
    assert region_of(765) == "england" and region_of(778) == "western_europe"
