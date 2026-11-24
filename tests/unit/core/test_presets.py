"""The Basic settings page's presets: who gets a picture and picture quality (Phase 12g)."""

from __future__ import annotations

import pytest

from kitshot.core import presets

DEFAULTS = {
    "cutoffs.share.a_ca": 6.0, "cutoffs.share.a_pa": 3.0, "cutoffs.share.b_ca": 40.0,
    "cutoffs.share.b_pa": 20.0, "cutoffs.share.c_ca": 80.0, "cutoffs.share.c_pa": "-",
    "cutoffs.share.above_b_ca": 15.0, "cutoffs.share.above_c_ca": 50.0,
    "cutoffs.share.below_c_ca": 50.0, "cutoffs.share.continental_b_ca": 30.0,
    "cutoffs.manual.a_ca": "-", "cutoffs.manual.a_pa": "-", "cutoffs.manual.b_ca": 131,
    "cutoffs.manual.b_pa": 151, "cutoffs.manual.c_ca": 116, "cutoffs.manual.c_pa": 141,
    "cutoffs.everywhere.a_ca": 155, "cutoffs.everywhere.a_pa": 170,
    "cutoffs.everywhere.b_ca": 140, "cutoffs.everywhere.b_pa": 155,
    "cutoffs.everywhere.c_ca": "-", "cutoffs.everywhere.c_pa": "-",
}  # fmt: skip


def test_reach_scales_every_share_and_moves_every_number_together() -> None:
    options = presets.options("reach", DEFAULTS)
    assert options["default"] == DEFAULTS
    fewer, more = options["fewer"], options["more"]
    assert fewer["cutoffs.share.a_ca"] == 3.0 and more["cutoffs.share.a_ca"] == 9.0
    assert fewer["cutoffs.share.above_b_ca"] == 7.5 and more["cutoffs.share.a_pa"] == 4.5
    assert more["cutoffs.share.c_ca"] == 100.0  # 120% capped
    assert fewer["cutoffs.everywhere.a_ca"] == 165 and more["cutoffs.everywhere.a_ca"] == 145
    assert more["cutoffs.manual.b_ca"] == 121
    assert set(fewer) == set(presets.REACH_KEYS) == set(DEFAULTS)


def test_a_switched_off_cell_stays_off() -> None:
    for name, values in presets.options("reach", DEFAULTS).items():
        assert values["cutoffs.share.c_pa"] == "-", name
        assert values["cutoffs.everywhere.c_ca"] == "-", name


def test_reach_stays_within_the_ranges() -> None:
    edge = dict(DEFAULTS, **{"cutoffs.everywhere.a_pa": 195, "cutoffs.everywhere.b_ca": 4})
    edge |= {"cutoffs.share.a_ca": 0.1}
    options = presets.options("reach", edge)
    assert options["fewer"]["cutoffs.everywhere.a_pa"] == 200
    assert options["more"]["cutoffs.everywhere.b_ca"] == 1
    assert options["fewer"]["cutoffs.share.a_ca"] == 0.1


def test_the_current_preset_or_none_when_set_by_hand() -> None:
    values = dict(DEFAULTS) | presets.QUALITY["medium"]
    assert presets.current("reach", values, DEFAULTS) == "default"
    assert presets.current("quality", values, DEFAULTS) == "medium"
    more = values | presets.values_of("reach", "more", DEFAULTS)
    assert presets.current("reach", more, DEFAULTS) == "more"
    assert presets.current("reach", more | {"cutoffs.everywhere.b_ca": 1}, DEFAULTS) is None
    assert presets.current("reach", values | {"cutoffs.share.a_ca": 6}, DEFAULTS) == "default"
    assert presets.current("quality", values | {"groups.a.candidates": 5}, DEFAULTS) is None


def test_unknown_kinds_and_names_are_refused() -> None:
    with pytest.raises(KeyError):
        presets.values_of("colour", "red", DEFAULTS)
    with pytest.raises(KeyError):
        presets.values_of("quality", "ultra", DEFAULTS)


def test_history_labels_name_the_preset() -> None:
    assert presets.label("reach", "more") == "preset.reach.more"
    assert presets.parse_label("preset.quality.low") == ("quality", "low")
    for other in ("edit", "defaults", "auto-configure", "preset.colour.red", "preset.reach"):
        assert presets.parse_label(other) is None
