"""The Basic settings page's presets: who gets a picture and picture quality (Phase 12g)."""

from __future__ import annotations

import pytest

from facegen.core import presets

DEFAULTS = {
    "groups.a.ca_above": 155, "groups.a.pa_above": 170,
    "groups.b.league_ca_above": 130, "groups.b.league_pa_above": 150,
    "groups.b.any_ca_above": 140, "groups.b.any_pa_above": 155,
    "groups.c.league_ca_above": 115, "groups.c.league_pa_above": 140,
}  # fmt: skip


def test_reach_moves_every_threshold_together_from_its_default() -> None:
    options = presets.options("reach", DEFAULTS)
    assert options["default"] == DEFAULTS
    assert options["fewer"]["groups.a.ca_above"] == 165
    assert options["more"]["groups.c.league_pa_above"] == 130
    assert all(options["more"][k] == v - 10 for k, v in DEFAULTS.items())
    assert set(options["fewer"]) == set(presets.REACH_KEYS)


def test_reach_stays_within_the_thresholds_range() -> None:
    edge = dict(DEFAULTS, **{"groups.a.pa_above": 195, "groups.c.league_ca_above": 4})
    options = presets.options("reach", edge)
    assert options["fewer"]["groups.a.pa_above"] == 200
    assert options["more"]["groups.c.league_ca_above"] == 0


def test_the_current_preset_or_none_when_set_by_hand() -> None:
    values = dict(DEFAULTS) | presets.QUALITY["medium"]
    assert presets.current("reach", values, DEFAULTS) == "default"
    assert presets.current("quality", values, DEFAULTS) == "medium"
    more = values | presets.values_of("reach", "more", DEFAULTS)
    assert presets.current("reach", more, DEFAULTS) == "more"
    assert presets.current("reach", more | {"groups.b.any_ca_above": 1}, DEFAULTS) is None
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
