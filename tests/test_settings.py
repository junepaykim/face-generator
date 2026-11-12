from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from facegen.settings import (
    PROJECT_ROOT,
    SettingsError,
    build_settings,
    check_override,
    flatten,
    load_defaults,
    load_settings,
    reset_overrides,
    set_override,
)


def test_defaults_load_into_typed_settings() -> None:
    s = load_settings()
    assert s.groups.a.ca_above == 155
    assert s.groups.b.my_academy is True
    assert s.groups.c.published_long_edge == 512
    assert s.age.b == ((15, "W02"), (19, "W04"), (25, "W07"), (29, "W09"))
    assert (s.reference.enabled, s.reference.scale, s.reference.other_clubs) == (True, 0.1, False)
    assert s.my_club.banned_traits == frozenset({"freckles", "muscular"})
    assert (s.body.short_below_cm, s.body.tall_from_cm) == (174, 191)
    assert (s.body.slim_strength_at_most, s.body.muscular_strength_from) == (4, 14)
    assert s.my_club.soften_skin == 2
    assert s.age.a_my_club[0] == (15, "W01") and s.age.a_my_club[-1] == (34, "W10")
    assert s.eligibility.mens_football_only is True
    assert s.eligibility.continental_competition_ids == frozenset()
    assert s.overridden == frozenset()


def test_relative_data_dir_resolves_against_the_project() -> None:
    s = load_settings()
    assert s.paths.data_dir == PROJECT_ROOT / "data"
    assert s.paths.db_path == PROJECT_ROOT / "data" / "facegen.db"
    assert s.paths.games_dir.name == "games"


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": [2]}}, "e": True}) == {
        "a.b": 1,
        "a.c.d": [2],
        "e": True,
    }


def test_override_changes_one_value(conn: sqlite3.Connection) -> None:
    result = set_override(conn, "groups.a.ca_above", 150)
    assert result.groups.a.ca_above == 150
    again = load_settings(conn)
    assert again.groups.a.ca_above == 150
    assert again.groups.a.pa_above == 170
    assert again.overridden == frozenset({"groups.a.ca_above"})


def test_override_of_a_list(conn: sqlite3.Connection) -> None:
    set_override(conn, "eligibility.continental_competition_ids", [1301385, 1301388])
    assert load_settings(conn).eligibility.continental_competition_ids == {1301385, 1301388}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("groups.a.nonsense", 1),  # unknown
        ("groups.a.ca_above", "150"),  # string for int
        ("groups.a.ca_above", True),  # bool is not an int here
        ("groups.a.detail_pass", 1),  # int for bool
        ("eligibility.continental_competition_ids", [1, "2"]),
        ("paths.data_dir", "/tmp/elsewhere"),  # paths are fixed
    ],
)
def test_bad_overrides_are_refused(conn: sqlite3.Connection, key: str, value: object) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, key, value)
    assert conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("groups.a.ca_above", 201),
        ("groups.b.candidates", 0),
        ("groups.c.published_long_edge", 10),
        ("age.b.W04", 15),  # first ages must rise from W02 (15) to W04
        ("age.b.W07", 30),  # past W09 (29)
        ("age.a_my_club.W01", -1),
        ("age.c.W02", "15"),  # must be a whole number
        ("reference.scale", 0),
        ("reference.scale", 1.5),
        ("my_club.soften_skin", 3),
        ("my_club.banned_traits", [1, 2]),  # text, not numbers
        ("my_club.banned_traits", ["Freckles"]),  # lower case
        ("ui.port", 0),
    ],
)
def test_out_of_range_overrides_are_refused(
    conn: sqlite3.Connection, key: str, value: object
) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, key, value)
    assert load_settings(conn).overridden == frozenset()


def test_stale_stored_override_is_skipped_with_a_warning(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, 'now')",
        ("groups.z.removed", json.dumps(3)),
    )
    with pytest.warns(UserWarning, match="ignoring stored setting"):
        s = load_settings(conn)
    assert s.overridden == frozenset()


def test_reset_overrides(conn: sqlite3.Connection) -> None:
    set_override(conn, "groups.a.ca_above", 150)
    set_override(conn, "groups.a.pa_above", 160)
    reset_overrides(conn, ["groups.a.ca_above"])
    assert load_settings(conn).overridden == frozenset({"groups.a.pa_above"})
    reset_overrides(conn)
    assert load_settings(conn).groups.a.pa_above == 170


def test_check_override_returns_the_value() -> None:
    defaults = load_defaults()
    assert check_override("schedule.poll_seconds", 30, defaults) == 30


def test_build_settings_with_custom_defaults_file(tmp_path: Path) -> None:
    text = (PROJECT_ROOT / "config" / "default.toml").read_text()
    custom = tmp_path / "custom.toml"
    custom.write_text(text.replace('data_dir = "data"', f'data_dir = "{tmp_path.as_posix()}/d"'))
    s = load_settings(defaults_path=custom)
    assert s.paths.data_dir == tmp_path / "d"
    assert build_settings(dict(s.values)).paths.data_dir == tmp_path / "d"


def test_the_content_rule_is_not_a_setting() -> None:
    keys = " ".join(load_defaults())
    for word in ("nsfw", "content", "rating", "negative", "wardrobe", "safe"):
        assert word not in keys


def test_text_list_override(conn: sqlite3.Connection) -> None:
    s = set_override(conn, "my_club.banned_traits", ["freckles", "mole"])
    assert s.my_club.banned_traits == frozenset({"freckles", "mole"})
    s = set_override(conn, "my_club.banned_traits", [])
    assert s.my_club.banned_traits == frozenset()


def test_reference_can_be_switched_off(conn: sqlite3.Connection) -> None:
    s = set_override(conn, "reference.enabled", False)
    assert s.reference.enabled is False
    assert set_override(conn, "reference.scale", 0.2).reference.scale == 0.2


def test_banned_traits_must_be_sheet_traits(conn: sqlite3.Connection) -> None:
    """Free text in the negative prompt could weaken the content rule (AGENTS.md rule 3)."""
    for bad in (["shirt"], ["freckles", "clothes"]):
        with pytest.raises(SettingsError):
            set_override(conn, "my_club.banned_traits", bad)
    s = set_override(conn, "my_club.banned_traits", ["glasses", "mole"])
    assert s.my_club.banned_traits == frozenset({"glasses", "mole"})


def test_body_thresholds_are_checked(conn: sqlite3.Connection) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, "body.short_below_cm", 200)  # above tall_from_cm
    with pytest.raises(SettingsError):
        set_override(conn, "body.slim_strength_at_most", 14)  # not below muscular
    s = set_override(conn, "body.tall_from_cm", 193)
    assert s.body.tall_from_cm == 193


def test_several_overrides_are_checked_together(conn: sqlite3.Connection) -> None:
    from facegen.settings import set_overrides

    # Each alone would put short_below_cm above tall_from_cm; together they are fine.
    s = set_overrides(conn, {"body.short_below_cm": 195, "body.tall_from_cm": 200})
    assert (s.body.short_below_cm, s.body.tall_from_cm) == (195, 200)
    with pytest.raises(SettingsError):
        set_overrides(conn, {"groups.a.ca_above": 150, "groups.b.candidates": 0})
    # Nothing of a refused batch is stored.
    assert load_settings(conn).groups.a.ca_above == 155


def test_entering_the_default_again_clears_the_override(conn: sqlite3.Connection) -> None:
    from facegen.settings import merge_overrides, set_overrides

    set_overrides(conn, {"groups.a.ca_above": 150, "groups.a.pa_above": 160})
    assert merge_overrides(conn, {"groups.a.ca_above": 155}).overridden == {"groups.a.pa_above"}
    s = set_overrides(conn, {"groups.a.ca_above": 155})
    assert s.overridden == frozenset({"groups.a.pa_above"})
    assert load_settings(conn).overridden == frozenset({"groups.a.pa_above"})


def test_where_the_ui_listens_is_not_a_ui_setting(conn: sqlite3.Connection) -> None:
    """A browser must not be able to open the tool to the network."""
    from facegen.settings import is_fixed

    for key, value in (("ui.host", "0.0.0.0"), ("ui.port", 9000)):
        assert is_fixed(key)
        with pytest.raises(SettingsError, match="defaults file"):
            set_override(conn, key, value)


def test_published_sizes() -> None:
    assert load_settings().groups.published_sizes() == {"a": 1024, "b": 768, "c": 512}


def test_auto_groups(conn: sqlite3.Connection) -> None:
    # Phase 4 switched groups B and C on.
    assert load_settings().schedule.auto_groups == frozenset({"a", "b", "c"})
    assert set_override(conn, "schedule.auto_groups", ["a"]).schedule.auto_groups == {"a"}
    assert set_override(conn, "schedule.auto_groups", []).schedule.auto_groups == frozenset()
    with pytest.raises(SettingsError):
        set_override(conn, "schedule.auto_groups", ["d"])
