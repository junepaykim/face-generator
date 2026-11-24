from __future__ import annotations

from dataclasses import replace

import pytest

from facegen.core.age import TABLES, WORDINGS, age_wording, audience, drawn_as_man, table_name
from facegen.core.classify import Group
from facegen.persistence.settings import Settings

# docs/spec.md section 3: real age -> age wording, per table (user, 2026-11-11).
SPEC = {
    "a_my_club": {
        "W01": range(15, 17), "W02": range(17, 19), "W03": range(19, 22),
        "W04": range(22, 24), "W05": range(24, 26), "W06": range(26, 27),
        "W07": range(27, 29), "W08": range(29, 31), "W09": range(31, 34), "W10": range(34, 46),
    },
    "a_other_club": {
        "W02": range(15, 19), "W03": range(19, 22), "W04": range(22, 25),
        "W07": range(25, 29), "W09": range(29, 34), "W10": range(34, 46),
    },
    "b": {"W02": range(15, 19), "W04": range(19, 25), "W07": range(25, 29), "W09": range(29, 46)},
    "c": {"W03": range(15, 22), "W06": range(22, 28), "W08": range(28, 46)},
}  # fmt: skip
TABLE_ARGS = {
    "a_my_club": (Group.A, True),
    "a_other_club": (Group.A, False),
    "b": (Group.B, False),
    "c": (Group.C, False),
}


@pytest.mark.parametrize("table", sorted(SPEC))
def test_wordings_match_the_spec_tables(settings: Settings, table: str) -> None:
    group, at_my_club = TABLE_ARGS[table]
    for wording, ages in SPEC[table].items():
        for real in ages:
            assert age_wording(real, group, at_my_club, settings.age) == wording, (table, real)


def test_every_player_at_my_club_takes_my_clubs_table(settings: Settings) -> None:
    """A youth team's group B or a pin there changes no wording (2026-11-13)."""
    for group in (Group.B, Group.C):
        for real in (16, 20, 26, 35):
            assert age_wording(real, group, True, settings.age) == age_wording(
                real, Group.A, True, settings.age
            )


@pytest.mark.parametrize(
    ("group", "at_my_club", "expected"),
    [
        (Group.A, True, "W01"),
        (Group.A, False, "W02"),
        (Group.B, False, "W02"),
        (Group.C, False, "W03"),
    ],
)
def test_ages_below_the_first_row_use_it(
    settings: Settings, group: Group, at_my_club: bool, expected: str
) -> None:
    assert age_wording(13, group, at_my_club, settings.age) == expected


def test_unknown_age_or_group_has_no_wording(settings: Settings) -> None:
    assert age_wording(None, Group.A, True, settings.age) is None
    assert age_wording(20, None, True, settings.age) is None


def test_table_names() -> None:
    assert table_name(Group.A, True) == "a_my_club"
    assert table_name(Group.A, False) == "a_other_club"
    assert table_name(Group.B, False) == "b"
    assert table_name(Group.B, True) == "a_my_club"
    assert table_name("c", False) == "c"
    assert table_name(Group.B, True, staff=True) == "staff"
    assert table_name(None, True, staff=True) is None
    assert table_name(Group.B, False, media=True) == "media"
    assert table_name(Group.A, True, media=True) == "media"
    assert {
        table_name(g, m, staff, media=media)
        for g in Group
        for m in (True, False)
        for staff in (True, False)
        for media in (True, False)
    } == set(TABLES)


# My club's staff (user, 2026-11-12): very wide bands, never W01, most staff in W03-W07.
STAFF = {
    "W02": range(16, 25), "W03": range(25, 32), "W04": range(32, 39), "W05": range(39, 46),
    "W06": range(46, 53), "W07": range(53, 61), "W08": range(61, 68), "W09": range(68, 75),
    "W10": range(75, 95),
}  # fmt: skip


def _manual_staff(settings: Settings) -> Settings:
    """The settings with my club's staff aged by the manual table (Auto by default)."""
    return _manual(settings, "staff")


def _manual(settings: Settings, *tables: str) -> Settings:
    """The settings with these tables set look by look (Custom) instead of spread."""
    from facegen.persistence.settings import build_settings

    return build_settings(dict(settings.values) | {f"age.{t}.mode": "manual" for t in tables})


def test_staff_have_their_own_wide_bands_whatever_their_group(settings: Settings) -> None:
    manual = _manual_staff(settings)
    for group in Group:
        for wording, ages in STAFF.items():
            for real in ages:
                assert age_wording(real, group, True, manual.age, staff=True) == wording, real
    assert age_wording(14, Group.A, True, manual.age, staff=True) == "W02"  # never W01
    assert all(w != "W01" for _, w in manual.age.staff)


# The staff looks by default (user, 2026-11-22): W02 to W10 spread over the ages 28 to 64.
SPREAD = {
    "W02": range(14, 29), "W03": range(29, 34), "W04": range(34, 39), "W05": range(39, 44),
    "W06": range(44, 49), "W07": range(49, 54), "W08": range(54, 59), "W09": range(59, 64),
    "W10": range(64, 95),
}  # fmt: skip


def test_staff_are_spread_over_28_to_64_by_default(settings: Settings) -> None:
    assert settings.age.staff_spread is not None
    for group in Group:
        for wording, ages in SPREAD.items():
            for real in ages:
                assert age_wording(real, group, True, settings.age, staff=True) == wording, real


def test_staff_ages_up_to_100_are_allowed_in_the_tables(settings: Settings) -> None:
    from facegen.persistence.settings import build_settings

    values = dict(settings.values) | {"age.staff.mode": "manual"}
    values["age.staff.W10"] = 99
    assert build_settings(values).age.staff[-1] == (99, "W10")


def test_wordings_run_from_youngest_to_oldest() -> None:
    assert list(WORDINGS) == [f"W{n:02d}" for n in range(1, 11)]
    # The under-18 looks carry the content rule's extra negative terms; adults do not.
    assert [k for k, w in WORDINGS.items() if w.young] == [f"W0{n}" for n in range(1, 7)]


def test_the_words_each_look_draws_are_unchanged() -> None:
    """Phase 17l changes how the age tables are shown and edited, never what reaches the prompt
    (ground rule 2): each look's words, word for word. A change here is the user's to approve."""
    assert {key: (w.text, w.young) for key, w in WORDINGS.items()} == {
        "W01": ("child, young girl, youthful face, round face", True),
        "W02": ("aged down, young girl, youthful face, round face", True),
        "W03": ("young girl, youthful face, round face", True),
        "W04": ("young teenage girl, teenage, youthful face, round face", True),
        "W05": ("young teenage girl, teenage, youthful face", True),
        "W06": ("teenage girl, teenage", True),
        "W07": ("young adult woman", False),
        "W08": ("young woman, adult", False),
        "W09": ("adult woman, mature face", False),
        "W10": ("mature female, adult woman, mature face", False),
    }


def test_the_mens_wordings_are_gone_but_still_recognised() -> None:
    """Everyone is drawn as a woman since 2026-11-14; a stored M1 to M4 band shows a man."""
    assert not any(k.startswith("M") for k in WORDINGS)
    assert all(drawn_as_man(b) for b in ("M1", "M2", "M3", "M4"))
    assert not any(drawn_as_man(b) for b in ("W01", "W07", "W10", None))


# Journalists and press officers by default (user, 2026-11-22, Phase 17l): Auto, W04 to W10
# over 30-60, from the measured ages (journalists 27-63 in every save, docs/research.md).
MEDIA_SPREAD = {
    "W04": range(16, 31), "W05": range(31, 37), "W06": range(37, 43), "W07": range(43, 49),
    "W08": range(49, 55), "W09": range(55, 60), "W10": range(60, 95),
}  # fmt: skip
# Their Custom table (user, 2026-11-14): the staff bands, from W04 up.
MEDIA = {
    "W04": range(16, 39), "W05": range(39, 46), "W06": range(46, 53), "W07": range(53, 61),
    "W08": range(61, 68), "W09": range(68, 75), "W10": range(75, 95),
}  # fmt: skip


def test_journalists_and_press_officers_are_spread_over_30_to_60_by_default(
    settings: Settings,
) -> None:
    spread = settings.age.spreads["media"]
    assert (spread.first, spread.last, spread.start, spread.end) == ("W04", "W10", 30, 60)
    for wording, ages in MEDIA_SPREAD.items():
        for real in ages:
            assert age_wording(real, Group.B, False, settings.age, media=True) == wording, real
            # Pinned to A as a regular at my club: the same table.
            assert age_wording(real, Group.A, True, settings.age, media=True) == wording, real


def test_journalists_and_press_officers_are_band_4_or_older(settings: Settings) -> None:
    custom = _manual(settings, "media")
    for wording, ages in MEDIA.items():
        for real in ages:
            assert age_wording(real, Group.B, False, custom.age, media=True) == wording, real
            # Pinned to A as a regular at my club: the same table.
            assert age_wording(real, Group.A, False, custom.age, media=True) == wording, real
    assert age_wording(20, Group.B, False, custom.age, media=True) == "W04"


def test_media_bands_are_the_staff_bands_from_w04_up(settings: Settings) -> None:
    """The media table copies the staff's manual table, whatever the staff's mode."""
    staff = {w: a for a, w in _manual_staff(settings).age.staff}
    media = {w: a for a, w in _manual(settings, "media").age.media}
    assert list(media) == [f"W{n:02d}" for n in range(4, 11)]
    assert {w: a for w, a in media.items() if w != "W04"} == {
        w: a for w, a in staff.items() if w in media and w != "W04"
    }


def test_audience_reads_a_players_row() -> None:
    assert audience({"is_staff": 1, "media_role": None}) == {"staff": True, "media": False}
    row = {"is_staff": 0, "media_role": "journalist"}
    assert audience(row) == {"staff": False, "media": True}


def test_mens_wordings_are_refused_in_a_table(settings: Settings) -> None:
    from facegen.persistence.settings import SettingsError, build_settings

    values = dict(settings.values)
    values["age.staff.M2"] = 40
    with pytest.raises(SettingsError, match="unknown age wording"):
        build_settings(values)


def test_custom_table(settings: Settings) -> None:
    s = replace(settings.age, b=((15, "W03"), (30, "W10")))
    assert age_wording(29, Group.B, False, s) == "W03"
    assert age_wording(30, Group.B, False, s) == "W10"


# --- my club's staff spread over an age span (Phase 16f) ---------------------------------------


def test_the_staff_spread_of_the_request() -> None:
    from facegen.core.age import band_ages, spread_table

    rows = spread_table("W02", "W10", 28, 64)
    assert rows == (
        (0, "W02"), (29, "W03"), (34, "W04"), (39, "W05"), (44, "W06"),
        (49, "W07"), (54, "W08"), (59, "W09"), (64, "W10"),
    )  # fmt: skip
    ages = band_ages(rows)
    assert ages[0] == ("W02", None, 28)  # 28 and under
    assert ages[1] == ("W03", 29, 33)
    assert ages[-2] == ("W09", 59, 63)
    assert ages[-1] == ("W10", 64, None)  # 64 and over


def test_a_leftover_year_goes_to_the_younger_looks() -> None:
    from facegen.core.age import spread_table

    # 3 looks between over 30-40: 9 years, 3 each; over 30-41: 10 years, 4, 3, 3.
    assert spread_table("W04", "W08", 30, 40) == (
        (0, "W04"), (31, "W05"), (34, "W06"), (37, "W07"), (40, "W08"),
    )  # fmt: skip
    assert spread_table("W04", "W08", 30, 41) == (
        (0, "W04"), (31, "W05"), (35, "W06"), (38, "W07"), (41, "W08"),
    )  # fmt: skip


def test_spreads_with_one_or_two_looks() -> None:
    from facegen.core.age import spread_table

    assert spread_table("W05", "W05", 20, 60) == ((0, "W05"),)
    assert spread_table("W05", "W06", 20, 60) == ((0, "W05"), (60, "W06"))


@pytest.mark.parametrize(
    ("first", "last", "start", "end"),
    [
        ("W10", "W02", 28, 64),  # looks out of order
        ("W02", "W10", 64, 28),  # ages out of order
        ("W02", "W10", 28, 28),
        ("W02", "W10", 28, 35),  # 6 years between for 7 looks
        ("W02", "M2", 28, 64),
    ],
)
def test_a_spread_that_cannot_work_is_refused(first: str, last: str, start: int, end: int) -> None:
    from facegen.core.age import spread_table

    with pytest.raises(ValueError):
        spread_table(first, last, start, end)


def test_the_auto_mode_fills_the_staff_table(settings: Settings) -> None:
    from facegen.persistence.settings import SettingsError, build_settings

    values = dict(settings.values) | {"age.staff.mode": "auto"}
    s = build_settings(values)
    assert s.age.staff[1] == (29, "W03") and s.age.staff[-1] == (64, "W10")
    assert s.age.staff_spread is not None and s.age.staff_spread.start == 28
    assert age_wording(31, Group.A, True, s.age, staff=True) == "W03"
    assert settings.age.staff_spread is not None  # auto by default (user, 2026-11-22)
    assert _manual_staff(settings).age.staff_spread is None  # manual: the table
    with pytest.raises(SettingsError, match=r"age\.staff: "):
        build_settings(values | {"age.staff.end": 30})


# --- Auto for every table, All -1 / All +1 (Phase 17l) -------------------------------------------


def test_only_staff_and_media_are_spread_by_default(settings: Settings) -> None:
    """The players' tables stay Custom with their looks (D9): the update re-draws nobody."""
    assert set(settings.age.spreads) == {"staff", "media"}
    for table in ("a_my_club", "a_other_club", "b", "c"):
        assert settings.values[f"age.{table}.mode"] == "manual"


# Each players' table's default spread keeps its first and last looks and where they start.
PLAYER_SPREADS = {
    "a_my_club": (("W01", "W10", 16, 34), (0, "W01"), (34, "W10")),
    "a_other_club": (("W02", "W10", 18, 34), (0, "W02"), (34, "W10")),
    "b": (("W02", "W09", 18, 29), (0, "W02"), (29, "W09")),
    "c": (("W03", "W08", 21, 28), (0, "W03"), (28, "W08")),
}


@pytest.mark.parametrize("table", TABLES)
def test_every_table_can_be_spread(settings: Settings, table: str) -> None:
    from facegen.core.age import spread_table
    from facegen.persistence.settings import SettingsError, build_settings

    values = dict(settings.values) | {f"age.{table}.mode": "auto"}
    s = build_settings(values)
    spread = s.age.spreads[table]
    rows = spread_table(spread.first, spread.last, spread.start, spread.end)
    assert s.age.table(table) == rows
    if table in PLAYER_SPREADS:
        expected, first, last = PLAYER_SPREADS[table]
        assert (spread.first, spread.last, spread.start, spread.end) == expected
        assert rows[0] == first and rows[-1] == last
    # Its own spread is checked, by the table's name.
    with pytest.raises(SettingsError, match=rf"age\.{table}: "):
        build_settings(values | {f"age.{table}.end": spread.start})
    # Custom again: the looks as they were set.
    assert _manual(s, table).age.table(table) == _manual(settings, table).age.table(table)


def test_a_table_spread_moves_each_player_to_the_spreads_look(settings: Settings) -> None:
    from facegen.persistence.settings import build_settings

    s = build_settings(dict(settings.values) | {"age.b.mode": "auto"})
    # W02 to W09 over 18-29: 18 and under W02, then 10 years over W03-W08 (2, 2, 2, 2, 1, 1).
    expected = {
        16: "W02",
        18: "W02",
        19: "W03",
        21: "W04",
        26: "W06",
        27: "W07",
        28: "W08",
        29: "W09",
    }
    for real, wording in expected.items():
        assert age_wording(real, Group.B, False, s.age) == wording, real
    # Other tables are untouched.
    assert age_wording(26, Group.C, False, s.age) == age_wording(26, Group.C, False, settings.age)


def test_the_age_modes_are_checked(settings: Settings) -> None:
    from facegen.persistence.settings import SettingsError, build_settings

    with pytest.raises(SettingsError, match="age.c.mode"):
        build_settings(dict(settings.values) | {"age.c.mode": "spread"})


def test_a_shift_moves_every_used_age_and_keeps_them_within_0_to_100() -> None:
    from facegen.core.age import NOT_USED, shift_table

    table = {"W01": NOT_USED, "W02": 15, "W03": 19, "W10": 99}
    assert shift_table(table, -1) == {"W01": NOT_USED, "W02": 14, "W03": 18, "W10": 98}
    assert shift_table(table, 1) == {"W01": NOT_USED, "W02": 16, "W03": 20, "W10": 100}
    assert shift_table(table, 2)["W10"] == 100  # kept at the limit
    assert shift_table({"W02": 0, "W03": 29}, -1) == {"W02": 0, "W03": 28}
