from __future__ import annotations

from dataclasses import replace

import pytest

from facegen.age import TABLES, WORDINGS, age_wording, table_name
from facegen.classify import Group
from facegen.settings import Settings

# docs/spec.md section 3: real age -> age wording, per table (user, 2026-11-11).
SPEC = {
    "a_my_club": {
        "W01": range(15, 17), "W02": range(17, 19), "W03": range(19, 22),
        "W04": range(22, 25), "W05": range(25, 27), "W07": range(27, 29),
        "W08": range(29, 31), "W09": range(31, 34), "W10": range(34, 46),
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
    "b": (Group.B, True),
    "c": (Group.C, False),
}


@pytest.mark.parametrize("table", sorted(SPEC))
def test_wordings_match_the_spec_tables(settings: Settings, table: str) -> None:
    group, at_my_club = TABLE_ARGS[table]
    for wording, ages in SPEC[table].items():
        for real in ages:
            assert age_wording(real, group, at_my_club, settings.age) == wording, (table, real)


def test_my_club_only_matters_in_group_a(settings: Settings) -> None:
    for group in (Group.B, Group.C):
        for real in (16, 20, 26, 35):
            assert age_wording(real, group, True, settings.age) == age_wording(
                real, group, False, settings.age
            )


@pytest.mark.parametrize(
    ("group", "at_my_club", "expected"),
    [
        (Group.A, True, "W01"),
        (Group.A, False, "W02"),
        (Group.B, False, "W02"),
        (Group.C, True, "W03"),
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
    assert table_name(Group.B, True) == "b"
    assert table_name("c", False) == "c"
    assert table_name(Group.B, True, staff=True) == "staff"
    assert table_name(None, True, staff=True) is None
    assert {
        table_name(g, m, staff) for g in Group for m in (True, False) for staff in (True, False)
    } == set(TABLES)


# My club's staff (user, 2026-11-12): very wide bands, never W01, most staff in W03-W07.
STAFF = {
    "W02": range(16, 25), "W03": range(25, 32), "W04": range(32, 39), "W05": range(39, 46),
    "W06": range(46, 53), "W07": range(53, 61), "W08": range(61, 68), "W09": range(68, 75),
    "W10": range(75, 95),
}  # fmt: skip


def test_staff_have_their_own_wide_bands_whatever_their_group(settings: Settings) -> None:
    for group in Group:
        for wording, ages in STAFF.items():
            for real in ages:
                assert age_wording(real, group, True, settings.age, staff=True) == wording, real
    assert age_wording(14, Group.A, True, settings.age, staff=True) == "W02"  # never W01
    assert all(w != "W01" for _, w in settings.age.staff)


def test_staff_ages_up_to_100_are_allowed_in_the_tables(settings: Settings) -> None:
    from facegen.settings import build_settings

    values = dict(settings.values)
    values["age.staff.W10"] = 99
    assert build_settings(values).age.staff[-1] == (99, "W10")


def test_wordings_run_from_youngest_to_oldest() -> None:
    assert list(WORDINGS) == [f"W{n:02d}" for n in range(1, 11)]
    # The under-18 looks carry the content rule's extra negative terms; adults do not.
    assert [k for k, w in WORDINGS.items() if w.young] == [f"W0{n}" for n in range(1, 7)]


def test_custom_table(settings: Settings) -> None:
    s = replace(settings.age, b=((15, "W03"), (30, "W10")))
    assert age_wording(29, Group.B, False, s) == "W03"
    assert age_wording(30, Group.B, False, s) == "W10"
