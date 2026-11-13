from __future__ import annotations

from dataclasses import replace

import pytest

from facegen.classify import Group, PlayerFacts, classify, rule_matches
from facegen.settings import Groups, Settings


def facts(**kw: object) -> PlayerFacts:
    base: dict[str, object] = {
        "unique_id": 1,
        "is_newgen": True,
        "ca": 60,
        "pa": 90,
        "in_scope": True,
        "at_my_club": False,
        "in_my_league": False,
        "in_continental": False,
        "academy": False,
        "pin": None,
    }
    base.update(kw)
    return PlayerFacts(**base)  # type: ignore[arg-type]


@pytest.fixture
def groups(settings: Settings) -> Groups:
    return settings.groups


def group_of(g: Groups, **kw: object) -> Group | None:
    return classify(facts(**kw), g).group


# --- who gets a picture --------------------------------------------------------------------


def test_everyone_at_my_club_is_group_a_and_tracked(groups: Groups) -> None:
    for newgen in (True, False):
        c = classify(facts(at_my_club=True, is_newgen=newgen, ca=40, pa=50), groups)
        assert c.group is Group.A
        assert c.tracked
        assert "at my club" in c.reasons


@pytest.mark.parametrize("photo", [True, None])
def test_pre_existing_players_with_a_photo_or_unknown_are_never_picked_automatically(
    groups: Groups, photo: bool | None
) -> None:
    c = classify(
        facts(is_newgen=False, ca=190, pa=200, in_my_league=True, in_continental=True, photo=photo),
        groups,
    )
    assert c.group is None
    assert not c.tracked


def test_newgen_below_every_bar_gets_nothing(groups: Groups) -> None:
    assert group_of(groups, ca=100, pa=130, in_my_league=True) is None


def test_players_outside_mens_football_are_ignored_even_with_a_pin(groups: Groups) -> None:
    c = classify(facts(in_scope=False, at_my_club=True, ca=190, pin="a"), groups)
    assert c.group is None
    assert c.reasons == ("not in men's football",)


# --- thresholds are strict, CA and PA joined with OR -----------------------------------------


@pytest.mark.parametrize(
    ("ca", "pa", "expected"),
    [
        (155, 90, None),  # equal is not above
        (156, 90, Group.A),
        (60, 170, None),
        (60, 171, Group.A),
    ],
)
def test_group_a_bars_are_strict(groups: Groups, ca: int, pa: int, expected: Group | None) -> None:
    # Elsewhere and not in my league, so only A and "any club" B rules can apply.
    got = group_of(groups, ca=ca, pa=pa)
    if expected is None:
        assert got is not Group.A
    else:
        assert got is expected


@pytest.mark.parametrize(
    ("ca", "pa", "expected"),
    [
        (140, 90, None),
        (141, 90, Group.B),
        (60, 155, None),
        (60, 156, Group.B),
    ],
)
def test_group_b_any_club_bars(groups: Groups, ca: int, pa: int, expected: Group | None) -> None:
    assert group_of(groups, ca=ca, pa=pa) is expected


@pytest.mark.parametrize("where", ["in_my_league", "in_continental"])
@pytest.mark.parametrize(
    ("ca", "pa", "expected"),
    [
        (130, 90, None),
        (131, 90, Group.B),
        (60, 150, None),
        (60, 151, Group.B),
    ],
)
def test_group_b_league_or_continental_bars(
    groups: Groups, where: str, ca: int, pa: int, expected: Group | None
) -> None:
    got = group_of(groups, ca=ca, pa=pa, **{where: True})
    if where == "in_continental" or expected is not None:
        assert got is expected
    else:
        assert got is not Group.B  # in my league, the C rule may still apply


@pytest.mark.parametrize(
    ("ca", "pa", "expected"),
    [
        (115, 90, None),
        (116, 90, Group.C),
        (60, 140, None),
        (60, 141, Group.C),
    ],
)
def test_group_c_bars(groups: Groups, ca: int, pa: int, expected: Group | None) -> None:
    assert group_of(groups, ca=ca, pa=pa, in_my_league=True) is expected


def test_group_c_needs_my_league_not_continental(groups: Groups) -> None:
    assert group_of(groups, ca=120, pa=145, in_continental=True) is None
    assert group_of(groups, ca=120, pa=145, in_my_league=True) is Group.C


def test_unknown_pa_counts_only_ca(groups: Groups) -> None:
    assert group_of(groups, ca=150, pa=None) is Group.B
    assert group_of(groups, ca=60, pa=None) is None


def test_player_takes_the_highest_group(groups: Groups) -> None:
    c = classify(facts(ca=160, pa=180, in_my_league=True), groups)
    assert c.group is Group.A
    # Only the reasons for the winning group are given.
    assert all("CA 160 > 155" in r or "PA 180 > 170" in r for r in c.reasons)


def test_thresholds_come_from_settings(groups: Groups) -> None:
    lower = replace(groups, c=replace(groups.c, league_pa_above=100))
    assert group_of(groups, pa=120, in_my_league=True) is None
    assert group_of(lower, pa=120, in_my_league=True) is Group.C


# --- academy and tracking ---------------------------------------------------------------------


def test_academy_newgen_at_my_club_is_a(groups: Groups) -> None:
    c = classify(facts(academy=True, at_my_club=True), groups)
    assert (c.group, c.tracked) == (Group.A, True)


def test_academy_newgen_who_left_drops_to_b_and_stays_tracked(groups: Groups) -> None:
    c = classify(facts(academy=True, ca=50, pa=60), groups)
    assert c.group is Group.B
    assert c.tracked
    assert "came through my academy" in c.reasons
    assert "tracked: came through my academy" in c.reasons


def test_academy_newgen_who_still_qualifies_for_a_stays_a(groups: Groups) -> None:
    c = classify(facts(academy=True, ca=170), groups)
    assert (c.group, c.tracked) == (Group.A, True)


def test_my_academy_setting_off_treats_them_like_any_newgen(groups: Groups) -> None:
    off = replace(groups, b=replace(groups.b, my_academy=False))
    c = classify(facts(academy=True, ca=50, pa=60), off)
    assert (c.group, c.tracked) == (None, False)


def test_only_group_a_and_academy_are_tracked(groups: Groups) -> None:
    assert classify(facts(ca=150), groups).tracked is False  # B
    assert classify(facts(ca=120, in_my_league=True), groups).tracked is False  # C
    assert classify(facts(ca=160), groups).tracked is True  # A


# --- pins ------------------------------------------------------------------------------------


def test_pin_never_overrides_my_club(groups: Groups) -> None:
    c = classify(facts(at_my_club=True, pin="never"), groups)
    assert (c.group, c.tracked) == (None, False)


def test_pin_brings_in_a_pre_existing_player(groups: Groups) -> None:
    c = classify(facts(is_newgen=False, pin="b"), groups)
    assert (c.group, c.tracked) == (Group.B, False)
    assert c.reasons == ("pinned to B",)


def test_pin_sets_the_group_exactly_even_below_the_rules(groups: Groups) -> None:
    c = classify(facts(ca=170, pin="c"), groups)
    assert (c.group, c.tracked) == (Group.C, False)


def test_pin_to_a_tracks(groups: Groups) -> None:
    assert classify(facts(is_newgen=False, pin="a"), groups).tracked is True


def test_pinned_academy_newgen_stays_tracked(groups: Groups) -> None:
    c = classify(facts(academy=True, pin="c"), groups)
    assert (c.group, c.tracked) == (Group.C, True)


def test_rule_matches_lists_every_rule_met(groups: Groups) -> None:
    found = rule_matches(facts(ca=170, pa=180, in_my_league=True, academy=True), groups)
    assert {g for g, _ in found} == {Group.A, Group.B, Group.C}


# --- my club's staff (user, 2026-11-12) ----------------------------------------------------------


def test_my_clubs_staff_are_group_a_and_tracked(groups: Groups) -> None:
    c = classify(facts(is_staff=True, at_my_club=True, is_newgen=False, ca=30), groups)
    assert (c.group, c.tracked, c.reasons) == (Group.A, True, ("on my club's staff",))


def test_the_newgen_rules_never_apply_to_staff(groups: Groups) -> None:
    # A created coach with a high staff CA, in my league: none of A, B or C's newgen rules.
    staff = {"is_staff": True, "is_newgen": True, "ca": 180, "pa": 190, "in_my_league": True}
    assert rule_matches(facts(**staff), groups) == []
    assert classify(facts(**staff, academy=True), groups).tracked is False
    c = classify(facts(**staff, at_my_club=True), groups)
    assert c.reasons == ("on my club's staff",)


def test_staff_pictures_turned_off(groups: Groups) -> None:
    mine = facts(is_staff=True, at_my_club=True)
    c = classify(mine, groups, staff=False)
    assert c.group is None and not c.tracked and "staff pictures are off" in c.reasons[0]
    # A pin still hand-picks them; players at my club are unaffected.
    assert classify(replace(mine, pin="b"), groups, staff=False).group is Group.B
    assert classify(facts(at_my_club=True), groups, staff=False).group is Group.A
    assert classify(replace(mine, pin="never"), groups).group is None


# --- pre-existing players without a photo (user, 2026-11-12) ---------------------------------


def no_photo(**kw: object) -> PlayerFacts:
    return facts(is_newgen=False, photo=False, **kw)


@pytest.mark.parametrize(
    ("kw", "group", "reason"),
    [
        ({"ca": 160}, Group.A, "no photo, CA 160 > 155"),
        ({"pa": 152, "in_my_league": True}, Group.B, "no photo, in my league, PA 152 > 150"),
        (
            {"ca": 131, "in_continental": True},
            Group.B,
            "no photo, in a continental competition, CA 131 > 130",
        ),
        ({"ca": 141}, Group.B, "no photo, CA 141 > 140"),
        ({"ca": 120, "in_my_league": True}, Group.C, "no photo, in my league, CA 120 > 115"),
        ({"ca": 120, "pa": 130}, None, None),
    ],
)
def test_pre_existing_player_without_a_photo_follows_the_newgen_thresholds(
    groups: Groups, kw: dict[str, object], group: Group | None, reason: str | None
) -> None:
    c = classify(no_photo(**kw), groups)
    assert c.group is group
    assert c.reasons == ((reason,) if reason else ())
    assert c.tracked is (group is Group.A)


def test_newgen_reasons_are_unchanged(groups: Groups) -> None:
    c = classify(facts(pa=152, in_my_league=True), groups)
    assert c.reasons == ("newgen in my league, PA 152 > 150",)


def test_no_photo_setting_off_leaves_them_out(groups: Groups) -> None:
    c = classify(no_photo(ca=190), groups, no_photo=False)
    assert c.group is None
    # Newgens and my club are unaffected.
    assert classify(facts(ca=190), groups, no_photo=False).group is Group.A
    assert classify(no_photo(at_my_club=True), groups, no_photo=False).group is Group.A


def test_the_academy_rules_stay_newgens_own(groups: Groups) -> None:
    c = classify(no_photo(academy=True), groups)
    assert c.group is None
    assert not c.tracked


def test_staff_without_a_photo_are_not_picked(groups: Groups) -> None:
    c = classify(no_photo(is_staff=True, ca=190), groups)
    assert c.group is None


def test_without_a_photo_at_my_club_is_group_a_as_everyone_there(groups: Groups) -> None:
    c = classify(no_photo(at_my_club=True, ca=160), groups)
    assert c.group is Group.A
    assert c.reasons == ("at my club", "no photo, CA 160 > 155")


def test_a_pin_still_overrides_a_player_without_a_photo(groups: Groups) -> None:
    assert classify(no_photo(ca=190, pin="never"), groups).group is None
    assert classify(no_photo(ca=60, pin="c"), groups).group is Group.C
