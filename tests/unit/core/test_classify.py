"""Who is picked for each group (spec section 2; Phase 16: league shares, "at least")."""

from __future__ import annotations

from dataclasses import replace

import pytest
from factories import facts

from kitshot.core.classify import (
    Group,
    PlayerFacts,
    Rules,
    Why,
    classify,
    rule_matches,
    whys_from_json,
    whys_to_json,
)
from kitshot.core.cutoffs import NO_CUT, Cut, Cutoffs, Met

# My league's cells as shares with the values they gave, the other leagues' shares, and the
# global numbers (the defaults: 155/170, 140/155, Light off).
LEAGUE = {
    "a_ca": Cut(100, 6.0),
    "a_pa": Cut(150, 3.0),
    "b_ca": Cut(80, 40.0),
    "b_pa": Cut(130, 20.0),
    "c_ca": Cut(50, 80.0),
    "c_pa": NO_CUT,
}
OTHER = {
    "above_b_ca": Cut(110, 15.0),
    "above_c_ca": Cut(70, 50.0),
    "below_c_ca": Cut(40, 50.0),
    "continental_b_ca": Cut(120, 30.0),
}
EVERYWHERE = {
    "a_ca": Cut(155),
    "a_pa": Cut(170),
    "b_ca": Cut(140),
    "b_pa": Cut(155),
    "c_ca": NO_CUT,
    "c_pa": NO_CUT,
}


@pytest.fixture
def cuts() -> Cutoffs:
    return Cutoffs(league=LEAGUE, other=OTHER, everywhere=EVERYWHERE)


def group_of(c: Cutoffs, **kw: object) -> Group | None:
    return classify(facts(**kw), c).group


# --- who gets a picture --------------------------------------------------------------------


def test_everyone_in_my_first_team_is_group_a_and_tracked(cuts: Cutoffs) -> None:
    for newgen in (True, False):
        c = classify(facts(at_my_club=True, is_newgen=newgen, ca=40, pa=50), cuts)
        assert c.group is Group.A
        assert c.tracked
        assert "in my club's first team" in c.reasons


@pytest.mark.parametrize("photo", [True, None])
def test_pre_existing_players_with_a_photo_or_unknown_are_never_picked_automatically(
    cuts: Cutoffs, photo: bool | None
) -> None:
    f = facts(is_newgen=False, ca=190, pa=200, in_my_league=True, in_continental=True, photo=photo)
    c = classify(replace(f, league_offset=1), cuts)
    assert c.group is None
    assert not c.tracked


def test_newgen_below_every_cut_gets_nothing(cuts: Cutoffs) -> None:
    assert group_of(cuts, ca=40, pa=120, in_my_league=True) is None


def test_players_outside_mens_football_are_ignored_even_with_a_pin(cuts: Cutoffs) -> None:
    c = classify(facts(in_scope=False, at_my_club=True, ca=190, pin="a"), cuts)
    assert c.group is None
    assert c.reasons == ("not in men's football",)


# --- every number means "at least"; CA and PA joined with OR ----------------------------------


@pytest.mark.parametrize(
    ("ca", "pa", "expected"),
    [(154, 90, None), (155, 90, Group.A), (60, 169, None), (60, 170, Group.A)],
)
def test_the_global_a_cut_is_at_least(
    cuts: Cutoffs, ca: int, pa: int, expected: Group | None
) -> None:
    got = group_of(cuts, ca=ca, pa=pa)  # elsewhere: only the global cuts apply
    if expected is None:
        assert got is not Group.A
    else:
        assert got is expected


@pytest.mark.parametrize(
    ("ca", "pa", "expected"),
    [(139, 90, None), (140, 90, Group.B), (60, 154, None), (60, 155, Group.B)],
)
def test_the_global_b_cut(cuts: Cutoffs, ca: int, pa: int, expected: Group | None) -> None:
    assert group_of(cuts, ca=ca, pa=pa) is expected


@pytest.mark.parametrize(
    ("ca", "pa", "expected"),
    [
        (99, 90, Group.B),
        (100, 90, Group.A),
        (60, 149, Group.B),
        (60, 150, Group.A),
        (79, 129, Group.C),
        (80, 90, Group.B),
        (60, 130, Group.B),
        (49, 120, None),
        (50, 90, Group.C),
    ],
)
def test_my_leagues_cuts(cuts: Cutoffs, ca: int, pa: int, expected: Group | None) -> None:
    assert group_of(cuts, ca=ca, pa=pa, in_my_league=True) is expected


def test_reasons_name_the_share_and_its_value(cuts: Cutoffs) -> None:
    c = classify(facts(ca=100, pa=90, in_my_league=True), cuts)
    assert c.reasons == ("newgen in my league, CA 100 (top 6%: 100+)",)
    c = classify(facts(ca=60, pa=160), cuts)  # global
    assert c.reasons == ("newgen, PA 160 (155+)",)
    manual = Cutoffs(league={"b_ca": Cut(98)}, everywhere=EVERYWHERE)
    assert classify(facts(ca=98, in_my_league=True), manual).reasons == (
        "newgen in my league, CA 98 (98+)",
    )


def test_the_league_above_below_and_continental(cuts: Cutoffs) -> None:
    assert group_of(cuts, ca=110, league_offset=1) is Group.B
    assert group_of(cuts, ca=109, league_offset=1) is Group.C
    assert group_of(cuts, ca=69, league_offset=1) is None
    assert group_of(cuts, ca=40, league_offset=-1) is Group.C
    assert group_of(cuts, ca=110, league_offset=-1) is Group.C  # no Standard rule below
    assert group_of(cuts, ca=120, in_continental=True) is Group.B
    assert group_of(cuts, ca=119, in_continental=True) is None
    # Two levels away, or a parallel division (offset 0, not my league): nothing.
    assert group_of(cuts, ca=139, league_offset=2) is None
    assert group_of(cuts, ca=139, league_offset=0) is None
    c = classify(facts(ca=110, pa=None, league_offset=1), cuts)
    assert c.reasons == ("newgen in the league above, CA 110 (top 15%: 110+)",)


def test_the_league_cuts_need_my_league(cuts: Cutoffs) -> None:
    assert group_of(cuts, ca=100, in_continental=True) is None
    assert group_of(cuts, ca=100, league_offset=-1) is Group.C


def test_unknown_pa_counts_only_ca(cuts: Cutoffs) -> None:
    assert group_of(cuts, ca=145, pa=None) is Group.B
    assert group_of(cuts, ca=60, pa=None) is None


def test_player_takes_the_highest_group(cuts: Cutoffs) -> None:
    c = classify(facts(ca=160, pa=180, in_my_league=True), cuts)
    assert c.group is Group.A
    # Only the reasons for the winning group are given.
    assert c.reasons == (
        "newgen, CA 160 (155+)",
        "newgen in my league, CA 160 (top 6%: 100+)",
    )


def test_a_switched_off_cell_picks_nobody(cuts: Cutoffs) -> None:
    off = replace(cuts, league={**LEAGUE, "c_ca": NO_CUT})
    assert group_of(cuts, ca=60, pa=90, in_my_league=True) is Group.C
    assert group_of(off, ca=60, pa=90, in_my_league=True) is None


def test_no_league_found_leaves_only_the_global_cuts() -> None:
    bare = Cutoffs(everywhere=EVERYWHERE)
    assert group_of(bare, ca=150, in_my_league=True, league_offset=1) is Group.B
    assert group_of(bare, ca=130, in_my_league=True, league_offset=1) is None


# --- academy and tracking ---------------------------------------------------------------------


def test_academy_newgen_at_my_club_is_a(cuts: Cutoffs) -> None:
    c = classify(facts(academy=True, at_my_club=True), cuts)
    assert (c.group, c.tracked) == (Group.A, True)


def test_academy_newgen_who_left_drops_to_b_and_stays_tracked(cuts: Cutoffs) -> None:
    c = classify(facts(academy=True, ca=50, pa=60), cuts)
    assert c.group is Group.B
    assert c.tracked
    assert "came through my academy" in c.reasons
    assert "tracked: came through my academy" in c.reasons


def test_academy_newgen_who_still_qualifies_for_a_stays_a(cuts: Cutoffs) -> None:
    c = classify(facts(academy=True, ca=170), cuts)
    assert (c.group, c.tracked) == (Group.A, True)


def test_academy_alumni_off_treats_them_like_any_newgen(cuts: Cutoffs) -> None:
    off = replace(cuts, academy_alumni=False)
    c = classify(facts(academy=True, ca=50, pa=60), off)
    assert (c.group, c.tracked) == (None, False)


def test_only_group_a_my_club_and_academy_are_tracked(cuts: Cutoffs) -> None:
    assert classify(facts(ca=145), cuts).tracked is False  # B
    assert classify(facts(ca=60, in_my_league=True), cuts).tracked is False  # C
    assert classify(facts(ca=160), cuts).tracked is True  # A


# --- pins ------------------------------------------------------------------------------------


def test_pin_never_overrides_my_club(cuts: Cutoffs) -> None:
    c = classify(facts(at_my_club=True, pin="never"), cuts)
    assert (c.group, c.tracked) == (None, False)


def test_pin_brings_in_a_pre_existing_player(cuts: Cutoffs) -> None:
    c = classify(facts(is_newgen=False, pin="b"), cuts)
    assert (c.group, c.tracked) == (Group.B, False)
    assert c.reasons == ("pinned to B",)


def test_pin_sets_the_group_exactly_even_below_the_rules(cuts: Cutoffs) -> None:
    c = classify(facts(ca=170, pin="c"), cuts)
    assert (c.group, c.tracked) == (Group.C, False)


def test_pin_to_a_tracks(cuts: Cutoffs) -> None:
    assert classify(facts(is_newgen=False, pin="a"), cuts).tracked is True


def test_pinned_academy_newgen_stays_tracked(cuts: Cutoffs) -> None:
    c = classify(facts(academy=True, pin="c"), cuts)
    assert (c.group, c.tracked) == (Group.C, True)


def test_rule_matches_lists_every_rule_met(cuts: Cutoffs) -> None:
    found = rule_matches(facts(ca=170, pa=180, in_my_league=True, academy=True), cuts)
    assert {g for g, _ in found} == {Group.A, Group.B, Group.C}


# --- my club's staff (user, 2026-11-12) ----------------------------------------------------------


def test_my_clubs_staff_are_group_a_and_tracked(cuts: Cutoffs) -> None:
    c = classify(facts(is_staff=True, at_my_club=True, is_newgen=False, ca=30), cuts)
    assert (c.group, c.tracked, c.reasons) == (Group.A, True, ("on my club's staff",))


def test_only_my_manager_needs_the_extra_staff_opt_in(cuts: Cutoffs) -> None:
    me = facts(is_staff=True, at_my_club=True, staff_job="manager (you)")
    excluded = classify(me, cuts)
    assert (excluded.group, excluded.tracked) == (None, False)
    assert excluded.why[0].code == "human_manager_off"
    assert classify(me, cuts, Rules(include_human_manager=True)).group is Group.A
    assert classify(me, cuts, Rules(staff=False, include_human_manager=True)).group is None
    assert classify(replace(me, pin="b"), cuts).group is Group.B
    assert classify(replace(me, staff_job="manager"), cuts).group is Group.A
    rival = replace(me, at_my_club=False, in_my_league=True, photo=False)
    assert classify(rival, cuts).group is Group.C


# --- facepacks at my club (user, 2026-11-26) ------------------------------------------------------


def test_my_clubs_people_keep_a_facepacks_picture_unless_it_is_ignored(cuts: Cutoffs) -> None:
    mine = facts(at_my_club=True, is_newgen=False, ca=170, pa=180, in_my_league=True, photo=True)
    staff = facts(is_staff=True, at_my_club=True, is_newgen=False, ca=30, photo=True)
    for person in (mine, staff):
        assert classify(person, cuts).group is not None  # no facepack: as before
        kept = classify(replace(person, facepack=True), cuts)
        assert (kept.group, kept.tracked) == (None, False)
        assert [w.code for w in kept.why] == ["facepack"]
        assert rule_matches(replace(person, facepack=True), cuts) == []
        ignored = classify(replace(person, facepack=True), cuts, Rules(ignore_facepacks=True))
        assert ignored == classify(person, cuts)
        assert classify(replace(person, facepack=True, pin="b"), cuts).group is Group.B
        assert classify(replace(person, facepack=True, pin="never"), cuts).why[0].code == (
            "pin_never"
        )
    # Outside my club a facepack changes nothing: a newgen's cut still picks them.
    elsewhere = facts(ca=170, pa=180, in_my_league=True, facepack=True)
    assert classify(elsewhere, cuts).group is Group.A


def test_the_player_rules_never_apply_to_staff(cuts: Cutoffs) -> None:
    # A created coach with a high staff CA, in my league: none of the player rules.
    staff = {"is_staff": True, "is_newgen": True, "ca": 180, "pa": 190, "in_my_league": True}
    assert rule_matches(facts(**staff, league_offset=1), cuts) == []
    assert classify(facts(**staff, academy=True), cuts).tracked is False
    c = classify(facts(**staff, at_my_club=True), cuts)
    assert c.reasons == ("on my club's staff",)


def test_staff_pictures_turned_off(cuts: Cutoffs) -> None:
    mine = facts(is_staff=True, at_my_club=True)
    c = classify(mine, cuts, Rules(staff=False))
    assert c.group is None and not c.tracked and "staff pictures are off" in c.reasons[0]
    # A pin still hand-picks them; players at my club are unaffected.
    assert classify(replace(mine, pin="b"), cuts, Rules(staff=False)).group is Group.B
    assert classify(facts(at_my_club=True), cuts, Rules(staff=False)).group is Group.A
    assert classify(replace(mine, pin="never"), cuts).group is None


# --- pre-existing players without a photo (user, 2026-11-12) ---------------------------------


def no_photo(**kw: object) -> PlayerFacts:
    return facts(is_newgen=False, photo=False, **kw)


@pytest.mark.parametrize(
    ("kw", "group", "reason"),
    [
        ({"ca": 160}, Group.A, "no photo, CA 160 (155+)"),
        (
            {"pa": 150, "in_my_league": True},
            Group.A,
            "no photo, in my league, PA 150 (top 3%: 150+)",
        ),
        (
            {"pa": 135, "in_my_league": True},
            Group.B,
            "no photo, in my league, PA 135 (top 20%: 130+)",
        ),
        (
            {"ca": 121, "in_continental": True},
            Group.B,
            "no photo, in a continental competition, CA 121 (top 30%: 120+)",
        ),
        ({"ca": 141}, Group.B, "no photo, CA 141 (140+)"),
        ({"ca": 60, "in_my_league": True}, Group.C, "no photo, in my league, CA 60 (top 80%: 50+)"),
        ({"ca": 120, "pa": 130}, None, None),
    ],
)
def test_pre_existing_player_without_a_photo_is_a_candidate(
    cuts: Cutoffs, kw: dict[str, object], group: Group | None, reason: str | None
) -> None:
    c = classify(no_photo(**kw), cuts)
    assert c.group is group
    assert c.reasons == ((reason,) if reason else ())
    assert c.tracked is (group is Group.A)


def test_no_photo_setting_off_leaves_them_out(cuts: Cutoffs) -> None:
    c = classify(no_photo(ca=190), cuts, Rules(no_photo=False))
    assert c.group is None
    # Newgens and my club are unaffected.
    assert classify(facts(ca=190), cuts, Rules(no_photo=False)).group is Group.A
    assert classify(no_photo(at_my_club=True), cuts, Rules(no_photo=False)).group is Group.A


def test_the_academy_rules_stay_newgens_own(cuts: Cutoffs) -> None:
    c = classify(no_photo(academy=True), cuts)
    assert c.group is None
    assert not c.tracked


def test_staff_without_a_photo_are_not_picked(cuts: Cutoffs) -> None:
    c = classify(no_photo(is_staff=True, ca=190), cuts)
    assert c.group is None


def test_without_a_photo_at_my_club_is_group_a_as_everyone_there(cuts: Cutoffs) -> None:
    c = classify(no_photo(at_my_club=True, ca=160), cuts)
    assert c.group is Group.A
    assert c.reasons == ("in my club's first team", "no photo, CA 160 (155+)")


def test_a_pin_still_overrides_a_player_without_a_photo(cuts: Cutoffs) -> None:
    assert classify(no_photo(ca=190, pin="never"), cuts).group is None
    assert classify(no_photo(ca=60, pin="c"), cuts).group is Group.C


# --- my club's youth teams and loans (user, 2026-11-13 and 2026-11-21) -------------------------


def youth(**kw: object) -> PlayerFacts:
    base: dict[str, object] = {"at_my_club": True, "youth": True, "is_newgen": False}
    return facts(**{**base, **kw})


def loanee(**kw: object) -> PlayerFacts:
    base: dict[str, object] = {"at_my_club": True, "loaned_out": True, "is_newgen": False}
    return facts(**{**base, **kw})


def test_my_youth_teams_are_group_b_and_tracked(cuts: Cutoffs) -> None:
    c = classify(youth(ca=40, pa=90), cuts)
    assert c.group is Group.B
    assert c.tracked is True
    assert c.reasons == ("in my club's youth or reserve team", "tracked while at my club")


def test_a_youth_player_reaching_my_leagues_or_the_global_a_cut_is_a(cuts: Cutoffs) -> None:
    c = classify(youth(ca=60, pa=150), cuts)
    assert c.group is Group.A and c.tracked is True
    assert c.reasons == ("youth team, PA 150 (top 3%: 150+)",)
    bare = Cutoffs(everywhere=EVERYWHERE)  # no league found
    c = classify(youth(ca=60, pa=170), bare)
    assert c.reasons == ("youth team, PA 170 (170+)",)
    assert classify(youth(ca=60, pa=150), bare).group is Group.B


def test_the_tiers_of_my_club_come_from_the_settings(cuts: Cutoffs) -> None:
    assert classify(youth(youth=False, ca=40, pa=90), cuts).group is Group.A
    assert classify(youth(ca=40, pa=90), replace(cuts, youth="a")).group is Group.A
    first = classify(youth(youth=False, ca=40, pa=90), replace(cuts, first_team="b"))
    assert (first.group, first.tracked) == (Group.B, True)
    assert first.reasons == ("in my club's first team", "tracked while at my club")
    # A first team set lower still reaches A by the cuts.
    assert classify(youth(youth=False, ca=100), replace(cuts, first_team="c")).group is Group.A


def test_academy_youth_take_my_youth_tier_and_stay_tracked(cuts: Cutoffs) -> None:
    """The academy's B applies once they leave (spec section 2); while in my youth teams their
    tier is the youth one. They are tracked as academy players either way."""
    c = classify(youth(is_newgen=True, academy=True, ca=40, pa=90), cuts)
    assert c.group is Group.B and c.tracked is True
    assert c.reasons == ("in my club's youth or reserve team", "tracked: came through my academy")
    light = classify(youth(is_newgen=True, academy=True, ca=40, pa=90), replace(cuts, youth="c"))
    assert (light.group, light.tracked) == (Group.C, True)
    assert "came through my academy" not in light.reasons
    gone = classify(facts(is_newgen=True, academy=True, ca=40, pa=90), replace(cuts, youth="c"))
    assert gone.group is Group.B and "came through my academy" in gone.reasons


def test_the_youth_rule_is_about_my_club(cuts: Cutoffs) -> None:
    # A youth player elsewhere follows the candidate rules alone.
    assert classify(facts(youth=True, ca=40, pa=90), cuts).group is None


def test_players_i_loaned_out_are_b_tracked_while_i_hold_the_contract(cuts: Cutoffs) -> None:
    c = classify(loanee(ca=40, pa=90), cuts)
    assert (c.group, c.tracked) == (Group.B, True)
    assert c.reasons == ("loaned out by my club", "tracked while I hold the contract")
    c = classify(loanee(ca=100, pa=90), cuts)
    assert c.group is Group.A and c.reasons == ("on loan, CA 100 (top 6%: 100+)",)
    assert classify(loanee(ca=40), replace(cuts, loans="c")).group is Group.C
    # Counted as not at my club (eligibility.loaned_out_at_my_club off): a player like any.
    assert classify(loanee(at_my_club=False, ca=40), cuts).group is None


# --- journalists and press officers (user, 2026-11-13) ----------------------------------------


def media(role: str, **kw: object) -> PlayerFacts:
    base: dict[str, object] = {"media": role, "is_newgen": False, "ca": 0, "pa": None}
    return facts(**{**base, **kw})


def test_local_journalists_and_my_press_officers_are_b_and_not_tracked(cuts: Cutoffs) -> None:
    c = classify(media("journalist", local_media=True), cuts)
    assert (c.group, c.tracked, c.reasons) == (
        Group.B,
        False,
        ("journalist from my club's country",),
    )
    c = classify(media("press officer", local_media=True, at_my_club=True), cuts)
    assert (c.group, c.tracked, c.reasons) == (Group.B, False, ("my club's press officer",))


def test_other_journalists_only_by_a_pin(cuts: Cutoffs) -> None:
    assert classify(media("journalist"), cuts).group is None
    c = classify(media("journalist", pin="b"), cuts)
    assert (c.group, c.tracked) == (Group.B, False)
    # Pinned to A, a journalist or press officer is a regular at my club: A and tracked
    # (user, 2026-11-14). Any other pin draws them once.
    for role, local in (("journalist", False), ("journalist", True), ("press officer", True)):
        c = classify(media(role, pin="a", local_media=local), cuts)
        assert (c.group, c.tracked) == (Group.A, True), role
        assert c.reasons == ("pinned to A", "tracked: a regular at my club")
    for pin in ("b", "c"):
        assert classify(media("journalist", pin=pin, local_media=True), cuts).tracked is False
    assert classify(media("journalist", pin="never"), cuts).group is None


def test_media_never_follow_the_player_rules(cuts: Cutoffs) -> None:
    f = media("journalist", is_newgen=True, ca=190, pa=200, in_my_league=True, photo=False)
    assert classify(f, cuts).group is None


def test_media_switches(cuts: Cutoffs) -> None:
    j = media("journalist", local_media=True)
    p = media("press officer", local_media=True, at_my_club=True)
    assert classify(j, cuts, Rules(journalists=False)).group is None
    assert classify(p, cuts, Rules(journalists=False)).group is Group.B
    assert classify(p, cuts, Rules(press_officers=False)).group is None


# --- reasons as codes (Phase 17h) ------------------------------------------------------------

CASES = {
    "league share": {"ca": 100, "pa": 90, "in_my_league": True},
    "global": {"ca": 60, "pa": 160},
    "above": {"ca": 110, "pa": 90, "league_offset": 1},
    "below": {"ca": 45, "pa": 60, "league_offset": -1, "academy": False},
    "continental": {"ca": 125, "pa": 90, "in_continental": True},
    "no photo": {"is_newgen": False, "photo": False, "ca": 160, "pa": 160},
    "academy": {"academy": True, "ca": 40, "pa": 60},
    "youth reaching A": {"at_my_club": True, "youth": True, "ca": 60, "pa": 150},
    "loaned out": {"at_my_club": True, "loaned_out": True, "ca": 60, "pa": 60},
    "first team": {"at_my_club": True, "ca": 60, "pa": 60},
    "staff": {"is_staff": True, "at_my_club": True, "is_newgen": False, "ca": 30},
    "manager": {"is_staff": True, "is_newgen": False, "in_my_league": True, "photo": False,
                "staff_job": "Manager"},
    "journalist": {"media": "journalist", "local_media": True, "is_newgen": False},
    "press officer pinned": {"media": "press officer", "local_media": True, "pin": "a"},
    "pinned": {"pin": "b"},
    "never": {"pin": "never"},
    "not men's": {"in_scope": False},
    "nothing": {"ca": 10, "pa": 20},
}  # fmt: skip


@pytest.mark.parametrize("case", CASES)
def test_each_reason_is_a_code_whose_text_is_the_sentence_stored_before(
    cuts: Cutoffs, case: str
) -> None:
    c = classify(facts(**CASES[case]), cuts)
    assert c.reasons == tuple(w.text for w in c.why)
    assert len(c.why) == len(c.reasons) and (case == "nothing") == (not c.why)
    assert all(w.complete for w in c.why)
    assert whys_from_json(whys_to_json(c.why)) == c.why


def test_a_cut_names_who_where_and_the_condition(cuts: Cutoffs) -> None:
    c = classify(facts(ca=100, pa=90, in_my_league=True), cuts)
    assert c.why == (Why("cut", Met("CA", 100, 100, 6.0), who="newgen", where="league"),)
    no_photo = classify(facts(is_newgen=False, photo=False, ca=160, pa=160), cuts)
    assert no_photo.why == (Why("cut", Met("CA", 160, 155), who="no_photo"),)
    youth = classify(facts(at_my_club=True, youth=True, ca=60, pa=150), cuts)
    assert youth.why == (Why("part_cut", Met("PA", 150, 150, 3.0), part="youth"),)
    kid = classify(facts(at_my_club=True, youth=True, ca=60, pa=60), cuts)
    assert kid.why == (Why("part", part="youth"), Why("tracked_at_club"))
    assert [w.tracking for w in kid.why] == [False, True]
    assert classify(facts(pin="b"), cuts).why == (Why("pin", group="b"),)


@pytest.mark.parametrize(
    "stored",
    [
        None,
        "",
        "not json",
        '{"code": "staff"}',  # not a list
        '[{"code": "made up"}]',
        '[{"code": "cut", "who": "newgen"}]',  # no condition
        '[{"code": "part", "part": "bench"}]',
        '[{"code": "pin", "group": "d"}]',
        '[{"value": 3}]',
    ],
)
def test_stored_codes_this_version_cannot_word_are_not_read(stored: str | None) -> None:
    """The pages then show the stored English text instead (turn off, don't guess)."""
    assert whys_from_json(stored) is None


def test_no_reason_at_all_reads_as_none_given() -> None:
    assert whys_from_json("[]") == ()
