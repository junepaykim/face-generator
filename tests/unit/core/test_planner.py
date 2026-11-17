from __future__ import annotations

from dataclasses import replace
from datetime import date

from facegen.core.classify import Classification, Group
from facegen.core.planner import (
    TIER_A,
    TIER_A_MY_CLUB,
    TIER_B,
    TIER_C,
    Cancel,
    Enqueue,
    ImageState,
    JobState,
    PlayerPlanState,
    Update,
    desired_job,
    plan,
    plan_player,
    queue_position,
    reference_scale,
)
from facegen.persistence.settings import ReferenceSettings, Settings

A = Classification(Group.A, True, ("at my club",))
A_NEWGEN = Classification(Group.A, True, ("newgen, CA 160 > 155",))
B = Classification(Group.B, False, ("newgen, PA 160 > 155",))
B_TRACKED = Classification(Group.B, True, ("came through my academy",))
C = Classification(Group.C, False, ("newgen in my league, CA 120 > 115",))
NONE = Classification(None, False, ())


def state(c: Classification = B, **kw: object) -> PlayerPlanState:
    base: dict[str, object] = {
        "unique_id": 1,
        "classification": c,
        "club_id": 10,
        "age_band": "W04",
        "at_my_club": False,
        "club_since": date(2025, 7, 1),
        "pa": 160,
    }
    base.update(kw)
    return PlayerPlanState(**base)  # type: ignore[arg-type]


def job(kind: str = "initial", group: Group = Group.B, **kw: object) -> JobState:
    base: dict[str, object] = {
        "job_id": 99,
        "kind": kind,
        "group": group,
        "status": "queued",
        "manual": False,
        "tier": TIER_B,
        "rank": -160.0,
        "target_club_id": 10,
        "target_band": "W04",
    }
    base.update(kw)
    return JobState(**base)  # type: ignore[arg-type]


def kind_of(s: PlayerPlanState) -> str | None:
    want = desired_job(s)
    return want.kind if want else None


# --- desired job -------------------------------------------------------------------------------


def test_eligible_player_without_a_picture_gets_an_initial_one() -> None:
    want = desired_job(state())
    assert want is not None
    assert (want.kind, want.group, want.target_club_id, want.target_band) == (
        "initial",
        Group.B,
        10,
        "W04",
    )


def test_free_agent_can_get_a_first_picture() -> None:
    assert kind_of(state(club_id=None)) == "initial"


def test_ineligible_player_needs_nothing() -> None:
    assert desired_job(state(NONE)) is None


def test_untracked_player_with_a_picture_is_drawn_once() -> None:
    img = ImageState(Group.B, club_id=5, age_band="W02")  # other club and band than now
    assert kind_of(state(B, image=img)) is None


def test_tracked_player_is_redrawn_on_a_club_change() -> None:
    img = ImageState(Group.A, club_id=5, age_band="W04")
    want = desired_job(state(A, image=img, club_id=10))
    assert want is not None and want.kind == "kit" and want.target_club_id == 10


def test_tracked_free_agent_waits_for_the_next_club() -> None:
    img = ImageState(Group.A, club_id=5, age_band="W02")
    assert kind_of(state(A, image=img, club_id=None, age_band="W04")) is None


def test_tracked_player_is_redrawn_on_a_new_age_band() -> None:
    img = ImageState(Group.A, club_id=10, age_band="W02")
    want = desired_job(state(A, image=img, age_band="W04"))
    assert want is not None and want.kind == "age"
    assert "W02 to W04" in want.reason


def test_unknown_age_band_does_not_trigger_a_redraw() -> None:
    img = ImageState(Group.A, club_id=10, age_band="W02")
    assert kind_of(state(A, image=img, age_band=None)) is None
    img = ImageState(Group.A, club_id=10, age_band=None)
    assert kind_of(state(A, image=img, age_band="W04")) is None


def test_club_change_and_band_change_make_one_kit_job() -> None:
    img = ImageState(Group.A, club_id=5, age_band="W02")
    assert kind_of(state(A, image=img, age_band="W04")) == "kit"


def test_academy_newgen_in_group_b_follows_transfers() -> None:
    img = ImageState(Group.A, club_id=5, age_band="W04")
    want = desired_job(state(B_TRACKED, image=img))
    assert want is not None and (want.kind, want.group) == ("kit", Group.B)


def test_promotion_into_a_is_redrawn_at_a_quality() -> None:
    for old in (Group.B, Group.C):
        img = ImageState(old, club_id=10, age_band="W04")
        want = desired_job(state(A_NEWGEN, image=img))
        assert want is not None and (want.kind, want.group) == ("upgrade", Group.A)


def test_promotion_from_c_to_b_is_not_redrawn() -> None:
    img = ImageState(Group.C, club_id=10, age_band="W04")
    assert kind_of(state(B, image=img)) is None


def test_demotion_is_not_redrawn() -> None:
    img = ImageState(Group.A, club_id=10, age_band="W04")
    assert kind_of(state(C, image=img)) is None


def test_up_to_date_picture_needs_nothing() -> None:
    img = ImageState(Group.A, club_id=10, age_band="W04")
    assert kind_of(state(A, image=img)) is None


# --- queue order -----------------------------------------------------------------------------


def test_queue_tiers() -> None:
    assert queue_position(Group.A, True, date(2026, 1, 1), 100)[0] == TIER_A_MY_CLUB
    assert queue_position(Group.A, False, None, 100)[0] == TIER_A
    assert queue_position(Group.B, True, None, 100)[0] == TIER_B  # only A sorts by arrival
    assert queue_position(Group.C, False, None, 100)[0] == TIER_C


def test_my_club_newest_arrivals_first() -> None:
    old = queue_position(Group.A, True, date(2020, 7, 1), 190)
    new = queue_position(Group.A, True, date(2026, 2, 1), 90)
    unknown = queue_position(Group.A, True, None, 190)
    assert sorted([old, unknown, new]) == [new, old, unknown]


def test_b_and_c_by_pa_descending() -> None:
    low = queue_position(Group.B, False, None, 141)
    high = queue_position(Group.B, False, None, 180)
    none = queue_position(Group.B, False, None, None)
    assert sorted([low, none, high]) == [high, low, none]


# --- reconciling with the queue ----------------------------------------------------------------


def test_enqueue_when_nothing_is_queued() -> None:
    action = plan_player(state())
    assert isinstance(action, Enqueue) and action.job.kind == "initial"


def test_matching_queued_job_means_no_change() -> None:
    assert plan_player(state(active_job=job())) is None


def test_running_job_is_left_alone() -> None:
    assert plan_player(state(NONE, active_job=job(status="running"))) is None
    assert plan_player(state(A, active_job=job(status="running"))) is None


def test_queued_job_is_cancelled_when_the_player_no_longer_qualifies() -> None:
    action = plan_player(state(NONE, active_job=job()))
    assert action == Cancel(99, 1, "no longer qualifies")


def test_queued_job_is_cancelled_when_no_longer_needed() -> None:
    img = ImageState(Group.B, club_id=10, age_band="W04")
    action = plan_player(state(B, image=img, active_job=job(kind="kit")))
    assert action == Cancel(99, 1, "no longer needed")


def test_manual_job_keeps_intent_despite_automatic_group_change() -> None:
    assert plan_player(state(NONE, manual_band="W04", active_job=job(manual=True))) is None
    assert (
        plan_player(state(A, manual_band="W04", active_job=job(manual=True, group=Group.C))) is None
    )


def test_manual_targets_refresh_without_losing_priority_or_reason() -> None:
    active = job("reroll", manual=True, tier=0, rank=-123, reason="requested: new face")
    action = plan_player(state(club_id=11, manual_band="W09", active_job=active))
    assert isinstance(action, Update)
    assert (action.job.kind, action.job.group, action.job.tier, action.job.rank) == (
        "reroll",
        Group.B,
        0,
        -123,
    )
    assert (action.job.target_club_id, action.job.target_band, action.job.reason) == (
        11,
        "W09",
        "requested: new face",
    )


def test_manual_request_is_cancelled_when_player_is_out_of_scope() -> None:
    action = plan_player(state(manual_allowed=False, active_job=job(manual=True)))
    assert action == Cancel(99, 1, "player no longer eligible")


def test_queued_job_follows_a_group_change() -> None:
    action = plan_player(state(A_NEWGEN, active_job=job()))
    assert isinstance(action, Update)
    assert (action.job_id, action.job.group, action.job.tier) == (99, Group.A, TIER_A)


def test_queued_job_follows_a_club_change() -> None:
    action = plan_player(state(club_id=11, active_job=job()))
    assert isinstance(action, Update) and action.job.target_club_id == 11


def test_queued_job_follows_a_new_rank() -> None:
    action = plan_player(state(pa=170, active_job=job()))
    assert isinstance(action, Update) and action.job.rank == -170.0


def test_failed_job_is_not_requeued_for_the_same_target() -> None:
    failed = job(status="failed")
    assert plan_player(state(last_failed_job=failed)) is None


def test_failed_job_does_not_block_a_new_target() -> None:
    failed = job(status="failed", target_club_id=5)
    assert isinstance(plan_player(state(last_failed_job=failed)), Enqueue)


def test_plan_collects_actions() -> None:
    states = [
        state(unique_id=1),
        state(NONE, unique_id=2, active_job=job(job_id=2)),
        state(A_NEWGEN, unique_id=3, active_job=job(job_id=3)),
        state(unique_id=4, active_job=job(job_id=4)),
        state(NONE, unique_id=5),
    ]
    p = plan(states)
    assert [a.unique_id for a in p.enqueue] == [1]
    assert [a.job_id for a in p.cancel] == [2]
    assert [a.job_id for a in p.update] == [3]
    assert not p.empty


def test_planning_the_result_again_changes_nothing() -> None:
    first = plan([state(unique_id=i, pa=150 + i) for i in range(5)])
    queued = [
        replace(
            state(unique_id=a.unique_id, pa=150 + a.unique_id),
            active_job=job(
                job_id=a.unique_id,
                kind=a.job.kind,
                group=a.job.group,
                tier=a.job.tier,
                rank=a.job.rank,
                target_club_id=a.job.target_club_id,
                target_band=a.job.target_band,
            ),
        )
        for a in first.enqueue
    ]
    assert plan(queued).empty


# --- reference picture ----------------------------------------------------------------------


def test_redraws_at_my_club_use_the_reference_by_default(settings: Settings) -> None:
    for kind in ("kit", "age", "upgrade", "redraw"):
        assert reference_scale(kind, True, settings.reference) == 0.1, kind


def test_first_pictures_and_new_faces_never_use_one(settings: Settings) -> None:
    for kind in ("initial", "reroll"):
        assert reference_scale(kind, True, settings.reference) is None, kind


def test_players_elsewhere_get_none_unless_switched_on(settings: Settings) -> None:
    assert reference_scale("kit", False, settings.reference) is None
    on = replace(settings.reference, other_clubs=True)
    assert reference_scale("kit", False, on) == 0.1


def test_switching_the_reference_off_stops_it_everywhere(settings: Settings) -> None:
    off = replace(settings.reference, enabled=False, other_clubs=True)
    for kind in ("kit", "age", "upgrade", "redraw"):
        assert reference_scale(kind, True, off) is None
        assert reference_scale(kind, False, off) is None


# --- the other gender (gender.py) ---------------------------------------------------------------


def test_a_picture_of_a_man_gets_a_new_face_tracked_or_not() -> None:
    """Staff and media drawn as men on 2026-11-13 are drawn as women again (2026-11-14)."""
    for band in ("M1", "M2", "M3", "M4"):
        s = state(A, age_band="W05", at_my_club=True, image=ImageState(Group.A, 10, band))
        want = desired_job(s)
        assert want is not None and (want.kind, want.reason) == ("reroll", "drawn as a woman")
        assert want.target_band == "W05"
        assert reference_scale(want.kind, True, _reference()) is None  # new face: no reference
    # Untracked people (journalists drawn once) are re-drawn too, and with no club.
    journalist = state(B, club_id=None, media=True, image=ImageState(Group.B, None, "M2"))
    assert kind_of(journalist) == "reroll"
    # A woman's picture in another band is an ordinary age re-draw for the tracked only.
    older = state(A, age_band="W06", at_my_club=True, image=ImageState(Group.A, 10, "W05"))
    assert kind_of(older) == "age"
    assert kind_of(state(B, age_band="W06", image=ImageState(Group.B, 10, "W05"))) is None


def test_a_regular_journalist_is_redrawn_without_a_club() -> None:
    """A journalist pinned to A is tracked (classify.py) though they have no club or kit."""
    regular = Classification(Group.A, True, ("pinned to A", "tracked: a regular at my club"))
    img = ImageState(Group.A, None, "W04")
    assert kind_of(state(regular, club_id=None, media=True, image=img)) is None
    want = desired_job(state(regular, club_id=None, media=True, image=img, age_band="W05"))
    assert want is not None and (want.kind, want.reason) == ("age", "age band W04 to W05")
    upgrade = state(regular, club_id=None, media=True, image=ImageState(Group.B, None, "W04"))
    assert kind_of(upgrade) == "upgrade"
    # A press officer's club makes no kit re-draw either.
    moved = state(
        regular, club_id=7, media=True, age_band="W04", image=ImageState(Group.A, 10, "W04")
    )
    assert kind_of(moved) is None
    # A player without a club still waits for one.
    assert kind_of(state(A, club_id=None, image=ImageState(Group.A, 10, "M2"))) is None


def _reference() -> ReferenceSettings:
    return ReferenceSettings(enabled=True, scale=0.1, other_clubs=True)
