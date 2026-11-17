"""How long the queue takes, from how long recent pictures took (Phase 12c)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from facegen.core.estimate import Drawn, estimate_queue, seconds_per_picture

START = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)


def at(seconds: float) -> str:
    return (START + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def test_measured_times_give_each_groups_median() -> None:
    drawn = [Drawn("a", at(i * 60), s) for i, s in enumerate((20.0, 25.0, 23.0, 90.0))]
    drawn += [Drawn("b", at(1000 + i * 60), s) for i, s in enumerate((7.0, 6.0, 8.0))]
    assert seconds_per_picture(drawn) == {"a": 24.0, "b": 7.0}


def test_older_pictures_count_the_gap_since_the_picture_before() -> None:
    # Drawn one after another before the time was measured: 23 s, 23 s, then 7 s each.
    times = [0, 23, 46, 69, 76, 83, 90]
    groups = "aaaabbb"
    drawn = [Drawn(g, at(t), None) for g, t in zip(groups, times, strict=True)]
    assert seconds_per_picture(drawn) == {"a": 23.0, "b": 7.0}


def test_a_long_gap_is_a_pause_not_a_picture() -> None:
    drawn = [Drawn("b", at(t), None) for t in (0, 7, 14, 3600, 3607, 3614)]
    assert seconds_per_picture(drawn) == {"b": 7.0}  # the hour-long gap is left out


def test_the_order_given_does_not_matter_and_few_pictures_give_no_estimate() -> None:
    drawn = [Drawn("c", at(t), None) for t in (21, 0, 7, 14)]
    assert seconds_per_picture(drawn) == {"c": 7.0}
    assert seconds_per_picture(drawn[:3]) == {}  # two gaps: too few


def test_only_the_latest_pictures_count() -> None:
    """After a settings change (a smaller detail pass) the new times take over."""
    slow = [Drawn("a", at(i * 100), 60.0) for i in range(30)]
    fast = [Drawn("a", at(5000 + i * 100), 20.0) for i in range(30)]
    assert seconds_per_picture(slow + fast) == {"a": 20.0}


def test_the_queue_drawn_now_and_the_queue_held_for_generate_now() -> None:
    queued = {"a": (6, 2), "b": (30, 0), "c": (10, 3)}
    est = estimate_queue(queued, {"a": 23.0, "b": 7.0, "c": 7.5}, auto_groups=("a", "b"))
    assert [(g.group, g.will_draw, g.held) for g in est.groups] == [
        ("a", 6, 0),
        ("b", 30, 0),
        ("c", 3, 7),  # requests by hand are drawn whatever the group
    ]
    assert (est.queued, est.will_draw, est.held) == (46, 39, 7)
    assert est.seconds() == pytest.approx(6 * 23 + 30 * 7 + 3 * 7.5)
    assert est.seconds(held=True) == pytest.approx(7 * 7.5)


def test_a_group_with_no_times_yet_borrows_the_slowest_and_none_known_is_unknown() -> None:
    est = estimate_queue({"a": (2, 0), "c": (4, 0)}, {"a": 23.0}, auto_groups="abc")
    assert est.groups[2].each == 23.0
    unknown = estimate_queue({"a": (2, 0)}, {}, auto_groups="abc")
    assert unknown.seconds() is None and unknown.groups[0].each is None
    assert estimate_queue({}, {}, auto_groups="abc").seconds() == 0.0  # nothing to draw
