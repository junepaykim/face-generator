"""supervision: when the worker process is started again after a failure. Pure."""

from __future__ import annotations

from kitshot.runtime.supervision import DEFAULT_POLICY, FailureHistory, RestartPolicy

MIN = 60.0


def test_the_defaults_are_the_plan() -> None:
    assert DEFAULT_POLICY.backoff == (5.0, 30.0, 120.0)
    assert DEFAULT_POLICY.window == 30 * MIN
    assert DEFAULT_POLICY.max_failures == 3 and DEFAULT_POLICY.review_after == 2


def test_backoff_grows_and_three_failures_within_the_window_give_up() -> None:
    h = FailureHistory()
    first = h.record(0.0, None)
    assert (first.restart_in, first.recent, first.review) == (5.0, 1, False)
    assert h.record(1 * MIN, None).restart_in == 30.0
    third = h.record(2 * MIN, None)
    assert third.restart_in is None and third.recent == 3


def test_failures_spread_out_get_the_longest_wait_without_giving_up() -> None:
    h = FailureHistory()
    h.record(0.0, None)
    h.record(20 * MIN, None)
    third = h.record(45 * MIN, None)  # 20 and 45 min are within 30 min of each other; 0 is not
    assert third.restart_in == 120.0 and third.recent == 2
    assert h.record(70 * MIN, None).restart_in == 120.0  # still the same run of failures


def test_a_quiet_spell_starts_a_new_run() -> None:
    h = FailureHistory()
    h.record(0.0, None)
    h.record(10 * MIN, None)
    later = h.record(3 * 60 * MIN, None)
    assert later.restart_in == 5.0 and later.recent == 1


def test_a_job_drawn_during_two_failures_goes_to_review() -> None:
    h = FailureHistory()
    assert not h.record(0.0, 7).review
    assert not h.record(1 * MIN, 8).review  # another job
    assert h.record(40 * MIN, 7).review  # job 7's second, even outside the window


def test_clear_starts_counting_afresh_but_remembers_the_jobs() -> None:
    h = FailureHistory()
    for t in (0.0, 1.0, 2.0):
        h.record(t, 7 if t == 0.0 else None)
    h.clear()
    again = h.record(3.0, 7)
    assert again.restart_in == 5.0 and again.recent == 1
    assert again.review  # job 7 failed once before the user resumed


def test_a_custom_policy() -> None:
    h = FailureHistory(RestartPolicy(backoff=(0.1,), window=10.0, max_failures=2))
    assert h.record(0.0, None).restart_in == 0.1
    assert h.record(1.0, None).restart_in is None
    assert h.record(20.0, None).restart_in == 0.1  # the window has passed
