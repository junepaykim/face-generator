"""Pure GPU-worker restart and job-review policy; callers supply the time.

Failures include reported fatal CUDA faults and exits without a reason. Backoff is 5 s,
30 s, then 2 min; a gap longer than `window` starts a new run. `max_failures` failures
within `window` (three in 30 minutes) stop restarts until Resume clears the count.
A job affected by `review_after` failures (two) goes to review; that count survives Resume.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    backoff: tuple[float, ...] = (5.0, 30.0, 120.0)
    """Seconds before a restart: the first failure of a run, the second, the third and later."""
    window: float = 30 * 60.0
    max_failures: int = 3
    """This many failures within `window` stop the restarts."""
    review_after: int = 2
    """A job being drawn during this many failures goes to review."""


DEFAULT_POLICY = RestartPolicy()


@dataclass(frozen=True, slots=True)
class Decision:
    restart_in: float | None
    """Seconds to wait before starting a new worker process; None: give up."""
    review: bool
    """Send the job that was being drawn to review rather than drawing it again."""
    recent: int
    """Failures within the window, this one included."""


@dataclass
class FailureHistory:
    policy: RestartPolicy = DEFAULT_POLICY
    times: list[float] = field(default_factory=list)
    """When the failures within the window happened, oldest first."""
    run: int = 0
    """Failures in the current run: each within `window` of the one before."""
    last: float | None = None
    jobs: Counter[int] = field(default_factory=Counter)
    """Failures per job ID, for as long as the history is kept."""

    def record(self, now: float, job_id: int | None) -> Decision:
        """Count a failure at `now` while drawing `job_id` (None: between jobs) and decide."""
        p = self.policy
        if self.last is None or now - self.last >= p.window:
            self.run = 0
        self.run += 1
        self.last = now
        self.times = [t for t in self.times if now - t < p.window] + [now]
        review = False
        if job_id is not None:
            self.jobs[job_id] += 1
            review = self.jobs[job_id] >= p.review_after
        recent = len(self.times)
        if recent >= p.max_failures:
            return Decision(None, review, recent)
        return Decision(p.backoff[min(self.run, len(p.backoff)) - 1], review, recent)

    def clear(self) -> None:
        """Start counting afresh (the user resumed after the restarts stopped). Each job's
        count is kept, so a job that already failed once is not given two more tries."""
        self.times.clear()
        self.run = 0
        self.last = None
