"""From classified players to queue actions (docs/spec.md sections 2, 5 and 7). Pure.

For each player the planner works out the one job they need now, if any, and reconciles it
with the job already queued for them. There is at most one queued or running job per
player, so planning twice from the same state changes nothing.

Rules:
- An eligible player with no picture gets an initial picture.
- A player in group A whose picture was drawn at B or C quality is re-drawn (upgrade).
- A tracked player is re-drawn on a club change (new kit) or a new age band, i.e. a new
  age wording (age.py). A free agent has no kit to draw, so their re-draw waits until
  they join a club. A journalist or press officer wears no kit: one pinned to A is tracked
  (classify.py) and re-drawn on a new age band alone.
- Queued automatic jobs for players who no longer need them are cancelled. Running jobs
  are left alone; manual requests retain their intent but refresh their targets.
  Pictures are never deleted here.
- A job that failed its checks and went to review is not re-queued for the same target.
- A re-draw that keeps the character sheet may use the existing picture as a reference
  (`reference_scale`); by default only for players at my club.
- A picture with a legacy men's age wording gets a new face, tracked or not:
  everyone is drawn as a woman or girl (age.drawn_as_man).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from facegen.core.age import drawn_as_man
from facegen.core.classify import Classification, Group
from facegen.core.domain import JobKind, JobStatus
from facegen.core.settings_values import ReferenceSettings

# Queue order (spec section 2): manual requests; group A at my club, newest arrivals first;
# the rest of group A; group B; group C. B, C and the rest of A go by PA, highest first.
TIER_MANUAL = 0
TIER_A_MY_CLUB = 1
TIER_A = 2
TIER_B = 3
TIER_C = 4
_GROUP_TIER = {Group.A: TIER_A, Group.B: TIER_B, Group.C: TIER_C}


@dataclass(frozen=True, slots=True)
class ImageState:
    """The player's current picture."""

    group: Group
    club_id: int | None
    age_band: str | None


@dataclass(frozen=True, slots=True)
class JobState:
    job_id: int
    kind: JobKind
    group: Group
    status: JobStatus
    manual: bool
    tier: int
    rank: float
    target_club_id: int | None
    target_band: str | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PlayerPlanState:
    unique_id: int
    classification: Classification
    club_id: int | None
    age_band: str | None
    at_my_club: bool
    club_since: date | None
    pa: int | None
    image: ImageState | None = None
    active_job: JobState | None = None
    """The player's queued or running job."""
    last_failed_job: JobState | None = None
    manual_allowed: bool = True
    manual_band: str | None = None
    media: bool = False
    """A journalist or press officer: no kit, so no club is needed to re-draw them."""


@dataclass(frozen=True, slots=True)
class DesiredJob:
    kind: JobKind
    group: Group
    tier: int
    rank: float
    target_club_id: int | None
    target_band: str | None
    reason: str

    def same_target(self, job: JobState) -> bool:
        return (self.kind, self.group, self.target_club_id, self.target_band) == (
            job.kind,
            job.group,
            job.target_club_id,
            job.target_band,
        )

    def same_as(self, job: JobState) -> bool:
        return self.same_target(job) and (self.tier, self.rank) == (job.tier, job.rank)


@dataclass(frozen=True, slots=True)
class Enqueue:
    unique_id: int
    job: DesiredJob


@dataclass(frozen=True, slots=True)
class Update:
    job_id: int
    unique_id: int
    job: DesiredJob


@dataclass(frozen=True, slots=True)
class Cancel:
    job_id: int
    unique_id: int
    reason: str


type Action = Enqueue | Update | Cancel


@dataclass(frozen=True, slots=True)
class Plan:
    enqueue: tuple[Enqueue, ...] = ()
    update: tuple[Update, ...] = ()
    cancel: tuple[Cancel, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.enqueue or self.update or self.cancel)


# Jobs that keep the character sheet and replace an existing picture. Only these can use
# that picture as a reference; "initial" has none and "reroll" ("new face") changes the
# sheet. "redraw" is the UI's "new picture".
REFERENCE_KINDS = frozenset({"kit", "age", "upgrade", "redraw"})


def reference_scale(kind: str, at_my_club: bool, s: ReferenceSettings) -> float | None:
    """The reference-picture strength for a job, or None to draw without one.

    `at_my_club` is where the player is when the new picture is drawn.
    """
    if not s.enabled or kind not in REFERENCE_KINDS:
        return None
    if not at_my_club and not s.other_clubs:
        return None
    return s.scale


def queue_position(
    group: Group, at_my_club: bool, club_since: date | None, pa: int | None
) -> tuple[int, float]:
    """(tier, rank) for an automatic job; the queue runs lowest first."""
    if group is Group.A and at_my_club:
        return TIER_A_MY_CLUB, -float(club_since.toordinal() if club_since else 0)
    return _GROUP_TIER[group], -float(pa or 0)


def desired_job(s: PlayerPlanState) -> DesiredJob | None:
    """The one automatic job this player needs now, or None."""
    group = s.classification.group
    if group is None:
        return None
    tier, rank = queue_position(group, s.at_my_club, s.club_since, s.pa)

    def job(kind: JobKind, reason: str) -> DesiredJob:
        return DesiredJob(kind, group, tier, rank, s.club_id, s.age_band, reason)

    img = s.image
    if img is None:
        return job("initial", "first picture")
    if s.club_id is None and not s.media:
        return None  # a re-draw needs a kit; wait for the next club
    if drawn_as_man(img.age_band):
        return job("reroll", "drawn as a woman")
    if group is Group.A and img.group is not Group.A:
        return job("upgrade", f"promoted to A from {img.group.label}")
    if not s.classification.tracked:
        return None
    if s.club_id != img.club_id and not s.media:
        return job("kit", "changed club")
    if s.age_band is not None and img.age_band is not None and s.age_band != img.age_band:
        return job("age", f"age band {img.age_band} to {s.age_band}")
    return None


def plan_player(s: PlayerPlanState) -> Action | None:
    want = desired_job(s)
    active = s.active_job
    if active is not None:
        if active.status != "queued":
            return None
        if active.manual:
            if not s.manual_allowed:
                return Cancel(active.job_id, s.unique_id, "player no longer eligible")
            # Keep the user's kind, group and priority; refresh only snapshot targets.
            refreshed = DesiredJob(
                active.kind,
                active.group,
                active.tier,
                active.rank,
                s.club_id,
                s.manual_band,
                active.reason,
            )
            return (
                Update(active.job_id, s.unique_id, refreshed)
                if not refreshed.same_as(active)
                else None
            )
        if want is None:
            why = "no longer qualifies" if s.classification.group is None else "no longer needed"
            return Cancel(active.job_id, s.unique_id, why)
        if not want.same_as(active):
            return Update(active.job_id, s.unique_id, want)
        return None
    if want is None:
        return None
    failed = s.last_failed_job
    if failed is not None and want.same_target(failed):
        return None
    return Enqueue(s.unique_id, want)


def plan(states: Iterable[PlayerPlanState]) -> Plan:
    enqueue: list[Enqueue] = []
    update: list[Update] = []
    cancel: list[Cancel] = []
    for s in states:
        match plan_player(s):
            case Enqueue() as a:
                enqueue.append(a)
            case Update() as a:
                update.append(a)
            case Cancel() as a:
                cancel.append(a)
            case None:
                pass
    return Plan(tuple(enqueue), tuple(update), tuple(cancel))
