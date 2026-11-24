"""How long the queue takes to draw, from how long recent pictures took (Phase 12c)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from statistics import median

MAX_GAP_SECONDS = 600.0
"""Two pictures this close were drawn one after the other; a longer gap is a pause."""
RECENT = 30
"""Pictures per group the estimate looks at: recent ones follow a settings change."""
MIN_SAMPLES = 3


@dataclass(frozen=True, slots=True)
class Drawn:
    """A picture the worker finished, passed or sent to review."""

    group: str
    at: str
    """When it was stored, ISO 8601."""
    seconds: float | None
    """How long it took, measured since Phase 12c; None for older pictures."""


def seconds_per_picture(
    drawn: Iterable[Drawn], *, recent: int = RECENT, min_samples: int = MIN_SAMPLES
) -> dict[str, float]:
    """The typical seconds per picture of each group: the median of its latest pictures.

    A picture from before the time was measured counts the gap since the picture before it,
    if that is under MAX_GAP_SECONDS (the worker went straight on from one to the next). A
    group with fewer than `min_samples` pictures has no estimate.
    """
    samples: dict[str, list[float]] = {}
    previous: datetime | None = None
    for picture in sorted(drawn, key=lambda d: datetime.fromisoformat(d.at)):
        at = datetime.fromisoformat(picture.at)
        value = picture.seconds
        if value is None and previous is not None:
            gap = (at - previous).total_seconds()
            value = gap if 0 < gap < MAX_GAP_SECONDS else None
        previous = at
        if value is not None and value > 0:
            samples.setdefault(picture.group, []).append(value)
    return {g: median(v[-recent:]) for g, v in samples.items() if len(v) >= min_samples}


@dataclass(frozen=True, slots=True)
class GroupQueue:
    group: str
    queued: int
    manual: int
    """Of them, requested by hand: drawn whether or not the group is drawn automatically."""
    automatic: bool
    """The group's jobs are drawn without Generate now (schedule.auto_groups)."""
    each: float | None
    """Seconds per picture; None until the first pictures are drawn."""

    @property
    def will_draw(self) -> int:
        """Drawn without Generate now."""
        return self.queued if self.automatic else self.manual

    @property
    def held(self) -> int:
        """Waiting for Generate now."""
        return self.queued - self.will_draw


@dataclass(frozen=True, slots=True)
class QueueEstimate:
    groups: tuple[GroupQueue, ...]

    @property
    def queued(self) -> int:
        return sum(g.queued for g in self.groups)

    @property
    def will_draw(self) -> int:
        return sum(g.will_draw for g in self.groups)

    @property
    def held(self) -> int:
        return sum(g.held for g in self.groups)

    def seconds(self, *, held: bool = False) -> float | None:
        """Seconds of drawing for the jobs drawn without Generate now (or, with `held`, for
        those waiting for it); None when a group with such jobs has no estimate yet."""
        total = 0.0
        for g in self.groups:
            n = g.held if held else g.will_draw
            if n == 0:
                continue
            if g.each is None:
                return None
            total += n * g.each
        return total


def estimate_queue(
    queued: Mapping[str, tuple[int, int]],
    per_picture: Mapping[str, float],
    auto_groups: Iterable[str],
) -> QueueEstimate:
    """Per group A, B and C: (queued, of them manual) and its seconds per picture.

    A group with no estimate of its own borrows the slowest known one: a guess on the safe
    side, better than none while its first pictures are drawn.
    """
    auto = frozenset(auto_groups)
    fallback = max(per_picture.values(), default=None)
    groups = tuple(
        GroupQueue(
            group=g,
            queued=queued.get(g, (0, 0))[0],
            manual=queued.get(g, (0, 0))[1],
            automatic=g in auto,
            each=per_picture.get(g, fallback),
        )
        for g in ("a", "b", "c")
    )
    return QueueEstimate(groups)
