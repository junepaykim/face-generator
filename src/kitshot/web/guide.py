"""The first run as one guided flow on the dashboard (Phase 12f).

Five steps, each with its state: FM27's folder, the graphics card, the picture models, the first
save read, and the first pictures. A step asks only when it cannot work it out: the folder is
looked for, the card is asked in the background, a sync starts by itself. The models need the
user's one confirmation (AGENTS.md hard rule 4), and the card's suggested settings are applied
only on request; neither the card's suggestion nor a step this process does not run holds up the
rest.

The guide shows while setup is unfinished, and for a day after the first picture until it is
closed, so a new user sees how it ended; an install whose first pictures are older never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kitshot.runtime.contracts import ERROR_REASONS, WorkerStatus
from kitshot.setup import hardware
from kitshot.setup.hardware import Hardware
from kitshot.setup.models import DownloadStatus, ModelsStatus

FRESH_HOURS = 24.0
"""How long the finished guide stays after the first picture, unless closed."""
CLOSED = "guide-closed"
"""A file in the data folder: the user closed the finished guide."""

DONE, WORKING, ASK, WAIT, PROBLEM = "done", "working", "ask", "wait", "problem"


@dataclass(frozen=True, slots=True)
class Step:
    key: str
    """folder, gpu, models, sync or pictures."""
    state: str
    """done, working, ask (the user can or must act), wait (on an earlier step) or problem."""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Guide:
    steps: tuple[Step, ...]
    complete: bool

    @property
    def current(self) -> str | None:
        """The first step that is not done: the one the page draws attention to."""
        return next((s.key for s in self.steps if s.state != DONE), None)


@dataclass(frozen=True, slots=True)
class Facts:
    """What the guide is built from, gathered by the dashboard."""

    folder_ok: bool
    folder: str | None
    folder_gone: bool
    models: ModelsStatus | None
    """None where this process does not draw (no models to set up)."""
    download: DownloadStatus | None
    card_available: bool
    card: Hardware | None
    """None until the card has answered."""
    current_values: dict[str, Any]
    career: str | None
    game_date: str | None
    syncing: bool
    can_sync: bool
    watching: bool
    worker: WorkerStatus | None
    pictures: int
    first_picture_at: str | None
    closed: bool
    now: datetime


def _card_step(f: Facts) -> Step | None:
    if not f.card_available:
        return None
    if f.card is None:
        return Step("gpu", WORKING)
    if not f.card.has_gpu:
        return Step("gpu", PROBLEM, {"why": f.card.reason})
    preset = hardware.recommend(f.card.vram_gib)
    data = {"card": f.card.name, "gib": round(f.card.vram_gib or 0), "preset": preset.name}
    changes = hardware.changes(preset, f.current_values)
    if not changes:
        return Step("gpu", DONE, data)
    return Step("gpu", ASK, {**data, "changes": changes})


def _models_step(f: Facts) -> Step | None:
    if f.models is None:
        return None
    if f.models.ready:
        return Step("models", DONE)
    if f.download is not None and f.download.running:
        return Step("models", WORKING, {"download": f.download})
    return Step("models", ASK, {"models": f.models, "download": f.download})


def _sync_step(f: Facts) -> Step:
    if f.career is not None:
        return Step("sync", DONE, {"career": f.career, "date": f.game_date})
    if not f.folder_ok:
        return Step("sync", WAIT)
    if f.syncing:
        return Step("sync", WORKING)
    return Step("sync", ASK, {"watching": f.watching, "can_sync": f.can_sync})


def _pictures_step(f: Facts, models_done: bool) -> Step | None:
    w = f.worker
    if w is None:
        return None
    if f.pictures:
        return Step("pictures", DONE, {"count": f.pictures})
    if f.career is None or not models_done:
        return Step("pictures", WAIT)
    reason = w.reason
    if reason == "drawing":
        return Step("pictures", WORKING, {"name": w.values.get("name")})
    if reason in ("fm_running", "fm_unknown", "held"):
        return Step("pictures", ASK, {"why": reason})
    if reason == "paused":
        return Step("pictures", ASK, {"why": reason})
    if reason in ERROR_REASONS:
        return Step("pictures", PROBLEM)
    if reason == "queue_empty":
        return Step("pictures", ASK, {"why": "nobody"})
    return Step("pictures", WORKING, {"name": None})


def build(f: Facts) -> Guide | None:
    """The guide, or None when it is not shown."""
    folder = (
        Step("folder", DONE, {"folder": f.folder})
        if f.folder_ok
        else Step("folder", ASK, {"gone": f.folder_gone, "folder": f.folder})
    )
    models = _models_step(f)
    models_done = models is None or models.state == DONE
    steps = [
        s
        for s in (folder, _card_step(f), models, _sync_step(f), _pictures_step(f, models_done))
        if s is not None
    ]
    # The card's suggestion never holds the rest up.
    complete = all(s.state == DONE for s in steps if s.key != "gpu")
    if complete and (f.closed or not _fresh(f.first_picture_at, f.now)):
        return None
    return Guide(tuple(steps), complete)


def _fresh(first: str | None, now: datetime) -> bool:
    if not first:
        return False
    try:
        then = datetime.fromisoformat(first)
    except ValueError:
        return False
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds() < FRESH_HOURS * 3600


def closed(data_dir: Path) -> bool:
    return (data_dir / CLOSED).exists()


def close(data_dir: Path) -> None:
    """Remember that the user closed the finished guide."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / CLOSED).write_text("closed\n", encoding="utf-8")
