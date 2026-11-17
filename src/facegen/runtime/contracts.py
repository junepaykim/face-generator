"""Plain status and operation records, and the runtime control interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from facegen.core.job_records import Job

ExitReason = Literal["stopped", "paused", "error", "freed", "fault", "crashed"]
"""A requested stop, pause, job error, model unload, GPU fault or loop crash."""


@dataclass(frozen=True, slots=True)
class Exit:
    """Why a worker process's loop ended, for its parent (the same strings sent over the pipe)."""

    reason: ExitReason
    message: str | None = None
    job: Job | None = None
    """The job being drawn, for a job error or a GPU fault during one."""


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """What the dashboard shows about the worker."""

    state: str
    """starting, idle, paused, waiting, setup (models missing), running, error, gpu-fault or
    stopped."""
    detail: str
    current: Job | None
    paused: bool
    gpu_fault: bool
    """The GPU cannot be used again in this process: only a restart draws again."""
    generate_now: bool
    models_loaded: bool
    done: int
    failed: int
    last_error: str | None
    log: tuple[str, ...]
    reason: str = ""
    """Why the worker is in this state, as a code the pages word in their language
    ("fm_running", "drawing", "queue_empty"...; WORKER_REASONS); empty when unknown. `detail`
    keeps the English line for the log and the tray."""
    values: dict[str, Any] = field(default_factory=dict)
    """The reason's values: counts, a name, a job kind, an error text."""
    since: str | None = None
    """When this state and reason began (UTC, ISO 8601): for a job, when it started."""


WORKER_REASONS = (
    "starting", "no_career", "queue_empty", "held", "paused", "paused_error", "models_missing",
    "fm_running", "fm_unknown", "other_worker", "drawing", "next_job", "job_error", "retrying",
    "crashed", "gpu_fault", "gpu_fault_child", "restarting", "gave_up", "stopped",
)  # fmt: skip
"""Every WorkerStatus.reason: a page words each one (message catalog "worker.reason.<code>")."""


class Link(Protocol):
    """A worker process's line to its parent (worker_process.ChildLink)."""

    def line(self, text: str) -> None:
        """A line for the dashboard's log, time-stamped."""

    def status(self, read: Callable[[], WorkerStatus]) -> None:
        """The status changed: `read` gives it. The link reads and sends in one step, so
        two threads cannot send theirs in the wrong order."""


class WorkerControl(Protocol):
    """What the web app and "sync now" use: WorkerService in a thread, or
    worker_process.WorkerProxy for a worker process."""

    def pause(self) -> None: ...
    def resume(self) -> bool: ...
    def generate_now(self) -> bool: ...
    def wake(self) -> None: ...
    def status(self) -> WorkerStatus: ...
    def log(self, line: str) -> None: ...


@dataclass
class SyncStatus:
    running: bool = False
    started: str | None = None
    trigger: str | None = None
    message: str | None = None
    error: str | None = None
    unsupported: bool = False
    """The last sync failed because the save reader cannot read the save's build."""
    warnings: list[str] = field(default_factory=list)
    notice: str | None = None
    """Sync now read a save that was already synced: FM27 has not saved since."""


SYNCED, UNREADABLE, FAILED = "synced", "unreadable", "failed"
"""How a sync ended. UNREADABLE is recorded in the database and not retried until the save
changes; FAILED (no save, a save still being written, an error) is worth trying again."""

SyncOutcome = Literal["synced", "unreadable", "failed"]


@dataclass(frozen=True, slots=True)
class SyncRequest:
    """Paths and trigger only; the executor creates its own connection and source."""

    db_path: str
    defaults_path: str | None = None
    save_path: str | None = None
    trigger: str = "sync now"


@dataclass(frozen=True, slots=True)
class SyncResult:
    outcome: SyncOutcome
    message: str | None = None
    error: str | None = None
    unsupported: bool = False
    warnings: tuple[str, ...] = ()
    notice: str | None = None


class SyncControl(Protocol):
    """One claimed sync at a time, either in the caller's thread or in the background."""

    def start(self) -> bool: ...
    def run(self, save: Path | None = None, trigger: str = "sync now") -> SyncOutcome | None: ...
    def status(self) -> SyncStatus: ...
    def join(self, timeout: float | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class PollerStatus:
    state: str
    """starting, not set (FM27's folder), no save, synced, settling, due, syncing, error or
    stopped."""
    detail: str
    checked: str | None
    """When the folder was last looked at (local time)."""
