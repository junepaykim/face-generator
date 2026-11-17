"""Compatibility exports for the web app and existing service consumers."""

from facegen.pipeline.sync_operation import (
    describe_activation,
    describe_sync,
    unchanged_save_notice,
)
from facegen.runtime.contracts import (
    FAILED,
    SYNCED,
    UNREADABLE,
    Exit,
    ExitReason,
    Link,
    PollerStatus,
    SyncControl,
    SyncOutcome,
    SyncRequest,
    SyncResult,
    SyncStatus,
    WorkerControl,
    WorkerStatus,
)
from facegen.runtime.logging import LOG_FILE, LOG_LINES, log_to_file, logger
from facegen.runtime.logging import stamp as _stamp
from facegen.runtime.policy import BUSY_TIMEOUT_MS, FM_POLL_SECONDS
from facegen.runtime.poller_service import PollerService
from facegen.runtime.sync_service import SyncService
from facegen.runtime.worker_service import (
    CHILD_FAULT_DETAIL,
    GPU_FAULT_ADVICE,
    IDLE_POLL_SECONDS,
    IDLE_UNLOAD_SECONDS,
    MODELS_MISSING,
    WorkerService,
)

__all__ = [
    "BUSY_TIMEOUT_MS",
    "CHILD_FAULT_DETAIL",
    "FAILED",
    "FM_POLL_SECONDS",
    "GPU_FAULT_ADVICE",
    "IDLE_POLL_SECONDS",
    "IDLE_UNLOAD_SECONDS",
    "LOG_FILE",
    "LOG_LINES",
    "MODELS_MISSING",
    "SYNCED",
    "UNREADABLE",
    "Exit",
    "ExitReason",
    "Link",
    "PollerService",
    "PollerStatus",
    "SyncControl",
    "SyncOutcome",
    "SyncRequest",
    "SyncResult",
    "SyncService",
    "SyncStatus",
    "WorkerControl",
    "WorkerService",
    "WorkerStatus",
    "_stamp",
    "describe_activation",
    "describe_sync",
    "log_to_file",
    "logger",
    "unchanged_save_notice",
]
