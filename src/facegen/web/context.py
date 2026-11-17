"""Injected web services and the existing first-run setup state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from facegen.core.settings_values import Settings
from facegen.platform import platform_info
from facegen.runtime.contracts import SyncControl, WorkerControl
from facegen.runtime.poller_service import PollerService
from facegen.setup import folder_setup, hardware
from facegen.setup.models import DownloadStatus, ModelDownload, ModelsStatus


@dataclass
class AppContext:
    db_path: Path
    defaults_path: Path | None = None
    worker: WorkerControl | None = None
    """WorkerService in a thread, or worker_process.WorkerProxy for a worker process."""
    syncer: SyncControl | None = None
    poller: PollerService | None = None
    models: Callable[[], ModelsStatus] | None = None
    """The model files' status (models.status); None where no real GPU worker runs (fakes,
    --no-worker): then the models are not part of the setup."""
    download: ModelDownload | None = None
    hardware: Callable[[], hardware.Hardware] | None = None
    """Asks for the graphics card (hardware.probe, in a child process); asked once."""
    browse: Callable[[str | None], str | None] | None = None
    """Opens a folder dialog on this PC (folder_setup.browse); None where there is no display."""
    platform: platform_info.Kind | None = None
    """The system the folder check reads paths for; None: this one."""

    def setup(self, settings: Settings) -> SetupState:
        """What first-run setup still needs (Phase 10): FM27's folder, the model files."""
        folder = settings.paths.fm_user_dir
        return SetupState(
            folder_missing=folder is None,
            folder_gone=folder is not None and not folder_setup.RealDisk().is_dir(folder),
            models=self.models() if self.models is not None else None,
            download=self.download.status() if self.download is not None else None,
            browse=self.browse is not None,
        )


@dataclass(frozen=True, slots=True)
class SetupState:
    folder_missing: bool
    """FM27's folder is neither set nor found on this PC."""
    folder_gone: bool
    """It is set, but no such folder is there (moved, or a drive not connected)."""
    models: ModelsStatus | None
    download: DownloadStatus | None
    browse: bool
    """Browse can open a folder dialog here."""

    @property
    def needs_folder(self) -> bool:
        return self.folder_missing or self.folder_gone

    @property
    def needs_models(self) -> bool:
        return self.models is not None and not self.models.ready

    @property
    def state(self) -> str:
        """ready, needs-folder or needs-models."""
        if self.needs_folder:
            return "needs-folder"
        return "needs-models" if self.needs_models else "ready"
