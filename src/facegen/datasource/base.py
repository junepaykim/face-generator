"""The DataSource protocol: the only way the rest of the tool reads a save."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from facegen.datasource.records import Snapshot


class UnsupportedSaveError(Exception):
    """The save is from a game or build the reader cannot read. Existing pictures stay."""


class SaveReadError(Exception):
    """The save could not be read: damaged, mid-write, or a reader check failed."""


class DataSource(Protocol):
    def load(self, path: Path) -> Snapshot:
        """Read a save copy. Raises UnsupportedSaveError or SaveReadError."""
        ...
