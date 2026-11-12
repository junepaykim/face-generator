"""The DataSource protocol: the only way the rest of the tool reads a save."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from facegen.datasource.records import Snapshot


class UnsupportedSaveError(Exception):
    """The save is from a game or build the reader cannot read. Existing pictures stay.

    `build` is the save's build when the reader got that far; `reader` names the reader and
    its version, so the report can say what would need updating.
    """

    def __init__(self, message: str, *, build: str | None = None, reader: str | None = None):
        super().__init__(message)
        self.build = build
        self.reader = reader


class SaveReadError(Exception):
    """The save could not be read: damaged, mid-write, or a reader check failed."""


class SaveBusyError(SaveReadError):
    """The save kept changing while it was copied: the game is still writing it. Try later."""


class DataSource(Protocol):
    def load(self, path: Path) -> Snapshot:
        """Read a save copy. Raises UnsupportedSaveError or SaveReadError."""
        ...
