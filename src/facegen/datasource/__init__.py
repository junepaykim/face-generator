"""Save readers. Everything that touches fmsave sits behind DataSource."""

from facegen.datasource.base import DataSource, SaveReadError, UnsupportedSaveError
from facegen.datasource.records import (
    ClubRecord,
    Gender,
    LeagueTableRecord,
    PlayerRecord,
    SaveMeta,
    Snapshot,
)

__all__ = [
    "ClubRecord",
    "DataSource",
    "Gender",
    "LeagueTableRecord",
    "PlayerRecord",
    "SaveMeta",
    "SaveReadError",
    "Snapshot",
    "UnsupportedSaveError",
]
