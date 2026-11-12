"""Save readers. Everything that touches fmsave sits behind DataSource."""

from facegen.datasource.base import (
    DataSource,
    SaveBusyError,
    SaveReadError,
    UnsupportedSaveError,
)
from facegen.datasource.records import (
    ClubRecord,
    Gender,
    LeagueTableRecord,
    Personality,
    PlayerRecord,
    SaveMeta,
    Snapshot,
)

__all__ = [
    "ClubRecord",
    "DataSource",
    "Gender",
    "LeagueTableRecord",
    "Personality",
    "PlayerRecord",
    "SaveBusyError",
    "SaveMeta",
    "SaveReadError",
    "Snapshot",
    "UnsupportedSaveError",
]
