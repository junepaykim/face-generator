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
    MediaRecord,
    Personality,
    PlayerRecord,
    SaveMeta,
    Snapshot,
    StaffRecord,
)

__all__ = [
    "ClubRecord",
    "DataSource",
    "Gender",
    "LeagueTableRecord",
    "MediaRecord",
    "Personality",
    "PlayerRecord",
    "SaveBusyError",
    "SaveMeta",
    "SaveReadError",
    "Snapshot",
    "StaffRecord",
    "UnsupportedSaveError",
]
