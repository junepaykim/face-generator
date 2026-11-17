"""Save readers. Everything that touches fmsave sits behind DataSource."""

from facegen.core.records import (
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
from facegen.datasource.base import (
    DataSource,
    SaveBusyError,
    SaveReadError,
    UnsupportedSaveError,
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
