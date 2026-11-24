"""Plain queue records and selection policy shared by runtime and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from kitshot.core.domain import GroupName, JobKind
from kitshot.core.records import Personality


@dataclass(frozen=True, slots=True)
class Career:
    id: int
    key: str
    display_name: str
    game_date: date | None = None
    """The in-game date of the last save synced; it sets the season of a player's clothes."""


@dataclass(frozen=True, slots=True)
class Job:
    id: int
    unique_id: int
    kind: JobKind
    group: GroupName
    target_club_id: int | None
    target_band: str | None
    name: str
    at_my_club: bool
    goalkeeper: bool
    height_cm: int | None
    strength: int | None
    age: int | None
    manual: bool = False
    traits_version: int | None = None
    revision: int = 0
    personality: Personality | None = None
    staff: bool = False
    """On my club's staff: dressed by the wardrobe, aged by the staff table."""
    staff_job: str | None = None
    """The staff job in words, None for a player or where the save names none."""
    media_role: str | None = None
    """"journalist" or "press officer": dressed for their work, aged by the media table."""
    nation: int | None = None
    """The game's database ID of the person's nation (core/nations.py): a man's look."""


@dataclass(frozen=True, slots=True)
class JobFilter:
    groups: frozenset[str] | None = None
    """Only jobs in these groups; None for all."""
    my_club_only: bool = False
    manual_always: bool = False
    """Manual requests from the UI pass whatever the other conditions say."""

    def accepts(self, job: Job) -> bool:
        if self.manual_always and job.manual:
            return True
        return (self.groups is None or job.group in self.groups) and (
            job.at_my_club or not self.my_club_only
        )
