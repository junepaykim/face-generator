"""Plain records a DataSource returns. Nothing here depends on fmsave."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class Gender(StrEnum):
    MEN = "men"
    WOMEN = "women"
    UNKNOWN = "unknown"


def is_first_team_manager(job: str | None) -> bool:
    """Manager title, excluding assistants; the adapter also checks the team's slot."""
    return (job or "").strip().lower() in {"manager", "manager (you)", "head coach"}


@dataclass(frozen=True, slots=True)
class ClubRecord:
    club_id: int
    """The game's unique ID for the club: stable across saves."""
    name: str
    gender: Gender
    nation_id: int | None
    """Save-local ID of the nation whose league the club plays in."""


@dataclass(frozen=True, slots=True)
class Personality:
    """A player's eight personality attributes, each on the 1 to 20 scale."""

    adaptability: int
    ambition: int
    loyalty: int
    pressure: int
    professionalism: int
    sportsmanship: int
    temperament: int
    controversy: int


@dataclass(frozen=True, slots=True)
class PlayerRecord:
    unique_id: int
    """The game's unique ID for the person; what FM27's config.xml maps."""
    name: str
    birth_date: date | None
    age: int | None
    height_cm: int | None
    ca: int
    pa: int | None
    club_id: int | None
    """The club the player plays for (the loan club while on loan); None for a free agent."""
    loan_parent_club_id: int | None
    """The club holding the contract while on loan, else None."""
    home_grown_club_ids: tuple[int, ...]
    club_join_date: date | None
    """When the player arrived at the club they play for: the loan start while on loan.

    The save can hold a date after the in-game date; callers clamp it.
    """
    is_goalkeeper: bool
    strength: int | None
    """Strength on the 1 to 20 display scale."""
    personality: Personality | None = None
    """None when the save holds none for the player."""
    team_slot: int | None = None
    """The player's team at their club: 0 the first team, higher the club's other teams
    (reserves, under-23s, under-18s); None for a free agent or an unknown team."""


@dataclass(frozen=True, slots=True)
class StaffRecord:
    """A person on my club's staff who is not a player: coach, physio, scout, analyst, a
    board member, or the save's own human manager."""

    unique_id: int
    """The game's unique ID for the person; what FM27's config.xml maps, as for players."""
    name: str
    birth_date: date | None
    age: int | None
    club_id: int
    job: str | None
    """The job in lower-case words ("assistant manager"), or None where the save does not say."""
    ca: int | None
    pa: int | None
    """Staff ability on the players' 1 to 200 scale; None for the human manager."""
    club_join_date: date | None
    """When the contract at the club started; None for a role with no contract."""
    is_human_manager: bool = False
    personality: Personality | None = None
    female: bool | None = None
    """A woman, by her first name's entry in the game database (datasource/people.py); None
    where the save does not tell."""


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """A journalist, or a press officer attached to my club (datasource/people.py)."""

    unique_id: int
    """The game's unique ID for the person; what FM27's config.xml maps, as for players."""
    name: str
    birth_date: date | None
    age: int | None
    role: str
    """"journalist" or "press officer"."""
    nation_id: int | None
    """Save-local ID of the person's nation, as ClubRecord.nation_id."""
    club_id: int | None
    """A press officer's club; None for a journalist."""
    female: bool | None = None


@dataclass(frozen=True, slots=True)
class LeagueTableRecord:
    competition_id: int | None
    club_ids: tuple[int, ...]
    """First-team rows only."""


@dataclass(frozen=True, slots=True)
class SaveMeta:
    game: str
    build: str
    known_build: bool
    game_date: date | None
    save_name: str
    file_name: str
    warnings: tuple[str, ...] = ()
    skipped_players: int = 0
    """Player records dropped because they had no unique ID or repeated one."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything the tool reads from one save."""

    meta: SaveMeta
    managed_club_id: int | None
    clubs: Mapping[int, ClubRecord]
    players: tuple[PlayerRecord, ...]
    league_tables: tuple[LeagueTableRecord, ...] = ()
    fixture_counts: Mapping[int, Mapping[int, int]] = field(default_factory=dict)
    """Competition ID to club ID to first-team fixtures in the competition's latest season.

    Competition IDs are the game's editor database IDs where the save links one (stable
    across saves, what the continental settings list), else minus one minus the save-local ID.
    """
    staff: tuple[StaffRecord, ...] = ()
    """My club's staff and first-team managers at clubs in my current league."""
    staff_read: bool = True
    """False when the staff reader failed: the staff stored before are then left as they were,
    rather than counted as gone from my club."""
    media: tuple[MediaRecord, ...] = ()
    """Every journalist in the save, and my club's press officers."""
    media_read: bool = True
    """False when the media reader failed: the media people stored before are left as they
    were."""
