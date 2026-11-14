"""Builders for plain records and snapshots used across the tests. No save file needed."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date

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

# Nation 1 is "home"; nation 2 is abroad.

# Middling values: no attribute is at either end, so personality weights nothing.
PLAIN_PERSONALITY = Personality(
    adaptability=12,
    ambition=12,
    loyalty=11,
    pressure=11,
    professionalism=12,
    sportsmanship=11,
    temperament=13,
    controversy=5,
)

MY_CLUB = 100
RIVAL = 101  # my league
RIVAL_2 = 102  # my league
ABROAD = 200  # another nation's league, not continental
CONTINENTAL = 201  # abroad, plays in the continental competition
WOMENS = 300
MY_LEAGUE_COMP = 5000
ABROAD_COMP = 5001
CONTINENTAL_COMP = 9000

CREATED_FROM = 2_002_000_000  # first ID of the filler block of created people
NEWGEN = CREATED_FROM + 10_000  # newgens in tests start here
DB_PERSON = 2_000_100_000  # a database person above 2e9, below the gap

GAME_DATE = date(2026, 3, 1)


def club(club_id: int, gender: Gender = Gender.MEN, nation: int | None = 1) -> ClubRecord:
    return ClubRecord(club_id=club_id, name=f"Club {club_id}", gender=gender, nation_id=nation)


DEFAULT_CLUBS: dict[int, ClubRecord] = {
    MY_CLUB: club(MY_CLUB),
    RIVAL: club(RIVAL),
    RIVAL_2: club(RIVAL_2),
    ABROAD: club(ABROAD, nation=2),
    CONTINENTAL: club(CONTINENTAL, nation=2),
    WOMENS: club(WOMENS, Gender.WOMEN),
}


def player(
    unique_id: int,
    *,
    name: str | None = None,
    ca: int = 80,
    pa: int | None = 100,
    club_id: int | None = MY_CLUB,
    age: int | None = 20,
    birth_date: date | None = None,
    loan_parent: int | None = None,
    home_grown: tuple[int, ...] = (),
    join: date | None = None,
    goalkeeper: bool = False,
    personality: Personality | None = None,
    team_slot: int | None = 0,
) -> PlayerRecord:
    return PlayerRecord(
        unique_id=unique_id,
        name=name or f"Player {unique_id}",
        birth_date=birth_date or date(2006, 1, 1),
        age=age,
        height_cm=180,
        ca=ca,
        pa=pa,
        club_id=club_id,
        loan_parent_club_id=loan_parent,
        home_grown_club_ids=home_grown,
        club_join_date=join,
        is_goalkeeper=goalkeeper,
        strength=12,
        personality=personality,
        team_slot=team_slot if club_id is not None else None,
    )


def staff_member(
    unique_id: int,
    *,
    name: str | None = None,
    age: int | None = 45,
    job: str | None = "coach",
    club_id: int = MY_CLUB,
    ca: int | None = 50,
    join: date | None = None,
    human: bool = False,
    female: bool | None = None,
) -> StaffRecord:
    return StaffRecord(
        unique_id=unique_id,
        name=name or f"Staff {unique_id}",
        birth_date=date(1981, 1, 1),
        age=age,
        club_id=club_id,
        job=job,
        ca=ca,
        pa=None if ca is None else ca + 10,
        club_join_date=join,
        is_human_manager=human,
        female=female,
    )


def media_person(
    unique_id: int,
    *,
    role: str = "journalist",
    nation: int | None = 1,
    club_id: int | None = None,
    age: int | None = 50,
    female: bool | None = True,
) -> MediaRecord:
    return MediaRecord(
        unique_id=unique_id,
        name=f"{role.capitalize()} {unique_id}",
        birth_date=date(1976, 1, 1),
        age=age,
        role=role,
        nation_id=nation,
        club_id=club_id,
        female=female,
    )


def filler(career: str = "one", count: int = 60) -> list[PlayerRecord]:
    """A database person above 2e9, then a block of created people at a women's club.

    They give the newgen boundary a clear gap and the career a fingerprint; their names
    depend on `career`, as created people's names differ between careers.
    """
    people = [player(DB_PERSON, name="Database Person", club_id=WOMENS)]
    people += [
        player(CREATED_FROM + i, name=f"{career} created {i}", club_id=WOMENS, ca=40, pa=60)
        for i in range(count)
    ]
    return people


def snapshot(
    players: Iterable[PlayerRecord] = (),
    *,
    career: str = "one",
    managed: int | None = MY_CLUB,
    clubs: Mapping[int, ClubRecord] | None = None,
    game_date: date | None = GAME_DATE,
    file_name: str = "career.fm",
    tables: tuple[LeagueTableRecord, ...] | None = None,
    fixture_counts: Mapping[int, Mapping[int, int]] | None = None,
    staff: Iterable[StaffRecord] = (),
    staff_read: bool = True,
    media: Iterable[MediaRecord] = (),
    media_read: bool = True,
) -> Snapshot:
    if tables is None:
        tables = (
            LeagueTableRecord(MY_LEAGUE_COMP, (MY_CLUB, RIVAL, RIVAL_2)),
            LeagueTableRecord(ABROAD_COMP, (ABROAD, CONTINENTAL)),
        )
    if fixture_counts is None:
        fixture_counts = {CONTINENTAL_COMP: {CONTINENTAL: 6}}
    return Snapshot(
        meta=SaveMeta(
            game="FM27",
            build="27.0.0+2486695",
            known_build=True,
            game_date=game_date,
            save_name=file_name.removesuffix(".fm"),
            file_name=file_name,
        ),
        managed_club_id=managed,
        clubs=dict(clubs if clubs is not None else DEFAULT_CLUBS),
        players=tuple(filler(career)) + tuple(players),
        league_tables=tables,
        fixture_counts=fixture_counts,
        staff=tuple(staff),
        staff_read=staff_read,
        media=tuple(media),
        media_read=media_read,
    )
