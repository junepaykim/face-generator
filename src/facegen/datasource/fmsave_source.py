"""DataSource backed by fmsave. This is the only module that imports fmsave.

fmsave joins records on save-local `uid`s. This adapter translates them into the game's
stable unique IDs, so nothing past this point depends on a save's internal numbering.
The conversion helpers take fmsave records by duck typing, which keeps them testable
without a save file.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from facegen.datasource.base import SaveReadError, UnsupportedSaveError
from facegen.datasource.records import (
    ClubRecord,
    Gender,
    LeagueTableRecord,
    PlayerRecord,
    SaveMeta,
    Snapshot,
)

_GENDERS = {"MEN": Gender.MEN, "WOMEN": Gender.WOMEN}
_FIRST_TEAM_SLOTS = (0, None)


def competition_key(database_id: int | None, local_id: int) -> int:
    """The editor database ID when the save links one, else minus one minus the local ID."""
    return database_id if database_id is not None else -1 - local_id


def club_gender(coded: Any) -> Gender:
    if coded is None:
        return Gender.UNKNOWN
    return _GENDERS.get(coded.label.name, Gender.UNKNOWN)


def club_id_of(club: Any) -> int:
    """The club's stable ID: its unique ID, else (never seen so far) its save-local uid."""
    return club.unique_id if club.unique_id is not None else club.uid


def convert_clubs(clubs: Iterable[Any]) -> tuple[dict[int, ClubRecord], dict[int, int]]:
    """Club records by stable ID, and the save-local uid to stable ID map."""
    records: dict[int, ClubRecord] = {}
    uid_map: dict[int, int] = {}
    for c in clubs:
        club_id = club_id_of(c)
        uid_map[c.uid] = club_id
        records[club_id] = ClubRecord(
            club_id=club_id,
            name=c.name,
            gender=club_gender(c.gender),
            nation_id=c.nation_id,
        )
    return records, uid_map


def _age_on(birth: date, today: date) -> int:
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def convert_player(p: Any, uid_map: Mapping[int, int], game_date: date | None) -> PlayerRecord:
    def club(uid: int | None) -> int | None:
        return None if uid is None else uid_map.get(uid)

    age = p.age
    if age is None and p.birth_date is not None and game_date is not None:
        age = _age_on(p.birth_date, game_date)
    attributes = getattr(p, "attributes", None)
    return PlayerRecord(
        unique_id=p.unique_id,
        name=p.name or p.full_name or p.legal_name or f"Player {p.unique_id}",
        birth_date=p.birth_date,
        age=age,
        height_cm=p.height_cm or None,
        ca=p.ability.current,
        pa=p.ability.potential,
        club_id=club(p.club_uid),
        loan_parent_club_id=club(p.loan_parent_club_uid) if p.on_loan else None,
        home_grown_club_ids=tuple(
            uid_map[u] for u in p.home_grown_club_uids if u is not None and u in uid_map
        ),
        # On loan, fmsave's join date is the parent club's; the loan start is the arrival.
        club_join_date=p.loan_start if p.on_loan and p.loan_start else p.club_join_date,
        is_goalkeeper=bool(p.natural_positions) and p.natural_positions[0] == "GK",
        strength=attributes.strength if attributes is not None else None,
    )


def convert_players(
    players: Iterable[Any], uid_map: Mapping[int, int], game_date: date | None
) -> tuple[tuple[PlayerRecord, ...], int]:
    """Players with a unique ID, first record wins on a repeat; and how many were skipped."""
    seen: set[int] = set()
    out: list[PlayerRecord] = []
    skipped = 0
    for p in players:
        if p.unique_id is None or p.unique_id in seen:
            skipped += 1
            continue
        seen.add(p.unique_id)
        out.append(convert_player(p, uid_map, game_date))
    return tuple(out), skipped


def convert_league_tables(
    tables: Iterable[Any], uid_map: Mapping[int, int], comp_keys: Mapping[int, int]
) -> tuple[LeagueTableRecord, ...]:
    out = []
    for t in tables:
        cid = None if t.competition_id is None else comp_keys.get(t.competition_id)
        clubs = tuple(
            uid_map[r.club_uid]
            for r in t.rows
            if r.club_uid is not None and r.club_uid in uid_map and r.team_slot in _FIRST_TEAM_SLOTS
        )
        out.append(LeagueTableRecord(competition_id=cid, club_ids=clubs))
    return tuple(out)


def convert_fixtures(
    fixtures: Iterable[Any], uid_map: Mapping[int, int], comp_keys: Mapping[int, int]
) -> dict[int, dict[int, int]]:
    """First-team fixtures per club in each competition's latest season."""
    by_season: dict[int, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    for f in fixtures:
        if f.competition_id is None or f.competition_id not in comp_keys:
            continue
        season = f.season_start_year if f.season_start_year is not None else -1
        counts = by_season[comp_keys[f.competition_id]][season]
        for uid, slot in ((f.home_club_uid, f.home_team_slot), (f.away_club_uid, f.away_team_slot)):
            if uid is not None and uid in uid_map and slot in _FIRST_TEAM_SLOTS:
                counts[uid_map[uid]] += 1
    return {cid: dict(seasons[max(seasons)]) for cid, seasons in by_season.items()}


class FmsaveDataSource:
    """Reads a save copy with fmsave. Never point it at the game's own saves folder."""

    def load(self, path: Path) -> Snapshot:
        import fmsave

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                save = fmsave.open(path)
            except fmsave.UnsupportedGameError as exc:
                raise UnsupportedSaveError(str(exc)) from exc
            except fmsave.FmsaveError as exc:
                raise SaveReadError(str(exc)) from exc
            try:
                snapshot = self._read(save, fmsave)
            except fmsave.FmsaveError as exc:
                raise SaveReadError(str(exc)) from exc
        messages = tuple(dict.fromkeys(str(w.message) for w in caught))
        meta = replace(snapshot.meta, warnings=snapshot.meta.warnings + messages)
        return replace(snapshot, meta=meta)

    def _read(self, save: Any, fmsave: Any) -> Snapshot:
        info = save.info
        clubs, uid_map = convert_clubs(save.clubs())
        players, skipped = convert_players(save.players(), uid_map, info.game_date)
        managed = list(save.managed_clubs())
        managed_id = uid_map.get(managed[0].club_uid) if managed else None

        notes: list[str] = []
        if len(managed) > 1:
            notes.append(f"{len(managed)} managed clubs; using the first")
        # League and fixture data only decide "my league" and continental membership; a
        # reader failure there should not stop a sync.
        tables: tuple[LeagueTableRecord, ...] = ()
        fixture_counts: dict[int, dict[int, int]] = {}
        try:
            comp_keys = {c.id: competition_key(c.database_id, c.id) for c in save.competitions()}
            tables = convert_league_tables(save.league_tables(), uid_map, comp_keys)
            fixture_counts = convert_fixtures(save.fixtures(), uid_map, comp_keys)
        except fmsave.FmsaveError as exc:
            notes.append(f"competition data unavailable: {exc}")

        return Snapshot(
            meta=SaveMeta(
                game=info.game,
                build=info.build,
                known_build=info.known_build,
                game_date=info.game_date,
                save_name=info.save_name,
                file_name=info.file_name,
                warnings=tuple(notes),
                skipped_players=skipped,
            ),
            managed_club_id=managed_id,
            clubs=clubs,
            players=players,
            league_tables=tables,
            fixture_counts=fixture_counts,
        )
