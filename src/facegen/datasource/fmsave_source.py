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
from contextlib import ExitStack, suppress
from dataclasses import fields, replace
from datetime import date
from pathlib import Path
from typing import Any

from facegen.core.leagues import UEFA_CLUB_COMPETITIONS, domestic_membership, uefa_season
from facegen.core.records import (
    ClubRecord,
    CompetitionRules,
    Gender,
    LeagueTableRecord,
    MediaRecord,
    Personality,
    PlayerRecord,
    SaveMeta,
    Snapshot,
    StaffRecord,
    is_first_team_manager,
)
from facegen.datasource import league_rules, people
from facegen.datasource.base import SaveReadError, UnsupportedSaveError

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
            reputation=getattr(c, "reputation", None),
        )
    return records, uid_map


def _age_on(birth: date, today: date) -> int:
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def convert_personality(personality: Any) -> Personality | None:
    if personality is None:
        return None
    return Personality(**{f.name: getattr(personality, f.name) for f in fields(Personality)})


def nation_of(local_id: int | None, nations: Mapping[int, int] | None) -> int | None:
    """A save-local nation ID as the game's database ID (core/nations.py); None if unknown."""
    if local_id is None or not nations:
        return None
    return nations.get(local_id)


def convert_player(
    p: Any,
    uid_map: Mapping[int, int],
    game_date: date | None,
    nations: Mapping[int, int] | None = None,
) -> PlayerRecord:
    """`nations` maps the save-local nation IDs to database IDs (`Save.nations()`)."""

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
        personality=convert_personality(getattr(p, "personality", None)),
        team_slot=getattr(p, "team_slot", None),
        nation=nation_of(getattr(p, "nation_id", None), nations),
    )


def convert_players(
    players: Iterable[Any],
    uid_map: Mapping[int, int],
    game_date: date | None,
    nations: Mapping[int, int] | None = None,
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
        out.append(convert_player(p, uid_map, game_date, nations))
    return tuple(out), skipped


def staff_job(coded: Any) -> str | None:
    """A staff job in words ("director of football"), or None where the save does not name it."""
    label = getattr(coded, "label", None) if coded is not None else None
    if label is None or label.name == "UNKNOWN":
        return None
    return label.name.replace("_", " ").lower()


def convert_staff(
    staff: Iterable[Any],
    uid_map: Mapping[int, int],
    club_uid: int,
    game_date: date | None,
    *,
    manager_club_ids: frozenset[int] = frozenset(),
    nations: Mapping[int, int] | None = None,
) -> tuple[StaffRecord, ...]:
    """My club's staff and first-team managers of the supplied stable club IDs."""
    seen: set[int] = set()
    out: list[StaffRecord] = []
    for s in staff:
        mine = s.club_uid == club_uid
        league_manager = (
            uid_map.get(s.club_uid) in manager_club_ids
            and getattr(s, "team_slot", None) == 0
            and is_first_team_manager(staff_job(s.job))
        )
        if not (mine or league_manager) or s.unique_id is None or s.unique_id in seen:
            continue
        seen.add(s.unique_id)
        age = s.age
        if age is None and s.birth_date is not None and game_date is not None:
            age = _age_on(s.birth_date, game_date)
        ability = s.ability
        out.append(
            StaffRecord(
                unique_id=s.unique_id,
                name=s.name or s.full_name or s.legal_name or f"Staff {s.unique_id}",
                birth_date=s.birth_date,
                age=age,
                club_id=uid_map.get(s.club_uid, s.club_uid),
                job=staff_job(s.job),
                ca=ability.current if ability is not None else None,
                pa=ability.potential if ability is not None else None,
                club_join_date=s.contract_start,
                is_human_manager=bool(s.is_human_manager),
                personality=convert_personality(s.personality),
                nation=nation_of(getattr(s, "nation_id", None), nations),
            )
        )
    return tuple(out)


def convert_media(
    found: Iterable[people.PersonRecord],
    role: str,
    names: tuple[people.NameAt, people.NameAt, people.NameAt],
    flags: bytes,
    game_date: date | None,
    club_id: int | None = None,
) -> list[MediaRecord]:
    """Journalists' or press officers' records as MediaRecords."""
    out = []
    for r in found:
        age = _age_on(r.birth_date, game_date) if game_date is not None else None
        out.append(
            MediaRecord(
                unique_id=r.unique_id,
                name=people.display_name(r, *names) or f"{role.capitalize()} {r.unique_id}",
                birth_date=r.birth_date,
                age=age,
                role=role,
                nation_id=r.nation_id,
                club_id=club_id,
                female=people.name_is_female(flags, r.first_name_id),
            )
        )
    return out


def read_people(
    save: Any,
    managed_uid: int | None,
    managed_id: int | None,
    staff_first_names: Mapping[int, str | None],
    game_date: date | None,
) -> tuple[tuple[MediaRecord, ...], dict[int, bool]]:
    """Journalists, my club's press officers, and whether each of my club's staff is a woman.

    The pinned fmsave (0.12.6) reads none of these, so they come from the game database's bytes
    through datasource/people.py. This borrows fmsave's decompressed section, its name pools and
    its club index: internals of the pinned version (AGENTS.md rule 5), so any failure here is
    reported and the sync goes on without them. `staff_first_names` maps a staff member's
    unique ID to their first name as fmsave reads it.
    """
    context = save._context
    pools = context.name_pools()
    latest_year = game_date.year if game_date is not None else date.today().year
    with context.section("game_db") as db:

        def first(i: int) -> str | None:
            return pools.first_names.name_at(db, i)

        def surname(i: int) -> str | None:
            return pools.surnames.name_at(db, i)

        def common(i: int) -> str | None:
            return pools.common_names.name_at(db, i)

        names = (first, surname, common)
        flags = people.first_name_flags(db, pools.first_names.entry_count)
        if flags is None:
            # Without it every gender would read as unknown: keep what earlier syncs read.
            raise ValueError("the first names' gender table was not found in the game database")
        sizes = {
            "start": pools.end_offset,
            "first_names": pools.first_names.entry_count,
            "surnames": pools.surnames.entry_count,
            "latest_year": latest_year,
        }
        media = convert_media(
            people.journalists(db, **sizes), "journalist", names, flags, game_date
        )
        if managed_uid is not None:
            by_index = context.club_index().uid_by_club_index
            mine = [index for index, uid in by_index.items() if uid == managed_uid]
            media += convert_media(
                people.press_officers(db, mine, **sizes),
                "press officer",
                names,
                flags,
                game_date,
                managed_id,
            )
        by_text = people.female_by_text((n for n in staff_first_names.values() if n), first, flags)
        female = {uid: by_text[n] for uid, n in staff_first_names.items() if n and n in by_text}
    return tuple(media), female


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
    fixtures: Iterable[Any],
    uid_map: Mapping[int, int],
    comp_keys: Mapping[int, int],
    game_date: date | None = None,
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """First-team fixtures in the current UEFA edition or each other competition's latest
    season, and the start year of the selected fixtures where known."""
    by_season: dict[int, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    edition = uefa_season(game_date)
    for f in fixtures:
        if f.competition_id is None or f.competition_id not in comp_keys:
            continue
        cid = comp_keys[f.competition_id]
        if cid in UEFA_CLUB_COMPETITIONS and (edition is None or f.season_start_year != edition):
            continue
        season = f.season_start_year if f.season_start_year is not None else -1
        counts = by_season[cid][season]
        for uid, slot in ((f.home_club_uid, f.home_team_slot), (f.away_club_uid, f.away_team_slot)):
            if uid is not None and uid in uid_map and slot in _FIRST_TEAM_SLOTS:
                counts[uid_map[uid]] += 1
    counts = {cid: dict(seasons[max(seasons)]) for cid, seasons in by_season.items()}
    latest = {cid: max(seasons) for cid, seasons in by_season.items() if max(seasons) >= 0}
    return counts, latest


def read_league_rules(save: Any) -> CompetitionRules:
    """The save's own division lists and competition levels (datasource/league_rules.py).

    Like `read_people`, this borrows fmsave's decompressed game database: an internal of the
    pinned version (AGENTS.md rule 5), so the caller reports a failure and goes on without."""
    with save._context.section("game_db") as db:
        return league_rules.competition_rules(db)


def unknown_build_note(build: str, reader: str) -> str:
    """The sync warning for a build the reader has no layouts of its own for."""
    return (
        f"FM27 build {build} is not one {reader} was made for: it was read with the layouts "
        "of a build it knows. Check that the players and clubs look right; a newer fmsave "
        "may support this build (AGENTS.md rule 5)."
    )


class FmsaveDataSource:
    """Reads a save copy with fmsave. Never point it at the game's own saves folder."""

    def load(self, path: Path) -> Snapshot:
        import fmsave

        reader = f"fmsave {fmsave.__version__}"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                save = fmsave.open(path)
            except fmsave.UnsupportedGameError as exc:
                raise UnsupportedSaveError(str(exc), reader=reader) from exc
            except fmsave.FmsaveError as exc:
                raise SaveReadError(str(exc)) from exc
            # fmsave reads an FM27 build it has no layouts for with a known build's layouts
            # where the sections match, and raises UnsupportedGameError from any reader whose
            # section does not. On such a build a reader's failed layout check means the same
            # (FM27 27.1.0 and fmsave 0.11.0: "the save layout differs from what fmsave
            # expects"). Either way the build is unsupported; say which one. On a build fmsave
            # knows, a failed check is a damaged save: a read error.
            try:
                info = save.info
                try:
                    snapshot = self._read(save, fmsave)
                except (fmsave.UnsupportedGameError, fmsave.ReaderCheckError) as exc:
                    if isinstance(exc, fmsave.ReaderCheckError) and info.known_build:
                        raise SaveReadError(str(exc)) from exc
                    raise UnsupportedSaveError(
                        f"{info.game} build {info.build} cannot be read by {reader}: {exc}",
                        build=info.build,
                        reader=reader,
                    ) from exc
                except fmsave.FmsaveError as exc:
                    raise SaveReadError(str(exc)) from exc
            finally:
                save.close()
        notes: tuple[str, ...] = ()
        if not snapshot.meta.known_build:
            notes = (unknown_build_note(snapshot.meta.build, reader),)
        messages = tuple(dict.fromkeys(str(w.message) for w in caught))
        meta = replace(snapshot.meta, warnings=notes + snapshot.meta.warnings + messages)
        return replace(snapshot, meta=meta, read_domestic=snapshot.domestic)

    def _read(self, save: Any, fmsave: Any) -> Snapshot:
        info = save.info
        notes: list[str] = []
        # Nations only choose a man's look; without them the world's mix is drawn from.
        nations: dict[int, int] = {}
        try:
            nations = {n.nation_id: n.database_id for n in save.nations()}
        except fmsave.FmsaveError as exc:
            notes.append(f"nations unavailable, so men's looks ignore nationality: {exc}")
        clubs, uid_map = convert_clubs(save.clubs())
        players, skipped = convert_players(save.players(), uid_map, info.game_date, nations)
        managed = list(save.managed_clubs())
        managed_id = uid_map.get(managed[0].club_uid) if managed else None

        if len(managed) > 1:
            notes.append(f"{len(managed)} managed clubs; using the first")
        # League and fixture data only decide "my league", the leagues around it, continental
        # membership and the managers read; a reader failure there should not stop a sync.
        # Each reader is tried on its own. A competition's clubs come from its fixtures (a
        # table counts only for a competition with none), so without the fixtures nothing is
        # known: the sync then keeps the leagues and managers stored before.
        tables: tuple[LeagueTableRecord, ...] = ()
        fixture_counts: dict[int, dict[int, int]] = {}
        fixture_seasons: dict[int, int] = {}
        competitions_read = False
        kept = "so my league and its managers stay as the last sync read them"
        try:
            comp_keys = {c.id: competition_key(c.database_id, c.id) for c in save.competitions()}
        except fmsave.FmsaveError as exc:
            notes.append(f"competition data unavailable, {kept}: {exc}")
        else:
            try:
                fixture_counts, fixture_seasons = convert_fixtures(
                    save.fixtures(), uid_map, comp_keys, info.game_date
                )
                competitions_read = True
            except fmsave.FmsaveError as exc:
                notes.append(f"fixtures unavailable, {kept}: {exc}")
            if competitions_read:
                try:
                    tables = convert_league_tables(save.league_tables(), uid_map, comp_keys)
                except fmsave.FmsaveError as exc:
                    notes.append(f"league tables unavailable, so leagues come from fixtures: {exc}")
        with ExitStack() as section:
            # Both byte readers below read the game database: borrow it once, not twice.
            with suppress(Exception):  # each reader reports its own failure below
                section.enter_context(save._context.section("game_db"))
            rules: CompetitionRules | None = None
            try:
                rules = read_league_rules(save)
            except Exception as exc:  # noqa: BLE001 - experimental reader; the sync goes on
                notes.append(
                    "the save's league levels unavailable, so no league above or below "
                    f"({type(exc).__name__}: {exc})"
                )
            domestic = domestic_membership(
                managed_id, clubs, tables, fixture_counts, fixture_seasons, rules
            )
            # Staff-reader failures must not prevent player syncs.
            staff: tuple[StaffRecord, ...] = ()
            staff_read = True
            first_names: dict[int, str | None] = {}
            if managed:
                try:
                    all_staff = list(save.staff())
                    staff = convert_staff(
                        all_staff,
                        uid_map,
                        managed[0].club_uid,
                        info.game_date,
                        manager_club_ids=domestic.club_ids,
                        nations=nations,
                    )
                    selected = {s.unique_id for s in staff}
                    first_names = {
                        s.unique_id: getattr(s, "first_name", None)
                        for s in all_staff
                        if s.unique_id in selected
                    }
                except fmsave.FmsaveError as exc:
                    staff_read = False
                    notes.append(f"staff data unavailable, staff pictures left as they were: {exc}")
            # The byte reader fills gaps in fmsave; its failure must not stop a sync.
            media: tuple[MediaRecord, ...] = ()
            media_read = True
            try:
                media, female = read_people(
                    save,
                    managed[0].club_uid if managed else None,
                    managed_id,
                    first_names,
                    info.game_date,
                )
                staff = tuple(replace(s, female=female.get(s.unique_id)) for s in staff)
                media = tuple(replace(m, nation=nation_of(m.nation_id, nations)) for m in media)
            except Exception as exc:  # noqa: BLE001 - experimental reader; the sync goes on
                media_read = False
                notes.append(
                    "journalists, press officers and staff genders unavailable "
                    f"({type(exc).__name__}: {exc}); their pictures are left as they were"
                )

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
            fixture_seasons=fixture_seasons,
            competition_rules=rules,
            read_domestic=domestic,
            staff=staff,
            staff_read=staff_read,
            media=media,
            media_read=media_read,
            competitions_read=competitions_read,
        )
