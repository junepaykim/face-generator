"""Store a save snapshot, classify players and plan jobs in one transaction.

A dry run performs the same work and rolls it back.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any, NamedTuple

from facegen.core.age import age_wording
from facegen.core.classify import Classification, Group, PlayerFacts, Rules, classify
from facegen.core.domain import ACTIVE_JOB_STATUSES
from facegen.core.identity import (
    Person,
    best_career,
    career_key,
    detect_newgen_boundary,
    fingerprint,
    is_newgen,
    person_hash,
)
from facegen.core.leagues import clubs_in_competitions, my_league_competitions
from facegen.core.planner import ImageState, JobState, Plan, PlayerPlanState, plan
from facegen.core.records import Gender, Snapshot
from facegen.core.settings_values import Settings
from facegen.core.traits import personality_to_json
from facegen.datasource.photos import PhotoScan
from facegen.persistence import photo_index
from facegen.persistence.db import transaction
from facegen.persistence.jobs import apply_plan


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _date(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


@dataclass(frozen=True, slots=True)
class Career:
    id: int
    key: str
    display_name: str
    newgen_min_id: int | None
    newgen_id_source: str | None
    is_new: bool


@dataclass(frozen=True, slots=True)
class QueueEntry:
    job_id: int
    unique_id: int
    name: str
    club: str | None
    ca: int
    pa: int | None
    kind: str
    group: str
    tier: int
    reason: str
    player_reasons: tuple[str, ...]


@dataclass
class SyncReport:
    dry_run: bool
    career: Career
    save_file: str
    build: str
    game_date: date | None
    managed_club: str | None
    managed_club_gender: str | None
    my_league_clubs: int
    continental_clubs: int
    counts: dict[str, int]
    enqueued: Counter[tuple[str, str]]
    updated: int
    cancelled: int
    queue: list[QueueEntry]
    warnings: list[str] = field(default_factory=list)
    table: list[dict[str, Any]] = field(default_factory=list)
    new_at_my_club: int = 0
    """Players at my club who were not there at the career's previous sync (youth intake,
    signings, loans in). Their jobs go to the front of the automatic queue. 0 on a first sync."""
    new_staff: int = 0
    """People on my club's staff who were not at the career's previous sync. 0 on a first sync."""


def resolve_career(
    conn: sqlite3.Connection, snapshot: Snapshot, settings: Settings
) -> tuple[Career, list[str]]:
    """Find the career this save belongs to, or create it."""
    warnings: list[str] = []
    people = [Person(p.unique_id, p.name, p.birth_date) for p in snapshot.players]
    file_name = snapshot.meta.file_name
    current = {person_hash(p) for p in people} | set(fingerprint((), None, file_name))
    rows = conn.execute(
        "SELECT id, key, display_name, fingerprint, newgen_min_id, newgen_id_source FROM careers"
    ).fetchall()
    match = best_career(((r["id"], json.loads(r["fingerprint"])) for r in rows), current)
    detected = detect_newgen_boundary(p.unique_id for p in snapshot.players)
    now = _now()
    display = snapshot.meta.save_name or file_name

    if match is None:
        if detected is not None:
            boundary, source = detected, "detected"
        else:
            boundary, source = settings.newgen.fallback_min_unique_id, "fallback"
            warnings.append(
                "no clear gap between database and created IDs; using the configured "
                f"newgen boundary {boundary:,}"
            )
        fp = fingerprint(people, boundary, file_name)
        key = career_key(fp)
        taken = {r["key"] for r in rows}
        suffix = 2
        while key in taken:
            key = f"{career_key(fp)}-{suffix}"
            suffix += 1
        cur = conn.execute(
            "INSERT INTO careers (key, display_name, fingerprint, newgen_min_id, newgen_id_source,"
            " managed_club_id, last_game_date, last_save_file, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                display,
                json.dumps(fp),
                boundary,
                source,
                snapshot.managed_club_id,
                _iso(snapshot.meta.game_date),
                file_name,
                now,
                now,
            ),
        )
        assert cur.lastrowid is not None
        return Career(cur.lastrowid, key, display, boundary, source, True), warnings

    row = next(r for r in rows if r["id"] == match)
    boundary = row["newgen_min_id"]
    if detected is not None and boundary is not None and detected < boundary:
        warnings.append(
            f"IDs from {detected:,} look game-created but the stored newgen boundary is "
            f"{boundary:,}; check docs/research.md"
        )
    # Refresh the fingerprint so it follows the career as created people retire.
    conn.execute(
        "UPDATE careers SET fingerprint = ?, display_name = ?, managed_club_id = ?,"
        " last_game_date = ?, last_save_file = ?, updated_at = ? WHERE id = ?",
        (
            json.dumps(fingerprint(people, boundary, file_name)),
            display,
            snapshot.managed_club_id,
            _iso(snapshot.meta.game_date),
            file_name,
            now,
            match,
        ),
    )
    career = Career(match, row["key"], display, boundary, row["newgen_id_source"], is_new=False)
    return career, warnings


class _PlayerRow(NamedTuple):
    """Named snapshot facts in the players table; booleans use SQLite integers."""

    career_id: int
    unique_id: int
    name: str
    birth_date: str | None
    age: int | None
    height_cm: int | None
    ca: int
    pa: int | None
    is_goalkeeper: int
    strength: int | None
    personality: str | None
    club_id: int | None
    loan_parent_club_id: int | None
    home_grown_club_ids: str
    club_since: str | None
    is_newgen: int
    in_scope: int
    at_my_club: int
    in_my_league: int
    in_continental: int
    first_club_id: int | None
    academy: int
    present: int
    first_seen_sync: int
    last_seen_sync: int
    is_staff: int
    staff_job: str | None
    youth_team: int
    media_role: str | None
    local_media: int
    female: int | None


_PLAYER_COLUMNS = _PlayerRow._fields
_KEEP_ON_UPDATE = {"career_id", "unique_id", "first_seen_sync"}
_UPSERT_PLAYER = (
    f"INSERT INTO players ({', '.join(_PLAYER_COLUMNS)})"
    f" VALUES ({', '.join('?' * len(_PLAYER_COLUMNS))})"
    " ON CONFLICT (career_id, unique_id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in _PLAYER_COLUMNS if c not in _KEEP_ON_UPDATE)
)


@dataclass(frozen=True, slots=True)
class IngestStats:
    first_sync: bool
    my_league_clubs: int
    continental_clubs: int
    arrivals_at_my_club: int = 0
    staff_arrivals: int = 0


def ingest(
    conn: sqlite3.Connection,
    career: Career,
    snapshot: Snapshot,
    settings: Settings,
    sync_id: int,
) -> IngestStats:
    """Store the snapshot's clubs and people with the facts classification needs.

    Staff are stored as rows beside the players (`is_staff`), keyed on the same unique ID, and
    so are journalists and press officers (`media_role`). A person who is also a player (a
    player-coach) is stored once, as the player.
    """
    cid = career.id
    prev = {
        r["unique_id"]: r
        for r in conn.execute(
            "SELECT unique_id, is_newgen, in_scope, first_club_id, club_id, club_since,"
            " at_my_club, present, academy, is_staff, media_role, female FROM players"
            " WHERE career_id = ?",
            (cid,),
        )
    }
    first_sync = not prev
    my_club = snapshot.managed_club_id
    elig = settings.eligibility
    league_ids = my_league_competitions(
        my_club, snapshot.clubs, snapshot.league_tables, snapshot.fixture_counts
    )
    league_clubs = clubs_in_competitions(
        league_ids, snapshot.league_tables, snapshot.fixture_counts
    )
    continental_clubs = clubs_in_competitions(
        elig.continental_competition_ids, snapshot.league_tables, snapshot.fixture_counts
    )
    game_date = snapshot.meta.game_date

    rows = []
    arrivals = 0
    for p in snapshot.players:
        old = prev.get(p.unique_id)
        club = snapshot.clubs.get(p.club_id) if p.club_id is not None else None
        if club is not None:
            in_scope = club.gender is Gender.MEN or not elig.mens_football_only
        else:
            in_scope = bool(old and old["in_scope"])  # free agent: as last seen
        at_my_club = my_club is not None and (
            p.club_id == my_club
            or (elig.loaned_out_at_my_club and p.loan_parent_club_id == my_club)
        )
        newgen = is_newgen(
            p.unique_id,
            career.newgen_min_id,
            previously_newgen=bool(old and old["is_newgen"]),
            first_seen_mid_career=old is None and not first_sync,
        )
        # "First seen at my club" only counts for players who appear after the first sync;
        # on the first sync, only home-grown data establishes academy membership.
        first_club = old["first_club_id"] if old else (None if first_sync else p.club_id)
        academy = bool(old and old["academy"]) or (
            newgen
            and my_club is not None
            and (first_club == my_club or my_club in p.home_grown_club_ids)
        )
        if (
            at_my_club
            and not first_sync
            and not (old and old["present"] and old["at_my_club"] and not old["is_staff"])
        ):
            arrivals += 1
        club_since = _club_since(old, p.club_id, p.club_join_date, game_date)
        youth = p.club_id is not None and p.club_id == my_club and p.team_slot not in (None, 0)
        rows.append(
            _PlayerRow(
                career_id=cid,
                unique_id=p.unique_id,
                name=p.name,
                birth_date=_iso(p.birth_date),
                age=p.age,
                height_cm=p.height_cm,
                ca=p.ca,
                pa=p.pa,
                is_goalkeeper=int(p.is_goalkeeper),
                strength=p.strength,
                personality=personality_to_json(p.personality),
                club_id=p.club_id,
                loan_parent_club_id=p.loan_parent_club_id,
                home_grown_club_ids=json.dumps(list(p.home_grown_club_ids)),
                club_since=club_since,
                is_newgen=int(newgen),
                in_scope=int(in_scope),
                at_my_club=int(at_my_club),
                in_my_league=int(p.club_id in league_clubs),
                in_continental=int(p.club_id in continental_clubs),
                first_club_id=first_club,
                academy=int(academy),
                present=1,
                first_seen_sync=sync_id,
                last_seen_sync=sync_id,
                is_staff=0,
                staff_job=None,
                youth_team=int(youth),
                media_role=None,
                local_media=0,
                female=None,
            )
        )
    staff_rows = _staff_rows(
        career, snapshot, settings, sync_id, prev, first_sync, league_clubs, continental_clubs
    )
    staff_arrivals = sum(
        1
        for r in staff_rows
        if not first_sync
        and r.at_my_club
        and not ((old := prev.get(r.unique_id)) and old["present"] and old["at_my_club"])
    )
    rows.extend(staff_rows)
    rows.extend(_media_rows(career, snapshot, sync_id, prev))
    conn.executemany(_UPSERT_PLAYER, rows)
    # People a reader could not read this time are left as they were, not counted as gone.
    conn.execute(
        "UPDATE players SET present = 0 WHERE career_id = ? AND last_seen_sync != ?"
        + ("" if snapshot.staff_read else " AND is_staff = 0")
        + ("" if snapshot.media_read else " AND media_role IS NULL"),
        (cid, sync_id),
    )
    conn.executemany(
        "INSERT INTO clubs (career_id, club_id, name, gender, nation_id, in_my_league, continental)"
        " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (career_id, club_id) DO UPDATE SET"
        " name = excluded.name, gender = excluded.gender, nation_id = excluded.nation_id,"
        " in_my_league = excluded.in_my_league, continental = excluded.continental",
        [
            (
                cid,
                c.club_id,
                c.name,
                c.gender.value,
                c.nation_id,
                int(c.club_id in league_clubs),
                int(c.club_id in continental_clubs),
            )
            for c in snapshot.clubs.values()
        ],
    )
    return IngestStats(
        first_sync, len(league_clubs), len(continental_clubs), arrivals, staff_arrivals
    )


def _club_since(
    old: sqlite3.Row | None, club_id: int | None, joined: date | None, game_date: date | None
) -> str | None:
    """When the person arrived at their club: kept while they stay, else the join date, never
    after the in-game date (the save holds some join dates still to come)."""
    if old is not None and old["club_id"] == club_id and old["club_since"]:
        return old["club_since"]
    arrived = joined or game_date
    if arrived and game_date and arrived > game_date:
        arrived = game_date
    return _iso(arrived)


def _staff_rows(
    career: Career,
    snapshot: Snapshot,
    settings: Settings,
    sync_id: int,
    prev: dict[int, sqlite3.Row],
    first_sync: bool,
    league_clubs: set[int],
    continental_clubs: set[int],
) -> list[_PlayerRow]:
    """Snapshot facts for my staff and league managers."""
    my_club = snapshot.managed_club_id
    players = {p.unique_id for p in snapshot.players}
    rows = []
    for s in snapshot.staff:
        if s.unique_id in players:
            continue  # a player-coach is drawn as a player
        old = prev.get(s.unique_id)
        club = snapshot.clubs.get(s.club_id)
        if club is not None:
            in_scope = club.gender is Gender.MEN or not settings.eligibility.mens_football_only
        else:
            in_scope = bool(old and old["in_scope"])
        newgen = is_newgen(
            s.unique_id,
            career.newgen_min_id,
            previously_newgen=bool(old and old["is_newgen"]),
            first_seen_mid_career=old is None and not first_sync,
        )
        # Without the people reader this time, keep what an earlier sync read.
        female = s.female
        if female is None and not snapshot.media_read and old and old["female"] is not None:
            female = bool(old["female"])
        rows.append(
            _PlayerRow(
                career_id=career.id,
                unique_id=s.unique_id,
                name=s.name,
                birth_date=_iso(s.birth_date),
                age=s.age,
                height_cm=None,
                ca=s.ca or 0,
                pa=s.pa,
                is_goalkeeper=0,
                strength=None,
                personality=personality_to_json(s.personality),
                club_id=s.club_id,
                loan_parent_club_id=None,
                home_grown_club_ids="[]",
                club_since=_club_since(old, s.club_id, s.club_join_date, snapshot.meta.game_date),
                is_newgen=int(newgen),
                in_scope=int(in_scope),
                at_my_club=int(my_club is not None and s.club_id == my_club),
                in_my_league=int(s.club_id in league_clubs),
                in_continental=int(s.club_id in continental_clubs),
                first_club_id=old["first_club_id"] if old else (None if first_sync else s.club_id),
                academy=0,
                present=1,
                first_seen_sync=sync_id,
                last_seen_sync=sync_id,
                is_staff=1,
                staff_job="manager (you)" if s.is_human_manager else s.job,
                youth_team=0,
                media_role=None,
                local_media=0,
                female=None if female is None else int(female),
            )
        )
    return rows


def _media_rows(
    career: Career, snapshot: Snapshot, sync_id: int, prev: dict[int, sqlite3.Row]
) -> list[_PlayerRow]:
    """Snapshot facts for journalists and my club's press officers.

    They are never newgens or players: CA 0, no club for a journalist. A journalist from my
    club's country, or a press officer of my club, is `local_media`. Someone who is also a
    player or on my club's staff is stored once, as that.
    """
    my_club = snapshot.managed_club_id
    club = snapshot.clubs.get(my_club) if my_club is not None else None
    my_nation = club.nation_id if club is not None else None
    taken = {p.unique_id for p in snapshot.players} | {s.unique_id for s in snapshot.staff}
    rows = []
    for m in snapshot.media:
        if m.unique_id in taken:
            continue
        old = prev.get(m.unique_id)
        at_my_club = my_club is not None and m.club_id == my_club
        local = at_my_club if m.role == "press officer" else m.nation_id == my_nation
        rows.append(
            _PlayerRow(
                career_id=career.id,
                unique_id=m.unique_id,
                name=m.name,
                birth_date=_iso(m.birth_date),
                age=m.age,
                height_cm=None,
                ca=0,
                pa=None,
                is_goalkeeper=0,
                strength=None,
                personality=None,
                club_id=m.club_id,
                loan_parent_club_id=None,
                home_grown_club_ids="[]",
                club_since=_club_since(old, m.club_id, None, snapshot.meta.game_date),
                is_newgen=0,
                in_scope=1,
                at_my_club=int(at_my_club),
                in_my_league=0,
                in_continental=0,
                first_club_id=old["first_club_id"] if old else None,
                academy=0,
                present=1,
                first_seen_sync=sync_id,
                last_seen_sync=sync_id,
                is_staff=0,
                staff_job=None,
                youth_team=0,
                media_role=m.role,
                local_media=int(local and my_nation is not None),
                female=None if m.female is None else int(m.female),
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class Move:
    """A player whose group or tracking a classification changes, from the stored values."""

    unique_id: int
    group_before: str | None
    tracked_before: bool
    group_after: str | None
    tracked_after: bool


@dataclass(frozen=True, slots=True)
class PlanResult:
    plan: Plan
    classifications: dict[int, Classification]
    moved: tuple[Move, ...] = ()


def _job_state(r: sqlite3.Row) -> JobState:
    return JobState(
        job_id=r["id"],
        kind=r["kind"],
        group=Group(r["group_name"]),
        status=r["status"],
        manual=bool(r["manual"]),
        tier=r["tier"],
        rank=r["rank"],
        target_club_id=r["target_club_id"],
        target_band=r["target_band"],
        reason=r["reason"],
    )


def image_state(r: sqlite3.Row) -> ImageState:
    """What a current picture stands for, to the planner.

    A drawn picture shows the group, club and age band it was drawn for. A picture restored
    from the history stands for the player as they were when it was restored (manual.py), so
    it is re-drawn only after a later change.
    """
    if r["restored_at"]:
        return ImageState(
            Group(r["restored_group"] or r["group_name"]), r["restored_club_id"], r["restored_band"]
        )
    return ImageState(Group(r["group_name"]), r["club_id"], r["age_band"])


def _classify(
    conn: sqlite3.Connection, career_id: int, settings: Settings
) -> tuple[PlanResult, list[tuple[Any, ...]]]:
    """Classify every stored player and plan the queue, reading only.

    Returns the plan and the player rows whose group, tracking or reasons would change.
    """
    images = {
        r["unique_id"]: image_state(r)
        for r in conn.execute(
            "SELECT unique_id, group_name, club_id, age_band, restored_at, restored_group,"
            " restored_club_id, restored_band FROM images WHERE career_id = ? AND is_current = 1",
            (career_id,),
        )
    }
    active_in = ", ".join("?" * len(ACTIVE_JOB_STATUSES))
    active = {
        r["unique_id"]: _job_state(r)
        for r in conn.execute(
            f"SELECT * FROM jobs WHERE career_id = ? AND status IN ({active_in})",
            (career_id, *ACTIVE_JOB_STATUSES),
        )
    }
    failed = {
        r["unique_id"]: _job_state(r)
        for r in conn.execute(
            "SELECT * FROM jobs WHERE career_id = ? AND status = 'failed' ORDER BY id",
            (career_id,),
        )
    }

    # Who FM27 shows a photo for is known only once a sync has read its photo list.
    photo_status = photo_index.status(conn)
    photos_known = photo_status is not None and photo_status.game_known

    rules = Rules.of(settings)
    classifications: dict[int, Classification] = {}
    changed: list[tuple[Any, ...]] = []
    moved: list[Move] = []
    states: list[PlayerPlanState] = []
    for r in conn.execute(
        "SELECT p.*, ph.unique_id IS NOT NULL AS has_photo FROM players p"
        " LEFT JOIN photos ph ON ph.unique_id = p.unique_id WHERE p.career_id = ?",
        (career_id,),
    ):
        uid = r["unique_id"]
        staff = bool(r["is_staff"])
        if r["present"]:
            facts = PlayerFacts(
                unique_id=uid,
                is_newgen=bool(r["is_newgen"]),
                ca=r["ca"],
                pa=r["pa"],
                in_scope=bool(r["in_scope"]),
                at_my_club=bool(r["at_my_club"]),
                in_my_league=bool(r["in_my_league"]),
                in_continental=bool(r["in_continental"]),
                academy=bool(r["academy"]),
                pin=r["pin"],
                is_staff=staff,
                staff_job=r["staff_job"],
                photo=bool(r["has_photo"]) if photos_known else None,
                youth=bool(r["youth_team"]),
                media=r["media_role"],
                local_media=bool(r["local_media"]),
            )
            c = classify(facts, settings.groups, rules)
        else:
            gone = "no longer on my club's staff" if staff else "not in the latest save"
            c = Classification(None, False, (gone,))
        classifications[uid] = c
        group = c.group.value if c.group else None
        reasons = json.dumps(list(c.reasons))
        if (group, int(c.tracked), reasons) != (r["group_name"], r["tracked"], r["reasons"]):
            changed.append((group, int(c.tracked), reasons, career_id, uid))
            if (group, c.tracked) != (r["group_name"], bool(r["tracked"])):
                moved.append(Move(uid, r["group_name"], bool(r["tracked"]), group, c.tracked))
        if c.group is None and uid not in active:
            continue
        who = {"staff": staff, "media": r["media_role"] is not None}
        states.append(
            PlayerPlanState(
                unique_id=uid,
                classification=c,
                club_id=r["club_id"],
                age_band=age_wording(r["age"], c.group, bool(r["at_my_club"]), settings.age, **who),
                at_my_club=bool(r["at_my_club"]),
                club_since=_date(r["club_since"]),
                pa=r["pa"],
                image=images.get(uid),
                active_job=active.get(uid),
                last_failed_job=failed.get(uid),
                manual_allowed=bool(r["present"] and r["in_scope"] and r["pin"] != "never"),
                manual_band=(
                    age_wording(
                        r["age"], active[uid].group, bool(r["at_my_club"]), settings.age, **who
                    )
                    if uid in active
                    else None
                ),
                media=who["media"],
            )
        )
    return PlanResult(plan(states), classifications, tuple(moved)), changed


def classify_and_plan(conn: sqlite3.Connection, career_id: int, settings: Settings) -> PlanResult:
    """Classify every stored player of a career and bring the queue in line.

    Reads only the database, so a settings change can re-classify without a new save.
    """
    result, changed = _classify(conn, career_id, settings)
    conn.executemany(
        "UPDATE players SET group_name = ?, tracked = ?, reasons = ?"
        " WHERE career_id = ? AND unique_id = ?",
        changed,
    )
    apply_plan(conn, career_id, result.plan)
    return result


def queue_preview(conn: sqlite3.Connection, career_id: int, limit: int) -> list[QueueEntry]:
    rows = conn.execute(
        "SELECT j.id, j.unique_id, j.kind, j.group_name, j.tier, j.reason, p.name, p.ca, p.pa,"
        " p.reasons, c.name AS club FROM jobs j"
        " JOIN players p ON p.career_id = j.career_id AND p.unique_id = j.unique_id"
        " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        " WHERE j.career_id = ? AND j.status = 'queued' ORDER BY j.tier, j.rank, j.id LIMIT ?",
        (career_id, limit),
    )
    return [
        QueueEntry(
            job_id=r["id"],
            unique_id=r["unique_id"],
            name=r["name"],
            club=r["club"],
            ca=r["ca"],
            pa=r["pa"],
            kind=r["kind"],
            group=r["group_name"],
            tier=r["tier"],
            reason=r["reason"],
            player_reasons=tuple(json.loads(r["reasons"])),
        )
        for r in rows
    ]


def player_table(conn: sqlite3.Connection, career_id: int) -> list[dict[str, Any]]:
    """One row per present player with save facts, group and classification reasons."""
    rows = conn.execute(
        "SELECT p.unique_id, p.name, p.age, p.height_cm, p.ca, p.pa, c.name AS club,"
        " c.gender AS club_gender, p.in_my_league, p.in_continental, p.is_newgen,"
        " p.group_name, p.tracked, p.reasons FROM players p"
        " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        " WHERE p.career_id = ? AND p.present = 1 ORDER BY p.unique_id",
        (career_id,),
    )
    table = []
    for r in rows:
        row = dict(r)
        for key in ("in_my_league", "in_continental", "is_newgen", "tracked"):
            row[key] = bool(row[key])
        row["group"] = (row.pop("group_name") or "").upper()
        row["reasons"] = "; ".join(json.loads(row["reasons"]))
        table.append(row)
    return table


def _counts(conn: sqlite3.Connection, career_id: int, snapshot: Snapshot) -> dict[str, int]:
    genders = Counter(
        (
            snapshot.clubs[p.club_id].gender.value
            if p.club_id is not None and p.club_id in snapshot.clubs
            else "no club"
        )
        for p in snapshot.players
    )
    row = conn.execute(
        "SELECT"
        " SUM(in_scope AND media_role IS NULL) AS in_scope,"
        " SUM(in_scope AND is_newgen) AS newgens,"
        " SUM(group_name = 'a') AS a,"
        " SUM(group_name = 'a' AND at_my_club) AS a_my_club,"
        " SUM(group_name = 'b') AS b,"
        " SUM(group_name = 'c') AS c,"
        " SUM(tracked) AS tracked,"
        " SUM(at_my_club AND media_role IS NULL) AS at_my_club,"
        " SUM(academy) AS academy,"
        " SUM(youth_team) AS youth_team,"
        " SUM(media_role = 'journalist') AS journalists,"
        " SUM(media_role = 'journalist' AND local_media) AS local_journalists,"
        " SUM(media_role = 'press officer') AS press_officers,"
        " SUM(media_role IS NOT NULL AND group_name IS NOT NULL) AS media_picked"
        " FROM players WHERE career_id = ? AND present = 1",
        (career_id,),
    ).fetchone()
    counts = {k: v or 0 for k, v in dict(row).items()}
    counts.update(photo_index.player_counts(conn, career_id))
    counts.update(
        players=len(snapshot.players),
        men=genders.get("men", 0),
        women=genders.get("women", 0),
        unknown_gender=genders.get("unknown", 0),
        no_club=genders.get("no club", 0),
        skipped=snapshot.meta.skipped_players,
    )
    return counts


def run_sync(
    conn: sqlite3.Connection,
    snapshot: Snapshot,
    settings: Settings,
    *,
    save_file: str,
    save_mtime: float | None = None,
    dry_run: bool = False,
    preview: int = 20,
    with_table: bool = False,
    trigger: str | None = None,
    photos: PhotoScan | None = None,
    read_settings: Callable[[], Settings] | None = None,
    before_commit: Callable[[], None] | None = None,
) -> SyncReport:
    """Run one sync in a transaction; a dry run rolls everything back.

    `save_mtime` is the save's modification time as read. The career keeps the newest it has
    seen; the career with the newest save is the active one (spec section 8). `trigger` says
    who started the sync: 'poller', 'sync now' or 'command'. `photos` is a fresh look at who
    FM27 shows a photo for (photos.scan); without it the stored one stands. File-based sync
    supplies `read_settings` so changes saved while parsing are seen under the write lock.
    `before_commit` runs inside the transaction, once a non-dry run is ready to commit.
    """
    with transaction(conn, rollback=dry_run):
        if read_settings is not None:
            settings = read_settings()
        if photos is not None:
            photo_index.store(conn, photos)
        career, warnings = resolve_career(conn, snapshot, settings)
        cur = conn.execute(
            "INSERT INTO sync_runs (career_id, started_at, save_file, save_mtime, build,"
            " game_date, trigger) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                career.id,
                _now(),
                save_file,
                save_mtime,
                snapshot.meta.build,
                _iso(snapshot.meta.game_date),
                trigger,
            ),
        )
        sync_id = cur.lastrowid
        assert sync_id is not None
        if save_mtime is not None:
            conn.execute(
                "UPDATE careers SET last_save_mtime = MAX(COALESCE(last_save_mtime, ?), ?)"
                " WHERE id = ?",
                (save_mtime, save_mtime, career.id),
            )
        stats = ingest(conn, career, snapshot, settings, sync_id)
        result = classify_and_plan(conn, career.id, settings)
        counts = _counts(conn, career.id, snapshot)
        conn.execute(
            "UPDATE sync_runs SET finished_at = ?, counts = ? WHERE id = ?",
            (_now(), json.dumps(counts), sync_id),
        )

        managed_id = snapshot.managed_club_id
        managed = snapshot.clubs.get(managed_id) if managed_id is not None else None
        if managed is None:
            warnings.append("no managed club found: no one is 'at my club' or 'in my league'")
        elif managed.gender is not Gender.MEN and settings.eligibility.mens_football_only:
            warnings.append(
                f"the managed club ({managed.name}) is not a men's club, so with men's football "
                "only, 'my club' and 'my league' match nobody (decisions.md, open question 1)"
            )
        elif stats.my_league_clubs == 0:
            warnings.append("could not work out my club's league from the save")
        if not settings.eligibility.continental_competition_ids:
            warnings.append("no continental competition IDs are set (eligibility settings)")
        photo_status = photo_index.status(conn)
        if photo_status is not None:
            warnings.extend(photo_status.warnings)
        warnings.extend(snapshot.meta.warnings)

        report = SyncReport(
            dry_run=dry_run,
            career=career,
            save_file=save_file,
            build=snapshot.meta.build,
            game_date=snapshot.meta.game_date,
            managed_club=managed.name if managed else None,
            managed_club_gender=managed.gender.value if managed else None,
            my_league_clubs=stats.my_league_clubs,
            continental_clubs=stats.continental_clubs,
            counts=counts,
            enqueued=Counter((a.job.kind, a.job.group.value) for a in result.plan.enqueue),
            updated=len(result.plan.update),
            cancelled=len(result.plan.cancel),
            queue=queue_preview(conn, career.id, preview),
            warnings=warnings,
            table=player_table(conn, career.id) if with_table else [],
            new_at_my_club=stats.arrivals_at_my_club,
            new_staff=stats.staff_arrivals,
        )
        if not dry_run and before_commit is not None:
            before_commit()
        return report


def reclassify(
    conn: sqlite3.Connection, career_id: int, settings: Settings, *, dry_run: bool = False
) -> PlanResult:
    """Re-classify from the stored snapshot after a settings change (spec section 7)."""
    with transaction(conn, rollback=dry_run):
        return classify_and_plan(conn, career_id, settings)


@dataclass(frozen=True, slots=True)
class GroupChange:
    unique_id: int
    name: str
    club: str | None
    ca: int
    pa: int | None
    before: str | None
    after: str | None
    tracked_before: bool
    tracked_after: bool
    reasons: tuple[str, ...]


GROUP_COUNT_KEYS = ("a", "b", "c", "tracked")


@dataclass
class ChangeReport:
    """What re-classifying after a settings change did to the groups and the queue."""

    before: dict[str, int]
    after: dict[str, int]
    enqueue: Counter[tuple[str, str]]
    updated: int
    cancelled: int
    changes: list[GroupChange]
    """Players whose group or tracking changed, highest PA first, at most `limit`."""
    changed_total: int

    @property
    def empty(self) -> bool:
        return not (self.changed_total or self.enqueue or self.updated or self.cancelled)

    def summary(self) -> str:
        """One line: the group counts that changed and what happened to the queue."""
        parts = []
        counts = ", ".join(
            f"{'tracked' if k == 'tracked' else k.upper()} {self.before[k]} → {self.after[k]}"
            for k in GROUP_COUNT_KEYS
            if self.before[k] != self.after[k]
        )
        if counts:
            parts.append(counts)
        queue = [
            f"{n} {what}"
            for n, what in (
                (sum(self.enqueue.values()), "queued"),
                (self.updated, "updated"),
                (self.cancelled, "cancelled"),
            )
            if n
        ]
        if queue:
            parts.append("queue: " + ", ".join(queue))
        return "; ".join(parts) or "no player moved; the queue is unchanged"

    def to_json(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "enqueue": [[kind, group, n] for (kind, group), n in sorted(self.enqueue.items())],
            "updated": self.updated,
            "cancelled": self.cancelled,
            "changes": [{**asdict(c), "reasons": list(c.reasons)} for c in self.changes],
            "changed_total": self.changed_total,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ChangeReport:
        return cls(
            before=data["before"],
            after=data["after"],
            enqueue=Counter({(kind, group): n for kind, group, n in data["enqueue"]}),
            updated=data["updated"],
            cancelled=data["cancelled"],
            changes=[GroupChange(**{**c, "reasons": tuple(c["reasons"])}) for c in data["changes"]],
            changed_total=data["changed_total"],
        )


def _group_counts(classifications: dict[int, Classification]) -> dict[str, int]:
    counts = Counter(c.group.value for c in classifications.values() if c.group is not None)
    return {
        "a": counts["a"],
        "b": counts["b"],
        "c": counts["c"],
        "tracked": sum(1 for c in classifications.values() if c.tracked),
    }


def reclassify_with_report(
    conn: sqlite3.Connection, career_id: int, settings: Settings, *, limit: int = 50
) -> ChangeReport:
    """Re-classify and re-plan inside the caller's transaction, and report what moved.

    "Before" is the groups as stored, so one classification is enough: the counts before are
    the counts after with each move taken back.
    """
    result = classify_and_plan(conn, career_id, settings)
    after = _group_counts(result.classifications)
    before = dict(after)
    for m in result.moved:
        if m.group_after:
            before[m.group_after] -= 1
        if m.group_before:
            before[m.group_before] += 1
        before["tracked"] += int(m.tracked_before) - int(m.tracked_after)

    moves = {m.unique_id: m for m in result.moved}
    rows: dict[int, sqlite3.Row] = {}
    uids = list(moves)
    for i in range(0, len(uids), 500):
        chunk = uids[i : i + 500]
        rows.update(
            (r["unique_id"], r)
            for r in conn.execute(
                "SELECT p.unique_id, p.name, p.ca, p.pa, c.name AS club FROM players p"
                " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
                f" WHERE p.career_id = ? AND p.unique_id IN ({', '.join('?' * len(chunk))})",
                (career_id, *chunk),
            )
        )

    def sort_key(uid: int) -> tuple[int, int, int]:
        r = rows[uid]
        return -(r["pa"] or 0), -r["ca"], uid

    changes = []
    for uid in sorted(moves, key=sort_key)[:limit]:
        m, r = moves[uid], rows[uid]
        changes.append(
            GroupChange(
                unique_id=uid,
                name=r["name"],
                club=r["club"],
                ca=r["ca"],
                pa=r["pa"],
                before=m.group_before,
                after=m.group_after,
                tracked_before=m.tracked_before,
                tracked_after=m.tracked_after,
                reasons=result.classifications[uid].reasons,
            )
        )
    p = result.plan
    return ChangeReport(
        before=before,
        after=after,
        enqueue=Counter((e.job.kind, e.job.group.value) for e in p.enqueue),
        updated=len(p.update),
        cancelled=len(p.cancel),
        changes=changes,
        changed_total=len(moves),
    )


def record_failed_sync(
    conn: sqlite3.Connection,
    save_file: str,
    error: str,
    build: str | None = None,
    *,
    save_mtime: float | None = None,
    unsupported: bool = False,
    trigger: str | None = None,
) -> None:
    """Note a sync that could not read its save. Existing pictures and jobs are untouched.

    With `save_mtime`, the save watcher counts that version of the save as dealt with and
    does not try it again until it changes; leave it out for a failure worth retrying.
    """
    now = _now()
    with transaction(conn):
        conn.execute(
            "INSERT INTO sync_runs (started_at, finished_at, save_file, save_mtime, build, error,"
            " unsupported, trigger) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, now, save_file, save_mtime, build, error, int(unsupported), trigger),
        )


def sync_of_save(conn: sqlite3.Connection, save_file: str, save_mtime: float) -> sqlite3.Row | None:
    """The latest sync, successful or not, that dealt with this version of a save, if any.

    A failure recorded without a save time (the save was still being written) does not count.
    """
    return conn.execute(
        "SELECT id, error, unsupported FROM sync_runs WHERE save_file = ?"
        " AND save_mtime BETWEEN ? AND ? ORDER BY id DESC LIMIT 1",
        (save_file, save_mtime - 0.001, save_mtime + 0.001),
    ).fetchone()
