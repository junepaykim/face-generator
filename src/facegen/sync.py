"""Sync: store a save snapshot in the state database, classify players, and plan jobs.

This is the glue between the pure modules and the database. A sync runs in one transaction,
so a dry run is the same work rolled back at the end.

    resolve_career -> ingest -> classify_and_plan -> report
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from facegen.age import age_wording
from facegen.classify import Classification, Group, PlayerFacts, classify
from facegen.datasource.records import Gender, Snapshot
from facegen.db import ACTIVE_JOB_STATUSES, transaction
from facegen.identity import (
    Person,
    best_career,
    career_key,
    detect_newgen_boundary,
    fingerprint,
    is_newgen,
    person_hash,
)
from facegen.jobs import apply_plan as apply_plan
from facegen.leagues import clubs_in_competitions, my_league_competitions
from facegen.planner import ImageState, JobState, Plan, PlayerPlanState, plan
from facegen.settings import Settings


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


# --- career ------------------------------------------------------------------------------


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


# --- ingest ------------------------------------------------------------------------------

_PLAYER_COLUMNS = (
    "career_id",
    "unique_id",
    "name",
    "birth_date",
    "age",
    "height_cm",
    "ca",
    "pa",
    "is_goalkeeper",
    "strength",
    "club_id",
    "loan_parent_club_id",
    "home_grown_club_ids",
    "club_since",
    "is_newgen",
    "in_scope",
    "at_my_club",
    "in_my_league",
    "in_continental",
    "first_club_id",
    "academy",
    "present",
    "first_seen_sync",
    "last_seen_sync",
)
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


def ingest(
    conn: sqlite3.Connection,
    career: Career,
    snapshot: Snapshot,
    settings: Settings,
    sync_id: int,
) -> IngestStats:
    """Write the snapshot's clubs and players, with the facts the rules need."""
    cid = career.id
    prev = {
        r["unique_id"]: r
        for r in conn.execute(
            "SELECT unique_id, is_newgen, in_scope, first_club_id, club_id, club_since,"
            " at_my_club, present, academy FROM players WHERE career_id = ?",
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
        # before that, the home-grown data alone decides (decisions.md, open question 2).
        first_club = old["first_club_id"] if old else (None if first_sync else p.club_id)
        academy = bool(old and old["academy"]) or (
            newgen
            and my_club is not None
            and (first_club == my_club or my_club in p.home_grown_club_ids)
        )
        if at_my_club and not first_sync and not (old and old["present"] and old["at_my_club"]):
            arrivals += 1
        if old is not None and old["club_id"] == p.club_id and old["club_since"]:
            club_since = old["club_since"]
        else:
            arrived = p.club_join_date or game_date
            if arrived and game_date and arrived > game_date:
                arrived = game_date  # the save holds some join dates still to come
            club_since = _iso(arrived)
        rows.append(
            (
                cid,
                p.unique_id,
                p.name,
                _iso(p.birth_date),
                p.age,
                p.height_cm,
                p.ca,
                p.pa,
                int(p.is_goalkeeper),
                p.strength,
                p.club_id,
                p.loan_parent_club_id,
                json.dumps(list(p.home_grown_club_ids)),
                club_since,
                int(newgen),
                int(in_scope),
                int(at_my_club),
                int(p.club_id in league_clubs),
                int(p.club_id in continental_clubs),
                first_club,
                int(academy),
                1,
                sync_id,
                sync_id,
            )
        )
    conn.executemany(_UPSERT_PLAYER, rows)
    conn.execute(
        "UPDATE players SET present = 0 WHERE career_id = ? AND last_seen_sync != ?",
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
    return IngestStats(first_sync, len(league_clubs), len(continental_clubs), arrivals)


# --- classify and plan -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanResult:
    plan: Plan
    classifications: dict[int, Classification]


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

    classifications: dict[int, Classification] = {}
    changed: list[tuple[Any, ...]] = []
    states: list[PlayerPlanState] = []
    for r in conn.execute("SELECT * FROM players WHERE career_id = ?", (career_id,)):
        uid = r["unique_id"]
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
            )
            c = classify(facts, settings.groups)
        else:
            c = Classification(None, False, ("not in the latest save",))
        classifications[uid] = c
        group = c.group.value if c.group else None
        reasons = json.dumps(list(c.reasons))
        if (group, int(c.tracked), reasons) != (r["group_name"], r["tracked"], r["reasons"]):
            changed.append((group, int(c.tracked), reasons, career_id, uid))
        if c.group is None and uid not in active:
            continue
        states.append(
            PlayerPlanState(
                unique_id=uid,
                classification=c,
                club_id=r["club_id"],
                age_band=age_wording(r["age"], c.group, bool(r["at_my_club"]), settings.age),
                at_my_club=bool(r["at_my_club"]),
                club_since=_date(r["club_since"]),
                pa=r["pa"],
                image=images.get(uid),
                active_job=active.get(uid),
                last_failed_job=failed.get(uid),
                manual_allowed=bool(r["present"] and r["in_scope"] and r["pin"] != "never"),
                manual_band=(
                    age_wording(r["age"], active[uid].group, bool(r["at_my_club"]), settings.age)
                    if uid in active
                    else None
                ),
            )
        )
    return PlanResult(plan(states), classifications), changed


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


# --- reports -----------------------------------------------------------------------------


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
    """One row per present player: the Phase 0.1 exit table plus group and reasons."""
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
        " SUM(in_scope) AS in_scope,"
        " SUM(in_scope AND is_newgen) AS newgens,"
        " SUM(group_name = 'a') AS a,"
        " SUM(group_name = 'a' AND at_my_club) AS a_my_club,"
        " SUM(group_name = 'b') AS b,"
        " SUM(group_name = 'c') AS c,"
        " SUM(tracked) AS tracked,"
        " SUM(at_my_club) AS at_my_club,"
        " SUM(academy) AS academy"
        " FROM players WHERE career_id = ? AND present = 1",
        (career_id,),
    ).fetchone()
    counts = {k: v or 0 for k, v in dict(row).items()}
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
) -> SyncReport:
    """Run one sync in a transaction; a dry run rolls everything back.

    `save_mtime` is the save's modification time as read. The career keeps the newest it has
    seen; the career with the newest save is the active one (spec section 8). `trigger` says
    who started the sync: 'poller', 'sync now' or 'command'.
    """
    with transaction(conn, rollback=dry_run):
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
        warnings.extend(snapshot.meta.warnings)

        return SyncReport(
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
        )


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


@dataclass
class SettingsPreview:
    """How a settings change would move players between groups and change the queue."""

    before: dict[str, int]
    after: dict[str, int]
    enqueue: Counter[tuple[str, str]]
    updated: int
    cancelled: int
    changes: list[GroupChange]
    """Players whose group or tracking would change, best first, at most `limit`."""
    changed_total: int

    @property
    def empty(self) -> bool:
        return not (self.changed_total or self.enqueue or self.updated or self.cancelled)


def _group_counts(classifications: dict[int, Classification]) -> dict[str, int]:
    counts = Counter(c.group.value for c in classifications.values() if c.group is not None)
    return {
        "a": counts["a"],
        "b": counts["b"],
        "c": counts["c"],
        "tracked": sum(1 for c in classifications.values() if c.tracked),
    }


def preview_settings(
    conn: sqlite3.Connection,
    career_id: int,
    current: Settings,
    proposed: Settings,
    *,
    limit: int = 50,
) -> SettingsPreview:
    """What applying `proposed` would do, against the stored snapshot; writes nothing.

    Both classifications run inside one read transaction, so a sync or the worker writing
    meanwhile cannot make the two sides disagree.
    """
    conn.execute("BEGIN")
    try:
        before, _ = _classify(conn, career_id, current)
        after, _ = _classify(conn, career_id, proposed)
        moved = [
            uid
            for uid, c in after.classifications.items()
            if (c.group, c.tracked)
            != (before.classifications[uid].group, before.classifications[uid].tracked)
        ]
        wanted = set(moved)
        rows = {
            r["unique_id"]: r
            for r in conn.execute(
                "SELECT p.unique_id, p.name, p.ca, p.pa, c.name AS club FROM players p"
                " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
                " WHERE p.career_id = ?",
                (career_id,),
            )
            if r["unique_id"] in wanted
        }
    finally:
        conn.execute("ROLLBACK")

    def sort_key(uid: int) -> tuple[int, int]:
        r = rows[uid]
        return -(r["pa"] or 0), -r["ca"]

    changes = []
    for uid in sorted(moved, key=sort_key)[:limit]:
        b, a, r = before.classifications[uid], after.classifications[uid], rows[uid]
        changes.append(
            GroupChange(
                unique_id=uid,
                name=r["name"],
                club=r["club"],
                ca=r["ca"],
                pa=r["pa"],
                before=b.group.value if b.group else None,
                after=a.group.value if a.group else None,
                tracked_before=b.tracked,
                tracked_after=a.tracked,
                reasons=a.reasons or b.reasons,
            )
        )
    p = after.plan
    return SettingsPreview(
        before=_group_counts(before.classifications),
        after=_group_counts(after.classifications),
        enqueue=Counter((e.job.kind, e.job.group.value) for e in p.enqueue),
        updated=len(p.update),
        cancelled=len(p.cancel),
        changes=changes,
        changed_total=len(moved),
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
