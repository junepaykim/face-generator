"""Queue records and transitions. Mutations are serialized by SQLite write transactions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from facegen.age import age_wording
from facegen.db import ACTIVE_CAREER_ORDER, transaction
from facegen.domain import FinishedStatus, GroupName, JobKind
from facegen.planner import TIER_MANUAL, Plan
from facegen.settings import BodyThresholds, Settings
from facegen.traits import CharacterSheet, load_sheet, sheet_for


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobGone(RuntimeError):
    """The selected job was changed, cancelled, or claimed before it could start."""


@dataclass(frozen=True, slots=True)
class Career:
    id: int
    key: str
    display_name: str


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


def load_career(conn: sqlite3.Connection, key: str | None = None) -> Career | None:
    """The named career, or the active one: the career whose save was written last."""
    if key is not None:
        row = conn.execute(
            "SELECT id, key, display_name FROM careers WHERE key = ?", (key,)
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT id, key, display_name FROM careers ORDER BY {ACTIVE_CAREER_ORDER} LIMIT 1"
        ).fetchone()
    return Career(row["id"], row["key"], row["display_name"]) if row else None


def queued_jobs(
    conn: sqlite3.Connection, career_id: int, f: JobFilter, *, job_id: int | None = None
) -> list[Job]:
    """Queued jobs that match the filter, in queue order."""
    rows = conn.execute(
        "SELECT j.id, j.unique_id, j.kind, j.group_name, j.target_club_id, j.target_band,"
        " j.manual, j.traits_version, j.revision, p.name, p.at_my_club, p.is_goalkeeper,"
        " p.height_cm, p.strength, p.age"
        " FROM jobs j JOIN players p ON p.career_id = j.career_id AND p.unique_id = j.unique_id"
        " WHERE j.career_id = ? AND j.status = 'queued'"
        + (" AND j.id = ?" if job_id is not None else "")
        + " ORDER BY j.tier, j.rank, j.id",
        (career_id, job_id) if job_id is not None else (career_id,),
    ).fetchall()
    jobs = [
        Job(
            id=r["id"],
            unique_id=r["unique_id"],
            kind=r["kind"],
            group=r["group_name"],
            target_club_id=r["target_club_id"],
            target_band=r["target_band"],
            name=r["name"],
            at_my_club=bool(r["at_my_club"]),
            goalkeeper=bool(r["is_goalkeeper"]),
            height_cm=r["height_cm"],
            strength=r["strength"],
            age=r["age"],
            manual=bool(r["manual"]),
            traits_version=r["traits_version"],
            revision=r["revision"],
        )
        for r in rows
    ]
    return [j for j in jobs if f.accepts(j)]


def claim(conn: sqlite3.Connection, career_id: int, selected: Job) -> Job:
    """Validate and claim a selection in one write transaction; stale selections retry."""
    with transaction(conn):
        current = next(iter(queued_jobs(conn, career_id, JobFilter(), job_id=selected.id)), None)
        if current != selected:
            raise JobGone(f"job {selected.id} changed or is no longer queued")
        player = conn.execute(
            "SELECT present, in_scope, pin FROM players WHERE career_id = ? AND unique_id = ?",
            (career_id, selected.unique_id),
        ).fetchone()
        if not player["present"] or not player["in_scope"] or player["pin"] == "never":
            conn.execute(
                "UPDATE jobs SET status = 'cancelled', error = 'player no longer eligible',"
                " updated_at = ? WHERE id = ?",
                (now(), selected.id),
            )
            current = None
        else:
            conn.execute(
                "UPDATE jobs SET status = 'running', error = NULL, updated_at = ? WHERE id = ?",
                (now(), current.id),
            )
    if current is None:
        raise JobGone(f"job {selected.id}'s player is no longer eligible")
    return current


def release(conn: sqlite3.Connection, job_id: int, error: str) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE jobs SET status = 'queued', error = ?, updated_at = ?"
            " WHERE id = ? AND status = 'running'",
            (error, now(), job_id),
        )


def requeue_stale(conn: sqlite3.Connection) -> int:
    """Caller holds the worker lock. Preserve the interrupted job's sheet allocation."""
    with transaction(conn):
        return conn.execute(
            "UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'running'",
            (now(),),
        ).rowcount


def finish(
    conn: sqlite3.Connection,
    job_id: int,
    status: FinishedStatus,
    attempts: int,
    error: str | None = None,
) -> None:
    """Caller commits the picture and completion in the same transaction."""
    cur = conn.execute(
        "UPDATE jobs SET status = ?, attempts = ?, error = ?, updated_at = ?"
        " WHERE id = ? AND status = 'running'",
        (status, attempts, error, now(), job_id),
    )
    if cur.rowcount != 1:
        raise JobGone(f"job {job_id} is no longer running")


def allocate_sheet(
    conn: sqlite3.Connection, career: Career, job: Job, body: BodyThresholds
) -> tuple[int, CharacterSheet]:
    """Assign a sheet once per job, atomically with sampling, before GPU work starts."""
    with transaction(conn):
        row = conn.execute(
            "SELECT traits_version FROM jobs WHERE id = ? AND status = 'running'", (job.id,)
        ).fetchone()
        if row is None:
            raise JobGone(f"job {job.id} is no longer running")
        if row[0] is not None:
            found = load_sheet(conn, career.id, job.unique_id, row[0])
            if found is None:
                raise RuntimeError(f"job {job.id}'s assigned character sheet is missing")
            return found
        version, sheet = sheet_for(
            conn,
            career.id,
            career.key,
            job.unique_id,
            height_cm=job.height_cm,
            strength=job.strength,
            body=body,
            new_face=job.kind == "reroll",
            activate=False,
        )
        conn.execute("UPDATE jobs SET traits_version = ? WHERE id = ?", (version, job.id))
        return version, sheet


def apply_plan(conn: sqlite3.Connection, career_id: int, p: Plan) -> None:
    timestamp = now()
    conn.executemany(
        "INSERT INTO jobs (career_id, unique_id, kind, group_name, manual, tier, rank, status,"
        " target_club_id, target_band, reason, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, 0, ?, ?, 'queued', ?, ?, ?, ?, ?)",
        [
            (
                career_id,
                a.unique_id,
                a.job.kind,
                a.job.group.value,
                a.job.tier,
                a.job.rank,
                a.job.target_club_id,
                a.job.target_band,
                a.job.reason,
                timestamp,
                timestamp,
            )
            for a in p.enqueue
        ],
    )
    conn.executemany(
        "UPDATE jobs SET kind = ?, group_name = ?, tier = ?, rank = ?, target_club_id = ?,"
        " target_band = ?, reason = ?, updated_at = ?, revision = revision + 1"
        " WHERE id = ? AND status = 'queued'",
        [
            (
                a.job.kind,
                a.job.group.value,
                a.job.tier,
                a.job.rank,
                a.job.target_club_id,
                a.job.target_band,
                a.job.reason,
                timestamp,
                a.job_id,
            )
            for a in p.update
        ],
    )
    conn.executemany(
        "UPDATE jobs SET status = 'cancelled', error = ?, updated_at = ?"
        " WHERE id = ? AND status = 'queued'",
        [(a.reason, timestamp, a.job_id) for a in p.cancel],
    )


def queue_manual(
    conn: sqlite3.Connection,
    career_id: int,
    p: sqlite3.Row,
    kind: JobKind,
    group: str,
    rank: float,
    reason: str,
    settings: Settings,
) -> None:
    """Queue a manual job for the player, taking over their queued automatic job if any."""
    band = age_wording(p["age"], group, bool(p["at_my_club"]), settings.age)
    timestamp = now()
    values = (kind, group, TIER_MANUAL, rank, p["club_id"], band, reason)
    cur = conn.execute(
        "UPDATE jobs SET kind = ?, group_name = ?, manual = 1, tier = ?, rank = ?,"
        " target_club_id = ?, target_band = ?, reason = ?, error = NULL, updated_at = ?,"
        " traits_version = NULL, revision = revision + 1"
        " WHERE career_id = ? AND unique_id = ? AND status = 'queued'",
        (*values, timestamp, career_id, p["unique_id"]),
    )
    if cur.rowcount == 0:
        conn.execute(
            "INSERT INTO jobs (career_id, unique_id, kind, group_name, manual, tier, rank,"
            " status, target_club_id, target_band, reason, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 1, ?, ?, 'queued', ?, ?, ?, ?, ?)",
            (
                career_id,
                p["unique_id"],
                kind,
                group,
                TIER_MANUAL,
                rank,
                *values[4:],
                timestamp,
                timestamp,
            ),
        )
