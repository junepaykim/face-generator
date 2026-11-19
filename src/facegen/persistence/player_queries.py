"""SQLite reads for the player roster.

The page query selects the matching IDs using only the joins needed for filtering and
ordering. It fetches full rows for those IDs. The Review and club-name filters each use
a single lookup across the roster. See docs/research.md for the index measurements.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

PER_PAGE = 100

# The player's latest job that was not cancelled: queued, running, done or failed. "failed"
# here means the player is on the review list.
_LATEST_JOB = (
    "(SELECT jj.status FROM jobs jj WHERE jj.career_id = p.career_id"
    " AND jj.unique_id = p.unique_id AND jj.status != 'cancelled' ORDER BY jj.id DESC LIMIT 1)"
)
# The same test for the Review filter, as a list of players found once in the jobs table (not
# correlated with the player row, which asked the jobs for every player).
_IN_REVIEW = (
    "(p.career_id, p.unique_id) IN (SELECT rj.career_id, rj.unique_id FROM jobs rj"
    " WHERE rj.status = 'failed' AND rj.id = (SELECT MAX(rjj.id) FROM jobs rjj"
    " WHERE rjj.career_id = rj.career_id AND rjj.unique_id = rj.unique_id"
    " AND rjj.status != 'cancelled'))"
)
"""Its aliases are not the page's own (`j`), so `_count_from` joins nothing for it."""

SORTS = {
    "ca": "p.ca DESC, p.pa DESC",
    "pa": "p.pa DESC, p.ca DESC",
    "name": "p.name COLLATE NOCASE",
    "age": "p.age, p.name COLLATE NOCASE",
    "club": "c.name COLLATE NOCASE, p.ca DESC",
}


@dataclass(frozen=True, slots=True)
class PlayerFilter:
    q: str = ""
    club: str = ""
    club_id: int | None = None
    league: str = "any"
    ca_min: int | None = None
    ca_max: int | None = None
    pa_min: int | None = None
    pa_max: int | None = None
    origin: str = "any"
    role: str = "any"
    group: str = "any"
    picture: str = "any"
    photo: str = "any"
    scope: str = "mens"
    sort: str = "ca"
    page: int = 1


def _where(f: PlayerFilter) -> tuple[str, list[Any]]:
    clauses = ["p.career_id = ?", "p.present = 1"]
    args: list[Any] = []
    if f.q:
        if f.q.isdigit():
            clauses.append("(p.unique_id = ? OR p.name LIKE ?)")
            args += [int(f.q), f"%{f.q}%"]
        else:
            clauses.append("p.name LIKE ?")
            args.append(f"%{f.q}%")
    if f.club_id is not None:
        clauses.append("p.club_id = ?")
        args.append(f.club_id)
    elif f.club:
        # The matching clubs found once, as with the Review filter; a player without a club
        # (or whose club is not stored) matches neither way.
        clauses.append(
            "(p.career_id, p.club_id) IN (SELECT career_id, club_id FROM clubs WHERE name LIKE ?)"
        )
        args.append(f"%{f.club}%")
    match f.league:
        case "mine":
            clauses.append("p.in_my_league = 1")
        case "continental":
            clauses.append("p.in_continental = 1")
        case "other":
            clauses.append("p.in_my_league = 0 AND p.in_continental = 0")
    for col, low, high in (("p.ca", f.ca_min, f.ca_max), ("p.pa", f.pa_min, f.pa_max)):
        if low is not None:
            clauses.append(f"{col} >= ?")
            args.append(low)
        if high is not None:
            clauses.append(f"{col} <= ?")
            args.append(high)
    if f.origin == "newgen":
        clauses.append("p.is_newgen = 1")
    elif f.origin == "existing":
        clauses.append("p.is_newgen = 0")
    match f.role:
        case "players":
            clauses.append("p.is_staff = 0 AND p.media_role IS NULL")
        case "youth":
            clauses.append("p.youth_team = 1")
        case "staff":
            clauses.append("p.is_staff = 1")
        case "media":
            clauses.append("p.media_role IS NOT NULL")
    match f.group:
        case "grouped":
            clauses.append("p.group_name IS NOT NULL")
        case "a" | "b" | "c":
            clauses.append("p.group_name = ?")
            args.append(f.group)
        case "none":
            clauses.append("p.group_name IS NULL")
        case "pinned":
            clauses.append("p.pin IS NOT NULL")
    match f.picture:
        case "yes":
            clauses.append("i.id IS NOT NULL")
        case "no":
            clauses.append("i.id IS NULL")
        case "queued":
            clauses.append("j.id IS NOT NULL")
        case "review":
            clauses.append(_IN_REVIEW)
    match f.photo:
        case "game":
            clauses.append("ph.game = 1")
        case "facepack":
            clauses.append("ph.facepack = 1")
        case "none":
            clauses.append("ph.unique_id IS NULL")
    if f.scope == "mens":
        clauses.append("p.in_scope = 1")
    return " AND ".join(clauses), args


_JOINS = {
    "c": " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id",
    "i": " LEFT JOIN images i ON i.career_id = p.career_id AND i.unique_id = p.unique_id"
    "   AND i.is_current = 1",
    "j": " LEFT JOIN jobs j ON j.career_id = p.career_id AND j.unique_id = p.unique_id"
    "   AND j.status IN ('queued', 'running')",
    "ph": " LEFT JOIN photos ph ON ph.unique_id = p.unique_id",
}
_FROM = " FROM players p" + "".join(_JOINS.values())
_COLUMNS = (
    "SELECT p.unique_id, p.name, p.age, p.ca, p.pa, p.is_newgen, p.academy, p.at_my_club,"
    " p.in_my_league, p.in_continental, p.in_scope, p.group_name, p.tracked, p.reasons,"
    " p.pin, p.club_id, p.is_goalkeeper, p.is_staff, p.staff_job, p.media_role,"
    " p.youth_team, p.local_media, c.name AS club,"
    " i.id AS image_id, ph.game AS photo_game, ph.facepack AS photo_facepack,"
    " i.published_path, i.group_name AS image_group, j.status AS job_status,"
    f" j.kind AS job_kind, j.manual AS job_manual, {_LATEST_JOB} AS latest_job"
)


def _count_from(where: str) -> str:
    """`_FROM` with only the joins the filter uses. Each joins at most one row per player (a
    unique key: one club, one current image, one active job, one photo row), so the others
    cannot change the count; leaving them out makes counting 160,000 players several times
    faster."""
    used = (j for alias, j in _JOINS.items() if re.search(rf"\b{alias}\.", where))
    return " FROM players p" + "".join(used)


def _decode_player(r: sqlite3.Row) -> dict[str, Any]:
    row = dict(r)
    row["reasons"] = json.loads(row["reasons"] or "[]")
    return row


def players_page(
    conn: sqlite3.Connection, career_id: int, f: PlayerFilter, per_page: int = PER_PAGE
) -> tuple[list[dict[str, Any]], int]:
    """One page and the total number of matches. The unique ID breaks sort ties in both
    the page of IDs and the final result."""
    where, args = _where(f)
    total = conn.execute(
        f"SELECT COUNT(*){_count_from(where)} WHERE {where}", (career_id, *args)
    ).fetchone()[0]
    order = f"{SORTS[f.sort]}, p.unique_id"
    window = (
        f"SELECT p.unique_id{_count_from(f'{where} {order}')} WHERE {where}"
        f" ORDER BY {order} LIMIT ? OFFSET ?"
    )
    rows = conn.execute(
        f"{_COLUMNS}{_FROM} WHERE p.career_id = ? AND p.unique_id IN ({window}) ORDER BY {order}",
        (career_id, career_id, *args, per_page, (f.page - 1) * per_page),
    ).fetchall()
    return [_decode_player(r) for r in rows], total


def player_row(conn: sqlite3.Connection, career_id: int, uid: int) -> dict[str, Any] | None:
    """The stored player and current club, image, photo and job for the detail page."""
    r = conn.execute(
        "SELECT p.*, c.name AS club, lp.name AS loan_parent_club, i.id AS image_id,"
        " ph.game AS photo_game, ph.facepack AS photo_facepack,"
        f" j.status AS job_status, j.kind AS job_kind, j.manual AS job_manual,"
        f" {_LATEST_JOB} AS latest_job{_FROM}"
        " LEFT JOIN clubs lp ON lp.career_id = p.career_id AND lp.club_id = p.loan_parent_club_id"
        " WHERE p.career_id = ? AND p.unique_id = ?",
        (career_id, uid),
    ).fetchone()
    return _decode_player(r) if r is not None else None


def club_name(conn: sqlite3.Connection, career_id: int, club_id: int) -> str | None:
    row = conn.execute(
        "SELECT name FROM clubs WHERE career_id = ? AND club_id = ?", (career_id, club_id)
    ).fetchone()
    return row[0] if row else None
