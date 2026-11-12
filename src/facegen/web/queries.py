"""Read-only queries behind the web pages. Plain SQL over the state database; no web here.

The players list covers every present player of the career (33,000 in the Colchester save),
so it filters and pages in SQL.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Any
from urllib.parse import urlencode

from facegen.traits import CharacterSheet

PER_PAGE = 100

# The player's latest job that was not cancelled: queued, running, done or failed. "failed"
# here means the player is on the review list.
_LATEST_JOB = (
    "(SELECT jj.status FROM jobs jj WHERE jj.career_id = p.career_id"
    " AND jj.unique_id = p.unique_id AND jj.status != 'cancelled' ORDER BY jj.id DESC LIMIT 1)"
)

LEAGUES = {
    "any": "any league",
    "mine": "my league",
    "continental": "continental competition",
    "other": "elsewhere",
}
ORIGINS = {"any": "newgen or pre-existing", "newgen": "newgens", "existing": "pre-existing"}
GROUP_FILTERS = {
    "any": "any group",
    "grouped": "in a group",
    "a": "group A",
    "b": "group B",
    "c": "group C",
    "none": "no group",
    "pinned": "pinned",
}
PICTURE_FILTERS = {
    "any": "any picture state",
    "yes": "has a picture",
    "no": "no picture",
    "queued": "queued or drawing",
    "review": "in review",
}
SCOPES = {"mens": "men's football", "all": "all players"}
SORTS = {
    "ca": "p.ca DESC, p.pa DESC",
    "pa": "p.pa DESC, p.ca DESC",
    "name": "p.name COLLATE NOCASE",
    "age": "p.age, p.name COLLATE NOCASE",
    "club": "club COLLATE NOCASE, p.ca DESC",
}


def _int(value: str | None, low: int = 0, high: int = 10**12) -> int | None:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if low <= n <= high else None


def _choice(value: str | None, options: Mapping[str, Any], default: str) -> str:
    return value if value in options else default


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
    group: str = "any"
    picture: str = "any"
    scope: str = "mens"
    sort: str = "ca"
    page: int = 1

    @classmethod
    def from_query(cls, params: Mapping[str, str]) -> PlayerFilter:
        """From the page's query string; anything malformed falls back to its default."""
        get = params.get
        return cls(
            q=(get("q") or "").strip()[:100],
            club=(get("club") or "").strip()[:100],
            club_id=_int(get("club_id")),
            league=_choice(get("league"), LEAGUES, "any"),
            ca_min=_int(get("ca_min"), 0, 200),
            ca_max=_int(get("ca_max"), 0, 200),
            pa_min=_int(get("pa_min"), 0, 200),
            pa_max=_int(get("pa_max"), 0, 200),
            origin=_choice(get("origin"), ORIGINS, "any"),
            group=_choice(get("group"), GROUP_FILTERS, "any"),
            picture=_choice(get("picture"), PICTURE_FILTERS, "any"),
            scope=_choice(get("scope"), SCOPES, "mens"),
            sort=_choice(get("sort"), SORTS, "ca"),
            page=_int(get("page"), 1, 100_000) or 1,
        )

    def query(self, **changes: Any) -> str:
        """The query string for this filter with some fields changed; defaults are left out."""
        f = replace(self, **changes)
        default = PlayerFilter()
        return urlencode(
            {
                k: v
                for k, v in asdict(f).items()
                if v != getattr(default, k) and v is not None and v != ""
            }
        )

    @property
    def active(self) -> bool:
        """Whether any filter other than the page and sort is set."""
        default = PlayerFilter()
        return any(
            getattr(self, f.name) != getattr(default, f.name)
            for f in fields(self)
            if f.name not in ("page", "sort")
        )


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
        clauses.append("c.name LIKE ?")
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
            clauses.append(f"{_LATEST_JOB} = 'failed'")
    if f.scope == "mens":
        clauses.append("p.in_scope = 1")
    return " AND ".join(clauses), args


_FROM = (
    " FROM players p"
    " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
    " LEFT JOIN images i ON i.career_id = p.career_id AND i.unique_id = p.unique_id"
    "   AND i.is_current = 1"
    " LEFT JOIN jobs j ON j.career_id = p.career_id AND j.unique_id = p.unique_id"
    "   AND j.status IN ('queued', 'running')"
)


def _player_row(r: sqlite3.Row) -> dict[str, Any]:
    row = dict(r)
    row["reasons"] = json.loads(row["reasons"] or "[]")
    return row


def players_page(
    conn: sqlite3.Connection, career_id: int, f: PlayerFilter, per_page: int = PER_PAGE
) -> tuple[list[dict[str, Any]], int]:
    """One page of players matching the filter, and how many match in all."""
    where, args = _where(f)
    total = conn.execute(f"SELECT COUNT(*){_FROM} WHERE {where}", (career_id, *args)).fetchone()[0]
    rows = conn.execute(
        "SELECT p.unique_id, p.name, p.age, p.ca, p.pa, p.is_newgen, p.academy, p.at_my_club,"
        " p.in_my_league, p.in_continental, p.in_scope, p.group_name, p.tracked, p.reasons,"
        " p.pin, p.club_id, p.is_goalkeeper, c.name AS club, i.id AS image_id,"
        " i.published_path, i.group_name AS image_group, j.status AS job_status,"
        f" j.kind AS job_kind, j.manual AS job_manual, {_LATEST_JOB} AS latest_job"
        f"{_FROM} WHERE {where} ORDER BY {SORTS[f.sort]}, p.unique_id LIMIT ? OFFSET ?",
        (career_id, *args, per_page, (f.page - 1) * per_page),
    ).fetchall()
    return [_player_row(r) for r in rows], total


def player_detail(conn: sqlite3.Connection, career_id: int, uid: int) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT p.*, c.name AS club, lp.name AS loan_parent_club, i.id AS image_id,"
        f" j.status AS job_status, j.kind AS job_kind, j.manual AS job_manual,"
        f" {_LATEST_JOB} AS latest_job{_FROM}"
        " LEFT JOIN clubs lp ON lp.career_id = p.career_id AND lp.club_id = p.loan_parent_club_id"
        " WHERE p.career_id = ? AND p.unique_id = ?",
        (career_id, uid),
    ).fetchone()
    if r is None:
        return None
    player = _player_row(r)
    player["home_grown_club_ids"] = json.loads(player["home_grown_club_ids"] or "[]")
    images = [
        _image(row)
        for row in conn.execute(
            "SELECT i.*, c.name AS club FROM images i"
            " LEFT JOIN clubs c ON c.career_id = i.career_id AND c.club_id = i.club_id"
            " WHERE i.career_id = ? AND i.unique_id = ? ORDER BY i.version DESC",
            (career_id, uid),
        )
    ]
    sheet_row = conn.execute(
        "SELECT version, base_seed, sheet, created_at FROM traits"
        " WHERE career_id = ? AND unique_id = ? ORDER BY version DESC LIMIT 1",
        (career_id, uid),
    ).fetchone()
    sheet = None
    if sheet_row is not None:
        cs = CharacterSheet.from_json(sheet_row["sheet"], sheet_row["base_seed"])
        sheet = {
            "version": sheet_row["version"],
            "created_at": sheet_row["created_at"],
            "fields": {k: v for k, v in asdict(cs).items()},
        }
    jobs = [
        dict(j)
        for j in conn.execute(
            "SELECT id, kind, group_name, manual, status, reason, attempts, error, created_at,"
            " updated_at FROM jobs WHERE career_id = ? AND unique_id = ? ORDER BY id DESC LIMIT 20",
            (career_id, uid),
        )
    ]
    return {
        "player": player,
        "images": images,
        "current": next((i for i in images if i["is_current"]), None),
        "sheet": sheet,
        "jobs": jobs,
    }


def _image(row: sqlite3.Row) -> dict[str, Any]:
    image = dict(row)
    image["check"] = json.loads(image.pop("check_result") or "{}")
    image["render"] = json.loads(image.get("render") or "{}")
    image["rejected"] = bool(
        image["master_path"] and image["master_path"].endswith("-rejected.png")
    )
    return image


def counts(conn: sqlite3.Connection, career_id: int) -> dict[str, int]:
    row = conn.execute(
        "SELECT SUM(group_name = 'a') AS a, SUM(group_name = 'a' AND at_my_club) AS a_my_club,"
        " SUM(group_name = 'b') AS b, SUM(group_name = 'c') AS c, SUM(tracked) AS tracked,"
        " SUM(at_my_club) AS at_my_club, SUM(pin IS NOT NULL) AS pinned"
        " FROM players WHERE career_id = ? AND present = 1",
        (career_id,),
    ).fetchone()
    out = {k: v or 0 for k, v in dict(row).items()}
    pics = conn.execute(
        "SELECT COUNT(*) AS current, COUNT(published_path) AS published FROM images"
        " WHERE career_id = ? AND is_current = 1",
        (career_id,),
    ).fetchone()
    out.update(pictures=pics["current"], published=pics["published"])
    out["review"] = review_count(conn, career_id)
    jobs = dict(
        conn.execute(
            "SELECT status, COUNT(*) FROM jobs WHERE career_id = ?"
            " AND status IN ('queued', 'running') GROUP BY status",
            (career_id,),
        ).fetchall()
    )
    out.update(queued=jobs.get("queued", 0), running=jobs.get("running", 0))
    out["manual"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE career_id = ? AND status = 'queued' AND manual = 1",
        (career_id,),
    ).fetchone()[0]
    return out


def queue(conn: sqlite3.Connection, career_id: int, limit: int = 15) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT j.id, j.unique_id, j.kind, j.group_name, j.manual, j.tier, j.status, j.reason,"
            " p.name, p.at_my_club, c.name AS club FROM jobs j"
            " JOIN players p ON p.career_id = j.career_id AND p.unique_id = j.unique_id"
            " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
            " WHERE j.career_id = ? AND j.status IN ('queued', 'running')"
            " ORDER BY j.status = 'queued', j.tier, j.rank, j.id LIMIT ?",
            (career_id, limit),
        )
    ]


def last_sync(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    out = dict(row)
    out["counts"] = json.loads(out["counts"] or "{}")
    return out


def _review_where() -> str:
    return (
        "j.career_id = ? AND j.status = 'failed' AND j.id = (SELECT MAX(jj.id) FROM jobs jj"
        " WHERE jj.career_id = j.career_id AND jj.unique_id = j.unique_id"
        " AND jj.status != 'cancelled')"
    )


def review_count(conn: sqlite3.Connection, career_id: int) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM jobs j WHERE {_review_where()}", (career_id,)
    ).fetchone()[0]


def review_items(conn: sqlite3.Connection, career_id: int) -> list[dict[str, Any]]:
    """Players whose latest job failed its checks, with the best rejected try if kept."""
    rows = conn.execute(
        "SELECT j.id AS job_id, j.unique_id, j.kind, j.group_name, j.error, j.updated_at,"
        " p.name, p.age, p.in_scope, p.present, c.name AS club FROM jobs j"
        " JOIN players p ON p.career_id = j.career_id AND p.unique_id = j.unique_id"
        " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        f" WHERE {_review_where()} ORDER BY j.updated_at DESC, j.id DESC",
        (career_id,),
    ).fetchall()
    items = []
    for r in rows:
        item = dict(r)
        image = conn.execute(
            "SELECT * FROM images WHERE career_id = ? AND unique_id = ? AND is_current = 0"
            " AND master_path LIKE '%-rejected.png' ORDER BY version DESC LIMIT 1",
            (career_id, r["unique_id"]),
        ).fetchone()
        item["image"] = _image(image) if image is not None else None
        items.append(item)
    return items
