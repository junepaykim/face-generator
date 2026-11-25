"""Read models and filter presentation for the web pages."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from kitshot.core.estimate import Drawn
from kitshot.core.man_traits import ManSheet
from kitshot.core.traits import CharacterSheet, personality_from_json, personality_leans
from kitshot.persistence import photo_index, player_queries
from kitshot.persistence.jobs import QUEUE_ORDER
from kitshot.persistence.player_queries import SORTS
from kitshot.pipeline.manual import restorable

LEAGUES = {
    "any": "any league",
    "mine": "my league",
    "continental": "continental competition",
    "other": "elsewhere",
}
ORIGINS = {"any": "newgen or pre-existing", "newgen": "newgens", "existing": "pre-existing"}
ROLES = {
    "any": "everyone",
    "players": "players",
    "youth": "my club's youth teams",
    "staff": "my club's staff",
    "media": "journalists and press officers",
}
GROUP_FILTERS = {
    "any": "any group",
    "grouped": "in a group",
    "a": "group A",
    "b": "group B",
    "c": "group C",
    "none": "no group",
    "pinned": "pinned",
    "tracked": "auto-update",
}
PICTURE_FILTERS = {
    "any": "any picture state",
    "yes": "has a picture",
    "no": "no picture",
    "queued": "queued or drawing",
    "review": "in review",
    "old_kit": "drawn with older kit info",
}
# Whether FM27 shows a photo other than ours: its own, or a facepack's (photos.py).
PHOTO_FILTERS = {
    "any": "any photo",
    "game": "FM27 photo",
    "facepack": "facepack picture",
    "none": "no photo or facepack",
}
SCOPES = {"mens": "men's football", "all": "all players"}
# Who the list shows (Phase 17h): the people this app draws, or everyone in the save.
SHOWN = {"targets": "picture targets", "everyone": "everyone in the save"}
FOLDED_FILTERS = ("league", "origin", "role", "scope", "shown")
"""The Players filters kept under "More filters" besides the CA and PA ranges (Phase 17g)."""


def _int(value: str | None, low: int = 0, high: int = 10**12) -> int | None:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if low <= n <= high else None


def _choice(value: str | None, options: Mapping[str, Any], default: str) -> str:
    return value if value in options else default


MAX_PAGE = 100_000


def _page(value: str | None) -> int:
    """The page asked for: below 1, or not a number, is the first; a page past the last is
    clamped by `one_page` once the number of pages is known (Phase 17b)."""
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return 1
    return min(max(n, 1), MAX_PAGE)


def page_count(total: int, per_page: int) -> int:
    """How many pages `total` rows fill; an empty list still has its one page."""
    return max(1, -(-total // per_page))


@dataclass(frozen=True, slots=True)
class PlayerPageFilter(player_queries.PlayerFilter):
    """Roster filters with request parsing and URL generation for the Players page."""

    shown: str = "targets"
    """The page lists the picture targets unless asked for everyone (Phase 17h, spec 7)."""

    @classmethod
    def from_query(cls, params: Mapping[str, str]) -> PlayerPageFilter:
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
            role=_choice(get("role"), ROLES, "any"),
            group=_choice(get("group"), GROUP_FILTERS, "any"),
            picture=_choice(get("picture"), PICTURE_FILTERS, "any"),
            photo=_choice(get("photo"), PHOTO_FILTERS, "any"),
            scope=_choice(get("scope"), SCOPES, "mens"),
            shown=_choice(get("shown"), SHOWN, "targets"),
            sort=_choice(get("sort"), SORTS, "ca"),
            page=_page(get("page")),
        )

    def params(self, **changes: Any) -> dict[str, Any]:
        """This filter's fields with some changed, defaults left out: the query string's
        fields, and the pager's hidden ones."""
        f = replace(self, **changes)
        default = PlayerPageFilter()
        return {
            k: v
            for k, v in asdict(f).items()
            if v != getattr(default, k) and v is not None and v != ""
        }

    def query(self, **changes: Any) -> str:
        """The query string for this filter with some fields changed; defaults are left out."""
        return urlencode(self.params(**changes))

    @property
    def active(self) -> bool:
        """Whether any filter other than the page and sort is set."""
        default = PlayerPageFilter()
        return any(
            getattr(self, f.name) != getattr(default, f.name)
            for f in fields(self)
            if f.name not in ("page", "sort")
        )

    @property
    def folded(self) -> int:
        """How many of the filters folded under "More filters" are set (Phase 17g): the
        badge on its button; any at all opens it. A min–max range counts once."""
        default = PlayerPageFilter()
        lists = sum(getattr(self, name) != getattr(default, name) for name in FOLDED_FILTERS)
        ranges = (self.ca_min, self.ca_max), (self.pa_min, self.pa_max)
        return lists + sum(low is not None or high is not None for low, high in ranges)

    def cleared(self) -> str:
        """The query string without the filters, keeping the sort (the Clear button)."""
        return PlayerPageFilter(sort=self.sort).query()


def player_detail(
    conn: sqlite3.Connection, career_id: int, uid: int, data_dir: Path | None = None
) -> dict[str, Any] | None:
    """Everything the player page shows. With `data_dir`, each earlier picture says whether
    it can be restored (it passed its check and its file is still there)."""
    player = player_queries.player_row(conn, career_id, uid)
    if player is None:
        return None
    player["home_grown_club_ids"] = json.loads(player["home_grown_club_ids"] or "[]")
    personality = personality_from_json(player.pop("personality", None))
    player["personality"] = None if personality is None else asdict(personality)
    images = [
        _image(row)
        for row in conn.execute(
            "SELECT i.*, c.name AS club FROM images i"
            " LEFT JOIN clubs c ON c.career_id = i.career_id AND c.club_id = i.club_id"
            " WHERE i.career_id = ? AND i.unique_id = ? ORDER BY i.version DESC",
            (career_id, uid),
        )
    ]
    for image in images:
        image["restorable"] = (
            data_dir is not None
            and not image["is_current"]
            and restorable(image["master_path"], image["check"])
            and (data_dir / image["master_path"]).is_file()
        )
    # The sheet the current picture was drawn from (its version, and the men's sheet for a man);
    # without a picture, the one the next picture uses.
    current = next((i for i in images if i["is_current"]), None)
    shown = current["traits_version"] if current is not None else None
    sheet_row = conn.execute(
        "SELECT version, base_seed, sheet, man_sheet, created_at FROM traits"
        " WHERE career_id = ? AND unique_id = ? AND version = COALESCE(?,"
        " (SELECT active_traits_version FROM players WHERE career_id = ? AND unique_id = ?),"
        " (SELECT MAX(version) FROM traits WHERE career_id = ? AND unique_id = ?))",
        (career_id, uid, shown, career_id, uid, career_id, uid),
    ).fetchone()
    sheet = None
    if sheet_row is not None:
        man = current is not None and current["gender"] == "man" and sheet_row["man_sheet"]
        cs: CharacterSheet | ManSheet = (
            ManSheet.from_json(sheet_row["man_sheet"], sheet_row["base_seed"])
            if man
            else CharacterSheet.from_json(sheet_row["sheet"], sheet_row["base_seed"])
        )
        sheet = {
            "version": sheet_row["version"],
            "created_at": sheet_row["created_at"],
            "man": bool(man),
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
        "current": current,
        "sheet": sheet,
        "leans": [
            {"reason": lean.describe(), "expression": lean.expression}
            for lean in personality_leans(personality)
        ],
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
    out = photo_index.player_counts(conn, career_id, include_groups=True)
    pics = conn.execute(
        "SELECT COUNT(*) AS current, COUNT(published_path) AS published FROM images"
        " WHERE career_id = ? AND is_current = 1",
        (career_id,),
    ).fetchone()
    out.update(pictures=pics["current"], published=pics["published"])
    out["review"] = review_count(conn, career_id)
    return out


def queue(conn: sqlite3.Connection, career_id: int, limit: int = 15) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT j.id, j.unique_id, j.kind, j.group_name, j.manual, j.tier, j.status, j.reason,"
            " j.skipped,"
            " p.name, p.at_my_club, c.name AS club, i.master_path, i.model, i.gender FROM jobs j"
            " JOIN players p ON p.career_id = j.career_id AND p.unique_id = j.unique_id"
            " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
            " LEFT JOIN images i ON i.career_id = j.career_id AND i.unique_id = j.unique_id"
            "  AND i.is_current = 1"
            " WHERE j.career_id = ? AND j.status IN ('queued', 'running')"
            f" ORDER BY j.status = 'queued', {QUEUE_ORDER} LIMIT ?",
            (career_id, limit),
        )
    ]


TIMING_PICTURES = 400
"""The latest pictures the estimate reads (core.estimate keeps 30 per group of them)."""


def drawn_times(conn: sqlite3.Connection, limit: int = TIMING_PICTURES) -> list[Drawn]:
    """The latest pictures of every career with their drawing times: how fast this PC draws
    does not depend on the career."""
    out = []
    for group, at, render in conn.execute(
        "SELECT group_name, created_at, render FROM images ORDER BY id DESC LIMIT ?", (limit,)
    ):
        try:
            seconds = json.loads(render).get("seconds") if render else None
        except (ValueError, AttributeError):
            seconds = None
        out.append(Drawn(group, at, float(seconds) if seconds is not None else None))
    return out


def queued_count(conn: sqlite3.Connection, career_id: int) -> int:
    """Pictures waiting in the career's queue (the one being drawn is not counted)."""
    return conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE career_id = ? AND status = 'queued'", (career_id,)
    ).fetchone()[0]


def queued_by_group(conn: sqlite3.Connection, career_id: int) -> dict[str, tuple[int, int]]:
    """Group -> (queued jobs, of them requested by hand)."""
    return {
        group: (n, manual or 0)
        for group, n, manual in conn.execute(
            "SELECT group_name, COUNT(*), SUM(manual) FROM jobs"
            " WHERE career_id = ? AND status = 'queued' GROUP BY group_name",
            (career_id,),
        )
    }


def recent_pictures(
    conn: sqlite3.Connection, career_id: int, limit: int = 8
) -> list[dict[str, Any]]:
    """The latest pictures that passed their checks, newest first."""
    return [
        dict(r)
        for r in conn.execute(
            "SELECT i.id, i.unique_id, i.version, i.group_name, i.reason, i.created_at,"
            " i.is_current, i.master_path, p.name"
            " FROM images i JOIN players p"
            " ON p.career_id = i.career_id AND p.unique_id = i.unique_id"
            " WHERE i.career_id = ? AND i.master_path IS NOT NULL"
            " AND i.master_path NOT LIKE '%-rejected.png'"
            " ORDER BY i.id DESC LIMIT ?",
            (career_id, limit),
        )
    ]


GALLERY_PER_PAGE = 60
WHERE_FILTERS = {
    "any": "anywhere",
    "mine": "my club",
    "league": "my league",
    "other": "elsewhere",
}
GALLERY_SORTS = {
    "new": "i.id DESC",
    "name": "p.name COLLATE NOCASE, i.id DESC",
    "ca": "p.ca DESC, p.pa DESC, i.id DESC",
}
GALLERY_GROUPS = ("any", "a", "b", "c")


@dataclass(frozen=True, slots=True)
class GalleryFilter:
    """The picture wall's filters (Phase 12h): everyone's current picture."""

    q: str = ""
    where: str = "any"
    group: str = "any"
    role: str = "any"
    sort: str = "new"
    page: int = 1

    @classmethod
    def from_query(cls, params: Mapping[str, str]) -> GalleryFilter:
        """From the page's query string; anything malformed falls back to its default."""
        get = params.get
        group = get("group")
        return cls(
            q=(get("q") or "").strip()[:100],
            where=_choice(get("where"), WHERE_FILTERS, "any"),
            group=group if group in GALLERY_GROUPS else "any",
            role=_choice(get("role"), ROLES, "any"),
            sort=_choice(get("sort"), GALLERY_SORTS, "new"),
            page=_page(get("page")),
        )

    def params(self, **changes: Any) -> dict[str, Any]:
        """The fields with some changed, defaults left out (the query string, the pager)."""
        f = replace(self, **changes)
        default = GalleryFilter()
        return {k: v for k, v in asdict(f).items() if v != getattr(default, k)}

    def query(self, **changes: Any) -> str:
        """The query string with some fields changed; defaults are left out."""
        return urlencode(self.params(**changes))

    @property
    def active(self) -> bool:
        """Whether any filter other than the page and sort is set."""
        return replace(self, page=1, sort="new") != GalleryFilter()

    def cleared(self) -> str:
        """The query string without the filters, keeping the sort (the Clear button)."""
        return GalleryFilter(sort=self.sort).query()


def one_page[PageFilter: (PlayerPageFilter, GalleryFilter)](
    fetch: Callable[[PageFilter], tuple[list[dict[str, Any]], int]],
    f: PageFilter,
    per_page: int,
) -> tuple[PageFilter, list[dict[str, Any]], int, int]:
    """The filter's page of rows from `fetch(f)` (rows, total), as (filter, rows, total,
    pages). A page past the last one (typed into the pager, or the list has shrunk since)
    gives the last page, and the filter returned says so (Phase 17b)."""
    rows, total = fetch(f)
    pages = page_count(total, per_page)
    if f.page > pages:
        f = replace(f, page=pages)
        rows, total = fetch(f)
        pages = page_count(total, per_page)
    return f, rows, total, pages


def gallery_page(
    conn: sqlite3.Connection, career_id: int, f: GalleryFilter, per_page: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    """One page of current pictures matching the filter, and how many match in all. A
    current picture is the one FM27 shows, so it passed its checks."""
    per_page = per_page or GALLERY_PER_PAGE
    clauses = ["i.career_id = ?", "i.is_current = 1", "i.master_path IS NOT NULL"]
    args: list[Any] = [career_id]
    if f.q:
        name_match, name_args = player_queries.name_or_id_match(f.q)
        clauses.append(name_match)
        args += name_args
    match f.where:
        case "mine":
            clauses.append("p.at_my_club = 1")
        case "league":
            clauses.append("p.in_my_league = 1")
        case "other":
            clauses.append("p.at_my_club = 0 AND p.in_my_league = 0")
    if f.group != "any":
        clauses.append("i.group_name = ?")
        args.append(f.group)
    match f.role:
        case "players":
            clauses.append("p.is_staff = 0 AND p.media_role IS NULL")
        case "youth":
            clauses.append("p.youth_team = 1")
        case "staff":
            clauses.append("p.is_staff = 1")
        case "media":
            clauses.append("p.media_role IS NOT NULL")
    where = " AND ".join(clauses)
    joins = (
        " FROM images i JOIN players p ON p.career_id = i.career_id AND p.unique_id = i.unique_id"
    )
    total = conn.execute(f"SELECT COUNT(*){joins} WHERE {where}", args).fetchone()[0]
    rows = conn.execute(
        "SELECT i.id, i.unique_id, i.version, i.group_name, i.created_at, i.master_path, p.name,"
        " p.at_my_club, p.is_staff, p.staff_job, p.media_role, p.local_media, c.name AS club"
        f"{joins} LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        f" WHERE {where} ORDER BY {GALLERY_SORTS[f.sort]} LIMIT ? OFFSET ?",
        (*args, per_page, (f.page - 1) * per_page),
    ).fetchall()
    return [dict(r) for r in rows], total


def first_picture_at(conn: sqlite3.Connection, career_id: int) -> str | None:
    """When the career's first picture was made (the guide's finish, Phase 12f)."""
    row = conn.execute(
        "SELECT MIN(created_at) FROM images WHERE career_id = ? AND master_path IS NOT NULL"
        " AND master_path NOT LIKE '%-rejected.png'",
        (career_id,),
    ).fetchone()
    return row[0] if row else None


def current_picture_id(conn: sqlite3.Connection, career_id: int, unique_id: int) -> int | None:
    row = conn.execute(
        "SELECT id FROM images WHERE career_id = ? AND unique_id = ? AND is_current = 1"
        " AND master_path IS NOT NULL",
        (career_id, unique_id),
    ).fetchone()
    return row[0] if row else None


def careers(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every career, the active one (newest save) first."""
    from kitshot.persistence.db import ACTIVE_CAREER_ORDER

    return [
        dict(r)
        for r in conn.execute(
            "SELECT id, key, display_name, last_save_file, last_game_date FROM careers"
            f" ORDER BY {ACTIVE_CAREER_ORDER}"
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
    """Failed pictures (`jobs` as `j`): the player's latest job that was not cancelled failed
    its checks, and the player is not pinned to never (no retry is coming; the player's key
    finds the pin)."""
    return (
        "j.career_id = ? AND j.status = 'failed' AND j.id = (SELECT MAX(jj.id) FROM jobs jj"
        " WHERE jj.career_id = j.career_id AND jj.unique_id = j.unique_id"
        " AND jj.status != 'cancelled') AND NOT EXISTS (SELECT 1 FROM players np"
        " WHERE np.career_id = j.career_id AND np.unique_id = j.unique_id AND np.pin = 'never')"
    )


def _flagged_where() -> str:
    """Current pictures sent to Review (Phase 17m, `images` as `i`) whose player has no queued
    or running job: a replacement under way takes them off the list until it is done."""
    return (
        "i.career_id = ? AND i.flagged_at IS NOT NULL AND i.is_current = 1 AND NOT EXISTS"
        " (SELECT 1 FROM jobs a WHERE a.career_id = i.career_id AND a.unique_id = i.unique_id"
        " AND a.status IN ('queued', 'running'))"
    )


def review_count(conn: sqlite3.Connection, career_id: int) -> int:
    """Players in Review: their latest job failed its checks (unless they are pinned to never),
    or they were sent there."""
    return conn.execute(
        f"SELECT COUNT(*) FROM (SELECT j.unique_id FROM jobs j WHERE {_review_where()}"
        f" UNION SELECT i.unique_id FROM images i WHERE {_flagged_where()})",
        (career_id, career_id),
    ).fetchone()[0]


def review_items(conn: sqlite3.Connection, career_id: int) -> list[dict[str, Any]]:
    """Players whose latest job failed its checks and who are not pinned to never, with the best
    rejected try if kept, and players whose current picture was sent to Review (`flagged`,
    when; Phase 17m). Newest first: by the job's last change or the time it was sent."""
    flagged = dict(
        conn.execute(
            "SELECT unique_id, flagged_at FROM images WHERE career_id = ?"
            " AND flagged_at IS NOT NULL AND is_current = 1",
            (career_id,),
        ).fetchall()
    )
    rows = conn.execute(
        "SELECT j.id AS job_id, j.unique_id, j.kind, j.group_name, j.error, j.updated_at,"
        " p.name, p.age, p.in_scope, p.present, p.pin, c.name AS club FROM jobs j"
        " JOIN players p ON p.career_id = j.career_id AND p.unique_id = j.unique_id"
        " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        f" WHERE {_review_where()}",
        (career_id,),
    ).fetchall()
    sent = conn.execute(
        "SELECT NULL AS job_id, i.unique_id, NULL AS kind, i.group_name, NULL AS error,"
        " i.flagged_at AS updated_at, p.name, p.age, p.in_scope, p.present, p.pin,"
        " c.name AS club FROM images i"
        " JOIN players p ON p.career_id = i.career_id AND p.unique_id = i.unique_id"
        " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        f" WHERE {_flagged_where()}",
        (career_id,),
    ).fetchall()
    failed = {r["unique_id"] for r in rows}
    items = []
    for r in [*rows, *(r for r in sent if r["unique_id"] not in failed)]:
        item = dict(r)
        item["flagged"] = flagged.get(r["unique_id"])
        image = None
        if r["job_id"] is not None:
            image = conn.execute(
                "SELECT * FROM images WHERE career_id = ? AND unique_id = ? AND is_current = 0"
                " AND master_path LIKE '%-rejected.png' ORDER BY version DESC LIMIT 1",
                (career_id, r["unique_id"]),
            ).fetchone()
        item["image"] = _image(image) if image is not None else None
        # What FM27 shows now, to compare the candidates with (Phase 12i).
        item["current_id"] = current_picture_id(conn, career_id, r["unique_id"])
        items.append(item)
    items.sort(key=lambda it: (it["updated_at"] or "", it["job_id"] or 0), reverse=True)
    return items
