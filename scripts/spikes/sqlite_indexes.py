"""SQLite follow-up after Phase 12: measure the pages' queries before adding any index.

Works on a copy only (implementation plan, "SQLite optimization"): the database given is
copied with SQLite's backup API, opened read-only, the copy is migrated to this code's schema,
and each variant runs on its own copy in `--work`. Nothing is written to the source.
Optionally adds a made-up job history, since a young database has few jobs and no failed
ones, while the Review filter's cost grows with them.

For every variant (indexes added, or a query rewritten) and every query it records the
median time of the count and of the page separately, their query plans, and a hash of the
results, which must match the baseline's. It also records each index's size (dbstat) and
what the indexes cost the writes a sync makes: re-writing every person with the sync's own
upsert, and the classification's group update.

    uv run --no-sync python scripts/spikes/sqlite_indexes.py --db data/facegen.db \
        --work /tmp/sqlite-bench --jobs 40000 --out /tmp/sqlite-bench/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sqlite3
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kitshot.persistence import db, player_queries
from kitshot.persistence.player_queries import PER_PAGE, PlayerFilter
from kitshot.web import queries

REPEATS = 7


@dataclass(frozen=True)
class Variant:
    name: str
    indexes: tuple[str, ...] = ()
    review_rewrite: bool = False
    deferred_page: bool = False
    """The page's order and window first, over the players table and the joins its filter
    and order need; the page's other joins and the latest job only for those rows."""


CANDIDATE_RANK = (
    "CREATE INDEX players_rank ON players"
    " (career_id, present, in_scope, ca DESC, pa DESC, unique_id)"
)
CANDIDATE_RANK_NO_SCOPE = (
    "CREATE INDEX players_rank ON players (career_id, present, ca DESC, pa DESC, unique_id)"
)
CANDIDATE_CLUB = "CREATE INDEX players_by_club ON players (career_id, club_id)"
CANDIDATE_JOBS_LATEST = (
    "CREATE INDEX jobs_latest ON jobs (career_id, unique_id, id) WHERE status != 'cancelled'"
)
PROPOSAL_CA_PA = "CREATE INDEX players_by_ca_pa ON players (career_id, present, ca DESC, pa DESC)"
PROPOSAL_CLUB = "CREATE INDEX players_by_club ON players (career_id, present, club_id)"
PROPOSAL_JOBS = "CREATE INDEX jobs_latest_status ON jobs (career_id, unique_id, status, id DESC)"

VARIANTS = (
    Variant("baseline"),
    Variant("earlier proposal", (PROPOSAL_CA_PA, PROPOSAL_CLUB, PROPOSAL_JOBS)),
    Variant("rank index", (CANDIDATE_RANK,)),
    Variant("rank index without scope", (CANDIDATE_RANK_NO_SCOPE,)),
    Variant("club index", (CANDIDATE_CLUB,)),
    Variant("jobs partial index", (CANDIDATE_JOBS_LATEST,)),
    Variant("review rewrite", review_rewrite=True),
    Variant("rank + club + review rewrite", (CANDIDATE_RANK, CANDIDATE_CLUB), True),
    Variant("deferred page", deferred_page=True),
    Variant("deferred page + review rewrite", review_rewrite=True, deferred_page=True),
)


def filters(conn: sqlite3.Connection, career_id: int) -> dict[str, PlayerFilter]:
    """The Players page's representative filters: what a user clicks."""
    club_id, club_name = (
        conn.execute(
            "SELECT p.club_id, c.name FROM players p JOIN clubs c"
            " ON c.career_id = p.career_id AND c.club_id = p.club_id"
            " WHERE p.career_id = ? AND p.present = 1 AND p.at_my_club = 1 LIMIT 1",
            (career_id,),
        ).fetchone()
        or conn.execute(
            "SELECT p.club_id, c.name FROM players p JOIN clubs c"
            " ON c.career_id = p.career_id AND c.club_id = p.club_id"
            " WHERE p.career_id = ? AND p.present = 1 GROUP BY p.club_id"
            " ORDER BY COUNT(*) DESC LIMIT 1",
            (career_id,),
        ).fetchone()
    )
    base = PlayerFilter()
    targets = replace(base, shown="targets")
    return {
        "targets (page default)": targets,
        "targets, page 20": replace(targets, page=20),
        "targets sort club": replace(targets, sort="club"),
        "targets club link": replace(targets, club_id=club_id),
        "targets name search": replace(targets, q="son"),
        "default (CA)": base,
        "default, page 50": replace(base, page=50),
        "sort PA": replace(base, sort="pa"),
        "sort name": replace(base, sort="name"),
        "sort club": replace(base, sort="club"),
        "all players": replace(base, scope="all"),
        "club link": replace(base, club_id=club_id),
        "club name": replace(base, club=club_name[:6]),
        "my league": replace(base, league="mine"),
        "group A": replace(base, group="a"),
        "grouped": replace(base, group="grouped"),
        "CA 150+": replace(base, ca_min=150),
        "PA 170+ sort PA": replace(base, pa_min=170, sort="pa"),
        "staff": replace(base, role="staff"),
        "youth": replace(base, role="youth"),
        "media": replace(base, role="media"),
        "newgens": replace(base, origin="newgen"),
        "has picture": replace(base, picture="yes"),
        "queued": replace(base, picture="queued"),
        "review": replace(base, picture="review"),
        "no photo": replace(base, photo="none"),
        "name search": replace(base, q="son"),
    }


REVIEW_IN = (
    "(p.career_id, p.unique_id) IN (SELECT j.career_id, j.unique_id FROM jobs j"
    " WHERE j.status = 'failed' AND j.id = (SELECT MAX(jj.id) FROM jobs jj"
    " WHERE jj.career_id = j.career_id AND jj.unique_id = j.unique_id"
    " AND jj.status != 'cancelled'))"
)
"""The Review filter as a lookup in the jobs table: not correlated with the player row, so
SQLite builds the list once instead of asking the jobs for every player."""


def player_sql(
    f: PlayerFilter, career_id: int, rewrite: bool, deferred: bool
) -> tuple[str, str, list[Any]]:
    """Build the count and page using the roster's filters and columns. The count's parameters
    (the career first); the page's are those, then the window, then (deferred) the career."""
    where, args = player_queries._where(f, career_id)
    if rewrite:
        where = where.replace(f"{player_queries._LATEST_JOB} = 'failed'", REVIEW_IN)
    count = f"SELECT COUNT(*){player_queries._count_from(where)} WHERE {where}"
    order = f"{player_queries.SORTS[f.sort]}, p.unique_id"
    select = player_queries._COLUMNS + player_queries._FROM
    if not deferred:
        page = f"{select} WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?"
        return count, page, args
    window = (
        f"SELECT p.unique_id{player_queries._count_from(where + ' ' + order)} WHERE {where}"
        f" ORDER BY {order} LIMIT ? OFFSET ?"
    )
    page = f"{select} WHERE p.unique_id IN ({window}) AND p.career_id = ? ORDER BY {order}"
    return count, page, args


def timed(fn: Callable[[], Any], repeats: int = REPEATS) -> tuple[float, Any]:
    fn()  # warm the page cache
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000, result


def digest(rows: Any) -> str:
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()[:16]


def plan_of(conn: sqlite3.Connection, sql: str, args: list[Any]) -> list[str]:
    return [r[3] for r in conn.execute(f"EXPLAIN QUERY PLAN {sql}", args)]


def copy_db(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{dest}{suffix}").unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def add_job_history(path: Path, career_id: int, n_jobs: int, seed: int = 7) -> dict[str, int]:
    """A made-up history: done, cancelled and failed jobs for random present people, newest
    last, and no more than one queued job each (the jobs_one_active index)."""
    rng = random.Random(seed)
    conn = sqlite3.connect(path, isolation_level=None)
    uids = [
        r[0]
        for r in conn.execute(
            "SELECT unique_id FROM players WHERE career_id = ? AND present = 1", (career_id,)
        )
    ]
    busy = {
        r[0]
        for r in conn.execute(
            "SELECT unique_id FROM jobs WHERE career_id = ? AND status IN ('queued', 'running')",
            (career_id,),
        )
    }
    rows = []
    counts: dict[str, int] = {}
    now = "2026-11-17T00:00:00"
    for _ in range(n_jobs):
        uid = rng.choice(uids)
        status = rng.choices(["done", "cancelled", "failed", "queued"], weights=[80, 10, 6, 4])[0]
        if status == "queued":
            if uid in busy:
                status = "done"
            else:
                busy.add(uid)
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            (career_id, uid, "initial", rng.choice("abc"), 0, 3, rng.random(), status,
             "made up", now, now)
        )  # fmt: skip
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO jobs (career_id, unique_id, kind, group_name, manual, tier, rank, status,"
        " reason, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute("COMMIT")
    conn.close()
    return counts


def index_sizes(conn: sqlite3.Connection, names: list[str]) -> dict[str, int]:
    if not names:
        return {}
    marks = ", ".join("?" * len(names))
    return {
        name: size
        for name, size in conn.execute(
            f"SELECT name, SUM(pgsize) FROM dbstat WHERE name IN ({marks}) GROUP BY name", names
        )
    }


def write_costs(conn: sqlite3.Connection, career_id: int) -> dict[str, float]:
    """A re-sync's player writes (the sync's own upsert, same values) and the classification's
    group update, each in a transaction that is rolled back."""
    from kitshot.pipeline.sync import _PLAYER_COLUMNS, _UPSERT_PLAYER

    cols = ", ".join(_PLAYER_COLUMNS)
    rows = conn.execute(f"SELECT {cols} FROM players WHERE career_id = ?", (career_id,)).fetchall()
    groups = conn.execute(
        "SELECT group_name, tracked, reasons, career_id, unique_id FROM players"
        " WHERE career_id = ?",
        (career_id,),
    ).fetchall()
    out = {}
    classify = (
        "UPDATE players SET group_name = ?, tracked = ?, reasons = ?"
        " WHERE career_id = ? AND unique_id = ?"
    )
    for name, sql, data in (
        ("sync upsert", _UPSERT_PLAYER, rows),
        ("classify update", classify, groups),
    ):
        times = []
        for _ in range(3):
            conn.execute("BEGIN")
            start = time.perf_counter()
            conn.executemany(sql, data)
            times.append(time.perf_counter() - start)
            conn.execute("ROLLBACK")
        out[name] = round(statistics.median(times) * 1000, 1)
    # Moved players: a real sync changes CA, PA or the club of some, which moves index entries.
    sample = rows[:: max(1, len(rows) // 20000)]
    ca = _PLAYER_COLUMNS.index("ca")
    moved = [tuple(v + 1 if i == ca else v for i, v in enumerate(r)) for r in sample]
    times = []
    for _ in range(3):
        conn.execute("BEGIN")
        start = time.perf_counter()
        conn.executemany(_UPSERT_PLAYER, moved)
        times.append(time.perf_counter() - start)
        conn.execute("ROLLBACK")
    out[f"upsert {len(moved)} with CA changed"] = round(statistics.median(times) * 1000, 1)
    return out


def other_queries(conn: sqlite3.Connection, career_id: int) -> dict[str, Callable[[], Any]]:
    return {
        "dashboard counts": lambda: queries.counts(conn, career_id),
        "review count": lambda: queries.review_count(conn, career_id),
        "review items": lambda: len(queries.review_items(conn, career_id)),
        "gallery": lambda: queries.gallery_page(conn, career_id, queries.GalleryFilter())[1],
    }


def run_variant(source: Path, work: Path, variant: Variant, career_id: int) -> dict[str, Any]:
    path = work / f"{variant.name.replace(' ', '-').replace('+', 'and')}.db"
    shutil.copyfile(source, path)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    created: list[str] = []
    create_ms = {}
    for sql in variant.indexes:
        start = time.perf_counter()
        conn.execute(sql)
        name = sql.split()[2]
        create_ms[name] = round((time.perf_counter() - start) * 1000, 1)
        created.append(name)
    result: dict[str, Any] = {
        "indexes": list(variant.indexes),
        "create_ms": create_ms,
        "index_bytes": index_sizes(conn, created),
        "writes_ms": write_costs(conn, career_id),
        "players": {},
        "other": {},
    }
    for label, f in filters(conn, career_id).items():
        count_sql, page_sql, args = player_sql(
            f, career_id, variant.review_rewrite, variant.deferred_page
        )
        count_args = args
        page_args = [*args, PER_PAGE, (f.page - 1) * PER_PAGE]
        if variant.deferred_page:
            page_args.append(career_id)
        count_ms, total = timed(
            lambda sql=count_sql, a=count_args: conn.execute(sql, a).fetchone()[0]
        )
        page_ms, rows = timed(
            lambda sql=page_sql, a=page_args: [tuple(r) for r in conn.execute(sql, a)]
        )
        result["players"][label] = {
            "count_ms": round(count_ms, 2),
            "page_ms": round(page_ms, 2),
            "total": total,
            "hash": digest(rows),
            "count_plan": plan_of(conn, count_sql, count_args),
            "page_plan": plan_of(conn, page_sql, page_args),
        }
    for label, fn in other_queries(conn, career_id).items():
        ms, value = timed(fn)
        result["other"][label] = {"ms": round(ms, 2), "hash": digest(value)}
    conn.close()
    path.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="database to copy (read-only)")
    parser.add_argument("--work", type=Path, required=True, help="folder for the copies")
    parser.add_argument("--career", type=int, help="career id (default: the most players)")
    parser.add_argument("--jobs", type=int, default=0, help="made-up jobs to add to the copy")
    parser.add_argument("--only", action="append", help="run only these variants")
    parser.add_argument("--out", type=Path, help="write the results here as JSON")
    args = parser.parse_args()

    base = args.work / "base.db"
    copy_db(args.db, base)
    db.connect(base).close()  # the copy in this code's schema: its queries read new columns
    conn = sqlite3.connect(base)
    career_id = (
        args.career
        or conn.execute(
            "SELECT career_id FROM players WHERE present = 1 GROUP BY career_id"
            " ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()[0]
    )
    people = conn.execute(
        "SELECT COUNT(*) FROM players WHERE career_id = ? AND present = 1", (career_id,)
    ).fetchone()[0]
    conn.close()
    history = add_job_history(base, career_id, args.jobs) if args.jobs else {}
    results: dict[str, Any] = {
        "sqlite": sqlite3.sqlite_version,
        "career_id": career_id,
        "present_people": people,
        "made_up_jobs": history,
        "variants": {},
    }
    print(f"career {career_id}: {people} present people; made-up jobs {history or 'none'}")
    for variant in VARIANTS:
        if args.only and variant.name not in args.only:
            continue
        print(f"-- {variant.name}", flush=True)
        r = run_variant(base, args.work, variant, career_id)
        results["variants"][variant.name] = r
        baseline = results["variants"].get("baseline")
        for label, q in r["players"].items():
            same = ""
            if baseline is not None and variant.name != "baseline":
                b = baseline["players"][label]
                same = "" if (b["hash"], b["total"]) == (q["hash"], q["total"]) else "  DIFFERS"
            print(
                f"  {label:<20} count {q['count_ms']:8.2f} ms  page {q['page_ms']:8.2f} ms"
                f"  ({q['total']}){same}"
            )
        for label, q in r["other"].items():
            print(f"  {label:<20} {q['ms']:8.2f} ms")
        print(f"  writes {r['writes_ms']}  sizes {r['index_bytes']}  create {r['create_ms']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    base.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
