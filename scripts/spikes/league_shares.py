"""Phase 16a spike: the league ladder and the tier counts the share cutoffs give. Read only.

Runs the real sync (pipeline/sync.py) on save copies in data/snapshots/ into a scratch
database, with FM27's photo list read as a sync reads it (nothing in FM27's folders is
written), and reports for each save:

- my nation's leagues on the ladder: level, offset from mine, clubs, median reputation, median
  first-team CA and the best-known clubs (the save stores no competition names);
- each scope's candidates (newgens, and photo-less players) and the cut values in force;
- each tier's count with the defaults, and with Light for the league below on as well;
- with two saves of one career (in order), how many people changed tier between them;
- how long the ladder, the stats and the classification take.

    uv run --no-sync python scripts/spikes/league_shares.py SAVE [SAVE ...] \\
        [--career-pairs] [--cache DIR] [--out FILE.json]

`--cache DIR` keeps each parsed save as a pickle, so a second run skips fmsave. The scratch
database lives in a temporary folder and is deleted at the end.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from facegen.core.cutoffs import LEAGUE_CELLS, OTHER_CELLS
from facegen.core.leagues import competition_name, league_ladder
from facegen.core.records import Snapshot
from facegen.datasource import photos
from facegen.datasource.fmsave_source import FmsaveDataSource
from facegen.persistence import db, league_stats, photo_index
from facegen.persistence.settings import load_settings, set_overrides
from facegen.pipeline import reclassification
from facegen.pipeline.sync import run_sync


def load(path: Path, cache: Path | None) -> Snapshot:
    if cache is not None:
        cached = cache / (path.stem + ".pkl")
        if cached.is_file() and cached.stat().st_mtime >= path.stat().st_mtime:
            return pickle.loads(cached.read_bytes())
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = FmsaveDataSource().load(path)
    print(f"  read {path.name} in {time.perf_counter() - t:.1f} s", file=sys.stderr)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / (path.stem + ".pkl")).write_bytes(pickle.dumps(snap))
    return snap


def ladder_rows(snap: Snapshot) -> list[dict[str, Any]]:
    t = time.perf_counter()
    ladder = league_ladder(
        snap.managed_club_id,
        snap.clubs,
        snap.league_tables,
        snap.fixture_counts,
        seasons=snap.fixture_seasons,
        rules=snap.competition_rules,
    )
    took = time.perf_counter() - t
    ca: dict[int, list[int]] = defaultdict(list)
    for p in snap.players:
        if p.club_id is not None and p.team_slot == 0:
            ca[p.club_id].append(p.ca)
    rows = []
    for lg in ladder.leagues:
        best = sorted(lg.club_ids, key=lambda c: -(snap.clubs[c].reputation or 0))[:3]
        medians = [statistics.median(ca[c]) for c in lg.club_ids if ca.get(c)]
        rows.append(
            {
                "level": lg.level,
                "offset": ladder.offset(lg),
                "mine": ladder.mine is lg,
                "competition": lg.competition_id,
                "name": competition_name(lg.competition_id),
                "clubs": len(lg.club_ids),
                "median_reputation": lg.reputation,
                "median_first_team_ca": statistics.median(medians) if medians else None,
                "best_known": [snap.clubs[c].name for c in best],
            }
        )
    return [{"ladder_ms": round(took * 1000, 1), "from_rules": ladder.from_rules}, *rows]


def tier_counts(conn: Any, career_id: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT group_name, is_staff, media_role IS NOT NULL AS media, at_my_club, COUNT(*) n"
        " FROM players WHERE career_id = ? AND present = 1 AND group_name IS NOT NULL"
        " GROUP BY 1, 2, 3, 4",
        (career_id,),
    ).fetchall()
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        who = "staff" if r["is_staff"] else "media" if r["media"] else "players"
        where = "my club" if r["at_my_club"] else "elsewhere"
        out[r["group_name"]][f"{who} {where}"] += r["n"]
        out[r["group_name"]]["total"] += r["n"]
    return {g: dict(c) for g, c in sorted(out.items())}


def reasons(conn: Any, career_id: int) -> dict[str, dict[str, int]]:
    """How many people each kind of rule put in each tier (their first reason, numbers cut)."""
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for r in conn.execute(
        "SELECT group_name, reasons FROM players WHERE career_id = ? AND present = 1"
        " AND group_name IS NOT NULL",
        (career_id,),
    ):
        first = json.loads(r["reasons"])[0]
        out[r["group_name"]][first.split(", CA")[0].split(", PA")[0].split(" (")[0]] += 1
    return {g: dict(c.most_common()) for g, c in sorted(out.items())}


def scope_rows(conn: Any, career_id: int, settings: Any) -> dict[str, Any]:
    stats = league_stats.load(conn, career_id)
    cuts = reclassification.cutoffs_for(conn, career_id, settings)
    out: dict[str, Any] = {}
    for scope, s in stats.items():
        out[scope] = {
            "competitions": list(s.competitions),
            "newgens": s.newgen_ca.total,
            "no_photo": s.no_photo_ca.total,
            "candidates_ca": s.candidates("ca", settings.no_photo.enabled).total,
            "candidates_pa": s.candidates("pa", settings.no_photo.enabled).total,
        }
    out["cuts"] = {
        **{f"league.{c}": cuts.league_cut(c).value for c in LEAGUE_CELLS},
        **{c: cuts.other_cut(c).value for c in OTHER_CELLS},
    }
    out["named"] = [
        {"scope": f.scope, "name": f.name, "clubs": f.club_count, "best_known": list(f.clubs)}
        for f in league_stats.found_leagues(conn, career_id)
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("saves", nargs="+", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--career-pairs",
        action="store_true",
        help="sync all saves into one database, in order, to measure changes between them",
    )
    args = parser.parse_args()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = load_settings()
        scan = photos.scan(base.paths, None)
        print(f"  photo list: game_known={scan.index is not None}", file=sys.stderr)
        shared = db.connect(Path(tmp) / "shared.db") if args.career_pairs else None
        previous: dict[int, str | None] | None = None
        for n, path in enumerate(args.saves):
            snap = load(path, args.cache)
            conn = shared or db.connect(Path(tmp) / f"{n}.db")
            settings = load_settings(conn)
            t = time.perf_counter()
            report = run_sync(conn, snap, settings, save_file=path.name, photos=scan)
            sync_s = time.perf_counter() - t
            cid = report.career.id
            t = time.perf_counter()
            reclassification.reclassify(conn, cid, settings, dry_run=True)
            classify_s = time.perf_counter() - t
            t = time.perf_counter()
            league_stats.collect(
                conn,
                cid,
                {
                    s: tuple(json.loads(r["competitions"]))
                    for r in conn.execute(
                        "SELECT scope, competitions FROM league_stats WHERE career_id = ?",
                        (cid,),
                    )
                    for s in (r["scope"],)
                },
            )
            stats_s = time.perf_counter() - t
            row: dict[str, Any] = {
                "save": path.name,
                "game_date": str(snap.meta.game_date),
                "managed_club": report.managed_club,
                "photos_known": bool(
                    photo_index.status(conn) and photo_index.status(conn).game_known
                ),
                "ladder": ladder_rows(snap),
                "scopes": scope_rows(conn, cid, settings),
                "tiers_default": tier_counts(conn, cid),
                "reasons_default": reasons(conn, cid),
                "timing_s": {
                    "sync": round(sync_s, 2),
                    "classify_and_plan": round(classify_s, 2),
                    "league_stats": round(stats_s, 3),
                },
            }
            now = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT unique_id, group_name FROM players WHERE career_id = ? AND present = 1",
                    (cid,),
                )
            }
            if previous is not None:
                moves = Counter(
                    f"{previous.get(u) or '-'}->{g or '-'}"
                    for u, g in now.items()
                    if previous.get(u) != g and (previous.get(u) or g)
                )
                row["changed_since_previous"] = dict(moves.most_common())
            previous = now if args.career_pairs else None
            # Light for the league below as well as the league above, without a new save.
            with db.transaction(conn, rollback=True):
                on = set_overrides(conn, {"cutoffs.light_below": True})
                reclassification.classify_and_plan(conn, cid, on)
                row["tiers_light_below"] = tier_counts(conn, cid)
            # The previous rules, for comparison: the old numbers as "at least" N+1.
            with db.transaction(conn, rollback=True):
                old = replace(
                    settings,
                    cutoffs=replace(settings.cutoffs, mode="manual"),
                )
                reclassification.classify_and_plan(conn, cid, old)
                row["tiers_old_numbers"] = tier_counts(conn, cid)
            results.append(row)
            if shared is None:
                conn.close()
        if shared is not None:
            shared.close()
    text = json.dumps(results, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
