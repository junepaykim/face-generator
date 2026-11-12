"""Command-line entry points.

facegen sync [--save PATH] [--dry-run] [--show N] [--csv PATH]
facegen run [--group a|b|c ...] [--my-club] [--limit N] [--no-publish] [--now] [--dry-run]
facegen publish
facegen gallery
facegen status
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from facegen import db
from facegen.datasource.base import DataSource, SaveReadError, UnsupportedSaveError
from facegen.gallery import write_gallery
from facegen.kits import KitError, load_kits
from facegen.poller import copy_snapshot, find_latest_save
from facegen.publish import RELOAD_HINT, PublishError, PublishReport
from facegen.settings import Settings, load_settings
from facegen.sync import SyncReport, record_failed_sync, run_sync
from facegen.worker import (
    Career,
    JobFilter,
    Renderer,
    Tagger,
    Worker,
    WorkerBusy,
    fm_running,
    load_career,
)

_TIER_NAMES = {0: "manual", 1: "A, my club", 2: "A", 3: "B", 4: "C"}


def _default_source() -> DataSource:
    from facegen.datasource.fmsave_source import FmsaveDataSource

    return FmsaveDataSource()


def _default_renderer() -> Renderer:
    from facegen.worker import AnimagineRenderer

    return AnimagineRenderer()


def _default_tagger() -> Tagger:
    from facegen.qc import Eva02Tagger

    return Eva02Tagger()


@dataclass(frozen=True, slots=True)
class Gpu:
    """What `run` uses for the GPU and the FM27 check; the tests pass fakes."""

    renderer: Callable[[], Renderer] = _default_renderer
    tagger: Callable[[], Tagger] = _default_tagger
    fm_check: Callable[[], bool | None] = fm_running


def _is_within(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def format_report(r: SyncReport) -> str:
    c = r.counts
    lines = [
        f"{'DRY RUN: nothing was stored.' if r.dry_run else 'Sync stored.'}",
        f"Save        {r.save_file}  ({r.build}, in-game {r.game_date or 'unknown'})",
        f"Career      {r.career.key}{' (new)' if r.career.is_new else ''}  {r.career.display_name}",
        f"My club     {r.managed_club or 'none'}"
        + (f" ({r.managed_club_gender})" if r.managed_club_gender else ""),
        f"My league   {r.my_league_clubs} clubs; continental {r.continental_clubs} clubs",
        "Newgens     unique ID >= "
        + (f"{r.career.newgen_min_id:,}" if r.career.newgen_min_id is not None else "unknown")
        + f" ({r.career.newgen_id_source or 'unknown'})",
        "",
        f"Players     {c['players']:,} read ({c['skipped']} skipped without a unique ID)",
        (
            f"            men's clubs {c['men']:,}, women's clubs {c['women']:,}, "
            f"unknown {c['unknown_gender']:,}, no club {c['no_club']:,}"
        ),
        f"            in scope {c['in_scope']:,}, of which newgens {c['newgens']:,}",
        f"Groups      A {c['a']:,} ({c['a_my_club']:,} at my club)   B {c['b']:,}   C {c['c']:,}",
        f"Tracked     {c['tracked']:,}  (academy newgens {c['academy']:,})",
    ]
    if r.enqueued:
        parts = ", ".join(
            f"{kind} {group.upper()} {n:,}" for (kind, group), n in sorted(r.enqueued.items())
        )
        lines.append(f"Would queue {parts}" if r.dry_run else f"Queued      {parts}")
    else:
        lines.append("Queue       no new jobs")
    if r.updated or r.cancelled:
        lines.append(f"            {r.updated} jobs updated, {r.cancelled} cancelled")
    if r.queue:
        lines += ["", f"Queue, first {len(r.queue)}:"]
        for i, e in enumerate(r.queue, 1):
            pa = "?" if e.pa is None else e.pa
            why = "; ".join(e.player_reasons)
            lines.append(
                f"{i:>4}. {_TIER_NAMES.get(e.tier, e.group.upper()):<10} {e.kind:<8} "
                f"{e.name} ({e.club or 'no club'}) CA {e.ca} PA {pa}: {why}"
            )
    if r.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in r.warnings]
    return "\n".join(lines)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cmd_sync(
    args: argparse.Namespace,
    settings: Settings,
    conn: sqlite3.Connection,
    source: DataSource,
    out: TextIO,
) -> int:
    save: Path | None = args.save or find_latest_save(settings.paths.games_dir)
    if save is None:
        print(f"No save found in {settings.paths.games_dir}", file=out)
        return 2
    if not save.is_file():
        print(f"No such save: {save}", file=out)
        return 2
    if _is_within(save, settings.paths.snapshots_dir):
        snapshot_path = save
    else:
        snapshot_path = copy_snapshot(save, settings.paths.snapshots_dir)
    try:
        snapshot = source.load(snapshot_path)
    except (UnsupportedSaveError, SaveReadError) as exc:
        if not args.dry_run:
            record_failed_sync(conn, save.name, str(exc))
        print(f"Cannot read {save.name}: {exc}", file=out)
        print("Existing pictures and the queue are unchanged.", file=out)
        return 3
    report = run_sync(
        conn,
        snapshot,
        settings,
        save_file=save.name,
        save_mtime=save.stat().st_mtime,
        dry_run=args.dry_run,
        preview=args.show,
        with_table=args.csv is not None,
    )
    print(format_report(report), file=out)
    if args.csv is not None:
        write_csv(report.table, args.csv)
        print(f"\nWrote {len(report.table):,} players to {args.csv}", file=out)
    return 0


def cmd_status(settings: Settings, conn: sqlite3.Connection, out: TextIO) -> int:
    careers = conn.execute(
        "SELECT id, key, display_name, last_game_date, updated_at FROM careers ORDER BY updated_at"
    ).fetchall()
    if not careers:
        print("No careers yet. Run `facegen sync`.", file=out)
        return 0
    for c in careers:
        print(
            f"{c['key']}  {c['display_name']}  in-game {c['last_game_date']}  "
            f"last sync {c['updated_at']}",
            file=out,
        )
        groups = dict(
            conn.execute(
                "SELECT COALESCE(group_name, '-'), COUNT(*) FROM players"
                " WHERE career_id = ? AND present = 1 AND group_name IS NOT NULL"
                " GROUP BY group_name",
                (c["id"],),
            ).fetchall()
        )
        tracked = conn.execute(
            "SELECT COUNT(*) FROM players WHERE career_id = ? AND present = 1 AND tracked = 1",
            (c["id"],),
        ).fetchone()[0]
        jobs = conn.execute(
            "SELECT status, COUNT(*) FROM jobs WHERE career_id = ? GROUP BY status ORDER BY status",
            (c["id"],),
        ).fetchall()
        print(
            "  groups  "
            + "  ".join(f"{g.upper()} {groups.get(g, 0):,}" for g in ("a", "b", "c"))
            + f"  tracked {tracked:,}",
            file=out,
        )
        print("  jobs    " + ("  ".join(f"{s} {n:,}" for s, n in jobs) or "none"), file=out)
        pics = conn.execute(
            "SELECT COUNT(*) AS current, COUNT(published_path) AS published FROM images"
            " WHERE career_id = ? AND is_current = 1",
            (c["id"],),
        ).fetchone()
        print(f"  pictures {pics['current']:,} current, {pics['published']:,} published", file=out)
    last = conn.execute(
        "SELECT save_file, finished_at, error FROM sync_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if last is not None and last["error"]:
        print(f"Last sync failed ({last['save_file']}): {last['error']}", file=out)
    return 0


def _career(conn: sqlite3.Connection, key: str | None, out: TextIO) -> Career | None:
    career = load_career(conn, key)
    if career is None:
        print(f"No career {key!r}." if key else "No careers yet. Run `facegen sync`.", file=out)
    return career


def format_publish(report: PublishReport) -> list[str]:
    lines = []
    if report.changed:
        lines.append(
            f"Published   {report.written:,} pictures ({report.unchanged:,} already in place) to"
            f" {report.folder}; config.xml maps {report.mapped:,} players."
        )
        lines.append(RELOAD_HINT)
    else:
        lines.append(f"Published   nothing changed in {report.folder} ({report.mapped:,} mapped).")
    if report.missing:
        lines.append(f"Missing     master pictures for {len(report.missing)} players: not mapped.")
    return lines


def cmd_run(
    args: argparse.Namespace,
    settings: Settings,
    conn: sqlite3.Connection,
    gpu: Gpu,
    out: TextIO,
) -> int:
    career = _career(conn, args.career, out)
    if career is None:
        return 2
    try:
        kits = load_kits(args.kits)
    except KitError as exc:
        print(f"config/kits.toml: {exc}", file=out)
        return 2
    worker = Worker(
        conn,
        settings,
        kits,
        career,
        renderer_factory=gpu.renderer,
        tagger_factory=gpu.tagger,
        fm_check=gpu.fm_check,
        log=lambda line: print(line, file=out, flush=True),
    )
    f = JobFilter(groups=frozenset(args.group) if args.group else None, my_club_only=args.my_club)
    if args.dry_run:
        previews = worker.preview(f, args.limit)
        print(f"DRY RUN: {len(previews)} jobs would run; nothing is drawn or stored.", file=out)
        for i, p in enumerate(previews, 1):
            j = p.job
            print(
                f"{i:>4}. {j.name} ({j.unique_id}) {j.kind} {j.group.upper()}"
                f"{' GK' if j.goalkeeper else ''}: {p.wording}, {p.kit}"
                f"{'' if p.kit_known else ' (fallback kit)'}, {p.pose}",
                file=out,
            )
            print(f"      sheet: {p.sheet.describe()}", file=out)
            if p.prompt is not None:
                print(f"      + {p.prompt.positive}", file=out)
                print(f"      - {p.prompt.negative}", file=out)
        return 0
    try:
        summary = worker.run(
            f, limit=args.limit, publish=not args.no_publish, generate_now=args.now
        )
    except WorkerBusy as exc:
        print(str(exc), file=out)
        return 4
    except PublishError as exc:
        print(f"Publishing failed: {exc}", file=out)
        return 3
    print(file=out)
    print(
        f"Done        {len(summary.done):,} pictures, {len(summary.failed):,} sent to review",
        file=out,
    )
    if summary.waiting_for_reference:
        print(
            f"Waiting     {summary.waiting_for_reference:,} re-draws need a reference picture"
            " (Phase 5)",
            file=out,
        )
    if summary.stopped:
        print(f"Stopped     {summary.stopped}", file=out)
    if summary.publish is not None:
        for line in format_publish(summary.publish):
            print(line, file=out)
    elif summary.done:
        print("Not published (--no-publish). Run `facegen publish` when ready.", file=out)
    page = write_gallery(conn, career.id, career.key, career.display_name, settings.paths.data_dir)
    print(f"Gallery     {page}", file=out)
    return 5 if summary.stopped else 0


def cmd_publish(
    args: argparse.Namespace, settings: Settings, conn: sqlite3.Connection, out: TextIO
) -> int:
    career = _career(conn, args.career, out)
    if career is None:
        return 2
    worker = Worker(
        conn,
        settings,
        load_kits(),
        career,
        renderer_factory=_default_renderer,
        tagger_factory=_default_tagger,
    )
    try:
        report = worker.publish()
    except PublishError as exc:
        print(f"Publishing failed: {exc}", file=out)
        return 3
    for line in format_publish(report):
        print(line, file=out)
    return 0


def cmd_gallery(
    args: argparse.Namespace, settings: Settings, conn: sqlite3.Connection, out: TextIO
) -> int:
    career = _career(conn, args.career, out)
    if career is None:
        return 2
    page = write_gallery(conn, career.id, career.key, career.display_name, settings.paths.data_dir)
    print(page, file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facegen", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="defaults file (default: config/default.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sync", help="read a save and update the player list and queue")
    s.add_argument("--save", type=Path, help="save file (default: the newest in FM27's games)")
    s.add_argument("--dry-run", action="store_true", help="show what would change; store nothing")
    s.add_argument("--show", type=int, default=20, metavar="N", help="queue entries to list")
    s.add_argument("--csv", type=Path, metavar="PATH", help="write the per-player table here")
    sub.add_parser("status", help="careers, group counts and the queue")

    r = sub.add_parser("run", help="generate pictures for queued jobs on the GPU")
    r.add_argument("--group", action="append", choices=("a", "b", "c"), help="only this group")
    r.add_argument("--my-club", action="store_true", help="only players at my club")
    r.add_argument("--limit", type=int, metavar="N", help="stop after N jobs")
    r.add_argument("--no-publish", action="store_true", help="do not write into FM27's folder")
    r.add_argument("--now", action="store_true", help="generate even if FM27 is running")
    r.add_argument("--dry-run", action="store_true", help="print the prompts; no GPU, no storing")
    r.add_argument("--kits", type=Path, help="kit table (default: config/kits.toml)")
    for p in (r, sub.add_parser("publish", help="write current pictures and config.xml")):
        p.add_argument("--career", metavar="KEY", help="career key (default: last synced)")
    g = sub.add_parser("gallery", help="write the local page of generated pictures")
    g.add_argument("--career", metavar="KEY", help="career key (default: last synced)")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    source_factory: Callable[[], DataSource] = _default_source,
    gpu: Gpu | None = None,
    out: TextIO | None = None,
) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    paths = load_settings(defaults_path=args.config).paths
    conn = db.connect(paths.db_path)
    try:
        settings = load_settings(conn, defaults_path=args.config)
        if args.command == "sync":
            return cmd_sync(args, settings, conn, source_factory(), out)
        if args.command == "run":
            return cmd_run(args, settings, conn, gpu or Gpu(), out)
        if args.command == "publish":
            return cmd_publish(args, settings, conn, out)
        if args.command == "gallery":
            return cmd_gallery(args, settings, conn, out)
        return cmd_status(settings, conn, out)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
