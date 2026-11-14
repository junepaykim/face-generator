"""Command-line entry points.

facegen sync [--save PATH] [--dry-run] [--show N] [--csv PATH]
facegen run [--group a|b|c ...] [--my-club] [--limit N] [--no-publish] [--now] [--dry-run]
facegen publish
facegen gallery
facegen status
facegen serve [--port N] [--paused] [--no-worker] [--no-watch]
facegen worker [--generate-now]   (started by serve: the GPU worker in its own process)
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from facegen import db
from facegen.datasource.base import DataSource
from facegen.gallery import write_gallery
from facegen.kits import KitError, catalog_path, load_kits
from facegen.poller import UNSUPPORTED_ADVICE, NoSaveError, SaveUnreadableError, sync_save
from facegen.publish import RELOAD_HINT, PublishError, PublishReport, publish_active
from facegen.settings import Settings, load_settings
from facegen.sync import SyncReport
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
KIT_SOURCE_NOTES = {
    "table": "",
    "catalog": " (colours from FM27's kit image)",
    "fallback": " (fallback kit)",
}


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
    """What `run` and `worker` use for the GPU and the FM27 check; the tests pass fakes."""

    renderer: Callable[[], Renderer] = _default_renderer
    tagger: Callable[[], Tagger] = _default_tagger
    fm_check: Callable[[], bool | None] = fm_running
    program: tuple[str, ...] | None = None
    """How `serve` starts facegen for its worker process, before `[--config X] worker`;
    None: this Python with `-m facegen.cli`. The tests give a script that passes fakes."""


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
    if c.get("with_photo") or c.get("without_photo"):
        lines.append(
            f"No photo    {c['without_photo']:,} pre-existing players"
            f" ({c['with_photo']:,} have one); {c['without_photo_picked']:,} of them in a group"
            " outside my club"
        )
    if c.get("youth_team"):
        lines.append(
            f"Youth       {c['youth_team']:,} in my club's teams other than the first team"
        )
    if c.get("journalists") or c.get("press_officers"):
        lines.append(
            f"Media       {c['journalists']:,} journalists ({c['local_journalists']:,} from my"
            f" club's country), {c['press_officers']:,} press officers;"
            f" {c['media_picked']:,} in a group"
        )
    if r.new_at_my_club:
        lines.append(
            f"New         {r.new_at_my_club:,} at my club since the last sync: drawn first"
        )
    if r.new_staff:
        lines.append(f"            {r.new_staff:,} new on my club's staff since the last sync")
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
    try:
        report = sync_save(
            conn,
            settings,
            source,
            args.save,
            dry_run=args.dry_run,
            preview=args.show,
            with_table=args.csv is not None,
            trigger="command",
        )
    except NoSaveError as exc:
        print(str(exc), file=out)
        return 2
    except SaveUnreadableError as exc:
        print(str(exc), file=out)
        print(
            UNSUPPORTED_ADVICE
            if exc.unsupported
            else "Existing pictures and the queue are unchanged.",
            file=out,
        )
        return 3
    print(format_report(report), file=out)
    if args.csv is not None:
        write_csv(report.table, args.csv)
        print(f"\nWrote {len(report.table):,} players to {args.csv}", file=out)
    if not args.dry_run:
        # The newest save decides which career's pack FM27 is shown (spec section 8).
        try:
            activation = publish_active(conn, settings)
        except PublishError as exc:
            print(f"Publishing failed: {exc}", file=out)
            return 3
        if activation is not None and activation.changed:
            print(file=out)
            for line in format_publish(activation):
                print(line, file=out)
    return 0


def cmd_status(settings: Settings, conn: sqlite3.Connection, out: TextIO) -> int:
    careers = conn.execute(
        "SELECT id, key, display_name, last_game_date, updated_at FROM careers ORDER BY updated_at"
    ).fetchall()
    if not careers:
        print("No careers yet. Run `facegen sync`.", file=out)
    active = db.active_career_id(conn)
    for c in careers:
        print(
            f"{c['key']}  {c['display_name']}  in-game {c['last_game_date']}  "
            f"last sync {c['updated_at']}"
            + ("  (active: FM27 is shown this pack)" if c["id"] == active else ""),
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
        "SELECT save_file, finished_at, error, unsupported FROM sync_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if last is not None and last["error"]:
        print(f"Last sync failed ({last['save_file']}): {last['error']}", file=out)
        if last["unsupported"]:
            print(UNSUPPORTED_ADVICE, file=out)
    return 0


def _career(conn: sqlite3.Connection, key: str | None, out: TextIO) -> Career | None:
    career = load_career(conn, key)
    if career is None:
        print(f"No career {key!r}." if key else "No careers yet. Run `facegen sync`.", file=out)
    return career


def format_publish(report: PublishReport) -> list[str]:
    lines = []
    if not report.active:
        lines.append(
            f"Published   {report.written:,} pictures to {report.folder}, not mapped: another"
            " career's save is newer, so FM27 is shown that career's pack."
        )
    elif report.changed:
        if report.written or report.config_changed:
            lines.append(
                f"Published   {report.written:,} pictures ({report.unchanged:,} already in place)"
                f" to {report.folder}; config.xml maps {report.mapped:,} players."
            )
        if report.deactivated:
            lines.append(
                f"Packs       {report.folder.name} is the newest save's career; other careers'"
                f" packs now map nothing: {', '.join(report.deactivated)}."
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
        kits = load_kits(args.kits, catalog=catalog_path(settings.paths.data_dir))
    except KitError as exc:
        print(f"kits: {exc}", file=out)
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
                f"{KIT_SOURCE_NOTES[p.kit_source]}, {p.pose}"
                f"{'' if p.reference_scale is None else f', reference {p.reference_scale}'}",
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


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def cmd_serve(
    args: argparse.Namespace,
    settings: Settings,
    source_factory: Callable[[], DataSource],
    gpu: Gpu,
    out: TextIO,
    *,
    run_server: Callable[..., None] | None = None,
) -> int:
    """The web UI with the save watcher in the background and the GPU worker in a process of
    its own (`facegen worker`, supervised by worker_process.WorkerProxy), until Ctrl+C."""
    from facegen.service import PollerService, SyncService, log_to_file
    from facegen.web.app import AppContext, create_app
    from facegen.worker_process import STOP_TIMEOUT, WorkerProxy, worker_command

    host = settings.ui.host
    port = args.port or settings.ui.port
    if host not in LOOPBACK:
        print(f"ui.host is {host!r}: facegen serves on this PC only (127.0.0.1).", file=out)
        return 2
    try:  # the worker process loads them again; a broken table is reported here, at once
        load_kits(catalog=catalog_path(settings.paths.data_dir))
    except KitError as exc:
        print(f"kits: {exc}", file=out)
        return 2
    db_path = settings.paths.db_path
    print(f"Log: {log_to_file(settings.paths.data_dir)}", file=out, flush=True)
    worker = None
    if not args.no_worker:
        worker = WorkerProxy(
            worker_command(gpu.program, args.config), db_path=db_path, start_paused=args.paused
        )
    syncer = SyncService(
        db_path,
        source_factory,
        defaults_path=args.config,
        on_done=worker.wake if worker else (lambda: None),
        log=worker.log if worker else (lambda line: None),
    )
    poller = None if args.no_watch else PollerService(db_path, syncer, defaults_path=args.config)
    app = create_app(AppContext(db_path, args.config, worker, syncer, poller))
    url = f"http://{'[::1]' if host == '::1' else host}:{port}/"
    print(f"facegen UI at {url}  (Ctrl+C to stop)", file=out, flush=True)
    if poller is not None:
        print(f"Watching {settings.paths.games_dir} for new saves.", file=out, flush=True)
        poller.start()
    if worker is not None:
        worker.start()
    try:
        if run_server is None:
            import uvicorn

            uvicorn.run(app, host=host, port=port, log_level="warning")
        else:
            run_server(app, host=host, port=port)
    finally:
        if poller is not None:
            poller.stop(timeout=None)
        if worker is not None:
            print(
                f"Stopping the worker after the current job (at most {STOP_TIMEOUT:.0f} s)...",
                file=out,
                flush=True,
            )
            worker.stop(timeout=None)
    return 0


def cmd_worker(args: argparse.Namespace, settings: Settings, gpu: Gpu, out: TextIO) -> int:
    """The GPU worker as a child of `facegen serve`: commands on stdin, messages on stdout
    (worker_process). Not meant to be run by hand."""
    import faulthandler
    import signal

    from facegen.worker_process import protocol_streams, serve_child

    commands: Iterable[str] = sys.stdin
    process = out is sys.stdout
    if process:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is the parent's: it sends stop
        faulthandler.enable()  # a crash in native code leaves its stack on stderr, so in the log
        commands, out = protocol_streams(sys.stdin, sys.stdout)
    ended = serve_child(
        settings,
        defaults_path=args.config,
        renderer_factory=gpu.renderer,
        tagger_factory=gpu.tagger,
        fm_check=gpu.fm_check,
        commands=commands,
        out=out,
        generate_now=args.generate_now,
    )
    if process and ended.reason == "fault":
        # The CUDA context is dead, and torch tearing it down at exit could hang or abort.
        # Everything has been said and stored: leave without the interpreter's clean-up.
        sys.stderr.flush()
        os._exit(0)
    return 2 if ended.reason == "crashed" else 0


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
    v = sub.add_parser("serve", help="the web UI on 127.0.0.1, with the worker in the background")
    v.add_argument("--port", type=int, help="port (default: ui.port in the settings)")
    v.add_argument("--paused", action="store_true", help="start with the worker paused")
    v.add_argument("--no-worker", action="store_true", help="no GPU work")
    v.add_argument("--no-watch", action="store_true", help="do not watch the saves folder")
    w = sub.add_parser(
        "worker", help="the GPU worker process that serve starts (JSON lines on stdin and stdout)"
    )
    w.add_argument("--generate-now", action="store_true", help="draw the whole queue at once")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    source_factory: Callable[[], DataSource] = _default_source,
    gpu: Gpu | None = None,
    out: TextIO | None = None,
    run_server: Callable[..., None] | None = None,
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
        if args.command == "serve":
            conn.close()  # the server's threads open their own connections
            return cmd_serve(
                args, settings, source_factory, gpu or Gpu(), out, run_server=run_server
            )
        if args.command == "worker":
            conn.close()  # the worker opens its own
            return cmd_worker(args, settings, gpu or Gpu(), out)
        return cmd_status(settings, conn, out)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
