"""Command-line entry points.

facegen sync [--save PATH] [--dry-run] [--show N] [--csv PATH]
facegen run [--group a|b|c ...] [--my-club] [--limit N] [--no-publish] [--now] [--dry-run]
facegen publish
facegen gallery
facegen status
facegen serve [--port N] [--paused] [--no-worker] [--no-watch]
facegen worker [--generate-now]   (started by serve: the GPU worker in its own process)
facegen native [--install [--from DIR]]
facegen models [--download [--yes]]
facegen hardware [--json]
facegen app [--minimized] [--port N] [--paused]   (the dashboard in a window, with a tray icon)
facegen tray [--port N] [--paused]                (the tray icon; the dashboard in the browser)
facegen launch [--open-running] [--log PATH]      (what Start.bat runs after `uv sync`)
facegen shortcuts                                 (desktop and Start Menu shortcuts, Windows)
facegen update [--wait S] [--log PATH]            (what Start.bat --update runs)
facegen remove-pack [--yes]                       (what the uninstaller offers)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TextIO

from facegen.core.job_records import Career, JobFilter
from facegen.core.kit_values import KitError
from facegen.core.settings_values import Paths, Settings
from facegen.datasource.base import DataSource
from facegen.datasource.kits import load_kits
from facegen.persistence import db
from facegen.persistence.settings import folder_not_set_message, load_settings
from facegen.pipeline.gallery import write_gallery
from facegen.pipeline.publish import RELOAD_HINT, PublishError, PublishReport, publish_active
from facegen.pipeline.sync import SyncReport
from facegen.pipeline.worker import (
    Worker,
    WorkerBusy,
    load_career,
)
from facegen.platform import native
from facegen.platform.output import utf8_output
from facegen.runtime.dependencies import Gpu
from facegen.runtime.dependencies import default_renderer as _default_renderer
from facegen.runtime.dependencies import default_source as _default_source
from facegen.runtime.dependencies import default_tagger as _default_tagger
from facegen.runtime.poller import UNSUPPORTED_ADVICE, NoSaveError, SaveUnreadableError, sync_save
from facegen.setup import hardware, models, native_wheel
from facegen.setup import operations as setup_operations

_TIER_NAMES = {0: "manual", 1: "A, my club", 2: "A", 3: "B", 4: "C"}
KIT_SOURCE_NOTES = {
    "table": "",
    "catalog": " (colours from FM27's kit image)",
    "bundled": " (bundled colours from FM27's kit image)",
    "fallback": " (fallback kit)",
}


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
    print(f"FM27 folder   {settings.paths.describe('paths.fm_user_dir')}", file=out)
    print(f"FM27 install  {settings.paths.describe('paths.fm_install_dir')}", file=out)
    print(f"Rust code     {native.status().describe()}", file=out)
    try:
        print(f"Models        {models.status().describe()}", file=out)
    except Exception as exc:  # noqa: BLE001 - a status line, never a reason to stop
        print(f"Models        unknown ({type(exc).__name__}: {exc})", file=out)
    if settings.paths.fm_user_dir is None:
        print(folder_not_set_message(), file=out)
    print(file=out)
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


def cmd_native(
    args: argparse.Namespace,
    out: TextIO,
    *,
    host: native_wheel.Host | None = None,
    native_dir: Path | None = None,
    fetch: Callable[[str, Path], None] = native_wheel.download,
    installer: Callable[[Path], None] = native_wheel.uv_install,
    loads: Callable[[], bool] = native_wheel.imports_cleanly,
) -> int:
    return setup_operations.prepare_native(
        install=args.install,
        source=args.wheel_dir,
        report=partial(print, file=out),
        host=host,
        native_dir=native_dir,
        fetch=fetch,
        installer=installer,
        loads=loads,
    )


def cmd_models(
    args: argparse.Namespace,
    out: TextIO,
    *,
    cache: models.HubCache | None = None,
    command: Sequence[str] | None = None,
    ask: Callable[[str], str] = input,
    interactive: Callable[[], bool] = lambda: sys.stdin is not None and sys.stdin.isatty(),
) -> int:
    if args.child:
        return models.run_child(out)
    return setup_operations.prepare_models(
        download=args.download,
        confirmed=args.yes,
        report=partial(print, file=out),
        ask=ask,
        interactive=interactive,
        run_download=lambda: models.wait_and_print(models.ModelDownload(command), out),
        cache=cache,
    )


def cmd_hardware(
    args: argparse.Namespace,
    out: TextIO,
    *,
    detect: Callable[[], hardware.Hardware] | None = None,
    current: dict[str, object] | None = None,
) -> int:
    """The graphics card and the preset Auto-configure would apply (Settings page)."""
    found = (detect or hardware.detect)()
    if args.json:
        print(json.dumps(found.to_json(), ensure_ascii=True), file=out, flush=True)
        return 0
    print(f"Graphics card  {found.describe()}", file=out)
    if found.torch:
        print(f"torch          {found.torch}, CUDA {found.cuda or 'none'}", file=out)
    preset = hardware.recommend(found.vram_gib)
    print(f"Preset         {preset.title}: {preset.note}", file=out)
    if current is not None:
        for c in hardware.changes(preset, current):
            print(f"  would change {c.key}: {c.before} -> {c.after}", file=out)
    return 0 if found.has_gpu else 1


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
        kits = load_kits(args.kits, data_dir=settings.paths.data_dir)
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
    check = gpu.models_check()
    if not args.dry_run and check is not None and not check():
        print(
            "The models are not downloaded yet. Run `facegen models --download` (about"
            f" {models.gigabytes(models.total_size(models.FILES))}), or use the dashboard.",
            file=out,
        )
        return 2
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
    from facegen.runtime.composition import (
        GRACEFUL_SHUTDOWN_SECONDS,
        CannotStart,
        build_runtime,
    )
    from facegen.runtime.logging import log_to_file
    from facegen.runtime.worker_process import STOP_TIMEOUT

    try:
        rt = build_runtime(
            settings,
            source_factory,
            gpu,
            config=args.config,
            port=args.port,
            paused=args.paused,
            worker=not args.no_worker,
            watch=not args.no_watch,
        )
    except CannotStart as exc:
        print(exc, file=out)
        return 2
    print(f"Log: {log_to_file(settings.paths.data_dir)}", file=out, flush=True)
    print(f"facegen UI at {rt.url}  (Ctrl+C to stop)", file=out, flush=True)
    if rt.poller is not None:
        if settings.paths.games_dir is None:
            print(folder_not_set_message(), file=out, flush=True)
        else:
            print(f"Watching {settings.paths.games_dir} for new saves.", file=out, flush=True)
    rt.start()
    try:
        if run_server is None:
            import uvicorn

            uvicorn.run(
                rt.app,
                host=rt.host,
                port=rt.port,
                log_level="warning",
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
            )
        else:
            run_server(rt.app, host=rt.host, port=rt.port)
    finally:
        rt.stop_updates()
        for name, stop in rt.stop_steps():
            if name == "worker":
                print(
                    f"Stopping the worker after the current job (at most {STOP_TIMEOUT:.0f} s)...",
                    file=out,
                    flush=True,
                )
            stop()
    return 0


def cmd_worker(args: argparse.Namespace, settings: Settings, gpu: Gpu, out: TextIO) -> int:
    """The GPU worker as a child of `facegen serve`: commands on stdin, messages on stdout
    (worker_process). Not meant to be run by hand."""
    import faulthandler
    import signal

    from facegen.runtime.worker_process import protocol_streams, serve_child

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
        models_ready=gpu.models_check(),
    )
    if process and ended.reason == "fault":
        # The CUDA context is dead, and torch tearing it down at exit could hang or abort.
        # Everything has been said and stored: leave without the interpreter's clean-up.
        sys.stderr.flush()
        os._exit(0)
    return 2 if ended.reason == "crashed" else 0


def cmd_desktop(
    args: argparse.Namespace,
    source_factory: Callable[[], DataSource],
    gpu: Gpu | None,
) -> int:
    """`facegen app` and `facegen tray` (desktop.py). A failure while starting is written to
    the log and shown in a message box on Windows, as when started without a console."""
    from facegen.desktop import shell as desktop

    try:
        return desktop.run(args.command, args, source_factory=source_factory, gpu=gpu)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - shown, with where the details are
        desktop.report_crash(exc)
        return 1


def cmd_shortcuts(args: argparse.Namespace, data_dir: Path, out: TextIO) -> int:
    """Write facegen's shortcuts on the desktop and in the Start Menu (`Start.bat
    --shortcut`). Exit 2 off Windows, 3 when they could not be written."""
    from facegen.desktop import shortcuts

    if os.name != "nt":
        print("Shortcuts are for Windows. Here, start facegen with `facegen serve`.", file=out)
        return 2
    folders = shortcuts.windows_folders()
    app = shortcuts.this_app(data_dir, args.config)
    links = shortcuts.app_shortcuts(
        folders, pythonw=app.pythonw, root=app.root, icon=app.icon, config=app.config
    )
    if not links:
        print("Neither the desktop nor the Start Menu folder was found.", file=out)
        return 3
    try:
        shortcuts.write(links)
    except shortcuts.ShortcutError as exc:
        print(f"Not written: {exc}", file=out)
        return 3
    for link in links:
        print(f"Shortcut    {link.path}", file=out)
    return 0


def cmd_remove_pack(args: argparse.Namespace, paths: Paths, out: TextIO) -> int:
    """Delete facegen's pictures from FM27's graphics folder (graphics/facegen/), as the
    uninstaller offers. Asks first unless --yes. Exit 1 when not done."""
    from facegen.pipeline.publish import packs_root, remove_packs

    if paths.fm_user_dir is None:
        print("FM27's folder is not set, so there is no picture pack to remove.", file=out)
        return 0
    root = packs_root(paths.fm_user_dir)
    if not root.exists() and not root.is_symlink():
        print(f"There is no picture pack to remove ({root}).", file=out)
        return 0
    if not args.yes:
        answer = input(f"Delete {root} and every picture in it? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing was removed.", file=out)
            return 1
    try:
        remove_packs(paths.fm_user_dir, paths.data_dir)
    except OSError as exc:
        print(f"Not removed: {exc}", file=out)
        return 1
    print(f"Removed {root}. FM27 shows its own photos after a restart or Reload Skin.", file=out)
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
    v = sub.add_parser("serve", help="the web UI on 127.0.0.1, with the worker in the background")
    v.add_argument("--port", type=int, help="port (default: ui.port in the settings)")
    v.add_argument("--paused", action="store_true", help="start with the worker paused")
    v.add_argument("--no-worker", action="store_true", help="no GPU work")
    v.add_argument("--no-watch", action="store_true", help="do not watch the saves folder")
    w = sub.add_parser(
        "worker", help="the GPU worker process that serve starts (JSON lines on stdin and stdout)"
    )
    w.add_argument("--generate-now", action="store_true", help="draw the whole queue at once")
    n = sub.add_parser("native", help="whether the Rust extension is in use; install its wheel")
    n.add_argument(
        "--install",
        action="store_true",
        help="download this PC's prebuilt wheel, check its SHA-256 and install it",
    )
    n.add_argument(
        "--from",
        dest="wheel_dir",
        type=Path,
        metavar="DIR",
        help="with --install: take the wheel from this folder instead of downloading it",
    )
    m = sub.add_parser("models", help="the model files: what is missing; download them once")
    m.add_argument(
        "--download", action="store_true", help="download the missing files (asks first)"
    )
    m.add_argument("--yes", action="store_true", help="with --download: do not ask")
    m.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    h = sub.add_parser("hardware", help="the graphics card and the settings preset for it")
    h.add_argument("--json", action="store_true", help="one line of JSON (for facegen serve)")
    from facegen.desktop.shell import add_arguments

    add_arguments(sub.add_parser("app", help="the dashboard in a window, with a tray icon"))
    add_arguments(sub.add_parser("tray", help="the tray icon; the dashboard in the browser"))
    la = sub.add_parser("launch", help="what Start.bat runs after uv sync: wheel, models")
    la.add_argument(
        "--open-running",
        action="store_true",
        help="only ask a running facegen app to open its window (exit 1 if none runs)",
    )
    la.add_argument("--log", type=Path, metavar="PATH", help="also append the output here")
    sub.add_parser("shortcuts", help="desktop and Start Menu shortcuts to the app (Windows)")
    up = sub.add_parser(
        "update", help="what Start.bat --update runs: put the newest release over this folder"
    )
    up.add_argument(
        "--wait",
        type=float,
        default=180.0,
        metavar="SECONDS",
        help="how long to wait for a running facegen to quit (default 180)",
    )
    up.add_argument("--log", type=Path, metavar="PATH", help="also append the output here")
    rp = sub.add_parser(
        "remove-pack", help="delete facegen's pictures from FM27's graphics folder (uninstall)"
    )
    rp.add_argument("--yes", action="store_true", help="do not ask")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    source_factory: Callable[[], DataSource] = _default_source,
    gpu: Gpu | None = None,
    out: TextIO | None = None,
    run_server: Callable[..., None] | None = None,
) -> int:
    if out is None:
        utf8_output(sys.stdout, sys.stderr)
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    if args.command == "native":  # needs no settings or database
        return cmd_native(args, out)
    if args.command == "models":  # neither do these
        return cmd_models(args, out)
    if args.command == "hardware" and args.json:
        return cmd_hardware(args, out)
    if args.command in ("app", "tray"):  # takes its lock before it opens the database
        return cmd_desktop(args, source_factory, gpu)
    if args.command == "launch":  # Start.bat's: no database; it reads the settings itself
        from facegen.setup.launch import cmd_launch

        return cmd_launch(args, out)
    if args.command == "update":  # Start.bat --update's: no database either
        from facegen.setup.update import cmd_update

        return cmd_update(args, out)
    paths = load_settings(defaults_path=args.config).paths
    if args.command == "shortcuts":
        return cmd_shortcuts(args, paths.data_dir, out)
    if args.command == "remove-pack":
        return cmd_remove_pack(args, paths, out)
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
        if args.command == "hardware":
            return cmd_hardware(args, out, current=dict(settings.values))
        return cmd_status(settings, conn, out)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
