"""The local web UI (docs/spec.md section 7): dashboard, players, review and settings.

Server-rendered pages with plain forms and a little JavaScript; no script or font is loaded
from the network. The app is meant for 127.0.0.1 only and has no login, so:

- requests must name a loopback host (a page on another site cannot reach it through DNS
  rebinding), and a form posted from another site is refused (its Origin is not ours);
- pictures are served only from data/masters/, and thumbnails are cached in data/thumbs/.

Each request opens its own database connection; the worker and syncs run in their own
threads (service.py) and are woken when a request changes the queue.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

from facegen import db, manual, photos, settings_history
from facegen.age import table_name
from facegen.classify import Group
from facegen.manual import ManualError
from facegen.planner import REFERENCE_KINDS, reference_scale
from facegen.poller import UNSUPPORTED_ADVICE
from facegen.publish import RELOAD_HINT, PublishError
from facegen.service import (
    BUSY_TIMEOUT_MS,
    GPU_FAULT_ADVICE,
    PollerService,
    SyncService,
    WorkerControl,
)
from facegen.settings import Settings, SettingsError, load_defaults, load_settings
from facegen.settings_history import Outcome
from facegen.traits import TRAIT_TABLES, eyes_drawn_round
from facegen.web import queries
from facegen.web.forms import (
    blamed_keys,
    change_lines,
    input_text,
    parse_settings_changes,
    parse_settings_form,
    settings_sections,
)
from facegen.web.queries import PlayerFilter
from facegen.web.security import install_security
from facegen.worker import Career, load_career

HERE = Path(__file__).resolve().parent
THUMB_SIZES = (96, 192, 384)
_PICTURE_SUFFIXES = (".png", ".webp")

TIER_NAMES = {0: "manual", 1: "A, my club", 2: "A", 3: "B", 4: "C"}
KIND_NAMES = {
    "initial": "first picture",
    "kit": "new kit",
    "age": "new age band",
    "upgrade": "upgrade to A",
    "reroll": "new face",
    "redraw": "new picture",
}


@dataclass
class AppContext:
    db_path: Path
    defaults_path: Path | None = None
    worker: WorkerControl | None = None
    """WorkerService in a thread, or worker_process.WorkerProxy for a worker process."""
    syncer: SyncService | None = None
    poller: PollerService | None = None


# --- helpers ----------------------------------------------------------------------------------


def _safe_next(target: str | None, fallback: str) -> str:
    """A local path to go back to; anything else (another site) falls back."""
    if not target:
        return fallback
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return fallback
    return target


def _with_message(url: str, msg: str | None = None, err: str | None = None) -> str:
    base, _, query = url.partition("?")
    params = [p for p in query.split("&") if p and not p.startswith(("msg=", "err="))]
    if msg:
        params.append(f"msg={quote(msg)}")
    if err:
        params.append(f"err={quote(err)}")
    return base + ("?" + "&".join(params) if params else "")


def _redirect(url: str, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    return RedirectResponse(_with_message(url, msg, err), status_code=303)


def _within(path: Path, folder: Path) -> bool:
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


def make_thumb(source: Path, size: int, cache_dir: Path) -> Path:
    """A cached WebP thumbnail of a picture, `size` pixels on its long edge (3:4 pictures are
    `size` high). Masters never change once written."""
    st = source.stat()
    key = hashlib.sha1(f"{source.resolve()}|{st.st_mtime_ns}|{size}".encode()).hexdigest()[:20]
    dest = cache_dir / f"{key}-{size}.webp"
    if dest.is_file():
        return dest
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        img = im.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    img.save(tmp, format="WEBP", quality=85)
    os.replace(tmp, dest)
    return dest


def static_version(folder: Path) -> str:
    """A short hash of the static files. Pages link them with it, so a browser fetches the
    script and styles again after an update instead of using the ones it cached."""
    digest = hashlib.sha1()
    for path in sorted(folder.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def describe_job(kind: str) -> str:
    return KIND_NAMES.get(kind, kind)


def uses_reference(kind: str, at_my_club: bool, settings: Settings) -> bool:
    """Whether a job would draw with the player's current picture as a reference."""
    return reference_scale(kind, at_my_club, settings.reference) is not None


def create_app(ctx: AppContext) -> FastAPI:
    # FastAPI's own telemetry is off; nothing leaves this PC (AGENTS.md hard rule 6).
    app = FastAPI(
        title="facegen",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False},
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env
    env.globals.update(
        tier_name=lambda t: TIER_NAMES.get(t, str(t)),
        kind_name=describe_job,
        reload_hint=RELOAD_HINT,
        trait_tables=TRAIT_TABLES,
        uses_reference=uses_reference,
        static_version=static_version(HERE / "static"),
    )
    env.filters["upper_group"] = lambda g: (g or "-").upper()

    def settings_now(conn: sqlite3.Connection) -> Settings:
        return load_settings(conn, ctx.defaults_path)

    data_dir = load_settings(defaults_path=ctx.defaults_path).paths.data_dir
    masters_dir = data_dir / "masters"
    thumbs_dir = data_dir / "thumbs"

    install_security(app, templates)

    def get_conn() -> Iterator[sqlite3.Connection]:
        conn = db.connect(ctx.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
        try:
            yield conn
        finally:
            conn.close()

    Conn = Depends(get_conn)

    def page(request: Request, name: str, conn: sqlite3.Connection, **context: Any) -> Response:
        career = load_career(conn)
        return templates.TemplateResponse(
            request,
            name,
            {
                "career": career,
                "game_date": _game_date(conn, career),
                "review_count": queries.review_count(conn, career.id) if career else 0,
                "msg": request.query_params.get("msg"),
                "err": request.query_params.get("err"),
                "path": request.url.path,
                **context,
            },
        )

    def wake() -> None:
        if ctx.worker is not None:
            ctx.worker.wake()

    def need_career(conn: sqlite3.Connection) -> Career:
        career = load_career(conn)
        if career is None:
            raise HTTPException(409, "No career yet: sync a save first.")
        return career

    async def form_of(request: Request) -> FormData:
        return await request.form()

    # -- dashboard --

    def dashboard_context(conn: sqlite3.Connection) -> dict[str, Any]:
        career = load_career(conn)
        settings = settings_now(conn)
        return {
            "counts": queries.counts(conn, career.id) if career else None,
            "queue": queries.queue(conn, career.id) if career else [],
            "last_sync": queries.last_sync(conn),
            "careers": queries.careers(conn),
            "unsupported_advice": UNSUPPORTED_ADVICE,
            "worker": ctx.worker.status() if ctx.worker else None,
            "sync": ctx.syncer.status() if ctx.syncer else None,
            "poller": ctx.poller.status() if ctx.poller else None,
            "settings": settings,
            "auto_groups": sorted(settings.schedule.auto_groups),
            "photo_status": photos.status(conn),
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return page(request, "dashboard.html", conn, **dashboard_context(conn))

    @app.get("/status", response_class=HTMLResponse)
    def status(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return templates.TemplateResponse(
            request, "_status.html", {"career": load_career(conn), **dashboard_context(conn)}
        )

    @app.post("/worker/{action}")
    def worker_action(action: str) -> Response:
        if ctx.worker is None:
            return _redirect("/", err="The worker is not running in this process.")
        match action:
            case "pause":
                ctx.worker.pause()
            case "resume":
                if not ctx.worker.resume():
                    return _redirect("/", err=GPU_FAULT_ADVICE)
            case "generate-now":
                if not ctx.worker.generate_now():
                    return _redirect("/", err=GPU_FAULT_ADVICE)
            case _:
                raise HTTPException(404)
        return _redirect("/")

    @app.post("/sync")
    def sync_now() -> Response:
        if ctx.syncer is None:
            return _redirect("/", err="Syncing is not available in this process.")
        if not ctx.syncer.start():
            return _redirect("/", err="A sync is already running.")
        return _redirect("/", msg="Sync started: reading the newest save.")

    # -- players --

    @app.get("/players", response_class=HTMLResponse)
    def players(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        career = load_career(conn)
        f = PlayerFilter.from_query(request.query_params)
        rows, total = queries.players_page(conn, career.id, f) if career else ([], 0)
        club_name = None
        if f.club_id is not None and career is not None:
            r = conn.execute(
                "SELECT name FROM clubs WHERE career_id = ? AND club_id = ?",
                (career.id, f.club_id),
            ).fetchone()
            club_name = r[0] if r else str(f.club_id)
        pages = max(1, -(-total // queries.PER_PAGE))
        return page(
            request,
            "players.html",
            conn,
            f=f,
            rows=rows,
            total=total,
            pages=pages,
            club_name=club_name,
            leagues=queries.LEAGUES,
            origins=queries.ORIGINS,
            roles=queries.ROLES,
            group_filters=queries.GROUP_FILTERS,
            picture_filters=queries.PICTURE_FILTERS,
            photo_filters=queries.PHOTO_FILTERS,
            photo_status=photos.status(conn),
            scopes=queries.SCOPES,
            sorts=list(queries.SORTS),
            here=str(request.url.path) + ("?" + request.url.query if request.url.query else ""),
        )

    @app.post("/players/action")
    async def players_action(request: Request) -> Response:
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), "/players")
        try:
            uids = [int(x) for x in form.getlist("uid")]
        except ValueError:
            return _redirect(back, err="Bad player ID.")
        if not uids:
            return _redirect(back, err="Tick at least one player first.")
        action = form.get("action")

        def run() -> str:
            conn = db.connect(ctx.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
            try:
                career = need_career(conn)
                settings = settings_now(conn)
                if action == "generate":
                    group = str(form.get("group") or manual.DEFAULT_GROUP)
                    r = manual.generate(conn, career.id, uids, settings, group)
                    return f"Generate as group {group.upper()}: {r.summary()}."
                if action == "new-picture":
                    r = manual.new_pictures(conn, career.id, uids, settings)
                    return f"New picture, same character sheet: {r.summary()}."
                if action == "pin":
                    pin = str(form.get("pin") or "")
                    r = manual.set_pin(conn, career.id, uids, pin or None, settings)
                    if pin in ("a", "b", "c"):
                        return f"Pin to {pin.upper()} (a hand-pick, as Generate): {r.summary()}."
                    what = "pinned to never" if pin == "never" else "unpinned"
                    return f"{len(r.pinned)} player(s) {what}."
                raise ManualError(f"unknown action {action!r}")
            finally:
                conn.close()

        try:
            message = await run_in_threadpool(run)
        except (ManualError, HTTPException) as exc:
            return _redirect(back, err=getattr(exc, "detail", None) or str(exc))
        wake()
        return _redirect(back, msg=message)

    @app.get("/players/{uid}", response_class=HTMLResponse)
    def player(uid: int, request: Request, conn: sqlite3.Connection = Conn) -> Response:
        career = need_career(conn)
        detail = queries.player_detail(conn, career.id, uid, data_dir)
        if detail is None:
            raise HTTPException(404, f"No player {uid} in this career.")
        settings = settings_now(conn)
        p = detail["player"]
        group_for_redraw = p["group_name"] or (
            detail["current"]["group_name"] if detail["current"] else None
        )
        return page(
            request,
            "player.html",
            conn,
            d=detail,
            p=p,
            settings=settings,
            age_table=table_name(p["group_name"], bool(p["at_my_club"]), bool(p["is_staff"])),
            redraw_reference=uses_reference("redraw", bool(p["at_my_club"]), settings),
            eyes_round=bool(detail["sheet"])
            and eyes_drawn_round(
                detail["sheet"]["fields"]["eye_shape"],
                detail["sheet"]["fields"]["base_seed"],
                settings.my_club.round_eyes,
            ),
            group_for_redraw=group_for_redraw,
            photo_status=photos.status(conn),
            reference_kinds=REFERENCE_KINDS,
            groups=[g.value for g in Group],
        )

    @app.post("/players/{uid}/{action}")
    async def player_action(uid: int, action: str, request: Request) -> Response:
        form = await form_of(request)
        back = f"/players/{uid}"

        def run() -> str:
            conn = db.connect(ctx.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
            try:
                career = need_career(conn)
                settings = settings_now(conn)
                match action:
                    case "generate":
                        group = str(form.get("group") or manual.DEFAULT_GROUP)
                        r = manual.generate(conn, career.id, [uid], settings, group)
                        if r.skipped:
                            raise ManualError(r.skipped[0][1])
                        if r.pinned:
                            return f"Pinned to {group.upper()}; the picture is already good enough."
                        return f"Queued as group {group.upper()}."
                    case "new-face":
                        manual.new_face(conn, career.id, uid, settings)
                        return "New face queued: a new character sheet, drawn next."
                    case "new-picture":
                        manual.new_picture(conn, career.id, uid, settings)
                        return "New picture queued with the same character sheet."
                    case "pin":
                        pin = str(form.get("pin") or "")
                        r = manual.set_pin(conn, career.id, [uid], pin or None, settings)
                        if r.skipped:
                            raise ManualError(r.skipped[0][1])
                        if r.queued:
                            return f"Pinned to {pin.upper()} and queued as a request."
                        if pin in ("a", "b", "c"):
                            return f"Pinned to {pin.upper()}; the picture is already good enough."
                        if pin == "never":
                            return "Pinned to never; queued jobs cancelled."
                        return "Pin cleared; the rules decide."
                    case "remove":
                        removed = manual.remove_picture(conn, career.id, career.key, uid, settings)
                        return (
                            "Picture removed and pinned to never. " + RELOAD_HINT
                            if removed
                            else "Pinned to never; there was no published picture."
                        )
                    case "retry":
                        manual.retry(conn, career.id, uid, settings)
                        return "Queued again with new seeds."
                    case "restore":
                        try:
                            image_id = int(str(form.get("image_id") or ""))
                        except ValueError:
                            raise ManualError("Bad picture ID.") from None
                        r = manual.restore_picture(
                            conn, career.id, career.key, uid, image_id, settings
                        )
                        where = (
                            RELOAD_HINT
                            if r.publish.active
                            else "Not shown in FM27: another career's save is newer."
                        )
                        note = (
                            " Still pinned to never: it is not updated." if r.pinned_never else ""
                        )
                        return f"v{r.version} is the current picture again.{note} {where}"
                    case "cancel":
                        done = manual.cancel(conn, career.id, uid, settings)
                        return "Request cancelled." if done else "Nothing to cancel."
                raise ManualError(f"unknown action {action!r}")
            finally:
                conn.close()

        try:
            message = await run_in_threadpool(run)
        except (ManualError, HTTPException) as exc:
            target = _safe_next(str(form.get("next") or ""), back)
            return _redirect(target, err=getattr(exc, "detail", None) or str(exc))
        except PublishError as exc:
            return _redirect(back, err=f"Publishing failed: {exc}")
        wake()
        return _redirect(_safe_next(str(form.get("next") or ""), back), msg=message)

    # -- review --

    @app.get("/review", response_class=HTMLResponse)
    def review(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        career = load_career(conn)
        items = queries.review_items(conn, career.id) if career else []
        return page(request, "review.html", conn, items=items)

    # -- settings --
    # Each change is saved as it is made (the page's script posts to /settings/save) and can
    # be undone; settings_history.py keeps the history. Without the script, the page's Save
    # button posts the whole form.

    def history_context(conn: sqlite3.Connection) -> dict[str, Any]:
        defaults = load_defaults(ctx.defaults_path)
        entries = settings_history.entries(conn)
        return {
            "history": [
                {
                    "entry": e,
                    "lines": change_lines(e.before, e.after, defaults),
                    "summary": e.report.summary() if e.report else "",
                }
                for e in entries
            ],
            "latest_done": next((e.id for e in entries if not e.undone), None),
            "can_undo": settings_history.can_undo(conn),
            "can_redo": settings_history.can_redo(conn),
        }

    def outcome_message(verb: str, outcome: Outcome) -> str:
        """ "Saved: Group A · CA above 155 → 150. A 52 → 60; queue: 8 queued." and the like."""
        defaults = load_defaults(ctx.defaults_path)
        what = "; ".join(
            f"{line.label} {line.before} → {line.after}"
            if verb != "Undone"
            else f"{line.label} {line.after} → {line.before}"
            for e in outcome.entries
            for line in change_lines(e.before, e.after, defaults)
        )
        if any(e.label == "defaults" for e in outcome.entries):
            what = f"every setting back to its default ({what})"
        note = f"{verb}: {what}."
        if outcome.report is not None:
            summary = outcome.report.summary()
            note += f" {summary[:1].upper()}{summary[1:]}."
        if any(k.startswith("eligibility.") for e in outcome.entries for k in e.after):
            note += " Eligibility changes take effect at the next sync."
        return note

    def settings_page(
        request: Request,
        conn: sqlite3.Connection,
        values: dict[str, Any] | None = None,
        *,
        errors: dict[str, str] | None = None,
        error: str | None = None,
    ) -> Response:
        defaults = load_defaults(ctx.defaults_path)
        current = settings_now(conn)
        shown = dict(current.values) | (values or {})
        return page(
            request,
            "settings.html",
            conn,
            sections=settings_sections(shown, defaults),
            errors=errors or {},
            error=error,
            **history_context(conn),
        )

    @app.get("/settings", response_class=HTMLResponse)
    def settings_get(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return settings_page(request, conn)

    @app.post("/settings", response_class=HTMLResponse)
    async def settings_post(request: Request) -> Response:
        """The whole form (Save, without the page's script), Reset, Undo and Redo."""
        form = await form_of(request)
        action = form.get("action")

        def run() -> Response:
            conn = db.connect(ctx.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
            try:
                defaults = load_defaults(ctx.defaults_path)
                values: dict[str, Any] = {}
                try:
                    match action:
                        case "apply":
                            values, errors = parse_settings_form(form, defaults)
                            if errors:
                                return settings_page(request, conn, values, errors=errors)
                            verb = "Saved"
                            outcome = settings_history.save(
                                conn, values, defaults_path=ctx.defaults_path
                            )
                        case "defaults":
                            verb = "Saved"
                            outcome = settings_history.save_defaults(
                                conn, defaults_path=ctx.defaults_path
                            )
                        case "undo":
                            through = str(form.get("through") or "")
                            verb = "Undone"
                            outcome = settings_history.undo(
                                conn,
                                through=int(through) if through.isdigit() else None,
                                defaults_path=ctx.defaults_path,
                            )
                        case "redo":
                            verb = "Redone"
                            outcome = settings_history.redo(conn, defaults_path=ctx.defaults_path)
                        case _:
                            raise HTTPException(400, "unknown action")
                except SettingsError as exc:
                    if action == "apply":
                        return settings_page(request, conn, values, error=str(exc))
                    return _redirect("/settings", err=f"Not done: {exc}")
                if not outcome.changed:
                    nothing = {"undo": "Nothing to undo.", "redo": "Nothing to redo."}
                    return _redirect("/settings", msg=nothing.get(str(action), "Nothing changed."))
                wake()
                return _redirect("/settings", msg=outcome_message(verb, outcome))
            finally:
                conn.close()

        return await run_in_threadpool(run)

    @app.post("/settings/save")
    async def settings_save(request: Request) -> Response:
        """Autosave: the fields the page changed. Answers JSON for the page's script."""
        form = await form_of(request)

        def run() -> Response:
            conn = db.connect(ctx.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
            try:
                defaults = load_defaults(ctx.defaults_path)
                values, errors = parse_settings_changes(form, defaults)
                if errors:
                    return JSONResponse(
                        {"ok": False, "errors": errors, "error": "Not saved: check the field."}
                    )
                try:
                    outcome = settings_history.save(conn, values, defaults_path=ctx.defaults_path)
                except SettingsError as exc:
                    message = str(exc)
                    blamed = blamed_keys(message, list(values)) or list(values)
                    return JSONResponse(
                        {
                            "ok": False,
                            "errors": {k: message for k in blamed},
                            "error": f"Not saved: {message}",
                        }
                    )
                if outcome.changed:
                    wake()
                # The values as stored (lists lower-cased, numbers tidied), for the page to show.
                current = outcome.settings.values
                fields = {
                    key: {
                        "value": current[key],
                        "text": input_text(current[key]),
                        "changed": current[key] != defaults[key],
                    }
                    for key in values
                }
                history = history_context(conn)
                history_html = templates.get_template("_settings_history.html").render(**history)
                return JSONResponse(
                    {
                        "ok": True,
                        "changed": outcome.changed,
                        "message": outcome_message("Saved", outcome)
                        if outcome.changed
                        else "Nothing changed.",
                        "fields": fields,
                        "history_html": history_html,
                        "can_undo": history["can_undo"],
                        "can_redo": history["can_redo"],
                    }
                )
            finally:
                conn.close()

        return await run_in_threadpool(run)

    # -- pictures --

    def send_picture(path: Path, size: int | None) -> Response:
        if not _within(path, masters_dir) or path.suffix not in _PICTURE_SUFFIXES:
            raise HTTPException(404)
        if not path.is_file():
            raise HTTPException(404, "picture file missing")
        if size is not None:
            if size not in THUMB_SIZES:
                raise HTTPException(400, f"size must be one of {THUMB_SIZES}")
            path = make_thumb(path, size, thumbs_dir)
        media_type = "image/webp" if path.suffix == ".webp" else "image/png"
        return FileResponse(
            path, media_type=media_type, headers={"Cache-Control": "private, max-age=86400"}
        )

    @app.get("/pictures/{image_id}")
    def picture(
        image_id: int, size: int | None = None, conn: sqlite3.Connection = Conn
    ) -> Response:
        row = conn.execute("SELECT master_path FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None or not row["master_path"]:
            raise HTTPException(404)
        return send_picture(data_dir / row["master_path"], size)

    @app.get("/files")
    def files(path: str, size: int | None = None) -> Response:
        """A candidate picture, by its path relative to data/ as stored in check results."""
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts or not path.startswith("masters/"):
            raise HTTPException(404)
        return send_picture(data_dir / rel, size)

    return app


def _game_date(conn: sqlite3.Connection, career: Career | None) -> str | None:
    if career is None:
        return None
    row = conn.execute("SELECT last_game_date FROM careers WHERE id = ?", (career.id,)).fetchone()
    return row[0] if row else None
