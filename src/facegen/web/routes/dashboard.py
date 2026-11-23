"""Dashboard routes and their app-local helpers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from facegen.core.estimate import estimate_queue, seconds_per_picture
from facegen.core.job_records import Career
from facegen.core.settings_values import Settings
from facegen.i18n import Translator
from facegen.persistence import photo_index
from facegen.persistence.jobs import load_career
from facegen.runtime.contracts import ERROR_REASONS, SyncStatus, WorkerStatus
from facegen.web import guide, live, queries
from facegen.web.context import SetupState
from facegen.web.dependencies import AppDependencies, translator_for
from facegen.web.forms import describe_key, value_text
from facegen.web.messages import problem, seconds_since, worker_sentence
from facegen.web.presentation import Presentation
from facegen.web.responses import _redirect, _safe_next


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    ctx = deps.ctx
    settings_now = deps.settings_now
    Conn = deps.connection
    page = presentation.page

    def now_context(
        conn: sqlite3.Connection,
        career: Career | None,
        settings: Settings,
        worker: WorkerStatus | None,
        t: Translator,
    ) -> dict[str, Any]:
        """What the worker is doing and when the queue will be done (Phase 12c)."""
        per_picture = seconds_per_picture(queries.drawn_times(conn))
        estimate = (
            estimate_queue(
                queries.queued_by_group(conn, career.id),
                per_picture,
                settings.schedule.auto_groups,
            )
            if career
            else None
        )
        drawing = None
        if worker is not None and worker.current is not None and career is not None:
            job = worker.current
            elapsed = seconds_since(worker.since)
            each = per_picture.get(job.group) or max(per_picture.values(), default=None)
            drawing = {
                "job": job,
                "picture": queries.current_picture_id(conn, career.id, job.unique_id),
                "each": each,
                "percent": min(99, round(100 * elapsed / each)) if elapsed and each else None,
            }
        raw = None
        if worker is not None and worker.reason in ERROR_REASONS:
            raw = worker.values.get("error") or worker.last_error
        return {
            "worker_problem": problem("worker", "", raw, t),
            "worker_text": worker_sentence(worker, t) if worker else "",
            "drawing": drawing,
            "estimate": estimate,
            "recent": queries.recent_pictures(conn, career.id) if career else [],
        }

    def first_run(
        conn: sqlite3.Connection,
        career: Career | None,
        settings: Settings,
        setup: SetupState,
        worker: WorkerStatus | None,
        sync: SyncStatus | None,
        counts: dict[str, int] | None,
        t: Translator,
    ) -> guide.Guide | None:
        """The first run as one guided flow (Phase 12f); None once it is not shown."""
        facts = guide.Facts(
            folder_ok=not setup.needs_folder,
            folder=str(settings.paths.fm_user_dir) if settings.paths.fm_user_dir else None,
            folder_gone=setup.folder_gone,
            models=setup.models,
            download=setup.download,
            card_available=deps.card.available,
            card=deps.card.known(),
            current_values=dict(settings.values),
            career=career.display_name if career else None,
            game_date=career.game_date.isoformat() if career and career.game_date else None,
            syncing=bool(sync and sync.running),
            can_sync=sync is not None,
            watching=ctx.poller is not None,
            worker=worker,
            pictures=counts["pictures"] if counts else 0,
            first_picture_at=queries.first_picture_at(conn, career.id) if career else None,
            closed=guide.closed(deps.data_dir),
            now=datetime.now(UTC),
        )
        shown = guide.build(facts)
        if shown is not None and deps.card.known() is None:
            deps.card.start()  # a few seconds in the background; the page updates when done
        if shown is not None:
            for step in shown.steps:
                if "changes" in step.data:  # the card's suggested settings, for people to read
                    step.data["lines"] = [
                        (describe_key(c.key, t), value_text(c.before, t, c.key),
                         value_text(c.after, t, c.key))
                        for c in step.data["changes"]
                    ]  # fmt: skip
        return shown

    def dashboard_context(
        request: Request, conn: sqlite3.Connection, sync: SyncStatus | None
    ) -> dict[str, Any]:
        career = load_career(conn)
        settings = settings_now(conn)
        t = translator_for(request, settings)
        worker = ctx.worker.status() if ctx.worker else None
        counts = queries.counts(conn, career.id) if career else None
        last_sync = queries.last_sync(conn)
        sync_raw = (sync.error if sync and not sync.running else None) or (
            last_sync["error"] if last_sync and not last_sync.get("unsupported") else None
        )
        setup = ctx.setup(settings)
        download_error = setup.download.error if setup.download else None
        return {
            **now_context(conn, career, settings, worker, t),
            "career": career,
            "guide": first_run(conn, career, settings, setup, worker, sync, counts, t),
            "counts": counts,
            "queue": queries.queue(conn, career.id) if career else [],
            "last_sync": last_sync,
            "careers": queries.careers(conn),
            "worker": worker,
            "sync": sync,
            "sync_problem": problem("sync", t("problem.sync"), sync_raw, t),
            "download_problem": problem("download", t("problem.download"), download_error, t),
            "settings": settings,
            "photo_status": photo_index.status(conn),
            "setup": setup,
        }

    @router.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        # What app.js compares events with.
        sync = ctx.syncer.status() if ctx.syncer else None
        rev = live.snapshot(ctx, conn, deps.live_extra(), sync=sync)["status"]["rev"]
        return page(
            request,
            "dashboard.html",
            conn,
            rev=rev,
            update=deps.update_offer(),
            **dashboard_context(request, conn, sync),
        )

    @router.get("/status", response_class=HTMLResponse)
    def status(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        sync = ctx.syncer.status() if ctx.syncer else None
        return presentation.fragment(
            request, "_status.html", conn, **dashboard_context(request, conn, sync)
        )

    def log_context(conn: sqlite3.Connection) -> dict[str, Any]:
        return {
            "career": load_career(conn),
            "last_sync": queries.last_sync(conn),
            "worker": ctx.worker.status() if ctx.worker else None,
            "poller": ctx.poller.status() if ctx.poller else None,
        }

    @router.get("/log", response_class=HTMLResponse)
    def log(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        """The technical side: the worker's log and the last sync in full (Phase 12)."""
        rev = live.snapshot(ctx, conn, deps.live_extra())["log"]["rev"]
        return page(request, "log.html", conn, rev=rev, **log_context(conn))

    @router.get("/log/part", response_class=HTMLResponse)
    def log_part(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return presentation.fragment(request, "_log.html", conn, **log_context(conn))

    @router.post("/worker/{action}")
    async def worker_action(action: str, request: Request) -> Response:
        """Pause, Resume and Generate now, from the dashboard or the sidebar's card (Phase
        17d): back to the page the button was on ("next")."""
        if action not in ("pause", "resume", "generate-now"):
            raise HTTPException(404)
        form = await request.form()
        back = _safe_next(str(form.get("next") or ""), "/")
        t = await run_in_threadpool(deps.translator, request)
        worker = ctx.worker
        if worker is None:
            return _redirect(back, err=t("msg.worker_absent"))

        def act() -> bool:
            match action:
                case "pause":
                    worker.pause()
                    return True
                case "resume":
                    return worker.resume()
                case _:
                    return worker.generate_now()

        if not await run_in_threadpool(act):
            return _redirect(back, err=t("msg.gpu_fault_advice"))
        return _redirect(back)

    @router.post("/sync")
    def sync_now(request: Request) -> Response:
        t = deps.translator(request)
        if ctx.syncer is None:
            return _redirect("/", err=t("msg.sync_absent"))
        if not ctx.syncer.start():
            return _redirect("/", err=t("msg.sync_running"))
        return _redirect("/", msg=t("msg.sync_started"))

    return router
