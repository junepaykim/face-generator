"""Dashboard routes and their app-local helpers."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from facegen.core.estimate import estimate_queue, seconds_per_picture
from facegen.core.settings_values import Settings
from facegen.datasource import photos
from facegen.i18n import Translator
from facegen.pipeline.worker import Career, load_career
from facegen.runtime.contracts import WorkerStatus
from facegen.web import live, queries
from facegen.web.dependencies import AppDependencies
from facegen.web.messages import seconds_since, worker_sentence
from facegen.web.presentation import Presentation
from facegen.web.responses import _redirect


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
                "elapsed": elapsed,
                "each": each,
                "percent": min(99, round(100 * elapsed / each)) if elapsed and each else None,
            }
        return {
            "worker_text": worker_sentence(worker, t) if worker else "",
            "drawing": drawing,
            "estimate": estimate,
            "recent": queries.recent_pictures(conn, career.id) if career else [],
        }

    def dashboard_context(conn: sqlite3.Connection, t: Translator) -> dict[str, Any]:
        career = load_career(conn)
        settings = settings_now(conn)
        worker = ctx.worker.status() if ctx.worker else None
        return {
            **now_context(conn, career, settings, worker, t),
            "counts": queries.counts(conn, career.id) if career else None,
            "queue": queries.queue(conn, career.id) if career else [],
            "last_sync": queries.last_sync(conn),
            "careers": queries.careers(conn),
            "worker": worker,
            "sync": ctx.syncer.status() if ctx.syncer else None,
            "poller": ctx.poller.status() if ctx.poller else None,
            "settings": settings,
            "auto_groups": sorted(settings.schedule.auto_groups),
            "photo_status": photos.status(conn),
            "setup": ctx.setup(settings),
        }

    @router.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        t = deps.translator(request, conn)
        rev = live.snapshot(ctx, conn)["status"]["rev"]  # what app.js compares events with
        return page(request, "dashboard.html", conn, rev=rev, **dashboard_context(conn, t))

    @router.get("/status", response_class=HTMLResponse)
    def status(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        t = deps.translator(request, conn)
        return presentation.fragment(
            request, "_status.html", conn, career=load_career(conn), **dashboard_context(conn, t)
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
        rev = live.snapshot(ctx, conn)["log"]["rev"]
        return page(request, "log.html", conn, rev=rev, **log_context(conn))

    @router.get("/log/part", response_class=HTMLResponse)
    def log_part(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return presentation.fragment(request, "_log.html", conn, **log_context(conn))

    @router.post("/worker/{action}")
    def worker_action(action: str, request: Request) -> Response:
        t = deps.translator(request)
        if ctx.worker is None:
            return _redirect("/", err=t("msg.worker_absent"))
        match action:
            case "pause":
                ctx.worker.pause()
            case "resume":
                if not ctx.worker.resume():
                    return _redirect("/", err=t("msg.gpu_fault_advice"))
            case "generate-now":
                if not ctx.worker.generate_now():
                    return _redirect("/", err=t("msg.gpu_fault_advice"))
            case _:
                raise HTTPException(404)
        return _redirect("/")

    @router.post("/sync")
    def sync_now(request: Request) -> Response:
        t = deps.translator(request)
        if ctx.syncer is None:
            return _redirect("/", err=t("msg.sync_absent"))
        if not ctx.syncer.start():
            return _redirect("/", err=t("msg.sync_running"))
        return _redirect("/", msg=t("msg.sync_started"))

    return router
