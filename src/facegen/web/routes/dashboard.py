"""Dashboard routes and their app-local helpers."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from facegen.datasource import photos
from facegen.pipeline.worker import load_career
from facegen.platform import native
from facegen.runtime.poller import UNSUPPORTED_ADVICE
from facegen.runtime.worker_service import GPU_FAULT_ADVICE
from facegen.web import queries
from facegen.web.dependencies import AppDependencies
from facegen.web.presentation import Presentation
from facegen.web.responses import _redirect


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    ctx = deps.ctx
    settings_now = deps.settings_now
    Conn = deps.connection
    page = presentation.page
    templates = presentation.templates

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
            "native": native.status(),
            "setup": ctx.setup(settings),
        }

    @router.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return page(request, "dashboard.html", conn, **dashboard_context(conn))

    @router.get("/status", response_class=HTMLResponse)
    def status(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return templates.TemplateResponse(
            request, "_status.html", {"career": load_career(conn), **dashboard_context(conn)}
        )

    @router.post("/worker/{action}")
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

    @router.post("/sync")
    def sync_now() -> Response:
        if ctx.syncer is None:
            return _redirect("/", err="Syncing is not available in this process.")
        if not ctx.syncer.start():
            return _redirect("/", err="A sync is already running.")
        return _redirect("/", msg="Sync started: reading the newest save.")

    return router
