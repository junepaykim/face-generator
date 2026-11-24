"""The Log page (Phase 12, 17i): the worker's log, the save watcher and the last sync, the report
the page copies, and opening one of Kitshot's folders in Explorer."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

import kitshot
from kitshot.persistence.jobs import load_career
from kitshot.pipeline.publish import packs_root
from kitshot.platform import platform_info
from kitshot.runtime import report
from kitshot.web import live, queries
from kitshot.web.dependencies import AppDependencies, translator_for
from kitshot.web.messages import watcher_line
from kitshot.web.presentation import Presentation
from kitshot.web.responses import _redirect, _safe_next

CARD_WAIT_SECONDS = 15.0
"""How long the report waits for the graphics card's answer when nobody has asked yet."""

FOLDERS = ("logs", "data", "fm", "pack")
"""The folders Open folder may show, by name (Phase 17i; "fm" and "pack" for 17k): never a
path from the request."""


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    ctx = deps.ctx
    Conn = deps.connection
    page = presentation.page

    def log_context(request: Request, conn: sqlite3.Connection) -> dict[str, Any]:
        worker = ctx.worker.status() if ctx.worker else None
        poller = ctx.poller.status() if ctx.poller else None
        t = translator_for(request, deps.settings_now(conn))
        return {
            "career": load_career(conn),
            "last_sync": queries.last_sync(conn),
            "worker": worker,
            "problems": sum(1 for line in worker.log if line.level != "info") if worker else 0,
            "poller": poller,
            "watcher": watcher_line(poller, t) if poller else None,
        }

    @router.get("/log", response_class=HTMLResponse)
    def log(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        """The technical side: the worker's log and the last sync in full (Phase 12)."""
        rev = live.snapshot(ctx, conn, deps.live_extra())["log"]["rev"]
        return page(
            request,
            "log.html",
            conn,
            rev=rev,
            can_open_folder=ctx.open_folder is not None,
            **log_context(request, conn),
        )

    @router.get("/log/part", response_class=HTMLResponse)
    def log_part(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return presentation.fragment(request, "_log.html", conn, **log_context(request, conn))

    def report_facts() -> report.ReportFacts:
        conn = deps.connect()
        try:
            settings = deps.settings_now(conn)
            career = load_career(conn)
            last_sync = queries.last_sync(conn)
        finally:
            conn.close()
        card = deps.card.known()
        if card is None and deps.card.available:
            card = deps.card.wait(CARD_WAIT_SECONDS)
        offer = deps.update_offer()
        install = ""
        if offer is not None:
            install = "git checkout" if offer["status"].checkout else "installed"
        return report.ReportFacts(
            made_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            version=str(kitshot.__version__),
            install=install,
            system=platform_info.describe(),
            python=sys.version.split()[0],
            card=card,
            card_asked=deps.card.available,
            settings=settings,
            models_ready=ctx.models(settings.style.model).ready if ctx.models else None,
            career=career.display_name if career else None,
            worker=ctx.worker.status() if ctx.worker else None,
            poller=ctx.poller.status() if ctx.poller else None,
            watching=ctx.poller is not None,
            last_sync=last_sync,
            sync=ctx.syncer.status() if ctx.syncer else None,
            homes=[str(Path.home())],
        )

    @router.get("/log/report", response_class=PlainTextResponse)
    async def log_report() -> Response:
        """The report Copy report puts on the clipboard (D11): plain text, no pictures, the
        home folder hidden."""
        text = report.build(await run_in_threadpool(report_facts))
        return PlainTextResponse(text, headers={"Cache-Control": "no-store"})

    def folder_path(name: str) -> Path | None:
        conn = deps.connect()
        try:
            settings = deps.settings_now(conn)
        finally:
            conn.close()
        fm = settings.paths.fm_user_dir
        match name:
            case "logs":
                return settings.paths.data_dir / "logs"
            case "data":
                return settings.paths.data_dir
            case "fm":
                return fm
            case _:
                return packs_root(fm) if fm is not None else None

    @router.post("/folders/open")
    async def open_folder(request: Request) -> Response:
        """Show one of Kitshot's folders in Explorer, named in the form (FOLDERS); Windows
        only. A script asking for JSON gets {ok, error}; a plain form goes back."""
        form = await request.form()
        name = str(form.get("name") or "")
        if name not in FOLDERS:
            raise HTTPException(404)
        back = _safe_next(str(form.get("next") or ""), "/log")
        wants_json = "application/json" in request.headers.get("accept", "")
        t = await run_in_threadpool(deps.translator, request)
        opener = ctx.open_folder

        def answer(error: str | None) -> Response:
            if wants_json:
                return JSONResponse({"ok": error is None, "error": error})
            return _redirect(back, err=error) if error else _redirect(back)

        if opener is None:
            return answer(t("msg.folder_open.unavailable"))
        path = await run_in_threadpool(folder_path, name)
        if path is None or not await run_in_threadpool(path.is_dir):
            return answer(t("msg.folder_open.missing"))
        try:
            await run_in_threadpool(opener, path)
        except OSError:
            return answer(t("msg.folder_open.failed"))
        return answer(None)

    return router
