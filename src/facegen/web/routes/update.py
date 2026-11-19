"""The update offer's actions (Phase 13): check now, later, and update.

Updating runs only where the desktop app set `AppContext.install_update` (Windows, a folder
that is not a git checkout): it starts `Start.bat --update` in a console of its own and quits
facegen, as the files cannot be replaced while facegen runs.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from facegen.web.dependencies import AppDependencies
from facegen.web.forms import form_of
from facegen.web.presentation import Presentation
from facegen.web.responses import _redirect, _safe_next

ABOUT = "/settings/advanced#about"


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    ctx = deps.ctx

    @router.post("/update/check")
    async def check(request: Request) -> Response:
        """Ask GitHub now, whatever `updates.check` says: the user asked."""
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), ABOUT)
        t = await run_in_threadpool(deps.translator, request)
        if ctx.updates is None:
            return _redirect(back, err=t("msg.update.absent"))
        status = await run_in_threadpool(ctx.updates.check_now)
        if status.error:
            return _redirect(back, err=t("msg.update.failed"))
        if status.available:
            return _redirect(back, msg=t("msg.update.available", version=status.latest))
        return _redirect(back, msg=t("msg.update.latest", version=status.current))

    @router.post("/update/later")
    async def later(request: Request) -> Response:
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), "/")
        if ctx.updates is not None:
            version = form.get("version")
            await run_in_threadpool(ctx.updates.later, str(version) if version else None)
        return _redirect(back)

    @router.post("/update/install", response_class=HTMLResponse)
    async def install(request: Request) -> Response:
        t = await run_in_threadpool(deps.translator, request)
        if ctx.updates is None or ctx.install_update is None:
            return _redirect("/", err=t("msg.update.not_here"))
        status = await run_in_threadpool(ctx.updates.status)
        if not status.available:
            return _redirect("/", msg=t("msg.update.latest", version=status.current))

        def render() -> Response:
            conn: sqlite3.Connection = deps.connect()
            try:
                return presentation.page(request, "updating.html", conn, update=status)
            finally:
                conn.close()

        response = await run_in_threadpool(render)
        assert ctx.install_update is not None
        try:
            # Starts the update's console, then quits facegen a moment later: this page is
            # sent first.
            await run_in_threadpool(ctx.install_update)
        except OSError as exc:
            return _redirect("/", err=t("msg.update.not_started", error=str(exc)))
        return response

    return router
