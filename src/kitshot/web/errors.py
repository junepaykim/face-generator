"""Error pages (Phase 12e): what happened and a way back, instead of bare text or JSON.

Only for a browser asking for HTML; other clients keep FastAPI's answers. The error's own
text is folded away, and an unexpected one goes to the log with its traceback.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from kitshot.core.settings_values import Settings
from kitshot.persistence.settings import load_settings
from kitshot.runtime.logging import logger
from kitshot.web.dependencies import AppDependencies
from kitshot.web.presentation import Presentation


def _wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


def install_error_pages(app: FastAPI, deps: AppDependencies, presentation: Presentation) -> None:
    def settings() -> Settings:
        try:
            conn = deps.connect()
            try:
                return deps.settings_now(conn)
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 - the page still shows, with the defaults
            return load_settings(defaults_path=deps.ctx.defaults_path)

    async def page(request: Request, status: int, **context: Any) -> Response:
        texts = presentation.texts(request, await run_in_threadpool(settings))
        return presentation.templates.TemplateResponse(
            request, "error.html", {**texts, "status": status, **context}, status_code=status
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        if not _wants_html(request):
            return await http_exception_handler(request, exc)
        found = exc.status_code != 404
        return await page(
            request,
            exc.status_code,
            title_key="error_page.server" if found and exc.status_code >= 500 else None,
            text=str(exc.detail) if found else None,
            detail=f"{exc.status_code}: {exc.detail}",
        )

    @app.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception) -> Response:
        logger.error("unexpected error on %s", request.url.path, exc_info=exc)
        if not _wants_html(request):
            return PlainTextResponse("Internal Server Error", status_code=500)
        return await page(
            request,
            500,
            title_key="error_page.server",
            text=None,
            detail=f"{type(exc).__name__}: {exc}",
        )
