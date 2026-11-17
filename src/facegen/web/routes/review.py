"""Review routes and their app-local helpers."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from facegen.pipeline.worker import load_career
from facegen.web import queries
from facegen.web.dependencies import AppDependencies
from facegen.web.messages import review_problem
from facegen.web.presentation import Presentation


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    Conn = deps.connection
    page = presentation.page

    @router.get("/review", response_class=HTMLResponse)
    def review(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        career = load_career(conn)
        items = queries.review_items(conn, career.id) if career else []
        t = deps.translator(request, conn)
        for it in items:
            it["problem"] = review_problem(f"review-{it['unique_id']}", it.get("error"), t)
        return page(request, "review.html", conn, items=items)

    return router
