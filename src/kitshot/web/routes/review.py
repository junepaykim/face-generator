"""Review routes and their app-local helpers."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from kitshot.persistence.jobs import load_career
from kitshot.web import queries
from kitshot.web.dependencies import AppDependencies
from kitshot.web.messages import check_failure_text, review_problem
from kitshot.web.presentation import Presentation


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
            for c in it["image"]["check"].get("candidates", []) if it["image"] else []:
                c["why"] = [check_failure_text(f, t) for f in c.get("failures", [])]
        return page(request, "review.html", conn, career=career, items=items)

    return router
