"""The picture wall (Phase 12h): everyone's current picture, newest first, with filters."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from facegen.persistence.jobs import load_career
from facegen.web import live, queries
from facegen.web.dependencies import AppDependencies
from facegen.web.presentation import Presentation
from facegen.web.queries import GalleryFilter


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    Conn = deps.connection

    def wall(request: Request, conn: sqlite3.Connection) -> dict[str, Any]:
        career = load_career(conn)
        f = GalleryFilter.from_query(request.query_params)
        rows, total = queries.gallery_page(conn, career.id, f) if career else ([], 0)
        return {
            "f": f,
            "rows": rows,
            "total": total,
            "pages": max(1, -(-total // queries.GALLERY_PER_PAGE)),
            "has_career": career is not None,
            "career": career,
        }

    @router.get("/gallery", response_class=HTMLResponse)
    def gallery(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        rev = live.snapshot(deps.ctx, conn, deps.live_extra())["pictures"]["rev"]
        return presentation.page(
            request,
            "gallery.html",
            conn,
            rev=rev,
            wheres=queries.WHERE_FILTERS,
            groups=queries.GALLERY_GROUPS,
            roles=queries.ROLES,
            sorts=queries.GALLERY_SORTS,
            **wall(request, conn),
        )

    @router.get("/gallery/part", response_class=HTMLResponse)
    def gallery_part(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        """The wall without the page around it, for live updates."""
        return presentation.fragment(request, "_gallery.html", conn, **wall(request, conn))

    return router
