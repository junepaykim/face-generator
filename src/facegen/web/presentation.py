"""Templates, shared page context, display helpers and package resource paths."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from facegen.core.planner import reference_scale
from facegen.core.settings_values import Settings
from facegen.core.traits import TRAIT_TABLES
from facegen.persistence.settings import local_label
from facegen.pipeline.publish import RELOAD_HINT
from facegen.pipeline.worker import Career, load_career
from facegen.setup.models import REPOS, gigabytes
from facegen.web import queries
from facegen.web.dependencies import AppDependencies

HERE = Path(__file__).resolve().parent

TIER_NAMES = {0: "manual", 1: "A, my club", 2: "A", 3: "B", 4: "C"}
KIND_NAMES = {
    "initial": "first picture",
    "kit": "new kit",
    "age": "new age band",
    "upgrade": "upgrade to A",
    "reroll": "new face",
    "redraw": "new picture",
}


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


def create_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env
    env.globals.update(
        gigabytes=gigabytes,
        model_repos=REPOS,
        tier_name=lambda t: TIER_NAMES.get(t, str(t)),
        kind_name=describe_job,
        reload_hint=RELOAD_HINT,
        trait_tables=TRAIT_TABLES,
        uses_reference=uses_reference,
        static_version=static_version(HERE / "static"),
    )
    env.filters["upper_group"] = lambda g: (g or "-").upper()
    return templates


class Presentation:
    """Templates and context bound to the same app as the request dependencies."""

    def __init__(self, dependencies: AppDependencies, templates: Jinja2Templates) -> None:
        self.dependencies = dependencies
        self.templates = templates

    def page(
        self, request: Request, name: str, conn: sqlite3.Connection, **context: Any
    ) -> Response:
        ctx = self.dependencies.ctx
        career = load_career(conn)
        settings = self.dependencies.settings_now(conn)
        return self.templates.TemplateResponse(
            request,
            name,
            {
                "career": career,
                "game_date": _game_date(conn, career),
                "review_count": queries.review_count(conn, career.id) if career else 0,
                "msg": request.query_params.get("msg"),
                "err": request.query_params.get("err"),
                "path": request.url.path,
                "setup": ctx.setup(settings),
                "folder_text": request.query_params.get("folder"),
                "folder_now": settings.paths.fm_user_dir,
                "folder_origin": settings.paths.describe("paths.fm_user_dir"),
                "local_file": local_label(ctx.defaults_path),
                **context,
            },
        )


def _game_date(conn: sqlite3.Connection, career: Career | None) -> str | None:
    if career is None:
        return None
    row = conn.execute("SELECT last_game_date FROM careers WHERE id = ?", (career.id,)).fetchone()
    return row[0] if row else None
