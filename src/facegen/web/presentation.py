"""Templates, shared page context, display helpers and package resource paths."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from facegen.core.planner import reference_scale
from facegen.core.settings_values import Settings
from facegen.core.traits import TRAIT_TABLES
from facegen.i18n import GROUPS, Translator
from facegen.persistence.image_store import reference_problem
from facegen.persistence.jobs import load_career
from facegen.setup.models import REPOS, gigabytes
from facegen.web import live
from facegen.web.dependencies import AppDependencies, translator_for
from facegen.web.messages import (
    band_name,
    clock,
    drawing_key,
    duration,
    side_line,
    why_chips,
    why_lines,
    why_tracking,
)

HERE = Path(__file__).resolve().parent

JS_TEXTS = (
    "js.saving",
    "js.no_answer",
    "js.unsaved_confirm",
    "js.ticked",
    "js.done_at",
    "js.queued",
    "js.quick_no_answer",
    "js.quick_reload",
    "js.copied",
    "gallery.made",
    "dur.seconds",
    "dur.minutes_seconds",
    "dur.hours_minutes",
)
"""Catalog texts the page's script uses, handed to it in the page."""


def static_version(folder: Path) -> str:
    """A short hash of the static files. Pages link them with it, so a browser fetches the
    script and styles again after an update instead of using the ones it cached."""
    digest = hashlib.sha1()
    for path in sorted(folder.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def uses_reference(
    kind: str, at_my_club: bool, settings: Settings, current: Mapping[str, Any] | None
) -> bool:
    """Whether the job can use the current master as a reference."""
    return (
        reference_scale(kind, at_my_club, settings.reference) is not None
        and reference_problem(settings, current) is None
    )


def group_chip(group: str, tr: Translator) -> Markup:
    """A picture group as a coloured chip with its name: 정밀, 표준, 간편 (Detailed, Standard,
    Light), so the name reads as facegen's group, not the word. Hovering it says what the group
    is (data-tip, shown by app.js)."""
    group = (group or "").lower()
    if group not in GROUPS:
        return Markup('<span class="grp">-</span>')
    return Markup('<span class="grp grp-{0}" data-tip="{1}">{2}</span>').format(
        group, tr(f"term.{group}"), tr(f"group.{group}")
    )


class HtmlText:
    """Catalog texts for templates. The catalog is ours, so its own markup (a link, a code
    span) stands; every value filled in is escaped. {a}, {b} and {c} become group chips."""

    __slots__ = ("chips", "tr")

    def __init__(self, tr: Translator) -> None:
        self.tr = tr
        self.chips = {g: group_chip(g, tr) for g in GROUPS}

    def __call__(self, key: str, /, **values: Any) -> Markup:
        template = Markup(self.tr.template(key, values.get("count")))
        return template.format(**{**self.tr.values, **self.chips, **values})

    @property
    def lang(self) -> str:
        return self.tr.lang


def text_context(tr: Translator, settings: Settings) -> dict[str, Any]:
    """What every template needs to speak the request's language and wear its theme."""
    return {
        "t": HtmlText(tr),
        "tr": tr,
        "lang": tr.lang,
        "display": settings.display,
        "js_texts": {key: tr.template(key) for key in JS_TEXTS},
        "dur": lambda seconds: duration(seconds, tr),
        "clock": lambda seconds: clock(seconds, tr),
        # The sentence for the picture being drawn, by its job kind (Phase 17e).
        "drawing_key": lambda kind: drawing_key(kind, tr, "dash.worker.drawing"),
        "grp": lambda group: group_chip(group, tr),
        "band": lambda wording: band_name(wording, tr),
        # A player's reasons in the request's language (Phase 17h).
        "why_chips": lambda row: why_chips(row, tr),
        "why_lines": lambda row: why_lines(row, tr),
        "why_tracking": lambda row: why_tracking(row, tr),
    }


def create_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env
    env.globals.update(
        gigabytes=gigabytes,
        model_repos=REPOS,
        trait_tables=TRAIT_TABLES,
        uses_reference=uses_reference,
        static_version=static_version(HERE / "static"),
    )
    env.filters["upper_group"] = lambda g: (g or "-").upper()
    env.filters["picture_filename"] = lambda path: Path(path).name
    env.filters["num"] = lambda n: f"{n:,}" if isinstance(n, int) else n
    return templates


class Presentation:
    """Templates and context bound to the same app as the request dependencies."""

    def __init__(self, dependencies: AppDependencies, templates: Jinja2Templates) -> None:
        self.dependencies = dependencies
        self.templates = templates

    def texts(self, request: Request, settings: Settings) -> dict[str, Any]:
        return text_context(translator_for(request, settings), settings)

    def fragment(
        self, request: Request, name: str, conn: sqlite3.Connection, **context: Any
    ) -> Response:
        """A part of a page (the dashboard's status), in the request's language."""
        settings = context.get("settings") or self.dependencies.settings_now(conn)
        return self.templates.TemplateResponse(
            request, name, {**self.texts(request, settings), **context}
        )

    def page(
        self, request: Request, name: str, conn: sqlite3.Connection, **context: Any
    ) -> Response:
        ctx = self.dependencies.ctx
        career = context["career"] if "career" in context else load_career(conn)
        settings = context.get("settings") or self.dependencies.settings_now(conn)
        texts = self.texts(request, settings)
        # The sidebar (Phase 17d). The dashboard hands in the sync state it shows, so a page
        # reads it once.
        worker = ctx.worker.status() if ctx.worker else None
        if "sync" in context:
            sync = context["sync"]
        else:
            sync = ctx.syncer.status() if ctx.syncer else None
        side = live.sidebar(conn, career, worker, sync)
        return self.templates.TemplateResponse(
            request,
            name,
            {
                **texts,
                "career": career,
                "game_date": career.game_date if career else None,
                "side": side,
                "side_text": side_line(side, texts["tr"]),
                "update": (
                    context["update"] if "update" in context else self.dependencies.update_offer()
                ),
                "msg": request.query_params.get("msg"),
                "err": request.query_params.get("err"),
                "path": request.url.path,
                "here": _here(request),
                "setup": context["setup"] if "setup" in context else ctx.setup(settings),
                "folder_text": request.query_params.get("folder"),
                "folder_now": settings.paths.fm_user_dir,
                **context,
            },
        )


def _here(request: Request) -> str:
    """This page's local address, to come back to after a form."""
    return str(request.url.path) + ("?" + request.url.query if request.url.query else "")
