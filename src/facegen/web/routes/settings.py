"""Settings routes and their app-local helpers."""

from __future__ import annotations

import sqlite3
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from facegen.persistence.settings import SettingsError, folder_not_set_message, load_defaults
from facegen.pipeline import settings_history
from facegen.pipeline.settings_history import Outcome
from facegen.setup import hardware
from facegen.web.dependencies import AppDependencies
from facegen.web.forms import (
    blamed_keys,
    change_lines,
    describe_key,
    form_of,
    input_text,
    parse_settings_changes,
    parse_settings_form,
    settings_sections,
)
from facegen.web.presentation import Presentation
from facegen.web.responses import _redirect


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    ctx = deps.ctx
    settings_now = deps.settings_now
    Conn = deps.connection
    page = presentation.page
    wake = deps.wake
    templates = presentation.templates

    # Each change is saved as it is made (the page's script posts to /settings/save) and can
    # be undone; settings_history.py keeps the history. Without the script, the page's Save
    # button posts the whole form.

    def history_context(conn: sqlite3.Connection) -> dict[str, Any]:
        defaults = load_defaults(ctx.defaults_path)
        entries = settings_history.entries(conn)
        return {
            "history": [
                {
                    "entry": e,
                    "lines": change_lines(e.before, e.after, defaults),
                    "summary": e.report.summary() if e.report else "",
                }
                for e in entries
            ],
            "latest_done": next((e.id for e in entries if not e.undone), None),
            "can_undo": settings_history.can_undo(conn),
            "can_redo": settings_history.can_redo(conn),
        }

    def outcome_message(verb: str, outcome: Outcome) -> str:
        """ "Saved: Group A · CA above 155 → 150. A 52 → 60; queue: 8 queued." and the like."""
        defaults = load_defaults(ctx.defaults_path)
        what = "; ".join(
            f"{line.label} {line.before} → {line.after}"
            if verb != "Undone"
            else f"{line.label} {line.after} → {line.before}"
            for e in outcome.entries
            for line in change_lines(e.before, e.after, defaults)
        )
        if any(e.label == "defaults" for e in outcome.entries):
            what = f"every setting back to its default ({what})"
        note = f"{verb}: {what}."
        if outcome.report is not None:
            summary = outcome.report.summary()
            note += f" {summary[:1].upper()}{summary[1:]}."
        if any(k.startswith("eligibility.") for e in outcome.entries for k in e.after):
            note += " Eligibility changes take effect at the next sync."
        return note

    def settings_page(
        request: Request,
        conn: sqlite3.Connection,
        values: dict[str, Any] | None = None,
        *,
        errors: dict[str, str] | None = None,
        error: str | None = None,
    ) -> Response:
        defaults = load_defaults(ctx.defaults_path)
        current = settings_now(conn)
        shown = dict(current.values) | (values or {})
        folders = {
            "user": current.paths.describe("paths.fm_user_dir"),
            "install": current.paths.describe("paths.fm_install_dir"),
            "user_missing": current.paths.fm_user_dir is None,
            "not_set": folder_not_set_message(),
            "value": request.query_params.get("folder") or current.paths.fm_user_dir or "",
        }
        return page(
            request,
            "settings.html",
            conn,
            folders=folders,
            sections=settings_sections(shown, defaults),
            errors=errors or {},
            error=error,
            **history_context(conn),
        )

    @router.get("/settings", response_class=HTMLResponse)
    def settings_get(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return settings_page(request, conn)

    card_lock = threading.Lock()
    card: list[hardware.Hardware] = []

    def graphics_card(again: bool = False) -> hardware.Hardware | None:
        """The card, asked once per app (a child process imports torch for it)."""
        if ctx.hardware is None:
            return None
        with card_lock:
            if again or not card:
                card[:] = [ctx.hardware()]
            return card[0]

    @router.get("/hardware", response_class=HTMLResponse)
    async def hardware_page(request: Request) -> Response:
        found = await run_in_threadpool(graphics_card, request.query_params.get("again") == "1")

        def render() -> Response:
            conn = deps.connect()
            try:
                current = settings_now(conn)
                defaults = load_defaults(ctx.defaults_path)
                preset = hardware.recommend(found.vram_gib if found else None)
                lines = [
                    {
                        "label": describe_key(c.key),
                        "before": input_text(c.before),
                        "after": input_text(c.after),
                        "default": input_text(defaults[c.key]),
                    }
                    for c in hardware.changes(preset, current.values)
                ]
                return page(request, "hardware.html", conn, card=found, preset=preset, lines=lines)
            finally:
                conn.close()

        return await run_in_threadpool(render)

    @router.post("/settings/auto-configure")
    async def auto_configure(request: Request) -> Response:
        """The preset for this PC's card, as one settings change that one Undo reverts. Only
        picture settings: never the group thresholds (user, 2026-11-13)."""
        found = await run_in_threadpool(graphics_card)

        def run() -> Response:
            if found is None or not found.has_gpu:
                why = found.reason if found else "the card cannot be asked in this process"
                return _redirect(
                    "/hardware", err=f"Nothing changed: no CUDA graphics card ({why})."
                )
            preset = hardware.recommend(found.vram_gib)
            conn = deps.connect()
            try:
                outcome = settings_history.save(
                    conn, preset.changes, label="auto-configure", defaults_path=ctx.defaults_path
                )
            except SettingsError as exc:
                return _redirect("/hardware", err=f"Not done: {exc}")
            finally:
                conn.close()
            if not outcome.changed:
                return _redirect(
                    "/settings",
                    msg=f"Nothing to change: the settings already suit the {found.name}.",
                )
            wake()
            return _redirect(
                "/settings",
                msg=f"Auto-configured for the {found.name} ({preset.title}). "
                + outcome_message("Saved", outcome)
                + " Pictures already drawn are not drawn again. Undo puts it all back.",
            )

        return await run_in_threadpool(run)

    @router.post("/settings", response_class=HTMLResponse)
    async def settings_post(request: Request) -> Response:
        """The whole form (Save, without the page's script), Reset, Undo and Redo."""
        form = await form_of(request)
        action = form.get("action")

        def run() -> Response:
            conn = deps.connect()
            try:
                defaults = load_defaults(ctx.defaults_path)
                values: dict[str, Any] = {}
                try:
                    match action:
                        case "apply":
                            values, errors = parse_settings_form(form, defaults)
                            if errors:
                                return settings_page(request, conn, values, errors=errors)
                            verb = "Saved"
                            outcome = settings_history.save(
                                conn, values, defaults_path=ctx.defaults_path
                            )
                        case "defaults":
                            verb = "Saved"
                            outcome = settings_history.save_defaults(
                                conn, defaults_path=ctx.defaults_path
                            )
                        case "undo":
                            through = str(form.get("through") or "")
                            verb = "Undone"
                            outcome = settings_history.undo(
                                conn,
                                through=int(through) if through.isdigit() else None,
                                defaults_path=ctx.defaults_path,
                            )
                        case "redo":
                            verb = "Redone"
                            outcome = settings_history.redo(conn, defaults_path=ctx.defaults_path)
                        case _:
                            raise HTTPException(400, "unknown action")
                except SettingsError as exc:
                    if action == "apply":
                        return settings_page(request, conn, values, error=str(exc))
                    return _redirect("/settings", err=f"Not done: {exc}")
                if not outcome.changed:
                    nothing = {"undo": "Nothing to undo.", "redo": "Nothing to redo."}
                    return _redirect("/settings", msg=nothing.get(str(action), "Nothing changed."))
                wake()
                return _redirect("/settings", msg=outcome_message(verb, outcome))
            finally:
                conn.close()

        return await run_in_threadpool(run)

    @router.post("/settings/save")
    async def settings_save(request: Request) -> Response:
        """Autosave: the fields the page changed. Answers JSON for the page's script."""
        form = await form_of(request)

        def run() -> Response:
            conn = deps.connect()
            try:
                defaults = load_defaults(ctx.defaults_path)
                values, errors = parse_settings_changes(form, defaults)
                if errors:
                    return JSONResponse(
                        {"ok": False, "errors": errors, "error": "Not saved: check the field."}
                    )
                try:
                    outcome = settings_history.save(conn, values, defaults_path=ctx.defaults_path)
                except SettingsError as exc:
                    message = str(exc)
                    blamed = blamed_keys(message, list(values)) or list(values)
                    return JSONResponse(
                        {
                            "ok": False,
                            "errors": {k: message for k in blamed},
                            "error": f"Not saved: {message}",
                        }
                    )
                if outcome.changed:
                    wake()
                # The values as stored (lists lower-cased, numbers tidied), for the page to show.
                current = outcome.settings.values
                fields = {
                    key: {
                        "value": current[key],
                        "text": input_text(current[key]),
                        "changed": current[key] != defaults[key],
                    }
                    for key in values
                }
                history = history_context(conn)
                history_html = templates.get_template("_settings_history.html").render(**history)
                return JSONResponse(
                    {
                        "ok": True,
                        "changed": outcome.changed,
                        "message": outcome_message("Saved", outcome)
                        if outcome.changed
                        else "Nothing changed.",
                        "fields": fields,
                        "history_html": history_html,
                        "can_undo": history["can_undo"],
                        "can_redo": history["can_redo"],
                    }
                )
            finally:
                conn.close()

        return await run_in_threadpool(run)

    return router
