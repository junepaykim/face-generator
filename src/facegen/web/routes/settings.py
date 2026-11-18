"""Settings routes and their app-local helpers."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from facegen.core import presets
from facegen.datasource import photos
from facegen.i18n import Translator
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
    value_text,
)
from facegen.web.messages import report_summary, settings_error_text
from facegen.web.presentation import Presentation
from facegen.web.responses import _redirect, _safe_next

ADVANCED = "/settings/advanced"


def preset_title(kind: str, name: str, t: Translator) -> str:
    """ "Players at other clubs: More", "Picture quality: Best"."""
    return t(
        "settings.basic.preset",
        kind=t(f"settings.basic.{kind}"),
        option=t(f"settings.basic.{kind}_option.{name}"),
    )


def entry_title(label: str, t: Translator) -> str:
    """What a history entry was, when it was more than an edit; empty for an edit."""
    if label == "defaults":
        return t("history.defaults")
    if label == "auto-configure":
        return t("history.auto_configure")
    preset = presets.parse_label(label)
    if preset is not None:
        return preset_title(*preset, t)
    return ""


def posted_preset(
    form: Any, defaults: dict[str, Any]
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """The presets a form chose ("preset.reach" = "more"): their values, their history labels
    and errors by field name. An unknown kind is ignored; an unknown name is an error."""
    values: dict[str, Any] = {}
    labels: list[str] = []
    errors: dict[str, str] = {}
    for kind in presets.KINDS:
        name = form.get(f"preset.{kind}")
        if not isinstance(name, str) or not name:
            continue
        try:
            values |= presets.values_of(kind, name, defaults)
        except KeyError:
            errors[f"preset.{kind}"] = name
            continue
        labels.append(presets.label(kind, name))
    return values, labels, errors


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

    def history_context(conn: sqlite3.Connection, t: Translator) -> dict[str, Any]:
        defaults = load_defaults(ctx.defaults_path)
        entries = settings_history.entries(conn)
        return {
            "history": [
                {
                    "entry": e,
                    "title": entry_title(e.label, t),
                    "lines": change_lines(e.before, e.after, defaults, t),
                    "summary": report_summary(e.report, t) if e.report else "",
                }
                for e in entries
            ],
            "latest_done": next((e.id for e in entries if not e.undone), None),
            "can_undo": settings_history.can_undo(conn),
            "can_redo": settings_history.can_redo(conn),
        }

    def outcome_message(verb: str, outcome: Outcome, t: Translator) -> str:
        """ "Saved: Group A · CA above 155 → 150. A 52 → 60; queue: 8 queued." and the like.
        `verb` is saved, undone or redone. A preset is named, not listed key by key."""
        defaults = load_defaults(ctx.defaults_path)
        parts: list[str] = []
        for e in outcome.entries:
            if presets.parse_label(e.label):
                parts.append(entry_title(e.label, t))
                continue
            parts += [
                f"{line.label} {line.before} → {line.after}"
                if verb != "undone"
                else f"{line.label} {line.after} → {line.before}"
                for line in change_lines(e.before, e.after, defaults, t)
            ]
        what = "; ".join(parts)
        if any(e.label == "defaults" for e in outcome.entries):
            what = t("msg.settings.all_defaults", what=what)
        note = t(f"msg.settings.{verb}", what=what)
        if outcome.report is not None:
            summary = report_summary(outcome.report, t)
            note += f" {summary[:1].upper()}{summary[1:]}."
        if any(k.startswith("eligibility.") for e in outcome.entries for k in e.after):
            note += " " + t("msg.settings.eligibility")
        return note

    def folders_of(request: Request, current: Any) -> dict[str, Any]:
        return {
            "user": current.paths.describe("paths.fm_user_dir"),
            "install": current.paths.describe("paths.fm_install_dir"),
            "user_missing": current.paths.fm_user_dir is None,
            "not_set": folder_not_set_message(),
            "value": request.query_params.get("folder") or current.paths.fm_user_dir or "",
        }

    def basic_page(request: Request, conn: sqlite3.Connection) -> Response:
        """Who gets a picture, the quality preset and when to draw, on one screen (Phase 12g)."""
        defaults = load_defaults(ctx.defaults_path)
        current = settings_now(conn)
        deps.card.start()  # the recommended quality, once the card has answered
        card = deps.card.known()
        recommended = (
            hardware.recommend(card.vram_gib).name if card is not None and card.has_gpu else None
        )
        values = current.values
        return page(
            request,
            "settings_basic.html",
            conn,
            folders=folders_of(request, current),
            reach=presets.current("reach", values, defaults),
            reach_options=list(presets.REACH_STEPS),
            quality=presets.current("quality", values, defaults),
            quality_options=list(presets.QUALITY),
            recommended=recommended,
            card=card,
            values=values,
            can_undo=settings_history.can_undo(conn),
            can_redo=settings_history.can_redo(conn),
        )

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
        t = deps.translator(request, conn)
        shown = dict(current.values) | (values or {})
        return page(
            request,
            "settings.html",
            conn,
            info={
                # Moved here from the dashboard (Phase 12): about this run, not the pictures.
                "worker": ctx.worker.status() if ctx.worker else None,
                "watching": ctx.poller is not None,
                "photos": photos.status(conn),
            },
            folders=folders_of(request, current),
            sections=settings_sections(shown, defaults, t),
            errors=errors or {},
            error=error,
            **history_context(conn, t),
        )

    @router.get("/settings", response_class=HTMLResponse)
    def settings_get(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return basic_page(request, conn)

    @router.get("/settings/advanced", response_class=HTMLResponse)
    def settings_advanced(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        return settings_page(request, conn)

    graphics_card = deps.card.get  # asked once per app, shared with the dashboard's guide

    @router.get("/hardware", response_class=HTMLResponse)
    async def hardware_page(request: Request) -> Response:
        found = await run_in_threadpool(graphics_card, request.query_params.get("again") == "1")

        def render() -> Response:
            conn = deps.connect()
            try:
                current = settings_now(conn)
                t = deps.translator(request, conn)
                defaults = load_defaults(ctx.defaults_path)
                preset = hardware.recommend(found.vram_gib if found else None)
                lines = [
                    {
                        "label": describe_key(c.key, t),
                        "before": value_text(c.before, t, c.key),
                        "after": value_text(c.after, t, c.key),
                        "default": value_text(defaults[c.key], t, c.key),
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
        picture settings: never the group thresholds (user, 2026-11-13). From the dashboard's
        first-run guide it goes back there (`next`)."""
        form = await form_of(request)
        done = _safe_next(str(form.get("next") or ""), "/settings")
        found = await run_in_threadpool(graphics_card)
        t = await run_in_threadpool(deps.translator, request)

        def run() -> Response:
            if found is None or not found.has_gpu:
                why = found.reason if found else t("hardware.cannot_ask")
                return _redirect("/hardware", err=t("msg.hardware.no_card", why=why))
            preset = hardware.recommend(found.vram_gib)
            conn = deps.connect()
            try:
                outcome = settings_history.save(
                    conn, preset.changes, label="auto-configure", defaults_path=ctx.defaults_path
                )
            except SettingsError as exc:
                said = settings_error_text(str(exc), t)
                return _redirect("/hardware", err=t("msg.settings.not_done", error=said))
            finally:
                conn.close()
            if not outcome.changed:
                return _redirect(done, msg=t("msg.hardware.suits", card=found.name))
            wake()
            return _redirect(
                done,
                msg=t(
                    "msg.hardware.configured",
                    card=found.name,
                    preset=t(f"hardware.preset.{preset.name}.title"),
                    outcome=outcome_message("saved", outcome, t),
                ),
            )

        return await run_in_threadpool(run)

    @router.post("/settings", response_class=HTMLResponse)
    async def settings_post(request: Request) -> Response:
        """The whole form (Save, without the page's script), Reset, Undo and Redo."""
        form = await form_of(request)
        action = form.get("action")
        back = _safe_next(str(form.get("next") or ""), ADVANCED)

        def run() -> Response:
            conn = deps.connect()
            try:
                defaults = load_defaults(ctx.defaults_path)
                t = deps.translator(request, conn)
                values: dict[str, Any] = {}
                try:
                    match action:
                        case "apply":
                            values, errors = parse_settings_form(form, defaults, t)
                            if errors:
                                return settings_page(request, conn, values, errors=errors)
                            verb = "saved"
                            outcome = settings_history.save(
                                conn, values, defaults_path=ctx.defaults_path
                            )
                        case "apply-some":
                            # The Basic page without its script: only the fields it shows.
                            values, errors = parse_settings_changes(form, defaults, t)
                            chosen, labels, bad = posted_preset(form, defaults)
                            if errors or bad:
                                return _redirect(back, err=t("msg.settings.check_field"))
                            verb = "saved"
                            outcome = settings_history.save(
                                conn,
                                chosen | values,
                                label=labels[0] if labels and not values else "edit",
                                defaults_path=ctx.defaults_path,
                            )
                        case "defaults":
                            verb = "saved"
                            outcome = settings_history.save_defaults(
                                conn, defaults_path=ctx.defaults_path
                            )
                        case "undo":
                            through = str(form.get("through") or "")
                            verb = "undone"
                            outcome = settings_history.undo(
                                conn,
                                through=int(through) if through.isdigit() else None,
                                defaults_path=ctx.defaults_path,
                            )
                        case "redo":
                            verb = "redone"
                            outcome = settings_history.redo(conn, defaults_path=ctx.defaults_path)
                        case _:
                            raise HTTPException(400, "unknown action")
                except SettingsError as exc:
                    said = settings_error_text(str(exc), t)
                    if action == "apply":
                        return settings_page(request, conn, values, error=said)
                    return _redirect(back, err=t("msg.settings.not_done", error=said))
                if not outcome.changed:
                    nothing = {"undo": "msg.settings.no_undo", "redo": "msg.settings.no_redo"}
                    key = nothing.get(str(action), "msg.settings.nothing")
                    return _redirect(back, msg=t(key))
                wake()
                return _redirect(back, msg=outcome_message(verb, outcome, t))
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
                t = deps.translator(request, conn)
                values, errors = parse_settings_changes(form, defaults, t)
                chosen, labels, bad = posted_preset(form, defaults)
                errors |= {k: t("msg.settings.check_field") for k in bad}
                if errors:
                    return JSONResponse(
                        {"ok": False, "errors": errors, "error": t("msg.settings.check_field")}
                    )
                # A preset alone is named in the history; with other fields it is an edit.
                label = labels[0] if len(labels) == 1 and not values else "edit"
                values = chosen | values
                try:
                    outcome = settings_history.save(
                        conn, values, label=label, defaults_path=ctx.defaults_path
                    )
                except SettingsError as exc:
                    message = str(exc)
                    blamed = blamed_keys(message, list(values)) or list(values)
                    said = settings_error_text(message, t)
                    return JSONResponse(
                        {
                            "ok": False,
                            "errors": {k: said for k in blamed},
                            "error": t("msg.settings.not_saved", error=said),
                        }
                    )
                # A new language or theme: the page shows it once it loads again. The texts
                # below are already in the language as saved.
                t = deps.translator(request, conn)
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
                chosen_now = {
                    kind: presets.current(kind, current, defaults) for kind in presets.KINDS
                }
                history = history_context(conn, t)
                history_html = templates.get_template("_settings_history.html").render(
                    **presentation.texts(request, outcome.settings), **history
                )
                return JSONResponse(
                    {
                        "ok": True,
                        "changed": outcome.changed,
                        "reload": any(k.startswith("display.") for k in values) and outcome.changed,
                        "message": outcome_message("saved", outcome, t)
                        if outcome.changed
                        else t("msg.settings.nothing"),
                        "fields": fields,
                        "presets": chosen_now,
                        "history_html": history_html,
                        "can_undo": history["can_undo"],
                        "can_redo": history["can_redo"],
                    }
                )
            finally:
                conn.close()

        return await run_in_threadpool(run)

    return router
