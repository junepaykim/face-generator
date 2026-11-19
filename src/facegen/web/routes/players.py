"""Players routes and their app-local helpers."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from facegen.core.age import audience, table_name
from facegen.core.classify import Group
from facegen.core.job_records import Career
from facegen.core.planner import REFERENCE_KINDS
from facegen.core.traits import eyes_drawn_round
from facegen.i18n import Translator
from facegen.persistence import photo_index, player_queries
from facegen.persistence.jobs import load_career
from facegen.pipeline import manual
from facegen.pipeline.manual import CareerChanged, ManualError
from facegen.pipeline.publish import PublishError
from facegen.runtime.logging import logger
from facegen.web import queries
from facegen.web.dependencies import AppDependencies, need_career, translator_for
from facegen.web.forms import form_of
from facegen.web.messages import manual_error_text, manual_summary, problem_line, review_problem
from facegen.web.presentation import Presentation, uses_reference
from facegen.web.queries import PlayerPageFilter
from facegen.web.responses import _redirect, _safe_next


def _form_career(conn: sqlite3.Connection, posted_id: object, t: Translator) -> Career:
    career = need_career(conn, t)
    if posted_id != str(career.id):
        raise CareerChanged()
    return career


def create_router(deps: AppDependencies, presentation: Presentation) -> APIRouter:
    router = APIRouter()
    settings_now = deps.settings_now
    data_dir = deps.data_dir
    Conn = deps.connection
    page = presentation.page
    wake = deps.wake

    @router.get("/players", response_class=HTMLResponse)
    def players(request: Request, conn: sqlite3.Connection = Conn) -> Response:
        career = load_career(conn)
        f = PlayerPageFilter.from_query(request.query_params)
        rows, total = player_queries.players_page(conn, career.id, f) if career else ([], 0)
        club_name = None
        if f.club_id is not None and career is not None:
            club_name = player_queries.club_name(conn, career.id, f.club_id)
            if club_name is None:
                club_name = str(f.club_id)
        pages = max(1, -(-total // player_queries.PER_PAGE))
        return page(
            request,
            "players.html",
            conn,
            career=career,
            f=f,
            rows=rows,
            total=total,
            pages=pages,
            club_name=club_name,
            leagues=queries.LEAGUES,
            origins=queries.ORIGINS,
            roles=queries.ROLES,
            group_filters=queries.GROUP_FILTERS,
            picture_filters=queries.PICTURE_FILTERS,
            photo_filters=queries.PHOTO_FILTERS,
            photo_status=photo_index.status(conn),
            scopes=queries.SCOPES,
            sorts=list(queries.SORTS),
            here=str(request.url.path) + ("?" + request.url.query if request.url.query else ""),
        )

    @router.post("/players/action")
    async def players_action(request: Request) -> Response:
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), "/players")
        t = await run_in_threadpool(deps.translator, request)
        try:
            uids = [int(x) for x in form.getlist("uid")]
        except ValueError:
            return _redirect(back, err=t("msg.player.bad_id"))
        if not uids:
            return _redirect(back, err=t("msg.player.tick_first"))
        action = form.get("action")

        def run() -> str:
            conn = deps.connect()
            try:
                career = _form_career(conn, form.get("career_id"), t)
                settings = settings_now(conn)
                if action == "generate":
                    group = str(form.get("group") or manual.DEFAULT_GROUP)
                    r = manual.generate(conn, career.id, uids, settings, group, active_only=True)
                    return t(
                        "msg.bulk.generate", group=t(f"group.{group}"), summary=manual_summary(r, t)
                    )
                if action == "new-picture":
                    r = manual.new_pictures(conn, career.id, uids, settings, active_only=True)
                    return t("msg.bulk.new_picture", summary=manual_summary(r, t))
                if action == "retry":  # the review page (Phase 12i)
                    r = manual.retry_many(conn, career.id, uids, settings, active_only=True)
                    return t("msg.bulk.retry", summary=manual_summary(r, t))
                if action == "new-face":
                    r = manual.new_faces(conn, career.id, uids, settings, active_only=True)
                    return t("msg.bulk.new_face", summary=manual_summary(r, t))
                if action == "pin":
                    pin = str(form.get("pin") or "")
                    r = manual.set_pin(
                        conn, career.id, uids, pin or None, settings, active_only=True
                    )
                    if pin in ("a", "b", "c"):
                        return t(
                            "msg.bulk.pin", group=t(f"group.{pin}"), summary=manual_summary(r, t)
                        )
                    what = "msg.bulk.pinned_never" if pin == "never" else "msg.bulk.unpinned"
                    return t(what, count=len(r.pinned))
                raise ManualError(f"unknown action {action!r}")
            finally:
                conn.close()

        try:
            message = await run_in_threadpool(run)
        except ManualError as exc:
            return _redirect(back, err=manual_error_text(exc, t))
        except HTTPException as exc:
            return _redirect(back, err=str(exc.detail))
        wake()
        return _redirect(back, msg=message)

    @router.get("/players/{uid}", response_class=HTMLResponse)
    def player(uid: int, request: Request, conn: sqlite3.Connection = Conn) -> Response:
        settings = settings_now(conn)
        t = translator_for(request, settings)
        career = need_career(conn, t)
        detail = queries.player_detail(conn, career.id, uid, data_dir)
        if detail is None:
            raise HTTPException(404, t("msg.player.none", uid=uid))
        for job in detail["jobs"]:  # an error folded under what happened (Phase 12e)
            job["problem"] = review_problem(f"job-{job['id']}", job.get("error"), t)
        p = detail["player"]
        group_for_redraw = p["group_name"] or (
            detail["current"]["group_name"] if detail["current"] else None
        )
        return page(
            request,
            "player.html",
            conn,
            career=career,
            d=detail,
            p=p,
            settings=settings,
            age_table=table_name(p["group_name"], bool(p["at_my_club"]), **audience(p)),
            redraw_reference=uses_reference("redraw", bool(p["at_my_club"]), settings),
            eyes_round=bool(detail["sheet"])
            and eyes_drawn_round(
                detail["sheet"]["fields"]["eye_shape"],
                detail["sheet"]["fields"]["base_seed"],
                settings.my_club.round_eyes,
            ),
            group_for_redraw=group_for_redraw,
            photo_status=photo_index.status(conn),
            reference_kinds=REFERENCE_KINDS,
            groups=[g.value for g in Group],
        )

    @router.post("/players/{uid}/{action}")
    async def player_action(uid: int, action: str, request: Request) -> Response:
        form = await form_of(request)
        back = f"/players/{uid}"
        t = await run_in_threadpool(deps.translator, request)

        def run() -> str:
            conn = deps.connect()
            try:
                career = _form_career(conn, form.get("career_id"), t)
                settings = settings_now(conn)
                match action:
                    case "generate":
                        group = str(form.get("group") or manual.DEFAULT_GROUP)
                        r = manual.generate(
                            conn, career.id, [uid], settings, group, active_only=True
                        )
                        if r.skipped:
                            raise ManualError(r.skipped[0][1])
                        if r.pinned:
                            return t("msg.player.pinned_good", group=t(f"group.{group}"))
                        return t("msg.player.queued_as", group=t(f"group.{group}"))
                    case "new-face":
                        manual.new_face(conn, career.id, uid, settings, active_only=True)
                        return t("msg.player.new_face")
                    case "new-picture":
                        manual.new_picture(conn, career.id, uid, settings, active_only=True)
                        return t("msg.player.new_picture")
                    case "pin":
                        pin = str(form.get("pin") or "")
                        r = manual.set_pin(
                            conn, career.id, [uid], pin or None, settings, active_only=True
                        )
                        if r.skipped:
                            raise ManualError(r.skipped[0][1])
                        if r.queued:
                            return t("msg.player.pinned_queued", group=t(f"group.{pin}"))
                        if pin in ("a", "b", "c"):
                            return t("msg.player.pinned_good", group=t(f"group.{pin}"))
                        if pin == "never":
                            return t("msg.player.pinned_never")
                        return t("msg.player.pin_cleared")
                    case "remove":
                        removed = manual.remove_picture(
                            conn, career.id, career.key, uid, settings, active_only=True
                        )
                        return (
                            t("msg.player.removed", reload=t("msg.reload_hint"))
                            if removed
                            else t("msg.player.nothing_published")
                        )
                    case "retry":
                        manual.retry(conn, career.id, uid, settings, active_only=True)
                        return t("msg.player.retry")
                    case "restore":
                        try:
                            image_id = int(str(form.get("image_id") or ""))
                        except ValueError:
                            raise ManualError(t("msg.player.bad_picture")) from None
                        r = manual.restore_picture(
                            conn, career.id, career.key, uid, image_id, settings, active_only=True
                        )
                        where = t(
                            "msg.reload_hint" if r.publish.active else "msg.player.other_career"
                        )
                        note = t("msg.player.still_never") if r.pinned_never else ""
                        return t("msg.player.restored", version=r.version, note=note, where=where)
                    case "pick":
                        try:
                            attempt = int(str(form.get("attempt") or ""))
                            index = int(str(form.get("index") or ""))
                        except ValueError:
                            raise ManualError(t("msg.player.bad_picture")) from None
                        picked = manual.pick_candidate(
                            conn,
                            career.id,
                            career.key,
                            uid,
                            attempt,
                            index,
                            settings,
                            active_only=True,
                        )
                        where = t(
                            "msg.reload_hint"
                            if picked.publish.active
                            else "msg.player.other_career"
                        )
                        return t("msg.player.picked", version=picked.version, where=where)
                    case "cancel":
                        done = manual.cancel(conn, career.id, uid, settings, active_only=True)
                        return t("msg.player.cancelled" if done else "msg.player.nothing_to_cancel")
                raise ManualError(f"unknown action {action!r}")
            finally:
                conn.close()

        try:
            message = await run_in_threadpool(run)
        except CareerChanged as exc:
            target = _safe_next(str(form.get("next") or ""), "/players")
            return _redirect(target, err=manual_error_text(exc, t))
        except ManualError as exc:
            target = _safe_next(str(form.get("next") or ""), back)
            return _redirect(target, err=manual_error_text(exc, t))
        except HTTPException as exc:
            target = _safe_next(str(form.get("next") or ""), back)
            return _redirect(target, err=str(exc.detail))
        except PublishError as exc:
            logger.error("publishing failed for player %s", uid, exc_info=exc)
            return _redirect(
                back, err=t("msg.player.publish_failed", error=problem_line(str(exc), t))
            )
        wake()
        return _redirect(_safe_next(str(form.get("next") or ""), back), msg=message)

    return router
