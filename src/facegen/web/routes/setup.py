"""Setup routes and their app-local helpers."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from facegen.i18n import Translator
from facegen.persistence.settings import SettingsError, load_settings, local_label, local_path
from facegen.platform import platform_info
from facegen.setup import folder_setup
from facegen.setup.models import gigabytes
from facegen.web import guide
from facegen.web.dependencies import AppDependencies
from facegen.web.forms import form_of
from facegen.web.messages import problem_line
from facegen.web.responses import _redirect, _safe_next


def create_router(deps: AppDependencies) -> APIRouter:
    router = APIRouter()
    ctx = deps.ctx
    wake = deps.wake

    def folder_back(target: str, text: str, err: str) -> RedirectResponse:
        """Back to the page with the error, and the folder as typed, so it can be corrected."""
        base, _, query = target.partition("?")
        params = [
            p for p in query.split("&") if p and not p.startswith(("msg=", "err=", "folder="))
        ]
        params += [f"err={quote(err)}", f"folder={quote(text)}"]
        return RedirectResponse(f"{base}?{'&'.join(params)}", status_code=303)

    def use_folder(text: str, back: str, t: Translator) -> Response:
        """Check the folder, write it to the local file, and use it at once."""
        kind = ctx.platform or platform_info.current()
        check = folder_setup.check_folder(text, kind, folder_setup.RealDisk())
        if check.folder is None:
            return folder_back(back, text, t("msg.folder.not_fm", why=check.message))
        target = local_path(ctx.defaults_path)
        try:
            folder_setup.write_local_path(
                target, "fm_user_dir", folder_setup.value_to_write(check.folder, kind)
            )
            now = load_settings(defaults_path=ctx.defaults_path)
        except (folder_setup.LocalFileError, OSError, SettingsError) as exc:
            return folder_back(
                back,
                text,
                t(
                    "msg.folder.not_saved",
                    file=local_label(ctx.defaults_path),
                    error=problem_line(str(exc), t),
                ).strip(),
            )
        if now.paths.fm_user_dir is None:  # cannot happen unless the file was changed meanwhile
            return folder_back(back, text, t("msg.folder.no_effect"))
        if ctx.poller is not None:
            ctx.poller.wake()
        syncing = ctx.syncer is not None and ctx.syncer.start()
        wake()
        note = f"{check.message} " if check.message else ""
        then = t(
            "msg.folder.reading"
            if syncing
            else "msg.folder.press_sync"
            if ctx.syncer is not None
            else "msg.folder.next_sync"
        )
        return _redirect(
            back,
            msg=note
            + t(
                "msg.folder.saved",
                folder=now.paths.fm_user_dir,
                file=local_label(ctx.defaults_path),
                then=then,
            ),
        )

    @router.post("/setup/folder")
    async def setup_folder(request: Request) -> Response:
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), "/")
        text = str(form.get("folder") or "")
        t = await run_in_threadpool(deps.translator, request)
        return await run_in_threadpool(use_folder, text, back, t)

    @router.post("/setup/browse")
    async def setup_browse(request: Request) -> Response:
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), "/")
        t = await run_in_threadpool(deps.translator, request)
        if ctx.browse is None:
            return _redirect(back, err=t("msg.folder.no_browse"))
        start = str(form.get("folder") or "").strip() or None
        chosen = await run_in_threadpool(ctx.browse, start)
        if not chosen:
            return _redirect(back, msg=t("msg.folder.none_chosen"))
        return await run_in_threadpool(use_folder, chosen, back, t)

    @router.post("/setup/guide")
    async def setup_guide_closed() -> Response:
        """The user closed the finished first-run guide (Phase 12f)."""
        await run_in_threadpool(guide.close, deps.data_dir)
        return _redirect("/")

    @router.post("/setup/models")
    async def setup_models(request: Request) -> Response:
        """The one confirmation before the model download (AGENTS.md hard rule 4)."""
        form = await form_of(request)
        t = await run_in_threadpool(deps.translator, request)
        if ctx.download is None or ctx.models is None:
            return _redirect("/", err=t("msg.models.absent"))
        if form.get("confirm") != "download":
            return _redirect("/", err=t("msg.models.confirm"))
        now = await run_in_threadpool(ctx.models)
        if now.ready:
            return _redirect("/", msg=t("msg.models.here"))
        if not now.enough_space:
            return _redirect(
                "/",
                err=t("msg.models.space", size=gigabytes(now.missing_bytes), folder=now.cache),
            )
        if not ctx.download.start():
            return _redirect("/", err=t("msg.models.running"))
        return _redirect("/", msg=t("msg.models.started", size=gigabytes(now.missing_bytes)))

    return router
