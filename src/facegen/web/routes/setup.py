"""Setup routes and their app-local helpers."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from facegen.persistence.settings import SettingsError, load_settings, local_label, local_path
from facegen.platform import platform_info
from facegen.setup import folder_setup
from facegen.setup.models import gigabytes
from facegen.web.dependencies import AppDependencies
from facegen.web.forms import form_of
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

    def use_folder(text: str, back: str) -> Response:
        """Check the folder, write it to the local file, and use it at once."""
        kind = ctx.platform or platform_info.current()
        check = folder_setup.check_folder(text, kind, folder_setup.RealDisk())
        if check.folder is None:
            return folder_back(back, text, f"Not FM27's folder: {check.message}")
        target = local_path(ctx.defaults_path)
        try:
            folder_setup.write_local_path(
                target, "fm_user_dir", folder_setup.value_to_write(check.folder, kind)
            )
            now = load_settings(defaults_path=ctx.defaults_path)
        except (folder_setup.LocalFileError, OSError, SettingsError) as exc:
            return folder_back(
                back, text, f"Not saved: {local_label(ctx.defaults_path)} {exc}".strip()
            )
        if now.paths.fm_user_dir is None:  # cannot happen unless the file was changed meanwhile
            return folder_back(back, text, "Not saved: the folder did not take effect.")
        if ctx.poller is not None:
            ctx.poller.wake()
        syncing = ctx.syncer is not None and ctx.syncer.start()
        wake()
        note = f"{check.message} " if check.message else ""
        then = (
            "Reading the newest save now."
            if syncing
            else "Press Sync now to read the newest save."
            if ctx.syncer is not None
            else "It is used from the next sync."
        )
        return _redirect(
            back,
            msg=f"{note}FM27's folder is {now.paths.fm_user_dir}, saved in"
            f" {local_label(ctx.defaults_path)}. {then}",
        )

    @router.post("/setup/folder")
    async def setup_folder(request: Request) -> Response:
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), "/")
        text = str(form.get("folder") or "")
        return await run_in_threadpool(use_folder, text, back)

    @router.post("/setup/browse")
    async def setup_browse(request: Request) -> Response:
        form = await form_of(request)
        back = _safe_next(str(form.get("next") or ""), "/")
        if ctx.browse is None:
            return _redirect(back, err="Browse is not available here: type or paste the folder.")
        start = str(form.get("folder") or "").strip() or None
        chosen = await run_in_threadpool(ctx.browse, start)
        if not chosen:
            return _redirect(back, msg="No folder chosen.")
        return await run_in_threadpool(use_folder, chosen, back)

    @router.post("/setup/models")
    async def setup_models(request: Request) -> Response:
        """The one confirmation before the model download (AGENTS.md hard rule 4)."""
        form = await form_of(request)
        if ctx.download is None or ctx.models is None:
            return _redirect("/", err="The models are not downloaded by this process.")
        if form.get("confirm") != "download":
            return _redirect("/", err="Not downloaded: confirm with the Download button.")
        now = await run_in_threadpool(ctx.models)
        if now.ready:
            return _redirect("/", msg="The models are here already.")
        if not now.enough_space:
            return _redirect(
                "/",
                err=f"Not enough free space: {gigabytes(now.missing_bytes)} and 1 GB to spare"
                f" are needed in {now.cache}.",
            )
        if not ctx.download.start():
            return _redirect("/", err="The download is running already.")
        return _redirect(
            "/",
            msg=f"Downloading the models ({gigabytes(now.missing_bytes)}); the dashboard"
            " shows how far it is. It can be stopped and goes on at the next start.",
        )

    return router
