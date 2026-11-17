"""Compose the local web UI with isolated services, templates, security and routers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from facegen.web.context import AppContext, SetupState
from facegen.web.dependencies import AppDependencies
from facegen.web.presentation import HERE, Presentation, create_templates, static_version
from facegen.web.routes import assets, dashboard, players, review, settings, setup
from facegen.web.security import install_security

__all__ = ["HERE", "AppContext", "SetupState", "create_app", "static_version"]


def create_app(ctx: AppContext) -> FastAPI:
    # FastAPI's own telemetry is off; nothing leaves this PC (AGENTS.md hard rule 6).
    app = FastAPI(
        title="facegen",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False},
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = create_templates()
    deps = AppDependencies(ctx)
    install_security(app, templates)
    presentation = Presentation(deps, templates)

    app.include_router(dashboard.create_router(deps, presentation))
    app.include_router(setup.create_router(deps))
    app.include_router(players.create_router(deps, presentation))
    app.include_router(review.create_router(deps, presentation))
    app.include_router(settings.create_router(deps, presentation))
    app.include_router(assets.create_router(deps))

    return app
