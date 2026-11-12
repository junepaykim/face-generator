"""Loopback host, exact-origin and CSRF checks for every state-changing request."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def origin_tuple(value: str) -> tuple[str, str, int] | None:
    """Parse an origin strictly, including the effective port; refuse opaque origins."""
    try:
        parts = urlsplit(value)
        if (
            parts.scheme not in ("http", "https")
            or parts.hostname not in LOOPBACK_HOSTS
            or parts.username is not None
            or parts.password is not None
            or parts.path
            or parts.query
            or parts.fragment
            or any(c.isspace() for c in value)
        ):
            return None
        port = parts.port
        return (
            parts.scheme,
            parts.hostname,
            port if port is not None else (443 if parts.scheme == "https" else 80),
        )
    except ValueError:
        return None


def install_security(app: FastAPI, templates: Jinja2Templates) -> None:
    # This is a local, unauthenticated application. A per-process unpredictable form
    # token is sufficient: other origins cannot read it under the same-origin policy.
    # No cookie is used, so another service on localhost cannot overwrite our token.
    token = secrets.token_urlsafe(32)
    templates.env.globals["csrf_token"] = token

    @app.middleware("http")
    async def local_only(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        hosts = request.headers.getlist("host")
        target = (
            origin_tuple(f"{request.scope['scheme']}://{hosts[0]}") if len(hosts) == 1 else None
        )
        if target is None:
            return Response("facegen only answers on a loopback host", status_code=403)
        if request.method not in SAFE_METHODS:
            origins = request.headers.getlist("origin")
            if origins and (len(origins) != 1 or origin_tuple(origins[0]) != target):
                return Response("cross-site request refused", status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return Response("cross-site request refused", status_code=403)
            supplied = request.headers.get("x-csrf-token")
            if supplied is None:
                # Cache the body before parsing so the downstream route can parse it too.
                await request.body()
                form = await request.form()
                supplied = form.get("csrf_token")
            if not isinstance(supplied, str) or not secrets.compare_digest(
                supplied.encode("utf-8"), token.encode("ascii")
            ):
                return Response(
                    "Invalid form token; reload the page and try again.", status_code=403
                )
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-store"
        return response
