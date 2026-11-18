"""The live events stream (Phase 12d)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from facegen.web import live
from facegen.web.dependencies import AppDependencies


def create_router(deps: AppDependencies) -> APIRouter:
    router = APIRouter()
    ctx = deps.ctx

    async def _pause(seconds: float) -> None:
        """Wait between looks, but notice at once when the server stops."""
        end = time.monotonic() + seconds
        while not deps.closing.is_set() and (left := end - time.monotonic()) > 0:
            await asyncio.sleep(min(0.1, left))

    def look() -> dict[str, dict[str, Any]]:
        conn = deps.connect()
        try:
            return live.snapshot(ctx, conn, deps.live_extra())
        finally:
            conn.close()

    @router.get("/events")
    async def events(request: Request) -> StreamingResponse:
        """Server-sent events: one per topic whenever its part of a page has changed, the
        first at once. Ends when the page goes, when the server stops (deps.closing), or
        after live.STREAM_SECONDS; the page then reconnects by itself."""

        t = await run_in_threadpool(deps.translator, request)

        async def stream() -> AsyncIterator[str]:
            yield f"retry: {live.RETRY_MS}\n\n"
            sent: dict[str, str] = {}
            started = last_ping = time.monotonic()
            while not deps.closing.is_set():
                if await request.is_disconnected():
                    return
                now = time.monotonic()
                if now - started >= live.STREAM_SECONDS:
                    return
                for topic, data in (await run_in_threadpool(look)).items():
                    if sent.get(topic) != data["rev"]:
                        sent[topic] = data["rev"]
                        if data.get("state"):  # the sidebar dot's title, in the page's language
                            data["label"] = t(f"worker.state.{data['state']}")
                        yield live.event(topic, data)
                        last_ping = now
                if now - last_ping >= live.PING_SECONDS:
                    yield ": ping\n\n"
                    last_ping = now
                await _pause(live.TICK_SECONDS)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
