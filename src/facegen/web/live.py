"""Live pages (Phase 12d): server-sent events say when a page part has changed.

The stream (`GET /events`) sends one event per topic whenever the fingerprint of what that
part shows changes: "status" (the dashboard's status block, with the worker's state and the
review count for the sidebar), "log" (the Log page) and "pictures" (the Gallery, Phase 12h). The page then fetches its part as
before. Nothing is pushed between changes but a ping; the picture's progress bar moves in the
page itself.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from facegen.persistence.image_store import picture_revision
from facegen.persistence.jobs import load_career
from facegen.runtime.contracts import SyncStatus
from facegen.web import queries
from facegen.web.context import AppContext

TICK_SECONDS = 1.0
"""How often a stream looks for a change."""
PING_SECONDS = 15.0
"""A comment line this often keeps a quiet stream open."""
STREAM_SECONDS = 600.0
"""A stream ends after this long and the page opens another at once: no stream outlives a
restart of the server for long."""
RETRY_MS = 1500
"""How soon the page reconnects after a stream ends or the server restarts."""

TOPICS = ("status", "log", "pictures")


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=repr, ensure_ascii=True)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def snapshot(
    ctx: AppContext, conn: sqlite3.Connection, extra: Any = None, *, sync: SyncStatus | None = None
) -> dict[str, dict[str, Any]]:
    """Per topic, what its event carries; "rev" changes when the part would look different.

    Cheap on purpose: the services' status in memory and a few MAX/COUNT lookups, about once
    a second for each open page. The initial dashboard supplies the sync status it renders.
    """
    worker = ctx.worker.status() if ctx.worker else None
    if sync is None and ctx.syncer is not None:
        sync = ctx.syncer.status()
    download = ctx.download.status() if ctx.download else None
    poller = ctx.poller.status() if ctx.poller else None
    career = load_career(conn)
    db = list(
        conn.execute(
            "SELECT (SELECT MAX(id) FROM images), (SELECT MAX(updated_at) FROM jobs),"
            " (SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')),"
            " (SELECT MAX(id) FROM sync_runs),"
            " (SELECT finished_at FROM sync_runs ORDER BY id DESC LIMIT 1),"
            " (SELECT MAX(id) FROM settings_history),"
            " (SELECT COUNT(*) FROM settings_history WHERE undone = 1)"
        ).fetchone()
    )
    review = queries.review_count(conn, career.id) if career else 0
    # Any picture made, restored or removed changes the current set's sum of ids.
    current = list(
        conn.execute(
            "SELECT MAX(id), COUNT(*), TOTAL(id) FROM images WHERE is_current = 1"
        ).fetchone()
    )
    if career:
        current.append(picture_revision(conn, career.id))
    shown = None
    if worker is not None:
        shown = [
            worker.state,
            worker.reason,
            worker.values,
            worker.since,
            worker.current.id if worker.current else None,
            worker.paused,
            worker.gpu_fault,
            worker.generate_now,
            worker.last_error,
        ]
    status = {
        "extra": extra,  # the first-run guide's own state (Phase 12f): the card's answer
        "career": career.id if career else None,
        "pictures": current,
        "db": db,
        "worker": shown,
        "sync": repr(sync),
        "download": repr(download),
        "review": review,
    }
    log = {
        "lines": [len(worker.log), worker.log[-1] if worker.log else None] if worker else None,
        "sync": db[3:5],
        "poller": repr(poller),
    }
    return {
        "status": {
            "rev": _digest(status),
            "state": worker.state if worker else None,
            "review": review,
        },
        "log": {"rev": _digest(log)},
        "pictures": {"rev": _digest([career.id if career else None, current, db[3]])},
    }


def event(topic: str, data: dict[str, Any]) -> str:
    """One server-sent event."""
    return f"event: {topic}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"
