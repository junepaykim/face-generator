"""Execute one sync with owned resources, then publish the active career's pack.

Ingestion and planning commit before publication begins. A later process adapter must not
promise to roll back either a committed sync or files already published when it stops.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from facegen.datasource.base import DataSource
from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline.publish import PublishError, PublishReport, publish_active
from facegen.pipeline.sync import SyncReport, sync_of_save
from facegen.runtime.contracts import (
    FAILED,
    SYNCED,
    UNREADABLE,
    SyncOutcome,
    SyncRequest,
    SyncResult,
)
from facegen.runtime.policy import BUSY_TIMEOUT_MS
from facegen.runtime.poller import (
    UNSUPPORTED_ADVICE,
    NoSaveError,
    SaveUnreadableError,
    SaveVersion,
    find_latest_save,
    sync_save,
)


def describe_sync(report: SyncReport) -> str:
    queued = sum(report.enqueued.values())
    text = (
        f"Synced {report.save_file} (in-game {report.game_date or 'unknown'}): "
        f"{queued} queued, {report.updated} updated, {report.cancelled} cancelled."
    )
    if report.new_at_my_club:
        text += f" {report.new_at_my_club} new at my club, drawn first."
    if report.new_staff:
        text += f" {report.new_staff} new on my club's staff."
    return text


def unchanged_save_notice(conn: sqlite3.Connection, games_dir: Path | None) -> str | None:
    """What to tell the user when the newest save is one that is already synced, else None.

    facegen reads the save file, not the running game: a signing or a new member of staff is
    in it only once FM27 has saved (spec section 6).
    """
    newest = find_latest_save(games_dir) if games_dir is not None else None
    if newest is None:
        return None
    version = SaveVersion.of(newest)
    row = sync_of_save(conn, newest.name, version.mtime)
    if row is None or row["error"]:
        return None
    return (
        f"FM27 has not saved since {newest.name} was last synced (saved {version.saved_at()}), "
        "so this sync found nothing new. facegen reads the save file, not the running game: "
        "a signing or a new member of staff reaches it when FM27 saves. Save the game (or wait "
        "for its autosave), then press Sync now again."
    )


def describe_activation(report: PublishReport | None) -> str | None:
    """A line for the log when a sync changed what FM27 is shown, else None."""
    if report is None or not (report.deactivated or report.config_changed or report.written):
        return None
    parts = [f"FM27 pack {report.folder.name}: {report.mapped} picture(s) mapped"]
    if report.deactivated:
        parts.append(f"other careers' packs unmapped ({', '.join(report.deactivated)})")
    return "; ".join(parts) + ". FM27 loads it at its next start or on Reload Skin."


def execute_sync(
    request: SyncRequest,
    source_factory: Callable[[], DataSource],
    *,
    log: Callable[[str], None] = lambda line: None,
    on_result: Callable[[SyncResult], None] = lambda result: None,
) -> SyncResult:
    """Run in this thread and log in execution order.

    After closing the connection, deliver the final record even when a source or log
    callback raises out of execution. This lets an adapter release its claim without
    swallowing the exception or announcing normal completion.
    """
    defaults_path = Path(request.defaults_path) if request.defaults_path is not None else None
    save = Path(request.save_path) if request.save_path is not None else None
    trigger = request.trigger
    message = error = None
    unsupported = False
    outcome: SyncOutcome = FAILED
    warnings: list[str] = []
    notice = None
    conn = db.connect(Path(request.db_path), busy_timeout_ms=BUSY_TIMEOUT_MS)
    try:
        settings = load_settings(conn, defaults_path)
        if save is None and trigger == "sync now":
            notice = unchanged_save_notice(conn, settings.paths.games_dir)
        report = sync_save(conn, settings, source_factory(), save, preview=0, trigger=trigger)
        outcome = SYNCED
        message = describe_sync(report)
        warnings = list(report.warnings)
        log(message)
        if notice:
            log(notice)
        try:
            if line := describe_activation(publish_active(conn, settings)):
                log(line)
        except PublishError as exc:
            warnings.append(f"publishing the active career's pack failed: {exc}")
            log(f"Publishing failed: {exc}")
    except SaveUnreadableError as exc:
        error = str(exc)
        outcome = FAILED if exc.busy else UNREADABLE
        unsupported = exc.unsupported
        log(error + (f" {UNSUPPORTED_ADVICE}" if unsupported else ""))
    except NoSaveError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a background sync reports, never raises
        error = f"sync failed: {type(exc).__name__}: {exc}"
        log(error)
    finally:
        conn.close()
        result = SyncResult(
            outcome=outcome,
            message=message,
            error=error,
            unsupported=unsupported,
            warnings=tuple(warnings),
            notice=notice if outcome == SYNCED else None,
        )
        on_result(result)
    return result
