"""Execute one sync with owned resources, then publish the active career's pack.

Copying and parsing come first, then ingestion and planning in one database transaction;
publication begins after its commit. The transaction records any publication still due.
The child may abandon a sync until the transaction enters its final commit. A crash
after that point leaves the pack eligible for publication again when facegen restarts.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from facegen.core.settings_values import Settings
from facegen.datasource import photos
from facegen.datasource import snapshots as snapshots_io
from facegen.datasource.base import DataSource, SaveBusyError, SaveReadError, UnsupportedSaveError
from facegen.datasource.snapshots import SaveVersion, find_latest_save
from facegen.persistence import db, photo_index, publication
from facegen.persistence.settings import folder_not_set_message, load_settings
from facegen.pipeline.publish import PublishError, PublishReport, publish_active
from facegen.pipeline.sync import SyncReport, record_failed_sync, run_sync, sync_of_save
from facegen.runtime.contracts import (
    FAILED,
    SYNCED,
    UNREADABLE,
    SyncOutcome,
    SyncRequest,
    SyncResult,
)
from facegen.runtime.policy import BUSY_TIMEOUT_MS

UNSUPPORTED_ADVICE = (
    "The save reader (fmsave, pinned in pyproject.toml) cannot read this game build. Existing "
    "pictures and the queue are unchanged, and nothing new is queued from this save. When an "
    "fmsave release supports the build: bump the pin, re-run the Phase 0.1 checks and note the "
    "result in docs/research.md (AGENTS.md rule 5)."
)


class NoSaveError(LookupError):
    """There is no save to read."""


class FolderNotSetError(NoSaveError):
    """FM27's user folder is neither set nor found, so there is no saves folder to look in."""

    def __init__(self) -> None:
        super().__init__(folder_not_set_message())


class SaveUnreadableError(RuntimeError):
    """The save could not be copied or parsed; existing pictures and the queue are unchanged."""

    def __init__(self, save: Path, cause: Exception) -> None:
        super().__init__(f"Cannot read {save.name}: {cause}")
        self.save = save
        self.unsupported = isinstance(cause, UnsupportedSaveError)
        """The game or build is one the save reader cannot read (see UNSUPPORTED_ADVICE)."""
        self.busy = isinstance(cause, SaveBusyError)
        """The game was still writing the save; a later try should work."""


def sync_save(
    conn: sqlite3.Connection,
    settings: Settings,
    source: DataSource,
    save: Path | None = None,
    *,
    dry_run: bool = False,
    preview: int = 20,
    with_table: bool = False,
    trigger: str | None = None,
    read_settings: Callable[[], Settings] | None = None,
    before_commit: Callable[[], None] | None = None,
) -> SyncReport:
    """Copy a save (default: the newest in FM27's games folder), parse the copy and sync it.

    A save already inside data/snapshots/ is read in place. A save that cannot be read is
    recorded as a failed sync (except in a dry run) and raises SaveUnreadableError. The
    recorded save time is the copy's, which keeps the source's time from when it was copied.
    Each sync also looks again at who FM27 shows a photo for (photos.py): its own photo list
    and the user's facepacks, read again only if they changed.
    """
    snapshots = settings.paths.snapshots_dir
    if save is None:
        games = settings.paths.games_dir
        if games is None:
            raise FolderNotSetError()
        save = find_latest_save(games)
        if save is None:
            raise NoSaveError(f"No save found in {games}")
    if not save.is_file():
        raise NoSaveError(f"No such save: {save}")
    save_mtime: float | None = None
    try:
        if save.resolve().is_relative_to(snapshots.resolve()):
            snapshot_path = save
        else:
            snapshot_path = snapshots_io.copy_snapshot(save, snapshots)
        save_mtime = snapshot_path.stat().st_mtime
        snapshot = source.load(snapshot_path)
    except (UnsupportedSaveError, SaveReadError) as exc:
        if not dry_run:
            record_failed_sync(
                conn,
                save.name,
                str(exc),
                getattr(exc, "build", None),
                # A save still being written is tried again; any other failure waits for the
                # save to change.
                save_mtime=None if isinstance(exc, SaveBusyError) else save_mtime,
                unsupported=isinstance(exc, UnsupportedSaveError),
                trigger=trigger,
            )
        raise SaveUnreadableError(save, exc) from exc

    def record_publication() -> None:
        # In the same transaction as the new active career and queue.
        publication.mark_pending(conn)
        if before_commit is not None:
            before_commit()

    return run_sync(
        conn,
        snapshot,
        settings,
        save_file=save.name,
        save_mtime=save_mtime,
        dry_run=dry_run,
        preview=preview,
        with_table=with_table,
        trigger=trigger,
        read_settings=read_settings,
        before_commit=record_publication,
        photos=photos.scan(settings.paths, photo_index.stored_state(conn)),
    )


def sync_numbers(report: SyncReport) -> dict[str, int]:
    """What the sync did to the queue, by name (runtime.contracts.SYNC_NUMBERS): the pages word
    these (Phase 17e), while `describe_sync` keeps the English line for the log and the tray."""
    return {
        "queued": sum(report.enqueued.values()),
        "new_at_my_club": report.new_at_my_club,
        "new_staff": report.new_staff,
        "updated": report.updated,
        "cancelled": report.cancelled,
    }


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
        "so this sync found nothing new. Kitshot reads the save file, not the running game: "
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
    before_commit: Callable[[], None] = lambda: None,
    on_committed: Callable[[], None] = lambda: None,
    on_result: Callable[[SyncResult], None] = lambda result: None,
) -> SyncResult:
    """Run in this thread and log in execution order.

    `before_commit` protects the final commit from a simultaneous Quit. `on_committed`
    runs once SQLite has committed and before anything is
    logged or published. After closing the connection, deliver the final record even when a
    source or log callback raises out of execution. This lets an adapter release its claim
    without swallowing the exception or announcing normal completion.
    """
    defaults_path = Path(request.defaults_path) if request.defaults_path is not None else None
    save = Path(request.save_path) if request.save_path is not None else None
    trigger = request.trigger
    message = error = None
    unsupported = False
    outcome: SyncOutcome = FAILED
    warnings: list[str] = []
    notice = None
    numbers: dict[str, int] = {}
    conn = db.connect(Path(request.db_path), busy_timeout_ms=BUSY_TIMEOUT_MS)
    try:
        settings = load_settings(conn, defaults_path)
        recovering = trigger == "publish recovery"
        if recovering:
            # The database already contains the sync, even if its save file is now gone.
            before_commit()
            on_committed()
        else:
            if save is None and trigger == "sync now":
                notice = unchanged_save_notice(conn, settings.paths.games_dir)
            report = sync_save(
                conn,
                settings,
                source_factory(),
                save,
                preview=0,
                trigger=trigger,
                read_settings=lambda: load_settings(conn, defaults_path),
                before_commit=before_commit,
            )
            on_committed()
            message = describe_sync(report)
            numbers = sync_numbers(report)
            warnings = list(report.warnings)
            log(message)
            if notice:
                log(notice)
        outcome = SYNCED
        try:
            if not recovering or publication.pending(conn):
                activation = publish_active(
                    conn, settings, read_settings=lambda: load_settings(conn, defaults_path)
                )
                line = describe_activation(activation)
                if line:
                    log(line)
            if recovering:
                message = "Recovered the active career's FM27 pack."
                log(message)
        except (PublishError, OSError) as exc:
            if recovering:
                outcome = FAILED
                error = f"publishing the active career's pack failed: {exc}"
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
        outcome = FAILED
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
            numbers=numbers if outcome == SYNCED else {},
        )
        on_result(result)
    return result
