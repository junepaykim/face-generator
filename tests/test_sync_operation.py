"""The sync execution boundary owns resources and preserves service ordering."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import pytest
from factories import player, snapshot

from facegen.core.records import Snapshot
from facegen.core.settings_values import Settings
from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline import sync_operation
from facegen.pipeline.publish import PublishError
from facegen.runtime.contracts import FAILED, SYNCED, SyncRequest, SyncResult
from facegen.runtime.sync_service import SyncService


def save_file(config_file: Path) -> Path:
    settings = load_settings(defaults_path=config_file)
    games = settings.paths.games_dir
    assert games is not None
    games.mkdir(parents=True)
    save = games / "career.fm"
    save.write_bytes(b"a synthetic save")
    return save


class Source:
    def load(self, path: Path) -> Snapshot:
        return snapshot([player(1001)])


def test_request_and_result_contain_only_message_data() -> None:
    request = SyncRequest("state.db", "defaults.toml", "save.fm", "poller")
    assert json.loads(json.dumps(asdict(request))) == {
        "db_path": "state.db",
        "defaults_path": "defaults.toml",
        "save_path": "save.fm",
        "trigger": "poller",
    }
    result = SyncResult(SYNCED, "done", None, False, ("a warning",), "unchanged")
    assert json.loads(json.dumps(asdict(result))) == {
        "outcome": "synced",
        "message": "done",
        "error": None,
        "unsupported": False,
        "warnings": ["a warning"],
        "notice": "unchanged",
    }


def test_sync_commits_before_publishing_and_closes_before_completion(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save = save_file(config_file)
    settings = load_settings(defaults_path=config_file)
    connections: list[sqlite3.Connection] = []
    events: list[str] = []
    connect = db.connect

    def record_connection(path: Path, *, busy_timeout_ms: int) -> sqlite3.Connection:
        connection = connect(path, busy_timeout_ms=busy_timeout_ms)
        connections.append(connection)
        return connection

    def publish(connection: sqlite3.Connection, settings: Settings) -> None:
        events.append("publish")
        with closing(connect(settings.paths.db_path)) as reader:
            assert reader.execute("SELECT name FROM players WHERE unique_id=1001").fetchone()
            assert reader.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        raise PublishError("synthetic publish failure")

    def completed() -> None:
        assert not syncer.status().running
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[-1].execute("SELECT 1")
        events.append("done")

    monkeypatch.setattr(sync_operation.db, "connect", record_connection)
    monkeypatch.setattr(sync_operation, "publish_active", publish)
    syncer = SyncService(
        settings.paths.db_path,
        Source,
        defaults_path=config_file,
        on_done=completed,
        log=events.append,
    )
    assert syncer.run(save) == SYNCED
    status = syncer.status()
    assert status.started is not None
    assert status.trigger == "sync now" and status.error is None
    assert status.warnings[-1] == (
        "publishing the active career's pack failed: synthetic publish failure"
    )
    assert events == [
        status.message,
        "publish",
        "Publishing failed: synthetic publish failure",
        "done",
    ]


def test_sync_operation_reports_source_failure_and_closes_connection(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_file(config_file)
    settings = load_settings(defaults_path=config_file)
    connection = db.connect(settings.paths.db_path)
    monkeypatch.setattr(sync_operation.db, "connect", lambda *args, **kwargs: connection)
    lines: list[str] = []

    def broken_source() -> Source:
        raise RuntimeError("synthetic reader failure")

    result = sync_operation.execute_sync(
        SyncRequest(str(settings.paths.db_path), str(config_file)), broken_source, log=lines.append
    )
    assert result == SyncResult(FAILED, error="sync failed: RuntimeError: synthetic reader failure")
    assert lines == [result.error]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_run_blocks_and_claim_rejects_both_entry_points(config_file: Path) -> None:
    save = save_file(config_file)
    settings = load_settings(defaults_path=config_file)
    entered = threading.Event()
    release = threading.Event()
    outcomes: list[str | None] = []
    completions: list[bool] = []

    class WaitingSource(Source):
        def load(self, path: Path) -> Snapshot:
            assert path != save and path.parent == settings.paths.snapshots_dir
            entered.set()
            assert release.wait(10)
            return super().load(path)

    syncer = SyncService(
        settings.paths.db_path,
        WaitingSource,
        defaults_path=config_file,
        on_done=lambda: completions.append(syncer.status().running),
    )
    thread = threading.Thread(target=lambda: outcomes.append(syncer.run(save)), daemon=True)
    thread.start()
    try:
        assert entered.wait(10)
        assert syncer.status().running and thread.is_alive()
        assert not syncer.start()
        assert syncer.run(save, "poller") is None
        assert not outcomes and not completions
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert outcomes == [SYNCED] and completions == [False]


def test_raising_log_callback_releases_claim_without_announcing_completion(
    config_file: Path,
) -> None:
    save_file(config_file)
    settings = load_settings(defaults_path=config_file)
    completions: list[bool] = []

    def broken_source() -> Source:
        raise RuntimeError("synthetic reader failure")

    def broken_log(line: str) -> None:
        raise ValueError("synthetic log failure")

    syncer = SyncService(
        settings.paths.db_path,
        broken_source,
        defaults_path=config_file,
        log=broken_log,
        on_done=lambda: completions.append(True),
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="synthetic log failure"):
            syncer.run()
        status = syncer.status()
        assert not status.running
        assert status.error == "sync failed: RuntimeError: synthetic reader failure"
        assert status.started is not None and status.trigger == "sync now"
    assert not completions


def test_interrupted_source_releases_claim_and_still_propagates(config_file: Path) -> None:
    save_file(config_file)
    settings = load_settings(defaults_path=config_file)
    completions: list[bool] = []

    class Interrupted(BaseException):
        pass

    def interrupted_source() -> Source:
        raise Interrupted

    syncer = SyncService(
        settings.paths.db_path,
        interrupted_source,
        defaults_path=config_file,
        on_done=lambda: completions.append(True),
    )
    with pytest.raises(Interrupted):
        syncer.run()
    assert not syncer.status().running
    assert syncer.status().error is None
    assert not completions
