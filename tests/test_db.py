from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from facegen import db

TABLES = {"careers", "clubs", "players", "traits", "images", "jobs", "settings", "sync_runs"}


def _job(conn: sqlite3.Connection, status: str, unique_id: int = 7) -> None:
    conn.execute(
        "INSERT INTO jobs (career_id, unique_id, kind, group_name, tier, rank, status, reason,"
        " created_at, updated_at) VALUES (1, ?, 'initial', 'a', 1, 0, ?, 'test', 'now', 'now')",
        (unique_id, status),
    )


@pytest.fixture
def with_career(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'c1', 'Career', '[]', 'now', 'now')"
    )
    return conn


def test_connect_creates_every_table(conn: sqlite3.Connection) -> None:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert TABLES <= names
    assert db.schema_version(conn) == len(db.MIGRATIONS)


def test_migrate_is_idempotent(conn: sqlite3.Connection) -> None:
    assert db.migrate(conn) == len(db.MIGRATIONS)
    assert db.migrate(conn) == len(db.MIGRATIONS)


def test_file_database_uses_wal(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "sub" / "facegen.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_newer_schema_is_refused(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) + 1}")
    with pytest.raises(RuntimeError, match="newer"):
        db.migrate(conn)


def test_one_active_job_per_player(with_career: sqlite3.Connection) -> None:
    _job(with_career, "queued")
    with pytest.raises(sqlite3.IntegrityError):
        _job(with_career, "running")


def test_finished_jobs_do_not_block_a_new_one(with_career: sqlite3.Connection) -> None:
    for status in ("done", "failed", "cancelled"):
        _job(with_career, status)
    _job(with_career, "queued")
    assert with_career.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 4


def test_job_kind_and_status_are_checked(with_career: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _job(with_career, "paused")


def test_one_current_image_per_player(with_career: sqlite3.Connection) -> None:
    sql = (
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model, prompt,"
        " negative_prompt, seed, is_current, created_at) VALUES (1, 7, ?, 'initial', 'a', 'm',"
        " 'p', 'n', 1, 1, 'now')"
    )
    with_career.execute(sql, (1,))
    with pytest.raises(sqlite3.IntegrityError):
        with_career.execute(sql, (2,))


def test_transaction_commits(with_career: sqlite3.Connection) -> None:
    with db.transaction(with_career):
        _job(with_career, "done")
    assert with_career.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_transaction_rolls_back_on_request(with_career: sqlite3.Connection) -> None:
    with db.transaction(with_career, rollback=True):
        _job(with_career, "done")
    assert with_career.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_transaction_rolls_back_on_error(with_career: sqlite3.Connection) -> None:
    with pytest.raises(ValueError), db.transaction(with_career):
        _job(with_career, "done")
        raise ValueError
    assert with_career.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
