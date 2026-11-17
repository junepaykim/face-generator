from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from facegen.persistence import db

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


def test_state_migration_preserves_history_constraints_and_restored_sheet() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    for script in db.MIGRATIONS[:5]:
        conn.executescript(script)
    conn.execute("PRAGMA user_version = 5")
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'c1', 'Career', '[]', 'now', 'now')"
    )
    conn.execute(
        "INSERT INTO players (career_id, unique_id, name, ca, is_newgen, in_scope, at_my_club,"
        " in_my_league, in_continental, academy) VALUES (1, 7, 'Player', 100, 1, 1, 0, 1, 0, 1)"
    )
    conn.executemany(
        "INSERT INTO traits VALUES (1, 7, ?, 42, '{}', 'now')",
        [(1,), (2,)],
    )
    conn.executemany(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model, prompt,"
        " negative_prompt, seed, age_band, traits_version, is_current, created_at, render,"
        " check_result, published_path, restored_at, restored_band, restored_group)"
        " VALUES (1, 7, ?, 'initial', 'b', 'm', 'p', 'n', 42, 'W04', ?, ?, 'now', '{}',"
        " '{\"passed\":true}', '/published.png', 'restored', 'W07', 'a')",
        [(1, 1, 1), (2, 2, 0)],
    )
    _job(conn, "queued")
    conn.execute("UPDATE jobs SET target_band = 'W07'")
    images_before = [tuple(r) for r in conn.execute("SELECT * FROM images ORDER BY id")]
    jobs_before = [tuple(r) for r in conn.execute("SELECT * FROM jobs")]
    db.migrate(conn)
    assert [tuple(r) for r in conn.execute("SELECT * FROM images ORDER BY id")] == images_before
    assert [tuple(r)[:-2] for r in conn.execute("SELECT * FROM jobs")] == jobs_before
    assert conn.execute("SELECT active_traits_version FROM players").fetchone()[0] == 1
    for table, column in (("images", "age_band"), ("jobs", "target_band")):
        assert {r[1]: r[2] for r in conn.execute(f"PRAGMA table_info({table})")}[column] == "TEXT"
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with pytest.raises(sqlite3.IntegrityError):
        _job(conn, "running")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE images SET is_current = 1 WHERE version = 2")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE jobs SET kind = 'anything'")
    assert db.migrate(conn) == len(db.MIGRATIONS)


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


def test_phase_4_migration_backfills_each_careers_newest_save() -> None:
    """A database from Phase 3 learns which career's save is newest from its sync history."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    for script in db.MIGRATIONS[:3]:
        conn.executescript(script)
    conn.execute("PRAGMA user_version = 3")
    for cid, key in ((1, "old"), (2, "new")):
        conn.execute(
            "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
            " VALUES (?, ?, ?, '[]', 'now', ?)",
            (cid, key, key, f"2026-11-1{cid}"),
        )
    conn.executemany(
        "INSERT INTO sync_runs (career_id, started_at, save_file, save_mtime, error)"
        " VALUES (?, 'now', 'x.fm', ?, ?)",
        [(1, 300.0, None), (1, 100.0, None), (2, 200.0, None), (2, 900.0, "unreadable")],
    )
    assert db.migrate(conn) == len(db.MIGRATIONS)
    newest = dict(conn.execute("SELECT key, last_save_mtime FROM careers").fetchall())
    assert newest == {"old": 300.0, "new": 200.0}  # failed syncs do not count
    # Career 1 synced earlier but its save is newer: it is the active one.
    assert db.active_career_id(conn) == 1
    assert conn.execute("SELECT unsupported, trigger FROM sync_runs").fetchone()[:] == (0, None)


def test_active_career_falls_back_to_the_last_synced(with_career: sqlite3.Connection) -> None:
    conn = with_career
    assert db.active_career_id(conn) == 1
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (2, 'c2', 'Two', '[]', 'now', 'now, and later')"
    )
    assert db.active_career_id(conn) == 2  # neither has a save time: the last synced
    conn.execute("UPDATE careers SET last_save_mtime = 5 WHERE id = 1")
    assert db.active_career_id(conn) == 1  # a known save time beats none


def test_phase_5_migration_republishes_square_pictures_and_can_record_a_restore() -> None:
    """Every picture before Phase 5 is square: they are marked unpublished, so the next
    publish writes them again as 3:4 crops."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    for script in db.MIGRATIONS[:4]:
        conn.executescript(script)
    conn.execute("PRAGMA user_version = 4")
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'c1', 'Career', '[]', 'now', 'now')"
    )
    conn.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model, prompt,"
        " negative_prompt, seed, master_path, published_path, is_current, created_at)"
        " VALUES (1, 7, 1, 'initial', 'a', 'm', 'p', 'n', 1, 'masters/c1/7/v001.png',"
        " '/fm/graphics/facegen/c1/7.png', 1, 'now')"
    )
    assert db.migrate(conn) == len(db.MIGRATIONS)
    row = conn.execute("SELECT * FROM images").fetchone()
    assert row["published_path"] is None and row["is_current"] == 1
    assert row["restored_at"] is None and row["restored_group"] is None
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE images SET restored_group = 'x'")


def test_migration_11_drops_drawn_male_and_the_mens_targets() -> None:
    """Everyone is drawn as a woman again (2026-11-14): queued jobs lose a men's band."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    for script in db.MIGRATIONS[:10]:
        conn.executescript(script)
    conn.execute("PRAGMA user_version = 10")
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'c1', 'Career', '[]', 'now', 'now')"
    )
    for uid, status, band in ((7, "queued", "M2"), (8, "queued", "W05"), (9, "done", "M3")):
        _job(conn, status, uid)
        conn.execute("UPDATE jobs SET target_band = ? WHERE unique_id = ?", (band, uid))
    assert db.migrate(conn) == len(db.MIGRATIONS) == 11
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(players)")}
    assert "drawn_male" not in columns and "female" in columns
    bands = dict(conn.execute("SELECT unique_id, target_band FROM jobs").fetchall())
    assert bands == {7: None, 8: "W05", 9: "M3"}  # finished jobs keep their record


def _history(conn: sqlite3.Connection, before: dict, after: dict) -> None:
    conn.execute(
        "INSERT INTO settings_history (at, label, before, after) VALUES ('now', 'edit', ?, ?)",
        (json.dumps(before), json.dumps(after)),
    )


def test_the_detail_pass_switch_is_renamed_to_its_size_at_connect(tmp_path: Path) -> None:
    """Phase 10c: a stored detail pass off is "off", on the size it drew at; the settings
    history follows, so an older change can still be undone. No schema number changes, so an
    older facegen on the same database still opens it."""
    path = tmp_path / "facegen.db"
    conn = db.connect(path)
    conn.executemany(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, 'then')",
        [
            ("groups.a.detail_pass", "false"),
            ("groups.b.detail_pass", "true"),
            ("groups.c.detail_pass", "7"),  # not a switch: dropped, the default applies
            ("groups.a.ca_above", "150"),
        ],
    )
    _history(conn, {"groups.a.detail_pass": None}, {"groups.a.detail_pass": False})
    _history(
        conn,
        {"groups.b.detail_pass": False, "groups.a.ca_above": None},
        {"groups.b.detail_pass": True, "groups.a.ca_above": 150},
    )
    _history(conn, {"staff.enabled": None}, {"staff.enabled": False})
    conn.close()

    conn = db.connect(path)
    assert db.schema_version(conn) == len(db.MIGRATIONS)
    stored = dict(conn.execute("SELECT key, value FROM settings").fetchall())
    assert stored == {
        "groups.a.detail_size": '"off"',
        "groups.b.detail_size": '"1344x1728"',
        "groups.a.ca_above": "150",
    }
    rows = [
        (json.loads(r["before"]), json.loads(r["after"]))
        for r in conn.execute("SELECT before, after FROM settings_history ORDER BY id")
    ]
    assert rows == [
        ({"groups.a.detail_size": None}, {"groups.a.detail_size": "off"}),
        (
            {"groups.a.ca_above": None, "groups.b.detail_size": "off"},
            {"groups.a.ca_above": 150, "groups.b.detail_size": "1344x1728"},
        ),
        ({"staff.enabled": None}, {"staff.enabled": False}),
    ]
    assert db.rename_settings(conn) == 0  # done once: nothing left


def test_an_old_key_written_again_by_an_older_facegen_wins(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('groups.a.detail_size', '\"1152x1472\"', 'a')"
    )
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('groups.a.detail_pass', 'false', 'b')"
    )
    assert db.rename_settings(conn) == 1
    assert conn.execute("SELECT key, value, updated_at FROM settings").fetchall()[0][:] == (
        "groups.a.detail_size",
        '"off"',
        "b",
    )


def test_with_nothing_to_rename_a_connection_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "facegen.db"
    db.connect(path).close()
    reader = db.connect(path)
    writer = sqlite3.connect(path, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")  # a sync holding the write lock
    try:
        quick = db.connect(path, busy_timeout_ms=100)  # would time out if it wanted to write
        quick.close()
    finally:
        writer.execute("ROLLBACK")
        writer.close()
        reader.close()
