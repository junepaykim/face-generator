from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.platform.locking import file_lock

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
    images_after = [tuple(r) for r in conn.execute("SELECT * FROM images ORDER BY id")]
    assert [r[:-2] for r in images_after] == images_before
    assert [r[-2] for r in images_after] == ["woman", "woman"]  # gender (migration 13)
    assert [r[-1] for r in images_after] == ["3:4", "3:4"]  # aspect (migration 14)
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


def test_an_older_database_is_migrated_by_one_connection_at_a_time(tmp_path: Path) -> None:
    path = tmp_path / "facegen.db"
    with closing(sqlite3.connect(path, isolation_level=None)) as old:
        for script in db.MIGRATIONS[:13]:
            old.executescript(script)
        old.execute("PRAGMA user_version = 13")
    results: list[object] = []

    def open_it() -> None:
        try:
            with closing(db.connect(path)) as conn:
                results.append(db.schema_version(conn))
        except Exception as exc:  # noqa: BLE001 - the test reports it
            results.append(exc)

    # Two of the app's processes open it together after an update: both wait for the lock,
    # and the second finds the first has migrated it, rather than adding the same columns.
    with file_lock(db.migration_lock(path)):
        threads = [threading.Thread(target=open_it) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(0.3)
        assert all(t.is_alive() for t in threads) and results == [], results
    for t in threads:
        t.join(20)
    assert results == [len(db.MIGRATIONS)] * 2


def test_an_up_to_date_database_opens_without_the_migration_lock(tmp_path: Path) -> None:
    path = tmp_path / "facegen.db"
    db.connect(path).close()
    db.migration_lock(path).unlink()
    with file_lock(db.migration_lock(path)):  # held elsewhere: no wait for it
        db.connect(path).close()


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
    assert db.migrate(conn) == len(db.MIGRATIONS)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(players)")}
    assert "drawn_male" not in columns and "female" in columns
    bands = dict(conn.execute("SELECT unique_id, target_band FROM jobs").fetchall())
    assert bands == {7: None, 8: "W05", 9: "M3"}  # finished jobs keep their record


def test_migration_12_adds_the_nation_empty_until_a_sync_reads_it() -> None:
    """Phase 15f: the nation's database ID, which picks a man's look."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    for script in db.MIGRATIONS[:11]:
        conn.executescript(script)
    conn.execute("PRAGMA user_version = 11")
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'c1', 'Career', '[]', 'now', 'now')"
    )
    assert db.migrate(conn) == len(db.MIGRATIONS)
    column = {r["name"]: r for r in conn.execute("PRAGMA table_info(players)")}["nation"]
    assert column["type"] == "INTEGER" and not column["notnull"] and column["dflt_value"] is None


def test_migration_13_records_who_each_picture_shows() -> None:
    """Phase 15g: pictures drawn with the 2026-11-13 men's wordings show men, every other
    earlier picture a woman; a sheet version keeps no men's sheet until drawn as a man."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    for script in db.MIGRATIONS[:12]:
        conn.executescript(script)
    conn.execute("PRAGMA user_version = 12")
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'c1', 'Career', '[]', 'now', 'now')"
    )
    for version, band in enumerate(("W05", "M2", None, "M4", "W10"), start=1):
        conn.execute(
            "INSERT INTO images (career_id, unique_id, version, reason, group_name, model,"
            " prompt, negative_prompt, seed, age_band, created_at)"
            " VALUES (1, 7, ?, 'initial', 'a', 'm', 'p', 'n', 1, ?, 'now')",
            (version, band),
        )
    conn.execute(
        "INSERT INTO traits (career_id, unique_id, version, base_seed, sheet, created_at)"
        " VALUES (1, 7, 1, 5, '{}', 'now')"
    )
    assert db.migrate(conn) == len(db.MIGRATIONS)
    genders = [r[0] for r in conn.execute("SELECT gender FROM images ORDER BY version")]
    assert genders == ["woman", "man", "woman", "man", "woman"]
    assert conn.execute("SELECT man_sheet FROM traits").fetchone()[0] is None
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE images SET gender = 'other'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE images SET gender = NULL")


def _at_version(version: int) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    for script in db.MIGRATIONS[:version]:
        conn.executescript(script)
    conn.execute(f"PRAGMA user_version = {version}")
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'c1', 'Career', '[]', 'now', 'now')"
    )
    return conn


def test_migration_15_keeps_a_database_with_pictures_drawing_3_by_4() -> None:
    """Pictures are 1:1 by default since 2026-11-22 (user); a database that already holds
    pictures (this PC's) keeps the old default as its own setting, unless it has one."""
    fresh = _at_version(14)
    assert db.migrate(fresh) == len(db.MIGRATIONS)
    assert load_settings(fresh).picture.aspect == "1:1"
    assert fresh.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0

    drawn = _at_version(14)
    drawn.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model,"
        " prompt, negative_prompt, seed, created_at)"
        " VALUES (1, 7, 1, 'initial', 'a', 'm', 'p', 'n', 1, 'now')"
    )
    assert db.migrate(drawn) == len(db.MIGRATIONS)
    s = load_settings(drawn)
    assert s.picture.aspect == "3:4" and s.overridden == {"picture.aspect"}

    chosen = _at_version(14)
    chosen.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model,"
        " prompt, negative_prompt, seed, created_at)"
        " VALUES (1, 7, 1, 'initial', 'a', 'm', 'p', 'n', 1, 'now')"
    )
    chosen.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('picture.aspect', '\"1:1\"', 'x')"
    )
    db.migrate(chosen)
    assert load_settings(chosen).picture.aspect == "1:1"  # a choice already made stays


def _history(conn: sqlite3.Connection, before: dict, after: dict) -> None:
    conn.execute(
        "INSERT INTO settings_history (at, label, before, after) VALUES ('now', 'edit', ?, ?)",
        (json.dumps(before), json.dumps(after)),
    )


def test_the_detail_pass_switch_is_renamed_to_its_size_at_connect(tmp_path: Path) -> None:
    """Phase 10c: a stored detail pass off is "off", on the size it drew at (named by scale
    since Phase 16: "x1.5"); the settings history follows, so an older change can still be
    undone. No schema number changes, so an older facegen on the same database still opens it."""
    path = tmp_path / "facegen.db"
    conn = db.connect(path)
    conn.executemany(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, 'then')",
        [
            ("groups.a.detail_pass", "false"),
            ("groups.b.detail_pass", "true"),
            ("groups.c.detail_pass", "7"),  # not a switch: dropped, the default applies
            ("staff.enabled", "false"),
        ],
    )
    _history(conn, {"groups.a.detail_pass": None}, {"groups.a.detail_pass": False})
    _history(
        conn,
        {"groups.b.detail_pass": False, "staff.enabled": None},
        {"groups.b.detail_pass": True, "staff.enabled": False},
    )
    _history(conn, {"media.journalists": None}, {"media.journalists": False})
    conn.close()

    conn = db.connect(path)
    assert db.schema_version(conn) == len(db.MIGRATIONS)
    stored = dict(conn.execute("SELECT key, value FROM settings").fetchall())
    assert stored == {
        "groups.a.detail_size": '"off"',
        "groups.b.detail_size": '"x1.5"',
        "staff.enabled": "false",
    }
    rows = [
        (json.loads(r["before"]), json.loads(r["after"]))
        for r in conn.execute("SELECT before, after FROM settings_history ORDER BY id")
    ]
    assert rows == [
        ({"groups.a.detail_size": None}, {"groups.a.detail_size": "off"}),
        (
            {"staff.enabled": None, "groups.b.detail_size": "off"},
            {"staff.enabled": False, "groups.b.detail_size": "x1.5"},
        ),
        ({"media.journalists": None}, {"media.journalists": False}),
    ]
    assert db.rename_settings(conn) == 0  # done once: nothing left


def test_the_old_cutoffs_become_the_new_keys_at_connect(tmp_path: Path) -> None:
    """Phase 16: an old "strictly above" N is "at least" N+1, so the same people stay picked;
    group A's and B's "any club" numbers become the global cuts, the league numbers the manual
    set; `youth.enabled` and `groups.b.my_academy` move to `cutoffs.my_club`, the detail sizes
    are named by scale, and the one switch for the leagues above and below becomes the league
    below's (the league above's is on by default). The settings history follows, so Undo still
    works across it."""
    path = tmp_path / "facegen.db"
    conn = db.connect(path)
    old = {
        "groups.a.ca_above": 150,
        "groups.a.pa_above": 200,  # above 200 picks nobody: switched off
        "groups.b.any_ca_above": 135,
        "groups.b.any_pa_above": 0,
        "groups.b.league_ca_above": 120,
        "groups.b.league_pa_above": 145,
        "groups.c.league_ca_above": 100,
        "groups.c.league_pa_above": 130,
        "groups.b.my_academy": False,
        "youth.enabled": False,
        "groups.a.detail_size": "1152x1472",
        "groups.b.detail_size": "x1.5",  # already new: left as it is
        "cutoffs.adjacent_light": True,  # one switch for both, before 2026-11-22
    }
    conn.executemany(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, 'then')",
        [(k, json.dumps(v)) for k, v in old.items()],
    )
    _history(conn, {"groups.a.ca_above": None, "youth.enabled": None}, {
        "groups.a.ca_above": 150, "youth.enabled": False,
    })  # fmt: skip
    _history(conn, {"groups.a.detail_size": "1344x1728"}, {"groups.a.detail_size": "1152x1472"})
    conn.close()

    conn = db.connect(path)
    stored = {k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM settings")}
    assert stored == {
        "cutoffs.everywhere.a_ca": 151,
        "cutoffs.everywhere.a_pa": "-",
        "cutoffs.everywhere.b_ca": 136,
        "cutoffs.everywhere.b_pa": 1,
        "cutoffs.manual.b_ca": 121,
        "cutoffs.manual.b_pa": 146,
        "cutoffs.manual.c_ca": 101,
        "cutoffs.manual.c_pa": 131,
        "cutoffs.my_club.academy_alumni": False,
        "cutoffs.my_club.youth": "a",
        "groups.a.detail_size": "smaller",
        "groups.b.detail_size": "x1.5",
        "cutoffs.light_below": True,  # the league above's is on by default
    }
    rows = [
        (json.loads(r["before"]), json.loads(r["after"]))
        for r in conn.execute("SELECT before, after FROM settings_history ORDER BY id")
    ]
    assert rows == [
        (
            {"cutoffs.everywhere.a_ca": None, "cutoffs.my_club.youth": None},
            {"cutoffs.everywhere.a_ca": 151, "cutoffs.my_club.youth": "a"},
        ),
        ({"groups.a.detail_size": "x1.5"}, {"groups.a.detail_size": "smaller"}),
    ]
    s = load_settings(conn)
    assert s.cutoffs.mode == "auto"  # the mode starts on Auto
    assert s.cutoffs.everywhere["a_ca"] == 151 and s.cutoffs.manual["b_ca"] == 121
    assert s.cutoffs.my_club.youth == "a" and s.groups.a.detail_size() == (1152, 1472)
    assert s.cutoffs.light_above and s.cutoffs.light_below
    assert db.rename_settings(conn) == 0


def test_an_old_key_written_again_by_an_older_facegen_wins(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('groups.a.detail_size', '\"smaller\"', 'a')"
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
