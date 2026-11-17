"""SQLite state database: schema, migrations and transactions.

The connection runs in autocommit mode; group writes with `transaction`. Migrations are
numbered by position in MIGRATIONS and recorded in PRAGMA user_version. A renamed setting is
put right at every connect instead (`rename_settings`), so an older facegen sharing the
database can still open it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from facegen.core.domain import ACTIVE_JOB_STATUSES, JOB_KINDS, JOB_STATUSES


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


_V1 = f"""
CREATE TABLE careers (
    id               INTEGER PRIMARY KEY,
    key              TEXT NOT NULL UNIQUE,     -- folder name for this career's picture pack
    display_name     TEXT NOT NULL,
    fingerprint      TEXT NOT NULL,            -- JSON list of hashes of created people
    newgen_min_id    INTEGER,                  -- unique IDs at or above this are newgens
    newgen_id_source TEXT,                     -- 'detected' or 'fallback'
    managed_club_id  INTEGER,
    last_game_date   TEXT,
    last_save_file   TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE clubs (
    career_id    INTEGER NOT NULL REFERENCES careers(id),
    club_id      INTEGER NOT NULL,             -- the game's unique ID for the club
    name         TEXT NOT NULL,
    gender       TEXT NOT NULL,                -- 'men', 'women' or 'unknown'
    nation_id    INTEGER,
    in_my_league INTEGER NOT NULL DEFAULT 0,
    continental  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (career_id, club_id)
);

CREATE TABLE players (
    career_id           INTEGER NOT NULL REFERENCES careers(id),
    unique_id           INTEGER NOT NULL,
    name                TEXT NOT NULL,
    birth_date          TEXT,
    age                 INTEGER,
    height_cm           INTEGER,
    ca                  INTEGER NOT NULL,
    pa                  INTEGER,
    is_goalkeeper       INTEGER NOT NULL DEFAULT 0,
    strength            INTEGER,
    club_id             INTEGER,
    loan_parent_club_id INTEGER,
    home_grown_club_ids TEXT NOT NULL DEFAULT '[]',
    club_since          TEXT,      -- club join date, else the in-game date first seen there
    is_newgen           INTEGER NOT NULL,
    in_scope            INTEGER NOT NULL,  -- men's football, or a free agent last seen there
    at_my_club          INTEGER NOT NULL,
    in_my_league        INTEGER NOT NULL,
    in_continental      INTEGER NOT NULL,
    first_club_id       INTEGER,   -- club when first seen; NULL if seen in the first sync
    academy             INTEGER NOT NULL,
    present             INTEGER NOT NULL DEFAULT 1,
    first_seen_sync     INTEGER,
    last_seen_sync      INTEGER,
    group_name          TEXT CHECK (group_name IN ('a', 'b', 'c')),
    tracked             INTEGER NOT NULL DEFAULT 0,
    reasons             TEXT NOT NULL DEFAULT '[]',
    pin                 TEXT CHECK (pin IN ('a', 'b', 'c', 'never')),
    PRIMARY KEY (career_id, unique_id)
);
CREATE INDEX players_by_group ON players (career_id, group_name);

CREATE TABLE traits (
    career_id  INTEGER NOT NULL REFERENCES careers(id),
    unique_id  INTEGER NOT NULL,
    version    INTEGER NOT NULL,
    base_seed  INTEGER NOT NULL,
    sheet      TEXT NOT NULL,              -- JSON character sheet
    created_at TEXT NOT NULL,
    PRIMARY KEY (career_id, unique_id, version)
);

CREATE TABLE images (
    id              INTEGER PRIMARY KEY,
    career_id       INTEGER NOT NULL REFERENCES careers(id),
    unique_id       INTEGER NOT NULL,
    version         INTEGER NOT NULL,
    reason          TEXT NOT NULL CHECK (reason IN {_in(JOB_KINDS)}),
    group_name      TEXT NOT NULL CHECK (group_name IN ('a', 'b', 'c')),
    club_id         INTEGER,
    age_band        INTEGER,
    traits_version  INTEGER,
    model           TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    master_path     TEXT,
    published_path  TEXT,
    check_result    TEXT,                  -- JSON
    is_current      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    UNIQUE (career_id, unique_id, version)
);
CREATE UNIQUE INDEX images_one_current ON images (career_id, unique_id) WHERE is_current = 1;

CREATE TABLE jobs (
    id             INTEGER PRIMARY KEY,
    career_id      INTEGER NOT NULL REFERENCES careers(id),
    unique_id      INTEGER NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN {_in(JOB_KINDS)}),
    group_name     TEXT NOT NULL CHECK (group_name IN ('a', 'b', 'c')),
    manual         INTEGER NOT NULL DEFAULT 0,
    tier           INTEGER NOT NULL,       -- queue order: lower first
    rank           REAL NOT NULL,          -- order within a tier: lower first
    status         TEXT NOT NULL CHECK (status IN {_in(JOB_STATUSES)}),
    target_club_id INTEGER,
    target_band    INTEGER,
    reason         TEXT NOT NULL,
    attempts       INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
-- At most one queued or running job per player: re-running a sync cannot duplicate work.
CREATE UNIQUE INDEX jobs_one_active ON jobs (career_id, unique_id)
    WHERE status IN {_in(ACTIVE_JOB_STATUSES)};
CREATE INDEX jobs_queue ON jobs (status, tier, rank, id);

CREATE TABLE settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,              -- JSON
    updated_at TEXT NOT NULL
);

CREATE TABLE sync_runs (
    id          INTEGER PRIMARY KEY,
    career_id   INTEGER REFERENCES careers(id),
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    save_file   TEXT NOT NULL,
    save_mtime  REAL,
    build       TEXT,
    game_date   TEXT,
    counts      TEXT,                      -- JSON
    error       TEXT
);
"""

# Store render settings (steps, CFG, sampler, size, detail pass) to reproduce each picture.
# Image paths are relative to the data directory.
_V2 = """
ALTER TABLE images ADD COLUMN render TEXT;     -- JSON
"""

_V3 = """
CREATE INDEX jobs_by_player ON jobs (career_id, unique_id, id);
"""

# last_save_mtime: newest save read for this career; the newest career's pack is active.
# trigger: 'poller', 'sync now' or 'command'.
# unsupported: the reader cannot read the save's game or build.
_V4 = """
ALTER TABLE careers ADD COLUMN last_save_mtime REAL;
ALTER TABLE sync_runs ADD COLUMN trigger TEXT;
ALTER TABLE sync_runs ADD COLUMN unsupported INTEGER NOT NULL DEFAULT 0;
UPDATE careers SET last_save_mtime = (
    SELECT MAX(s.save_mtime) FROM sync_runs s WHERE s.career_id = careers.id AND s.error IS NULL
);
CREATE INDEX sync_runs_by_save ON sync_runs (save_file, save_mtime);
"""

# restored_* records the player's group, club and age band at restoration, preventing an
# immediate redraw; only a later change makes the restored picture outdated.
# Pre-V5 pictures were square. Clearing published_path republishes them cropped to 3:4.
_V5 = """
ALTER TABLE images ADD COLUMN restored_at TEXT;
ALTER TABLE images ADD COLUMN restored_group TEXT CHECK (restored_group IN ('a', 'b', 'c'));
ALTER TABLE images ADD COLUMN restored_club_id INTEGER;
ALTER TABLE images ADD COLUMN restored_band TEXT;
UPDATE images SET published_path = NULL;
"""

_V6 = """
CREATE TABLE images_new (
    id              INTEGER PRIMARY KEY,
    career_id       INTEGER NOT NULL REFERENCES careers(id),
    unique_id       INTEGER NOT NULL,
    version         INTEGER NOT NULL,
    reason          TEXT NOT NULL CHECK (reason IN ('initial', 'kit', 'age', 'upgrade', 'reroll', 'redraw')),
    group_name      TEXT NOT NULL CHECK (group_name IN ('a', 'b', 'c')),
    club_id         INTEGER,
    age_band        TEXT,
    traits_version  INTEGER,
    model           TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    master_path     TEXT,
    published_path  TEXT,
    check_result    TEXT,                  -- JSON
    is_current      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    render          TEXT,
    restored_at     TEXT,
    restored_group  TEXT CHECK (restored_group IN ('a', 'b', 'c')),
    restored_club_id INTEGER,
    restored_band   TEXT,
    UNIQUE (career_id, unique_id, version)
);
INSERT INTO images_new (
    id, career_id, unique_id, version, reason, group_name, club_id, age_band,
    traits_version, model, prompt, negative_prompt, seed, master_path, published_path,
    check_result, is_current, created_at, render, restored_at, restored_group,
    restored_club_id, restored_band
) SELECT
    id, career_id, unique_id, version, reason, group_name, club_id, age_band,
    traits_version, model, prompt, negative_prompt, seed, master_path, published_path,
    check_result, is_current, created_at, render, restored_at, restored_group,
    restored_club_id, restored_band
FROM images;
DROP TABLE images;
ALTER TABLE images_new RENAME TO images;
CREATE UNIQUE INDEX images_one_current ON images (career_id, unique_id) WHERE is_current = 1;
CREATE TABLE jobs_new (
    id             INTEGER PRIMARY KEY,
    career_id      INTEGER NOT NULL REFERENCES careers(id),
    unique_id      INTEGER NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('initial', 'kit', 'age', 'upgrade', 'reroll', 'redraw')),
    group_name     TEXT NOT NULL CHECK (group_name IN ('a', 'b', 'c')),
    manual         INTEGER NOT NULL DEFAULT 0,
    tier           INTEGER NOT NULL,       -- queue order: lower first
    rank           REAL NOT NULL,          -- order within a tier: lower first
    status         TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    target_club_id INTEGER,
    target_band    TEXT,
    reason         TEXT NOT NULL,
    attempts       INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
INSERT INTO jobs_new (
    id, career_id, unique_id, kind, group_name, manual, tier, rank, status,
    target_club_id, target_band, reason, attempts, error, created_at, updated_at
) SELECT
    id, career_id, unique_id, kind, group_name, manual, tier, rank, status,
    target_club_id, target_band, reason, attempts, error, created_at, updated_at
FROM jobs;
DROP TABLE jobs;
ALTER TABLE jobs_new RENAME TO jobs;
CREATE UNIQUE INDEX jobs_one_active ON jobs (career_id, unique_id)
    WHERE status IN ('queued', 'running');
CREATE INDEX jobs_queue ON jobs (status, tier, rank, id);
CREATE INDEX jobs_by_player ON jobs (career_id, unique_id, id);
ALTER TABLE jobs ADD COLUMN traits_version INTEGER;
ALTER TABLE jobs ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE players ADD COLUMN active_traits_version INTEGER;
UPDATE players SET active_traits_version = COALESCE(
    (SELECT traits_version FROM images WHERE images.career_id = players.career_id
     AND images.unique_id = players.unique_id AND is_current = 1),
    (SELECT MAX(version) FROM traits WHERE traits.career_id = players.career_id
     AND traits.unique_id = players.unique_id)
);
"""

# Eight personality attributes (1 to 20), stored as JSON to weight character sheets.
# NULL until the next sync, or when the save has none.
_V7 = """
ALTER TABLE players ADD COLUMN personality TEXT;
"""

# Staff share player IDs and portrait mappings, so they use the same generation queue.
# staff_job is the job in words, NULL when absent from the save.
# History's before/after maps each changed key to its stored override, or null for none.
_V8 = """
ALTER TABLE players ADD COLUMN is_staff INTEGER NOT NULL DEFAULT 0;
ALTER TABLE players ADD COLUMN staff_job TEXT;
CREATE TABLE settings_history (
    id      INTEGER PRIMARY KEY,
    at      TEXT NOT NULL,
    label   TEXT NOT NULL,                 -- 'edit' or 'defaults'
    before  TEXT NOT NULL,                 -- JSON {key: override or null}
    after   TEXT NOT NULL,                 -- JSON {key: override or null}
    report  TEXT,                          -- JSON: what the change did to groups and queue
    undone  INTEGER NOT NULL DEFAULT 0
);
"""

# photos is keyed by unique ID across careers: FM27's photos and facepacks share that ID.
# photo_scan holds the last inventory (to skip unchanged inputs), catalog readability,
# and dashboard counts. Each photos row identifies a game or external facepack portrait.
_V9 = """
CREATE TABLE photos (
    unique_id INTEGER PRIMARY KEY,
    game      INTEGER NOT NULL DEFAULT 0,  -- FM27 ships a photo (its Addressables catalog)
    facepack  INTEGER NOT NULL DEFAULT 0   -- a config.xml outside graphics/facegen/ maps one
) WITHOUT ROWID;
CREATE TABLE photo_scan (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    state          TEXT NOT NULL,          -- JSON: catalog version, config.xml files, folders
    game_known     INTEGER NOT NULL,       -- 0: the catalog was unreadable; nobody counts as
                                           -- without a photo
    game           INTEGER NOT NULL,       -- how many IDs FM27 ships a photo for
    facepack       INTEGER NOT NULL,       -- how many IDs facepacks map
    facepack_files INTEGER NOT NULL,       -- how many config.xml files were read
    warnings       TEXT NOT NULL DEFAULT '[]',
    scanned_at     TEXT NOT NULL,
    changed_at     TEXT NOT NULL           -- when the IDs were last read again
);
"""

# youth_team: my club's reserves/under-23s/under-18s, group B unless they qualify for A.
# media_role: 'journalist' or 'press officer', NULL for players and staff.
# local_media: a journalist from my club's country or its press officer, drawn automatically.
# female: inferred from first name; NULL when unknown (players are not read).
# drawn_male: historical male rendering for staff/media, removed in _V11.
_V10 = """
ALTER TABLE players ADD COLUMN youth_team INTEGER NOT NULL DEFAULT 0;
ALTER TABLE players ADD COLUMN media_role TEXT CHECK (media_role IN ('journalist', 'press officer'));
ALTER TABLE players ADD COLUMN local_media INTEGER NOT NULL DEFAULT 0;
ALTER TABLE players ADD COLUMN female INTEGER;
ALTER TABLE players ADD COLUMN drawn_male INTEGER NOT NULL DEFAULT 0;
"""

# Remove historical male rendering. Clear queued male age targets (M1-M4) so the next plan
# chooses a female wording; current male pictures receive a new face through the planner.
_V11 = """
ALTER TABLE players DROP COLUMN drawn_male;
UPDATE jobs SET target_band = NULL
    WHERE status IN ('queued', 'running') AND target_band IN ('M1', 'M2', 'M3', 'M4');
"""


MIGRATIONS: tuple[str, ...] = (_V1, _V2, _V3, _V4, _V5, _V6, _V7, _V8, _V9, _V10, _V11)

# The active career: the one whose save was written last (spec section 8). A career synced
# without a save time (tests, old rows) falls back to when it was synced.
ACTIVE_CAREER_ORDER = "last_save_mtime IS NULL, last_save_mtime DESC, updated_at DESC, id DESC"


def active_career_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(f"SELECT id FROM careers ORDER BY {ACTIVE_CAREER_ORDER} LIMIT 1").fetchone()
    return row[0] if row else None


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations, each in its own transaction. Returns the new version."""
    version = schema_version(conn)
    if version > len(MIGRATIONS):
        raise RuntimeError(
            f"database schema {version} is newer than this code knows ({len(MIGRATIONS)})"
        )
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
    return schema_version(conn)


# Rename settings at every connect without raising the schema version: older installations
# may share this database. They warn and skip unknown keys but reject newer schemas.
RENAMED_SETTINGS = {
    f"groups.{g}.detail_pass": (f"groups.{g}.detail_size", {True: "1344x1728", False: "off"})
    for g in ("a", "b", "c")
}
"""Old key to (new key, value map). The old detail-pass switch used 1344x1728 when on."""


def rename_settings(conn: sqlite3.Connection) -> int:
    """Rename old setting keys in the overrides and in the settings history; return how many
    overrides changed. A quick read first: with nothing to rename, nothing is written, so a
    connection takes no write lock for it. An old key written again later (by an older facegen)
    replaces the new one, as the later change."""
    old_keys = list(RENAMED_SETTINGS)
    marks = " OR ".join("before LIKE ? OR after LIKE ?" for _ in old_keys)
    likes = [f'%"{k}"%' for k in old_keys for _ in (0, 1)]
    found = conn.execute(
        f"SELECT 1 FROM settings WHERE key IN ({', '.join('?' * len(old_keys))})"
        f" UNION ALL SELECT 1 FROM settings_history WHERE {marks} LIMIT 1",
        [*old_keys, *likes],
    ).fetchone()
    if found is None:
        return 0
    changed = 0
    with transaction(conn):
        for old_key, (new_key, values) in RENAMED_SETTINGS.items():
            row = conn.execute(
                "SELECT value, updated_at FROM settings WHERE key = ?", (old_key,)
            ).fetchone()
            if row is not None:
                old_value = json.loads(row[0])
                conn.execute("DELETE FROM settings WHERE key = ?", (old_key,))
                if isinstance(old_value, bool):
                    conn.execute(
                        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)"
                        " ON CONFLICT (key) DO UPDATE SET value = excluded.value,"
                        " updated_at = excluded.updated_at",
                        (new_key, json.dumps(values[old_value]), row[1]),
                    )
                changed += 1
            for entry in conn.execute(
                "SELECT id, before, after FROM settings_history WHERE before LIKE ? OR after LIKE ?",
                (f'%"{old_key}"%', f'%"{old_key}"%'),
            ).fetchall():
                sides = []
                for text in (entry[1], entry[2]):
                    side = json.loads(text)
                    if old_key in side:
                        value = side.pop(old_key)
                        side[new_key] = values.get(value) if isinstance(value, bool) else None
                    sides.append(json.dumps(side))
                conn.execute(
                    "UPDATE settings_history SET before = ?, after = ? WHERE id = ?",
                    (*sides, entry[0]),
                )
    return changed


def connect(path: Path | str, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open (creating if needed) and migrate the state database.

    `facegen serve` uses a longer busy timeout: its worker, its syncs and the web pages
    share the database, and a sync holds the write lock for a second or two.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    rename_settings(conn)
    return conn


@contextmanager
def transaction(
    conn: sqlite3.Connection, *, rollback: bool = False
) -> Iterator[sqlite3.Connection]:
    """Run a block in one write transaction. With rollback=True nothing is kept (dry runs)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("ROLLBACK" if rollback else "COMMIT")
