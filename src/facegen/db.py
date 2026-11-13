"""SQLite state database: schema, migrations and transactions.

The connection runs in autocommit mode; group writes with `transaction`. Migrations are
numbered by position in MIGRATIONS and recorded in PRAGMA user_version.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from facegen.domain import ACTIVE_JOB_STATUSES, JOB_KINDS, JOB_STATUSES


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

# Phase 2: the render settings of each picture (steps, CFG, sampler, size, detail pass), so
# any picture can be reproduced from its row. Paths in images are relative to the data dir.
_V2 = """
ALTER TABLE images ADD COLUMN render TEXT;     -- JSON
"""

# Phase 3: the web UI's player list and detail page look up each player's jobs, the latest
# first (queued, running, or failed and waiting in the review list).
_V3 = """
CREATE INDEX jobs_by_player ON jobs (career_id, unique_id, id);
"""

# Phase 4: automation.
# - careers.last_save_mtime: the newest save of the career a sync has read. The career with
#   the newest is the active one, whose picture pack FM27 is shown (spec section 8).
# - sync_runs.trigger: who started the sync ('poller', 'sync now', 'command').
# - sync_runs.unsupported: the save's game or build cannot be read by the save reader.
_V4 = """
ALTER TABLE careers ADD COLUMN last_save_mtime REAL;
ALTER TABLE sync_runs ADD COLUMN trigger TEXT;
ALTER TABLE sync_runs ADD COLUMN unsupported INTEGER NOT NULL DEFAULT 0;
UPDATE careers SET last_save_mtime = (
    SELECT MAX(s.save_mtime) FROM sync_runs s WHERE s.career_id = careers.id AND s.error IS NULL
);
CREATE INDEX sync_runs_by_save ON sync_runs (save_file, save_mtime);
"""

# Phase 5: updates for tracked players.
# - images.restored_*: "restore previous" on the player page makes an older picture current
#   again. It then stands for the player's group, club and age band at the time it was
#   restored, so the planner does not re-draw it straight away; the next change does.
# - Pictures are 3:4 portraits from now on (user, 2026-11-11). Every picture so far is square:
#   forget where they were published, so the next publish writes them again, cropped to 3:4.
_V5 = """
ALTER TABLE images ADD COLUMN restored_at TEXT;
ALTER TABLE images ADD COLUMN restored_group TEXT CHECK (restored_group IN ('a', 'b', 'c'));
ALTER TABLE images ADD COLUMN restored_club_id INTEGER;
ALTER TABLE images ADD COLUMN restored_band TEXT;
UPDATE images SET published_path = NULL;
"""

# State coordination: explicit sheet selection, resumable jobs, and TEXT wording keys.
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

# Phase 6: personality weights the character sheet (spec section 4). The player's eight
# personality attributes (1 to 20) as JSON; NULL until the next sync, or when the save has none.
_V7 = """
ALTER TABLE players ADD COLUMN personality TEXT;
"""

# My club's staff and the settings page's undo (user, 2026-11-12).
# - players.is_staff, players.staff_job: my club's staff are stored beside the players, keyed
#   on the same unique ID (FM27's config.xml maps both alike), so they are queued, drawn and
#   published by the same code. staff_job is the job in words, NULL where the save has none.
# - settings_history: each change saved on the settings page, undoable. `before` and `after`
#   map each key the change touched to its stored override, or null for none (the default).
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

MIGRATIONS: tuple[str, ...] = (_V1, _V2, _V3, _V4, _V5, _V6, _V7, _V8)

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
