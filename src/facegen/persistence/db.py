"""SQLite state database: schema, migrations and transactions.

The connection runs in autocommit mode; group writes with `transaction`. Migrations are
numbered by position in MIGRATIONS and recorded in PRAGMA user_version. A renamed setting is
put right at every connect instead (`rename_settings`), so an older facegen sharing the
database can still open it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from facegen.core.domain import ACTIVE_JOB_STATUSES, JOB_KINDS, JOB_STATUSES
from facegen.platform.locking import file_lock


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

# nation: the game's editor database ID of the person's nation (core/nations.py), which picks
# a man's look (Phase 15f); NULL until a sync reads it, or where the save does not give it.
_V12 = """
ALTER TABLE players ADD COLUMN nation INTEGER;
"""


# images.gender: who a picture shows, "woman" or "man" (`style.gender`, Phase 15g). Pictures
# drawn with the 2026-11-13 men's wordings (M1-M4) show men; every other earlier picture a woman.
# traits.man_sheet: the men's sheet (core/man_traits.py) of the same version, JSON, stored the
# first time that version is drawn as a man; NULL until then.
_V13 = """
ALTER TABLE images ADD COLUMN gender TEXT NOT NULL DEFAULT 'woman' CHECK (gender IN ('woman', 'man'));
UPDATE images SET gender = 'man' WHERE age_band IN ('M1', 'M2', 'M3', 'M4');
ALTER TABLE traits ADD COLUMN man_sheet TEXT;
"""


# Phase 16: league-share cutoffs and picture shapes.
# clubs.reputation: the club's reputation (1-10000) as last read. clubs.league_id: the league on
# my pyramid's ladder the club belongs to (core/leagues.py); league_offset: how many levels that
# league is above mine (1 the league above, -1 below, 0 mine or a parallel division); NULL off
# the ladder.
# league_stats: per career and scope ('league', 'above', 'below', 'continental'), the
# competitions found and the candidates' CA and PA histograms (JSON lists of 201 counts),
# newgens and photo-less players apart, as the last sync read them.
# images.aspect: the shape a picture was drawn in. Every earlier picture is 3:4, including the
# pre-V5 squares already published as 3:4 crops.
_V14 = """
ALTER TABLE clubs ADD COLUMN reputation INTEGER;
ALTER TABLE clubs ADD COLUMN league_id INTEGER;
ALTER TABLE clubs ADD COLUMN league_offset INTEGER;
CREATE TABLE league_stats (
    career_id    INTEGER NOT NULL REFERENCES careers(id),
    scope        TEXT NOT NULL CHECK (scope IN ('league', 'above', 'below', 'continental')),
    competitions TEXT NOT NULL,            -- JSON list of competition IDs
    newgen_ca    TEXT NOT NULL,            -- JSON list: candidates per value 0-200
    newgen_pa    TEXT NOT NULL,
    no_photo_ca  TEXT NOT NULL,
    no_photo_pa  TEXT NOT NULL,
    game_date    TEXT,
    PRIMARY KEY (career_id, scope)
);
ALTER TABLE images ADD COLUMN aspect TEXT NOT NULL DEFAULT '3:4' CHECK (aspect IN ('3:4', '1:1'));
"""


# Phase 16: pictures are 1:1 by default (`picture.aspect`, user 2026-11-22). A database that
# already holds pictures keeps drawing 3:4, so one career's pictures keep one shape: it gets the
# old default as its own setting, unless it has one.
_V15 = """
INSERT INTO settings (key, value, updated_at)
SELECT 'picture.aspect', '"3:4"', strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')
WHERE EXISTS (SELECT 1 FROM images)
  AND NOT EXISTS (SELECT 1 FROM settings WHERE key = 'picture.aspect');
"""


# Phase 17h: players.why, the reasons as codes and values (core/classify.py `Why`, JSON list),
# which the pages word in their language. NULL until the next classification fills it; until
# then the pages show the English `reasons`. players_pinned: the few pinned people, for the
# Players page's picture targets (persistence/player_queries.py); being partial, no query that
# does not ask for pins can take it.
_V16 = """
ALTER TABLE players ADD COLUMN why TEXT;
CREATE INDEX players_pinned ON players (career_id, unique_id) WHERE pin IS NOT NULL;
"""


MIGRATIONS: tuple[str, ...] = (
    _V1, _V2, _V3, _V4, _V5, _V6, _V7, _V8, _V9, _V10, _V11, _V12, _V13, _V14, _V15, _V16
)  # fmt: skip

# The active career: the one whose save was written last (spec section 8). A career synced
# without a save time (tests, old rows) falls back to when it was synced.
ACTIVE_CAREER_ORDER = "last_save_mtime IS NULL, last_save_mtime DESC, updated_at DESC, id DESC"


def active_career_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(f"SELECT id FROM careers ORDER BY {ACTIVE_CAREER_ORDER} LIMIT 1").fetchone()
    return row[0] if row else None


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations, each in its own transaction. Returns the new version.

    It reads the version first and does not lock it: `connect` holds `migration_lock` for a
    database that other connections may be opening at the same time."""
    version = schema_version(conn)
    if version > len(MIGRATIONS):
        raise RuntimeError(
            f"database schema {version} is newer than this code knows ({len(MIGRATIONS)})"
        )
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
    return schema_version(conn)


# Rename or convert settings at every connect without raising the schema version: older
# installations may share this database. They warn and skip unknown keys but reject newer schemas.


def _at_least(value: Any) -> Any:
    """An old "strictly above" number N as "at least" N+1, so the same people stay picked
    (Phase 16); above 200 picks nobody, so 200 becomes "-"."""
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    return value + 1 if value < 200 else "-"


_OLD_DETAIL = {"1344x1728": "x1.5", "1152x1472": "smaller"}


@dataclass(frozen=True, slots=True)
class Conversion:
    """An old setting and the key and value it becomes."""

    new_key: str
    convert: Callable[[Any], Any]
    old_values: tuple[Any, ...] = ()
    """For a conversion that keeps its key: the old values that need converting."""


CONVERTED_SETTINGS: dict[str, Conversion] = {
    **{
        f"groups.{g}.detail_pass": Conversion(
            f"groups.{g}.detail_size", lambda v: {True: "x1.5", False: "off"}.get(v)
        )
        for g in ("a", "b", "c")
    },
    **{
        f"groups.{g}.detail_size": Conversion(
            f"groups.{g}.detail_size", lambda v: _OLD_DETAIL.get(v, v), tuple(_OLD_DETAIL)
        )
        for g in ("a", "b", "c")
    },
    "groups.a.ca_above": Conversion("cutoffs.everywhere.a_ca", _at_least),
    "groups.a.pa_above": Conversion("cutoffs.everywhere.a_pa", _at_least),
    "groups.b.any_ca_above": Conversion("cutoffs.everywhere.b_ca", _at_least),
    "groups.b.any_pa_above": Conversion("cutoffs.everywhere.b_pa", _at_least),
    "groups.b.league_ca_above": Conversion("cutoffs.manual.b_ca", _at_least),
    "groups.b.league_pa_above": Conversion("cutoffs.manual.b_pa", _at_least),
    "groups.c.league_ca_above": Conversion("cutoffs.manual.c_ca", _at_least),
    "groups.c.league_pa_above": Conversion("cutoffs.manual.c_pa", _at_least),
    "groups.b.my_academy": Conversion("cutoffs.my_club.academy_alumni", lambda v: v),
    "youth.enabled": Conversion(
        "cutoffs.my_club.youth", lambda v: {True: "b", False: "a"}.get(v, v)
    ),
    "cutoffs.adjacent_light": Conversion("cutoffs.light_below", lambda v: v),
}
"""Old key to what it becomes. The detail-pass switch used 1344x1728 when on (now "x1.5").
Phase 16: thresholds "strictly above" N became "at least" N+1, group A's and B's "any club"
numbers the global cuts, the league numbers the manual set; `youth.enabled` off put the youth
teams in A with the first team. The one switch for the leagues above and below became two
(user, 2026-11-22): the league above's is on by default, so the old switch carries over to the
league below's."""
RENAMED_SETTINGS = CONVERTED_SETTINGS
"""The earlier name of CONVERTED_SETTINGS."""


def _history_marks() -> tuple[str, list[str]]:
    """A WHERE clause finding history entries that hold an old key or an old value."""
    likes: list[str] = []
    for old_key, c in CONVERTED_SETTINGS.items():
        if c.new_key != old_key:
            likes.append(f'%"{old_key}"%')
        else:
            likes += [f'%"{old_key}": {json.dumps(v)}%' for v in c.old_values]
    clause = " OR ".join("before LIKE ? OR after LIKE ?" for _ in likes)
    return clause, [x for like in likes for x in (like, like)]


def _stale_override(key: str, value: Any) -> bool:
    c = CONVERTED_SETTINGS.get(key)
    if c is None:
        return False
    return c.new_key != key or value in c.old_values


def rename_settings(conn: sqlite3.Connection) -> int:
    """Rename or convert old settings in the overrides and in the settings history; return
    how many overrides changed. A quick read first: with nothing to convert, nothing is written,
    so a connection takes no write lock for it. An old key written again later (by an older
    facegen) replaces the new one, as the later change. History entries are converted the same
    way, so Undo still works across the change."""
    old_keys = list(CONVERTED_SETTINGS)
    marks, likes = _history_marks()
    candidates = conn.execute(
        f"SELECT key, value FROM settings WHERE key IN ({', '.join('?' * len(old_keys))})",
        old_keys,
    ).fetchall()
    stale = [r for r in candidates if _stale_override(r[0], json.loads(r[1]))]
    history = conn.execute(
        f"SELECT 1 FROM settings_history WHERE {marks} LIMIT 1", likes
    ).fetchone()
    if not stale and history is None:
        return 0
    changed = 0
    with transaction(conn):
        for old_key, c in CONVERTED_SETTINGS.items():
            row = conn.execute(
                "SELECT value, updated_at FROM settings WHERE key = ?", (old_key,)
            ).fetchone()
            if row is not None and _stale_override(old_key, json.loads(row[0])):
                new_value = c.convert(json.loads(row[0]))
                conn.execute("DELETE FROM settings WHERE key = ?", (old_key,))
                if new_value is not None:
                    conn.execute(
                        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)"
                        " ON CONFLICT (key) DO UPDATE SET value = excluded.value,"
                        " updated_at = excluded.updated_at",
                        (c.new_key, json.dumps(new_value), row[1]),
                    )
                changed += 1
        rows = conn.execute(f"SELECT id, before, after FROM settings_history WHERE {marks}", likes)
        for entry in rows.fetchall():
            sides = []
            for text in (entry[1], entry[2]):
                side = json.loads(text)
                for old_key, c in CONVERTED_SETTINGS.items():
                    if old_key in side and (c.new_key != old_key or side[old_key] in c.old_values):
                        value = side.pop(old_key)
                        side[c.new_key] = None if value is None else c.convert(value)
                sides.append(json.dumps(side))
            conn.execute(
                "UPDATE settings_history SET before = ?, after = ? WHERE id = ?",
                (*sides, entry[0]),
            )
    return changed


def migration_lock(path: Path) -> Path:
    """The file `connect` locks while it migrates the database at `path`."""
    return path.with_name(path.name + ".migrate.lock")


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
    on_disk = str(path) != ":memory:"
    if on_disk and schema_version(conn) < len(MIGRATIONS):
        # The app's processes may open a new or older database at once (the first start, or
        # after an update): one switches it to WAL and migrates at a time, and the next reads
        # the version again under the lock, so it finds nothing left to apply instead of
        # applying the same migration twice (or failing to switch while the other does).
        with file_lock(migration_lock(Path(path))):
            conn.execute("PRAGMA journal_mode = WAL")
            migrate(conn)
    else:
        if on_disk:
            conn.execute("PRAGMA journal_mode = WAL")  # already, once it has been migrated
        migrate(conn)  # nothing to apply, or a newer schema, which it refuses
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
