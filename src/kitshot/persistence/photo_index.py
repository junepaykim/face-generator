"""The last successful FM photo scan and its indexed portrait IDs in SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kitshot.datasource.photos import PhotoScan


@dataclass(frozen=True, slots=True)
class PhotoStatus:
    """What the last stored scan found."""

    game_known: bool
    """FM27's photo list was read. If not, no one counts as without a photo."""
    game: int
    facepack: int
    facepack_files: int
    scanned_at: str
    changed_at: str
    """When the IDs last changed: a game update, or a facepack added, changed or removed."""
    warnings: tuple[str, ...]


def stored_state(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT state FROM photo_scan WHERE id = 1").fetchone()
    return json.loads(row[0]) if row else None


def store(conn: sqlite3.Connection, result: PhotoScan) -> bool:
    """Store a scan in the caller's transaction. Returns whether the IDs were rewritten."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    state = json.dumps(result.state)
    idx = result.index
    if result.error is not None:
        # An incomplete first scan has no known absences. After a success, keep its IDs,
        # counts and changed_at until the scanner can replace the entire index.
        conn.execute(
            "INSERT INTO photo_scan (id, state, game_known, game, facepack, facepack_files,"
            " warnings, scanned_at, changed_at) VALUES (1, ?, 0, 0, 0, 0, ?, ?, ?)"
            " ON CONFLICT (id) DO UPDATE SET state = excluded.state,"
            " warnings = excluded.warnings, scanned_at = excluded.scanned_at",
            (state, json.dumps([result.error]), now, now),
        )
        return False
    if idx is None:
        conn.execute("UPDATE photo_scan SET state = ?, scanned_at = ? WHERE id = 1", (state, now))
        return False
    conn.execute("DELETE FROM photos")
    game = idx.game or frozenset()
    conn.executemany(
        "INSERT INTO photos (unique_id, game, facepack) VALUES (?, ?, ?)",
        ((uid, int(uid in game), int(uid in idx.facepack)) for uid in sorted(game | idx.facepack)),
    )
    conn.execute(
        "INSERT INTO photo_scan (id, state, game_known, game, facepack, facepack_files,"
        " warnings, scanned_at, changed_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (id) DO UPDATE SET state = excluded.state,"
        " game_known = excluded.game_known, game = excluded.game,"
        " facepack = excluded.facepack, facepack_files = excluded.facepack_files,"
        " warnings = excluded.warnings, scanned_at = excluded.scanned_at,"
        " changed_at = excluded.changed_at",
        (
            state,
            int(idx.game is not None),
            len(game),
            len(idx.facepack),
            len(result.state["configs"]),
            json.dumps(list(idx.warnings)),
            now,
            now,
        ),
    )
    return True


def status(conn: sqlite3.Connection) -> PhotoStatus | None:
    """The last stored scan, or None before the first sync that scanned."""
    r = conn.execute(
        "SELECT game_known, game, facepack, facepack_files, warnings, scanned_at, changed_at"
        " FROM photo_scan WHERE id = 1"
    ).fetchone()
    if r is None:
        return None
    return PhotoStatus(
        game_known=bool(r["game_known"]),
        game=r["game"],
        facepack=r["facepack"],
        facepack_files=r["facepack_files"],
        scanned_at=r["scanned_at"],
        changed_at=r["changed_at"],
        warnings=tuple(json.loads(r["warnings"])),
    )


_GROUP_COUNTS = """
    COUNT(*) FILTER (WHERE group_name = 'a') AS a,
    COUNT(*) FILTER (WHERE group_name = 'a' AND at_my_club) AS a_my_club,
    COUNT(*) FILTER (WHERE group_name = 'b') AS b,
    COUNT(*) FILTER (WHERE group_name = 'c') AS c,
    SUM(tracked) AS tracked, COUNT(pin) AS pinned,
    COUNT(*) FILTER (WHERE group_name = 'a' AND is_staff) AS a_staff,
    SUM(youth_team) AS youth,
    COUNT(*) FILTER (WHERE group_name IS NOT NULL AND media_role IS NOT NULL) AS media,
"""


def player_counts(
    conn: sqlite3.Connection, career_id: int, *, include_groups: bool = False
) -> dict[str, int]:
    """A career's pre-existing players in scope, staff left out: with a photo, without one,
    and those without one that the rules put in a group outside my club (unpinned). All 0
    until FM27's photo list has been read. The dashboard also counts groups across the whole
    present roster, sharing the roster scan with the photo counts."""
    s = status(conn)
    known = s is not None and s.game_known
    if not known and not include_groups:
        return {"with_photo": 0, "without_photo": 0, "without_photo_picked": 0}
    row = conn.execute(
        f"""
        WITH roster AS (
            SELECT group_name, at_my_club, is_staff, tracked, pin, youth_team, media_role,
                   CASE WHEN :known AND in_scope = 1 AND is_newgen = 0
                             AND is_staff = 0 AND media_role IS NULL THEN unique_id END AS photo_id
            FROM players WHERE career_id = :career AND present = 1
        )
        SELECT {_GROUP_COUNTS if include_groups else ""}
               COUNT(ph.unique_id) AS with_photo,
               COUNT(p.photo_id) - COUNT(ph.unique_id) AS without_photo,
               COUNT(*) FILTER (WHERE p.group_name IS NOT NULL AND p.at_my_club = 0
                                AND p.pin IS NULL AND p.photo_id IS NOT NULL
                                AND ph.unique_id IS NULL) AS without_photo_picked
        FROM roster p LEFT JOIN photos ph ON ph.unique_id = p.photo_id
        {"" if include_groups else "WHERE p.photo_id IS NOT NULL"}
        """,
        {"known": known, "career": career_id},
    ).fetchone()
    return {k: v or 0 for k, v in dict(row).items()}
