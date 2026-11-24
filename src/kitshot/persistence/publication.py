"""Durable intent to publish the active pack after a successful save sync."""

from __future__ import annotations

import sqlite3


def mark_pending(conn: sqlite3.Connection) -> None:
    """Called inside the sync transaction. Multiple syncs need just one fresh pack."""
    # An older build can still use the same database: this private table is additive,
    # and neither the schema of the shared tables nor their migration version changes.
    conn.execute("CREATE TABLE IF NOT EXISTS pending_publication (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT OR IGNORE INTO pending_publication (id) VALUES (1)")


def pending(conn: sqlite3.Connection) -> bool:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'pending_publication'"
    ).fetchone()
    return (
        exists is not None
        and conn.execute("SELECT 1 FROM pending_publication").fetchone() is not None
    )


def finish(conn: sqlite3.Connection) -> None:
    """A complete active pack is on disk and the publisher still owns the DB write lock."""
    if pending(conn):
        conn.execute("DELETE FROM pending_publication")
