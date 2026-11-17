"""Raw settings override writes on a caller-owned database connection."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any


def write_overrides(conn: sqlite3.Connection, state: Mapping[str, Any], *, updated_at: str) -> None:
    """Store values exactly, deleting keys whose value is None, without committing.

    Callers validate values and decide whether a default should be stored or removed.
    History restoration can therefore preserve an explicit override equal to a default.
    """
    for key, value in state.items():
        if value is None:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        else:
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT (key)"
                " DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, json.dumps(value), updated_at),
            )
