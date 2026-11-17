"""Saving settings from the UI, with undo and redo (docs/spec.md section 7).

The settings page saves each change as soon as it is made; there is no Apply step. Every save
is one entry in `settings_history`: for each key it touched, the stored override before and
after (None: no override, so the default applies). Undo puts back an entry's `before` for its
keys, redo its `after`; both re-classify at once, as the change itself did. Pictures are never
deleted by any of them.

The history is a stack. A new change drops the entries that were undone (no redo past a new
change), and only the latest HISTORY_LIMIT entries are kept.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from facegen.core.settings_values import Settings
from facegen.persistence.db import transaction
from facegen.persistence.jobs import load_career
from facegen.persistence.settings import load_defaults, load_settings, read_overrides, set_overrides
from facegen.persistence.settings_store import write_overrides
from facegen.pipeline.sync import ChangeReport, reclassify_with_report

HISTORY_LIMIT = 20

_RECLASSIFY_PREFIXES = ("groups.", "age.", "staff.", "no_photo.")


@dataclass(frozen=True, slots=True)
class Entry:
    id: int
    at: str
    label: str
    """'edit' (a field on the page), 'defaults' (every setting back to its default) or
    'auto-configure' (the preset for the graphics card, hardware.py)."""
    before: dict[str, Any]
    """Key to its stored override before the change; None for no override."""
    after: dict[str, Any]
    undone: bool
    report: ChangeReport | None
    """What the change did to the groups and the queue; None when it re-classified nothing."""

    @property
    def keys(self) -> list[str]:
        return sorted(self.after)


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of a save, an undo or a redo."""

    settings: Settings
    entries: tuple[Entry, ...] = ()
    """The entries saved, undone or redone; empty when nothing changed."""
    report: ChangeReport | None = None
    """What re-classifying did; None with no career yet or nothing to re-classify."""

    @property
    def changed(self) -> bool:
        return bool(self.entries)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _entry(r: sqlite3.Row) -> Entry:
    report = json.loads(r["report"]) if r["report"] else None
    return Entry(
        id=r["id"],
        at=r["at"],
        label=r["label"],
        before=json.loads(r["before"]),
        after=json.loads(r["after"]),
        undone=bool(r["undone"]),
        report=ChangeReport.from_json(report) if report else None,
    )


def entries(conn: sqlite3.Connection, limit: int = HISTORY_LIMIT) -> list[Entry]:
    """The latest entries, newest first, undone ones included (they can be redone)."""
    rows = conn.execute(
        "SELECT * FROM settings_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_entry(r) for r in rows]


def can_undo(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM settings_history WHERE undone = 0 LIMIT 1").fetchone()
        is not None
    )


def can_redo(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM settings_history WHERE undone = 1 LIMIT 1").fetchone()
        is not None
    )


def _write(conn: sqlite3.Connection, state: Mapping[str, Any]) -> None:
    """Store each key's override, or remove it where the state is None."""
    now = datetime.now(UTC).isoformat()
    write_overrides(conn, state, updated_at=now)


def _reclassify(
    conn: sqlite3.Connection, settings: Settings, keys: list[str]
) -> ChangeReport | None:
    if not any(k.startswith(_RECLASSIFY_PREFIXES) for k in keys):
        return None
    career = load_career(conn)
    if career is None:
        return None
    return reclassify_with_report(conn, career.id, settings)


def save(
    conn: sqlite3.Connection,
    changes: Mapping[str, Any],
    *,
    label: str = "edit",
    defaults_path: Path | None = None,
) -> Outcome:
    """Store changes, re-classify, and record them as one undoable entry.

    The changes are checked together (settings.set_overrides) and nothing is stored unless all
    are valid: SettingsError otherwise. Values equal to the current ones are left out; a value
    equal to its default removes the override.
    """
    with transaction(conn):
        current = load_settings(conn, defaults_path)
        # An unknown key is kept, so set_overrides refuses it by name.
        changes = {
            k: v for k, v in changes.items() if k not in current.values or v != current.values[k]
        }
        if not changes:
            return Outcome(current)
        before = read_overrides(conn)
        applied = set_overrides(conn, changes, defaults_path)
        after = read_overrides(conn)
        keys = sorted(changes)
        report = _reclassify(conn, applied, keys)
        conn.execute("DELETE FROM settings_history WHERE undone = 1")
        cur = conn.execute(
            "INSERT INTO settings_history (at, label, before, after, report) VALUES (?, ?, ?, ?, ?)",
            (
                _now(),
                label,
                json.dumps({k: before.get(k) for k in keys}),
                json.dumps({k: after.get(k) for k in keys}),
                json.dumps(report.to_json()) if report else None,
            ),
        )
        conn.execute(
            "DELETE FROM settings_history WHERE id NOT IN"
            " (SELECT id FROM settings_history ORDER BY id DESC LIMIT ?)",
            (HISTORY_LIMIT,),
        )
        row = conn.execute(
            "SELECT * FROM settings_history WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return Outcome(applied, (_entry(row),), report)


def save_defaults(conn: sqlite3.Connection, *, defaults_path: Path | None = None) -> Outcome:
    """Every setting the UI can change back to its default, as one undoable entry."""
    defaults = load_defaults(defaults_path)
    current = load_settings(conn, defaults_path)
    return save(
        conn,
        {k: defaults[k] for k in current.overridden},
        label="defaults",
        defaults_path=defaults_path,
    )


def _step(
    conn: sqlite3.Connection, targets: list[sqlite3.Row], *, undo: bool, defaults_path: Path | None
) -> Outcome:
    """Put back `before` (undo) or `after` (redo) of each target, in the order given.

    Runs in the caller's transaction. The result is checked as a whole: SettingsError (and
    nothing kept) if it no longer makes valid settings, say after the defaults file changed.
    """
    for r in targets:
        _write(conn, json.loads(r["before"] if undo else r["after"]))
        conn.execute("UPDATE settings_history SET undone = ? WHERE id = ?", (int(undo), r["id"]))
    settings = load_settings(conn, defaults_path)
    keys = sorted({k for r in targets for k in json.loads(r["after"])})
    report = _reclassify(conn, settings, keys)
    ids = [r["id"] for r in targets]
    rows = conn.execute(
        f"SELECT * FROM settings_history WHERE id IN ({', '.join('?' * len(ids))})", ids
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    return Outcome(settings, tuple(_entry(by_id[i]) for i in ids), report)


def undo(
    conn: sqlite3.Connection, *, through: int | None = None, defaults_path: Path | None = None
) -> Outcome:
    """Undo the latest change; with `through`, every change back to and including that entry.

    Changes are undone newest first, so the settings return exactly to what they were before
    entry `through` was saved.
    """
    with transaction(conn):
        if through is None:
            targets = conn.execute(
                "SELECT * FROM settings_history WHERE undone = 0 ORDER BY id DESC LIMIT 1"
            ).fetchall()
        else:
            targets = conn.execute(
                "SELECT * FROM settings_history WHERE undone = 0 AND id >= ? ORDER BY id DESC",
                (through,),
            ).fetchall()
        if not targets:
            return Outcome(load_settings(conn, defaults_path))
        return _step(conn, targets, undo=True, defaults_path=defaults_path)


def redo(conn: sqlite3.Connection, *, defaults_path: Path | None = None) -> Outcome:
    """Redo the change undone last (the oldest undone entry)."""
    with transaction(conn):
        targets = conn.execute(
            "SELECT * FROM settings_history WHERE undone = 1 ORDER BY id LIMIT 1"
        ).fetchall()
        if not targets:
            return Outcome(load_settings(conn, defaults_path))
        return _step(conn, targets, undo=False, defaults_path=defaults_path)
