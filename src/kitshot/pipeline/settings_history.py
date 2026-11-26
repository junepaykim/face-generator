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

from kitshot.core.age import TABLES, WORDINGS, shift_table
from kitshot.core.cutoffs import convert_mode
from kitshot.core.settings_values import Settings
from kitshot.persistence import league_stats, publication
from kitshot.persistence.db import transaction
from kitshot.persistence.jobs import load_career
from kitshot.persistence.settings import (
    SettingsError,
    load_defaults,
    load_settings,
    merge_overrides,
    read_overrides,
    set_overrides,
)
from kitshot.persistence.settings_store import write_overrides
from kitshot.pipeline.reclassification import ChangeReport, reclassify_with_report

HISTORY_LIMIT = 20

_RECLASSIFY_PREFIXES = ("cutoffs.", "age.", "staff.", "no_photo.", "facepacks.", "media.")
_PUBLISH_PREFIXES = ("facepacks.",)
"""Changes to who FM27 is shown our picture of: the pack is published again (a sync's way)."""


@dataclass(frozen=True, slots=True)
class Entry:
    id: int
    at: str
    label: str
    """'edit' (a field on the page), 'defaults' (every setting back to its default),
    'auto-configure' (the preset for the graphics card, hardware.py), a preset's label
    (core/presets.py) or a shift's (`shift_label`: every age of a table one year up or down)."""
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
    if any(k.startswith(_PUBLISH_PREFIXES) for k in keys):
        publication.mark_pending(conn)
    return reclassify_with_report(conn, career.id, settings)


AGE_MODE_KEYS = tuple(f"age.{name}.mode" for name in TABLES)


def shift_label(table: str, delta: int) -> str:
    """How the history names a shift of a table's ages: "shift.b.-1"."""
    return f"shift.{table}.{delta:+d}"


def parse_shift_label(label: str) -> tuple[str, int] | None:
    """(table, years) from a history label, or None for any other label."""
    kind, _, rest = label.partition(".")
    table, _, delta = rest.rpartition(".")
    if kind != "shift" or table not in TABLES:
        return None
    try:
        return table, int(delta)
    except ValueError:
        return None


def shifted_ages(values: Mapping[str, Any], shifts: Mapping[str, int]) -> dict[str, Any]:
    """Every used age of each custom table in `shifts` moved by its years (All -1 / All +1,
    Phase 17l). A table in auto mode is left as it is: its ages are the spread's."""
    out: dict[str, Any] = {}
    for table, delta in shifts.items():
        if table not in TABLES:
            raise SettingsError(f"unknown age table {table!r}")
        if not delta or values.get(f"age.{table}.mode") == "auto":
            continue
        keys = {look: f"age.{table}.{look}" for look in WORDINGS}
        moved = shift_table({look: values[key] for look, key in keys.items()}, delta)
        out |= {keys[look]: age for look, age in moved.items()}
    return out


def mode_changes(
    conn: sqlite3.Connection,
    changes: Mapping[str, Any],
    current: Settings,
    defaults_path: Path | None = None,
) -> dict[str, Any]:
    """The changes with each mode switch carried out (Phase 16): my league's cutoffs switched
    between Auto and Manual convert each cell from the latest save (core/cutoffs.py
    `convert_mode`); an age table switched from Auto to Custom ("manual") copies its spread
    into the looks, so the switch alone moves nobody (my club's staff since Phase 16f, every
    table since 17l). Values the changes set themselves are kept, and the other values they
    set count: a share edited in the same save is the one converted, not the share before it."""
    out = dict(changes)
    mode = changes.get("cutoffs.mode")
    switching = mode is not None and mode != current.cutoffs.mode
    to_custom = [key for key in AGE_MODE_KEYS if changes.get(key) == "manual"]
    if not switching and not to_custom:
        return out
    edits = {k: v for k, v in changes.items() if k != "cutoffs.mode" and k not in AGE_MODE_KEYS}
    basis = current
    if edits:
        try:
            basis = merge_overrides(conn, edits, defaults_path)
        except SettingsError:
            pass  # refused by name when the changes are stored
    if switching:
        career = load_career(conn)
        stats = league_stats.load(conn, career.id) if career is not None else {}
        converted = convert_mode(basis.cutoffs, mode, stats, no_photo=basis.no_photo.enabled)
        out = converted | out
    for key in to_custom:
        table = key.split(".")[1]
        if table not in basis.age.spreads:
            continue
        firsts = {wording: first for first, wording in basis.age.table(table)}
        copied = {f"age.{table}.{w}": firsts.get(w, "-") for w in WORDINGS}
        out = copied | out
    return out


def save(
    conn: sqlite3.Connection,
    changes: Mapping[str, Any],
    *,
    label: str = "edit",
    defaults_path: Path | None = None,
    convert: bool = False,
    shifts: Mapping[str, int] | None = None,
) -> Outcome:
    """Store changes, re-classify, and record them as one undoable entry.

    The changes are checked together (settings.set_overrides) and nothing is stored unless all
    are valid: SettingsError otherwise. Values equal to the current ones are left out; a value
    equal to its default removes the override. With `convert` (an edit on the page), a mode
    switch also converts the values it switches between (`mode_changes`), in the same entry.
    `shifts` moves every used age of the named custom tables by its years (`shifted_ages`),
    after the changes, in the same entry; one table's shift alone is named in the history.
    """
    with transaction(conn):
        current = load_settings(conn, defaults_path)
        if convert:
            changes = mode_changes(conn, changes, current, defaults_path)
        shifts = {table: delta for table, delta in (shifts or {}).items() if delta}
        if shifts:
            if not changes and len(shifts) == 1:
                label = shift_label(*next(iter(shifts.items())))
            changes = dict(changes) | shifted_ages(dict(current.values) | dict(changes), shifts)
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
