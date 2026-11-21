"""Character-sheet selection and storage; callers own the transaction."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from facegen.core.man_traits import ManSheet, sample_man_sheet
from facegen.core.records import Personality
from facegen.core.traits import (
    DEFAULT_BODY,
    DEFAULT_PERSONALITY_WEIGHT,
    BodyThresholds,
    CharacterSheet,
    sample_sheet,
)


def next_sheet_version(conn: sqlite3.Connection, career_id: int, unique_id: int) -> int:
    return conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM traits WHERE career_id = ? AND unique_id = ?",
        (career_id, unique_id),
    ).fetchone()[0]


def select_sheet(conn: sqlite3.Connection, career_id: int, unique_id: int, version: int) -> None:
    conn.execute(
        "UPDATE players SET active_traits_version = ? WHERE career_id = ? AND unique_id = ?",
        (version, career_id, unique_id),
    )


def load_sheet(
    conn: sqlite3.Connection, career_id: int, unique_id: int, version: int | None = None
) -> tuple[int, CharacterSheet] | None:
    """Load an explicit version, or the selected sheet (latest for legacy/unselected rows)."""
    if version is None:
        selected = conn.execute(
            "SELECT active_traits_version FROM players WHERE career_id = ? AND unique_id = ?",
            (career_id, unique_id),
        ).fetchone()
        version = selected[0] if selected else None
    row = conn.execute(
        "SELECT version, base_seed, sheet FROM traits WHERE career_id = ? AND unique_id = ?"
        + (" AND version = ?" if version is not None else " ORDER BY version DESC LIMIT 1"),
        (career_id, unique_id, version) if version is not None else (career_id, unique_id),
    ).fetchone()
    if row is None:
        return None
    return row["version"], CharacterSheet.from_json(row["sheet"], row["base_seed"])


def store_sheet(
    conn: sqlite3.Connection, career_id: int, unique_id: int, version: int, sheet: CharacterSheet
) -> None:
    conn.execute(
        "INSERT INTO traits (career_id, unique_id, version, base_seed, sheet, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            career_id,
            unique_id,
            version,
            sheet.base_seed,
            sheet.to_json(),
            datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


def sheet_for(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    unique_id: int,
    *,
    height_cm: int | None,
    strength: int | None,
    body: BodyThresholds = DEFAULT_BODY,
    personality: Personality | None = None,
    personality_weight: int = DEFAULT_PERSONALITY_WEIGHT,
    new_face: bool = False,
    activate: bool = True,
    manager: bool = False,
) -> tuple[int, CharacterSheet]:
    """Load the player's sheet, sampling and storing it on first use.

    `new_face` samples the next version instead (the UI's "new face", spec section 4).
    The worker uses `activate=False`: a pending or rejected reroll must not change the
    selected appearance before a picture passes its checks.
    """
    found = load_sheet(conn, career_id, unique_id)
    if found is not None and not new_face:
        return found
    version = next_sheet_version(conn, career_id, unique_id)
    sheet = sample_sheet(
        career_key,
        unique_id,
        version,
        height_cm=height_cm,
        strength=strength,
        body=body,
        personality=personality,
        personality_weight=personality_weight,
        manager=manager,
    )
    store_sheet(conn, career_id, unique_id, version, sheet)
    if activate:
        select_sheet(conn, career_id, unique_id, version)
    return version, sheet


def load_man_sheet(
    conn: sqlite3.Connection, career_id: int, unique_id: int, version: int
) -> ManSheet | None:
    """The men's sheet stored with a sheet version, or None if that version was never drawn as
    a man (or does not exist)."""
    row = conn.execute(
        "SELECT base_seed, man_sheet FROM traits"
        " WHERE career_id = ? AND unique_id = ? AND version = ?",
        (career_id, unique_id, version),
    ).fetchone()
    if row is None or row["man_sheet"] is None:
        return None
    return ManSheet.from_json(row["man_sheet"], row["base_seed"])


def man_sheet_for(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    unique_id: int,
    version: int,
    *,
    nation: int | None,
    personality: Personality | None = None,
    personality_weight: int = DEFAULT_PERSONALITY_WEIGHT,
) -> ManSheet:
    """The men's sheet of an existing sheet version, sampled and stored with it on first use,
    so a later change of nation or personality leaves his look as it is."""
    found = load_man_sheet(conn, career_id, unique_id, version)
    if found is not None:
        return found
    sheet = sample_man_sheet(
        career_key,
        unique_id,
        version,
        nation=nation,
        personality=personality,
        personality_weight=personality_weight,
    )
    cur = conn.execute(
        "UPDATE traits SET man_sheet = ? WHERE career_id = ? AND unique_id = ? AND version = ?",
        (sheet.to_json(), career_id, unique_id, version),
    )
    if cur.rowcount != 1:
        raise RuntimeError(f"no character sheet version {version} for {unique_id}")
    return sheet
