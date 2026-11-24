"""Persisted character sheets retain their identity and sampling inputs."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from factories import PLAIN_PERSONALITY

from kitshot.core.traits import sample_sheet
from kitshot.persistence.sheet_store import load_sheet, sheet_for


def test_sheet_for_samples_once_and_stores(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'k', 'K', '[]', '', '')"
    )
    v1, s1 = sheet_for(conn, 1, "k", 42, height_cm=180, strength=10)
    v2, s2 = sheet_for(conn, 1, "k", 42, height_cm=150, strength=1)  # facts changed: kept
    assert (v1, v2) == (1, 1) and s1 == s2
    assert load_sheet(conn, 1, 42) == (1, s1)
    v3, s3 = sheet_for(conn, 1, "k", 42, height_cm=180, strength=10, new_face=True)
    assert v3 == 2 and s3.base_seed != s1.base_seed
    assert load_sheet(conn, 1, 42) == (2, s3)
    assert conn.execute("SELECT COUNT(*) FROM traits").fetchone()[0] == 2


def test_sheet_for_weights_a_new_sheet_by_personality(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'k', 'K', '[]', '', '')"
    )
    p = replace(PLAIN_PERSONALITY, pressure=20, ambition=20, controversy=20)
    _, sheet = sheet_for(conn, 1, "k", 42, height_cm=180, strength=10, personality=p)
    assert sheet == sample_sheet("k", 42, 1, height_cm=180, strength=10, personality=p)
