"""Settings saved from the UI as they are made, with undo and redo (settings_history.py)."""

from __future__ import annotations

import sqlite3

import pytest
from factories import NEWGEN, RIVAL, player, snapshot

from facegen import settings_history as history
from facegen.settings import SettingsError, load_settings, read_overrides
from facegen.sync import run_sync

RIVAL_C = NEWGEN + 3  # my league, CA 120: group C until C's bar passes 120


def a_bar(conn: sqlite3.Connection) -> int:
    return load_settings(conn).groups.a.ca_above


def test_a_save_is_stored_and_recorded(conn: sqlite3.Connection) -> None:
    out = history.save(conn, {"groups.a.ca_above": 150})
    assert out.changed and a_bar(conn) == 150
    (entry,) = out.entries
    assert entry.before == {"groups.a.ca_above": None}  # no override: the default applied
    assert entry.after == {"groups.a.ca_above": 150}
    assert entry.label == "edit" and not entry.undone
    assert history.entries(conn) == [entry]
    assert out.report is None  # no career yet: nothing to re-classify


def test_values_already_in_force_are_not_a_change(conn: sqlite3.Connection) -> None:
    out = history.save(conn, {"groups.a.ca_above": 155})  # the default
    assert not out.changed and history.entries(conn) == []
    assert read_overrides(conn) == {}


def test_an_invalid_change_stores_nothing(conn: sqlite3.Connection) -> None:
    with pytest.raises(SettingsError):
        history.save(conn, {"groups.a.ca_above": 150, "groups.b.candidates": 0})
    with pytest.raises(SettingsError, match="unknown setting"):
        history.save(conn, {"groups.d.ca_above": 1})
    assert read_overrides(conn) == {} and history.entries(conn) == []


def test_undo_and_redo_walk_the_history(conn: sqlite3.Connection) -> None:
    history.save(conn, {"groups.a.ca_above": 150})
    history.save(conn, {"groups.a.ca_above": 140, "groups.a.pa_above": 160})
    assert history.can_undo(conn) and not history.can_redo(conn)

    out = history.undo(conn)
    assert [e.after for e in out.entries] == [{"groups.a.ca_above": 140, "groups.a.pa_above": 160}]
    assert a_bar(conn) == 150 and load_settings(conn).groups.a.pa_above == 170
    assert "groups.a.pa_above" not in read_overrides(conn)  # back to the default, not 170 stored
    assert history.can_redo(conn)

    history.undo(conn)
    assert read_overrides(conn) == {} and not history.can_undo(conn)
    assert not history.undo(conn).changed  # nothing left to undo

    history.redo(conn)
    assert a_bar(conn) == 150
    history.redo(conn)
    assert a_bar(conn) == 140 and not history.can_redo(conn)
    assert not history.redo(conn).changed


def test_undo_back_to_an_entry_undoes_every_later_change(conn: sqlite3.Connection) -> None:
    first = history.save(conn, {"groups.a.ca_above": 150}).entries[0]
    history.save(conn, {"groups.b.any_ca_above": 135})
    history.save(conn, {"groups.a.ca_above": 145})
    out = history.undo(conn, through=first.id)
    assert len(out.entries) == 3 and all(e.undone for e in history.entries(conn))
    assert read_overrides(conn) == {}
    # Redo brings them back one at a time, oldest first.
    history.redo(conn)
    assert read_overrides(conn) == {"groups.a.ca_above": 150}


def test_a_new_change_drops_what_was_undone(conn: sqlite3.Connection) -> None:
    history.save(conn, {"groups.a.ca_above": 150})
    history.save(conn, {"groups.a.ca_above": 140})
    history.undo(conn)
    history.save(conn, {"groups.a.pa_above": 165})
    assert not history.can_redo(conn)
    assert [e.after for e in history.entries(conn)] == [
        {"groups.a.pa_above": 165},
        {"groups.a.ca_above": 150},
    ]


def test_only_the_latest_entries_are_kept(conn: sqlite3.Connection) -> None:
    for n in range(history.HISTORY_LIMIT + 5):
        history.save(conn, {"groups.a.ca_above": 100 + n})
    entries = history.entries(conn, limit=100)
    assert len(entries) == history.HISTORY_LIMIT
    assert entries[0].after == {"groups.a.ca_above": 100 + history.HISTORY_LIMIT + 4}


def test_reset_to_defaults_is_one_undoable_change(conn: sqlite3.Connection) -> None:
    history.save(conn, {"groups.a.ca_above": 150})
    history.save(conn, {"my_club.soften_skin": 1, "staff.enabled": False})
    out = history.save_defaults(conn)
    (entry,) = out.entries
    assert entry.label == "defaults" and read_overrides(conn) == {}
    assert set(entry.keys) == {"groups.a.ca_above", "my_club.soften_skin", "staff.enabled"}
    history.undo(conn)
    assert read_overrides(conn) == {
        "groups.a.ca_above": 150,
        "my_club.soften_skin": 1,
        "staff.enabled": False,
    }
    history.redo(conn)
    assert read_overrides(conn) == {}
    assert not history.save_defaults(conn).changed  # already all defaults


def test_changes_reclassify_and_say_what_they_did(conn: sqlite3.Connection) -> None:
    settings = load_settings(conn)
    run_sync(conn, snapshot([player(RIVAL_C, ca=120, pa=125, club_id=RIVAL)]), settings,
             save_file="career.fm")  # fmt: skip
    queued = "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND unique_id = ?"
    assert conn.execute(queued, (RIVAL_C,)).fetchone()[0] == 1

    out = history.save(conn, {"groups.c.league_ca_above": 125})
    assert out.report is not None and out.report.summary() == "C 1 → 0; queue: 1 cancelled"
    assert conn.execute(queued, (RIVAL_C,)).fetchone()[0] == 0
    assert history.entries(conn)[0].report == out.report  # kept with the entry

    undone = history.undo(conn)
    assert undone.report is not None and undone.report.summary() == "C 0 → 1; queue: 1 queued"
    assert conn.execute(queued, (RIVAL_C,)).fetchone()[0] == 1


def test_settings_the_rules_do_not_read_are_saved_without_reclassifying(
    conn: sqlite3.Connection,
) -> None:
    run_sync(conn, snapshot([]), load_settings(conn), save_file="career.fm")
    out = history.save(conn, {"schedule.poll_seconds": 30})
    assert out.changed and out.report is None
    assert history.save(conn, {"staff.enabled": False}).report is not None
    assert history.save(conn, {"no_photo.enabled": False}).report is not None
