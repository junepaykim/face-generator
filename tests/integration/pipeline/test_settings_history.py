"""Settings saved from the UI as they are made, with undo and redo (settings_history.py)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from io import StringIO
from pathlib import Path

import pytest
from factories import MANUAL, MY_CLUB, NEWGEN, RIVAL, media_person, player, snapshot, staff_member

from facegen.cli import main
from facegen.core.records import Snapshot
from facegen.persistence import db
from facegen.persistence.settings import (
    SettingsError,
    load_settings,
    local_path,
    read_overrides,
    set_override,
    set_overrides,
)
from facegen.pipeline import settings_history as history
from facegen.pipeline.publish import config_xml, pack_dir, write_config
from facegen.pipeline.sync import run_sync
from facegen.pipeline.sync_operation import execute_sync
from facegen.runtime.contracts import SYNCED, SyncRequest
from facegen.setup.folder_setup import write_local_path

RIVAL_C = NEWGEN + 3  # my league, CA 120: group C until C's bar passes 120


def a_bar(conn: sqlite3.Connection) -> int:
    return load_settings(conn).cutoffs.everywhere["a_ca"]


def test_a_save_is_stored_and_recorded(conn: sqlite3.Connection) -> None:
    out = history.save(conn, {"cutoffs.everywhere.a_ca": 150})
    assert out.changed and a_bar(conn) == 150
    (entry,) = out.entries
    assert entry.before == {"cutoffs.everywhere.a_ca": None}  # no override: the default applied
    assert entry.after == {"cutoffs.everywhere.a_ca": 150}
    assert entry.label == "edit" and not entry.undone
    assert history.entries(conn) == [entry]
    assert out.report is None  # no career yet: nothing to re-classify


def test_values_already_in_force_are_not_a_change(conn: sqlite3.Connection) -> None:
    out = history.save(conn, {"cutoffs.everywhere.a_ca": 165})  # the default
    assert not out.changed and history.entries(conn) == []
    assert read_overrides(conn) == {}


def test_an_invalid_change_stores_nothing(conn: sqlite3.Connection) -> None:
    with pytest.raises(SettingsError):
        history.save(conn, {"cutoffs.everywhere.a_ca": 150, "groups.b.candidates": 0})
    with pytest.raises(SettingsError, match="unknown setting"):
        history.save(conn, {"groups.d.ca_above": 1})
    assert read_overrides(conn) == {} and history.entries(conn) == []


def test_undo_and_redo_walk_the_history(conn: sqlite3.Connection) -> None:
    history.save(conn, {"cutoffs.everywhere.a_ca": 150})
    history.save(conn, {"cutoffs.everywhere.a_ca": 140, "cutoffs.everywhere.a_pa": 160})
    assert history.can_undo(conn) and not history.can_redo(conn)

    out = history.undo(conn)
    assert [e.after for e in out.entries] == [
        {"cutoffs.everywhere.a_ca": 140, "cutoffs.everywhere.a_pa": 160}
    ]
    assert a_bar(conn) == 150 and load_settings(conn).cutoffs.everywhere["a_pa"] == 180
    assert "cutoffs.everywhere.a_pa" not in read_overrides(
        conn
    )  # back to the default, not 180 stored
    assert history.can_redo(conn)

    history.undo(conn)
    assert read_overrides(conn) == {} and not history.can_undo(conn)
    assert not history.undo(conn).changed  # nothing left to undo

    history.redo(conn)
    assert a_bar(conn) == 150
    history.redo(conn)
    assert a_bar(conn) == 140 and not history.can_redo(conn)
    assert not history.redo(conn).changed


def test_history_restores_explicit_overrides_even_when_defaults_change(
    conn: sqlite3.Connection, config_file: Path
) -> None:
    key = "cutoffs.everywhere.a_ca"
    set_override(conn, key, 165, config_file)
    history.save(conn, {key: 150}, defaults_path=config_file)

    history.undo(conn, defaults_path=config_file)
    assert read_overrides(conn) == {key: 165}  # Keep an explicitly stored default.

    text = config_file.read_text(encoding="utf-8")
    assert text.count("a_ca = 165") == 1
    config_file.write_text(text.replace("a_ca = 165", "a_ca = 150"), encoding="utf-8")
    history.redo(conn, defaults_path=config_file)
    assert read_overrides(conn) == {key: 150}  # Redo must not normalize through new defaults.
    history.undo(conn, defaults_path=config_file)
    assert read_overrides(conn) == {key: 165}


def test_failed_undo_rolls_back_override_writes_and_history(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    history.save(conn, {"cutoffs.everywhere.a_ca": 150})
    history.save(conn, {"cutoffs.everywhere.a_ca": 140, "cutoffs.everywhere.a_pa": 160})
    before = list(conn.execute("SELECT * FROM settings ORDER BY key"))
    entries = history.entries(conn)

    def fail(*args: object) -> None:
        raise RuntimeError("reclassification failed")

    monkeypatch.setattr(history, "_reclassify", fail)
    with pytest.raises(RuntimeError, match="reclassification failed"):
        history.undo(conn)

    assert list(conn.execute("SELECT * FROM settings ORDER BY key")) == before
    assert history.entries(conn) == entries


def test_undo_back_to_an_entry_undoes_every_later_change(conn: sqlite3.Connection) -> None:
    first = history.save(conn, {"cutoffs.everywhere.a_ca": 150}).entries[0]
    history.save(conn, {"cutoffs.everywhere.b_ca": 135})
    history.save(conn, {"cutoffs.everywhere.a_ca": 145})
    out = history.undo(conn, through=first.id)
    assert len(out.entries) == 3 and all(e.undone for e in history.entries(conn))
    assert read_overrides(conn) == {}
    # Redo brings them back one at a time, oldest first.
    history.redo(conn)
    assert read_overrides(conn) == {"cutoffs.everywhere.a_ca": 150}


def test_a_new_change_drops_what_was_undone(conn: sqlite3.Connection) -> None:
    history.save(conn, {"cutoffs.everywhere.a_ca": 150})
    history.save(conn, {"cutoffs.everywhere.a_ca": 140})
    history.undo(conn)
    history.save(conn, {"cutoffs.everywhere.a_pa": 165})
    assert not history.can_redo(conn)
    assert [e.after for e in history.entries(conn)] == [
        {"cutoffs.everywhere.a_pa": 165},
        {"cutoffs.everywhere.a_ca": 150},
    ]


def test_only_the_latest_entries_are_kept(conn: sqlite3.Connection) -> None:
    for n in range(history.HISTORY_LIMIT + 5):
        history.save(conn, {"cutoffs.everywhere.a_ca": 100 + n})
    entries = history.entries(conn, limit=100)
    assert len(entries) == history.HISTORY_LIMIT
    assert entries[0].after == {"cutoffs.everywhere.a_ca": 100 + history.HISTORY_LIMIT + 4}


def test_reset_to_defaults_is_one_undoable_change(conn: sqlite3.Connection) -> None:
    history.save(conn, {"cutoffs.everywhere.a_ca": 150})
    history.save(conn, {"my_club.soften_skin": 1, "staff.enabled": False})
    out = history.save_defaults(conn)
    (entry,) = out.entries
    assert entry.label == "defaults" and read_overrides(conn) == {}
    assert set(entry.keys) == {"cutoffs.everywhere.a_ca", "my_club.soften_skin", "staff.enabled"}
    history.undo(conn)
    assert read_overrides(conn) == {
        "cutoffs.everywhere.a_ca": 150,
        "my_club.soften_skin": 1,
        "staff.enabled": False,
    }
    history.redo(conn)
    assert read_overrides(conn) == {}
    assert not history.save_defaults(conn).changed  # already all defaults


def test_changes_reclassify_and_say_what_they_did(conn: sqlite3.Connection) -> None:
    set_overrides(conn, MANUAL)  # my league's cutoffs as numbers, not shares
    settings = load_settings(conn)
    run_sync(conn, snapshot([player(RIVAL_C, ca=120, pa=125, club_id=RIVAL)]), settings,
             save_file="career.fm")  # fmt: skip
    queued = "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND unique_id = ?"
    assert conn.execute(queued, (RIVAL_C,)).fetchone()[0] == 1

    out = history.save(conn, {"cutoffs.manual.c_ca": 126})
    assert out.report is not None and out.report.summary() == "C 1 → 0; queue: 1 cancelled"
    assert conn.execute(queued, (RIVAL_C,)).fetchone()[0] == 0
    assert history.entries(conn)[0].report == out.report  # kept with the entry

    undone = history.undo(conn)
    assert undone.report is not None and undone.report.summary() == "C 0 → 1; queue: 1 queued"
    assert conn.execute(queued, (RIVAL_C,)).fetchone()[0] == 1


def test_history_reads_a_report_written_before_the_reclassification_move(
    conn: sqlite3.Connection,
) -> None:
    stored = """{
        "before": {"a": 0, "b": 0, "c": 0, "tracked": 0},
        "after": {"a": 0, "b": 1, "c": 0, "tracked": 0},
        "enqueue": [["initial", "b", 1]], "updated": 0, "cancelled": 0, "changed_total": 1,
        "changes": [{"unique_id": 123, "name": "Player 123", "club": null, "ca": 135, "pa": null,
                     "before": null, "after": "b", "tracked_before": false, "tracked_after": false,
                     "reasons": ["CA 135 > 130"]}]
    }"""
    conn.execute(
        "INSERT INTO settings_history (at, label, before, after, report)"
        " VALUES ('now', 'edit', ?, ?, ?)",
        ('{"cutoffs.everywhere.b_ca":null}', '{"cutoffs.everywhere.b_ca":130}', stored),
    )

    (entry,) = history.entries(conn)
    assert entry.report is not None
    assert entry.report.enqueue == {("initial", "b"): 1}
    assert entry.report.changes[0].reasons == ("CA 135 > 130",)
    assert entry.report.to_json() == json.loads(stored)


def test_settings_the_rules_do_not_read_are_saved_without_reclassifying(
    conn: sqlite3.Connection,
) -> None:
    run_sync(conn, snapshot([]), load_settings(conn), save_file="career.fm")
    out = history.save(conn, {"schedule.poll_seconds": 30})
    assert out.changed and out.report is None
    assert history.save(conn, {"staff.enabled": False}).report is not None
    assert history.save(conn, {"no_photo.enabled": False}).report is not None


def test_youth_and_media_edits_undo_and_redo_keep_groups_and_queue_in_sync(
    conn: sqlite3.Connection,
) -> None:
    youth, journalist, press, prospect = 1010, 1011, 1012, 1013
    people = snapshot(
        [player(youth, team_slot=2), player(prospect, team_slot=2, pa=130)],
        media=[
            media_person(journalist),
            media_person(press, role="press officer", club_id=MY_CLUB),
        ],
    )
    run_sync(conn, people, load_settings(conn), save_file="career.fm")

    def assert_groups(expected: dict[int, str | None]) -> None:
        people = dict(conn.execute("SELECT unique_id, group_name FROM players"))
        assert {uid: people[uid] for uid in expected} == expected
        queued = conn.execute(
            "SELECT unique_id, group_name FROM jobs WHERE status = 'queued' ORDER BY unique_id"
        )
        assert [tuple(row) for row in queued] == sorted(
            (uid, group) for uid, group in expected.items() if group is not None
        )

    groups = {youth: "b", journalist: "b", press: "b", prospect: "b"}
    assert_groups(groups)
    for key, value, ordered_ids, disabled_group in (
        ("cutoffs.my_club.youth", "a", [prospect, youth], "a"),
        ("media.journalists", False, [journalist], None),
        ("media.press_officers", False, [press], None),
    ):
        disabled = {**groups, **dict.fromkeys(ordered_ids, disabled_group)}
        edited = history.save(conn, {key: value}).report
        assert edited is not None
        assert [(c.unique_id, c.before, c.after) for c in edited.changes] == [
            (uid, "b", disabled_group) for uid in ordered_ids
        ]
        assert_groups(disabled)
        undone = history.undo(conn).report
        assert undone is not None
        assert [(c.unique_id, c.before, c.after) for c in undone.changes] == [
            (uid, disabled_group, "b") for uid in ordered_ids
        ]
        assert_groups(groups)
        assert history.redo(conn).report == edited
        assert_groups(disabled)
        groups = disabled


@pytest.mark.parametrize("entrypoint", ["desktop", "cli"])
def test_save_sync_keeps_a_settings_edit_made_while_reading_the_snapshot(
    config_file: Path,
    entrypoint: str,
) -> None:
    settings = load_settings(defaults_path=config_file)
    save = settings.paths.snapshots_dir / "career.fm"
    save.parent.mkdir(parents=True)
    save.write_bytes(b"synthetic save")
    coach = 1122
    people = snapshot(staff=[staff_member(coach)])
    new_fm_folder = config_file.parent / "other-fm-folder"
    obsolete_pack = pack_dir(new_fm_folder, "previous-career")
    write_config(obsolete_pack, [coach], new_fm_folder)
    active_jobs = "SELECT COUNT(*) FROM jobs WHERE unique_id = ? AND status = 'queued'"
    with closing(db.connect(settings.paths.db_path)) as editor:
        run_sync(editor, people, settings, save_file=save.name)
        assert editor.execute(active_jobs, (coach,)).fetchone()[0] == 1

        class Source:
            def load(self, path: Path) -> Snapshot:
                history.save(editor, {"staff.enabled": False}, defaults_path=config_file)
                write_local_path(local_path(config_file), "fm_user_dir", str(new_fm_folder))
                assert editor.execute(active_jobs, (coach,)).fetchone()[0] == 0
                return people

        if entrypoint == "desktop":
            request = SyncRequest(str(settings.paths.db_path), str(config_file), str(save))
            assert execute_sync(request, Source).outcome == SYNCED
        else:
            out = StringIO()
            assert (
                main(
                    ["--config", str(config_file), "sync", "--save", str(save)],
                    source_factory=Source,
                    out=out,
                )
                == 0
            ), out.getvalue()
        assert not load_settings(editor, config_file).staff.enabled
        assert (
            editor.execute(
                "SELECT group_name FROM players WHERE unique_id = ?", (coach,)
            ).fetchone()[0]
            is None
        )
        assert editor.execute(active_jobs, (coach,)).fetchone()[0] == 0
        assert (obsolete_pack / "config.xml").read_text(encoding="utf-8") == config_xml([])


def test_a_change_saved_before_phase_16_can_be_undone_after_it(conn: sqlite3.Connection) -> None:
    """The old keys are converted in the overrides and in the history together (Phase 16):
    undoing an old "CA above 150" puts back the default "at least" cut."""
    import json as _json

    from facegen.persistence.db import rename_settings

    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('groups.a.ca_above', '150', 'then')"
    )
    conn.execute(
        "INSERT INTO settings_history (at, label, before, after) VALUES ('then', 'edit', ?, ?)",
        (_json.dumps({"groups.a.ca_above": None}), _json.dumps({"groups.a.ca_above": 150})),
    )
    assert rename_settings(conn) == 1
    assert load_settings(conn).cutoffs.everywhere["a_ca"] == 151  # the same people as "> 150"
    out = history.undo(conn)
    assert [e.after for e in out.entries] == [{"cutoffs.everywhere.a_ca": 151}]
    assert read_overrides(conn) == {}
    assert load_settings(conn).cutoffs.everywhere["a_ca"] == 165
    history.redo(conn)
    assert read_overrides(conn) == {"cutoffs.everywhere.a_ca": 151}
