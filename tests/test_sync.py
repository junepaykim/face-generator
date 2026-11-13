from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date

import pytest
from factories import (
    ABROAD,
    CONTINENTAL,
    CONTINENTAL_COMP,
    CREATED_FROM,
    DEFAULT_CLUBS,
    MY_CLUB,
    NEWGEN,
    PLAIN_PERSONALITY,
    RIVAL,
    WOMENS,
    club,
    player,
    snapshot,
)

from facegen.datasource.records import Gender, PlayerRecord, Snapshot
from facegen.photos import PhotoIndex, PhotoScan
from facegen.planner import TIER_B, TIER_C
from facegen.settings import Settings, load_settings, set_override
from facegen.sync import SyncReport, reclassify, record_failed_sync, run_sync, sync_of_save
from facegen.traits import personality_from_json

# A small world: pre-existing players have low IDs, newgens start at NEWGEN.
MINE_OLD = 1001  # pre-existing, at my club
MINE_NEW = NEWGEN + 1  # newgen at my club
STAR_ELSEWHERE = 1002  # pre-existing star abroad: picked only if known to have no photo
RIVAL_NEWGEN = NEWGEN + 2  # my league, PA 152 -> B
RIVAL_C = NEWGEN + 3  # my league, CA 120 -> C
ABROAD_A = NEWGEN + 4  # abroad, CA 160 -> A
ABROAD_LOW = NEWGEN + 5  # abroad, nothing
WOMAN = NEWGEN + 6  # women's club, PA 190 -> ignored
CONT_NEWGEN = NEWGEN + 7  # continental club, CA 131 -> B when the competition is listed


def world() -> list[PlayerRecord]:
    return [
        player(MINE_OLD, ca=120, pa=125, age=29, join=date(2023, 7, 1)),
        player(MINE_NEW, ca=60, pa=110, age=16, join=date(2026, 2, 1)),
        player(STAR_ELSEWHERE, ca=190, pa=195, club_id=ABROAD),
        player(RIVAL_NEWGEN, ca=90, pa=152, club_id=RIVAL),
        player(RIVAL_C, ca=120, pa=125, club_id=RIVAL),
        player(ABROAD_A, ca=160, pa=170, club_id=ABROAD),
        player(ABROAD_LOW, ca=60, pa=90, club_id=ABROAD),
        player(WOMAN, ca=100, pa=190, club_id=WOMENS),
        player(CONT_NEWGEN, ca=131, pa=135, club_id=CONTINENTAL),
    ]


def sync(conn: sqlite3.Connection, snap: Snapshot, settings: Settings, **kw: object) -> SyncReport:
    return run_sync(conn, snap, settings, save_file=snap.meta.file_name, **kw)  # type: ignore[arg-type]


def groups(conn: sqlite3.Connection) -> dict[int, str | None]:
    return {
        r["unique_id"]: r["group_name"]
        for r in conn.execute("SELECT unique_id, group_name FROM players WHERE group_name NOTNULL")
    }


def queued(conn: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    rows = conn.execute("SELECT unique_id, kind, group_name FROM jobs WHERE status = 'queued'")
    return {r["unique_id"]: (r["kind"], r["group_name"]) for r in rows}


def player_row(conn: sqlite3.Connection, uid: int) -> sqlite3.Row:
    return conn.execute("SELECT * FROM players WHERE unique_id = ?", (uid,)).fetchone()


def add_image(
    conn: sqlite3.Connection, uid: int, group: str, club_id: int | None, band: str
) -> None:
    conn.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, club_id, age_band,"
        " model, prompt, negative_prompt, seed, is_current, created_at)"
        " VALUES (1, ?, 1, 'initial', ?, ?, ?, 'm', 'p', 'n', 1, 1, 'now')",
        (uid, group, club_id, band),
    )
    conn.execute("UPDATE jobs SET status = 'done' WHERE unique_id = ?", (uid,))


# --- first sync --------------------------------------------------------------------------------


def test_academy_membership_survives_managed_club_changes(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    uid = NEWGEN + 80
    origin = player(uid, club_id=RIVAL, home_grown=(MY_CLUB,))
    sync(conn, snapshot([origin]), settings)
    assert player_row(conn, uid)["academy"] == 1
    # Even if home-grown facts disappear, established academy tracking is permanent.
    moved = replace(origin, home_grown_club_ids=())
    sync(conn, replace(snapshot([moved]), managed_club_id=ABROAD), settings)
    row = player_row(conn, uid)
    assert (row["academy"], row["tracked"], row["group_name"]) == (1, 1, "b")
    reclassify(conn, 1, settings)
    assert player_row(conn, uid)["tracked"] == 1


def test_manual_jobs_refresh_current_targets_and_cancel_when_absent(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    from facegen import manual
    from facegen.age import age_wording

    uid = 1001
    sync(conn, snapshot([player(uid, age=19)]), settings)
    manual.generate(conn, 1, [uid], settings, "c", now=123)
    before = conn.execute("SELECT * FROM jobs WHERE status = 'queued'").fetchone()
    changed = player(uid, age=34, club_id=RIVAL)
    sync(conn, snapshot([changed]), settings)
    after = conn.execute("SELECT * FROM jobs WHERE status = 'queued'").fetchone()
    for field in ("id", "kind", "group_name", "manual", "tier", "rank", "reason"):
        assert before[field] == after[field]
    assert after["target_club_id"] == RIVAL
    assert after["target_band"] == age_wording(34, "c", False, settings.age)
    # Existing, low-ability pre-existing player stays hand-picked after transfer.
    assert after["manual"] == 1
    sync(conn, snapshot([]), settings)
    assert (
        conn.execute("SELECT status FROM jobs WHERE id = ?", (after["id"],)).fetchone()[0]
        == "cancelled"
    )


def test_manual_job_is_cancelled_on_transfer_out_of_mens_football(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    from facegen import manual

    sync(conn, snapshot([player(1001)]), settings)
    manual.new_face(conn, 1, 1001, settings)
    sync(conn, snapshot([player(1001, club_id=WOMENS)]), settings)
    assert not queued(conn)


def test_first_sync_classifies_and_queues(conn: sqlite3.Connection, settings: Settings) -> None:
    report = sync(conn, snapshot(world()), settings)

    assert report.career.is_new
    assert report.career.newgen_min_id == CREATED_FROM
    assert report.career.newgen_id_source == "detected"
    assert groups(conn) == {
        MINE_OLD: "a",
        MINE_NEW: "a",
        RIVAL_NEWGEN: "b",
        RIVAL_C: "c",
        ABROAD_A: "a",
    }
    assert queued(conn) == {
        MINE_OLD: ("initial", "a"),
        MINE_NEW: ("initial", "a"),
        RIVAL_NEWGEN: ("initial", "b"),
        RIVAL_C: ("initial", "c"),
        ABROAD_A: ("initial", "a"),
    }
    c = report.counts
    assert (c["a"], c["a_my_club"], c["b"], c["c"], c["tracked"]) == (3, 2, 1, 1, 3)
    assert report.my_league_clubs == 3


def test_queue_order_follows_the_spec(conn: sqlite3.Connection, settings: Settings) -> None:
    report = sync(conn, snapshot(world()), settings)
    # My club newest first, then the rest of A, then B, then C.
    assert [e.unique_id for e in report.queue] == [
        MINE_NEW,
        MINE_OLD,
        ABROAD_A,
        RIVAL_NEWGEN,
        RIVAL_C,
    ]


def test_second_identical_sync_changes_nothing(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    jobs_before = conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    report = sync(conn, snapshot(world()), settings)
    assert not report.career.is_new
    assert not report.enqueued and report.updated == 0 and report.cancelled == 0
    assert [tuple(r) for r in conn.execute("SELECT * FROM jobs ORDER BY id")] == [
        tuple(r) for r in jobs_before
    ]
    assert conn.execute("SELECT COUNT(*) FROM careers").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sync_runs").fetchone()[0] == 2


def test_dry_run_stores_nothing(conn: sqlite3.Connection, settings: Settings) -> None:
    report = sync(conn, snapshot(world()), settings, dry_run=True, with_table=True)
    assert report.dry_run
    assert sum(report.enqueued.values()) == 5
    assert len(report.queue) == 5
    assert any(row["unique_id"] == MINE_OLD and row["group"] == "A" for row in report.table)
    for table in ("careers", "players", "clubs", "jobs", "sync_runs"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_player_table_has_the_phase_0_columns(conn: sqlite3.Connection, settings: Settings) -> None:
    report = sync(conn, snapshot(world()), settings, with_table=True)
    row = next(r for r in report.table if r["unique_id"] == RIVAL_NEWGEN)
    assert row["club"] == f"Club {RIVAL}"
    assert row["club_gender"] == "men"
    assert row["in_my_league"] is True
    assert row["in_continental"] is False
    assert row["is_newgen"] is True
    assert row["group"] == "B"
    assert "in my league" in row["reasons"]
    expected = {"unique_id", "name", "age", "height_cm", "ca", "pa", "club", "club_gender"}
    assert expected <= set(row)


def test_personality_is_stored_and_kept_up_to_date(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    people = [
        replace(p, personality=PLAIN_PERSONALITY) if p.unique_id == MINE_OLD else p for p in world()
    ]
    sync(conn, snapshot(people), settings)
    assert personality_from_json(player_row(conn, MINE_OLD)["personality"]) == PLAIN_PERSONALITY
    assert player_row(conn, MINE_NEW)["personality"] is None
    calmer = replace(PLAIN_PERSONALITY, temperament=20)
    people = [replace(p, personality=calmer) if p.unique_id == MINE_OLD else p for p in world()]
    sync(conn, snapshot(people), settings)
    assert personality_from_json(player_row(conn, MINE_OLD)["personality"]) == calmer


# --- careers -------------------------------------------------------------------------------------


def test_renamed_save_is_the_same_career(conn: sqlite3.Connection, settings: Settings) -> None:
    first = sync(conn, snapshot(world(), file_name="a.fm"), settings)
    second = sync(conn, snapshot(world(), file_name="renamed.fm"), settings)
    assert second.career.key == first.career.key
    assert not second.career.is_new


def test_another_career_gets_its_own_state(conn: sqlite3.Connection, settings: Settings) -> None:
    first = sync(conn, snapshot(world(), career="one"), settings)
    other = sync(conn, snapshot(world(), career="two"), settings)
    assert other.career.is_new and other.career.key != first.career.key
    assert other.enqueued == first.enqueued  # same players, separate queue
    n = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0]
    assert n == 10


def test_fallback_boundary_when_no_gap(conn: sqlite3.Connection, settings: Settings) -> None:
    snap = snapshot([player(MINE_OLD)])
    snap = replace(snap, players=tuple(p for p in snap.players if p.unique_id < CREATED_FROM))
    report = sync(conn, snap, settings)
    assert report.career.newgen_id_source == "fallback"
    assert report.career.newgen_min_id == settings.newgen.fallback_min_unique_id
    assert any("no clear gap" in w for w in report.warnings)


# --- newgens, academy and tracking ----------------------------------------------------------------


def test_player_appearing_mid_career_is_a_newgen(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    youth = player(4242, ca=50, pa=100, age=15)  # a low, reused ID at my club
    sync(conn, snapshot([*world(), youth]), settings)
    row = player_row(conn, 4242)
    assert row["is_newgen"] == 1
    assert row["first_club_id"] == MY_CLUB
    assert row["academy"] == 1
    assert row["group_name"] == "a" and row["tracked"] == 1


def test_first_sync_uses_home_grown_data_for_the_academy(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    grown = player(NEWGEN + 50, ca=50, pa=60, club_id=ABROAD, home_grown=(MY_CLUB,))
    signed = player(NEWGEN + 51, ca=50, pa=60, club_id=MY_CLUB)
    sync(conn, snapshot([grown, signed]), settings)
    assert player_row(conn, NEWGEN + 50)["academy"] == 1
    assert player_row(conn, NEWGEN + 50)["group_name"] == "b"
    # Seen at my club in the first sync, but that alone does not make them academy.
    assert player_row(conn, NEWGEN + 51)["academy"] == 0
    assert player_row(conn, NEWGEN + 51)["first_club_id"] is None


def test_academy_newgen_who_leaves_drops_to_b_and_is_redrawn_in_the_new_kit(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    youth = player(NEWGEN + 60, ca=50, pa=100, age=17)
    sync(conn, snapshot([*world(), youth]), settings)
    add_image(conn, NEWGEN + 60, "a", MY_CLUB, band="W02")

    sold = replace(youth, club_id=ABROAD)
    report = sync(conn, snapshot([*world(), sold]), settings)
    row = player_row(conn, NEWGEN + 60)
    assert (row["group_name"], row["tracked"], row["academy"]) == ("b", 1, 1)
    assert queued(conn)[NEWGEN + 60] == ("kit", "b")
    assert report.enqueued == {("kit", "b"): 1}


def test_untracked_player_changing_club_keeps_the_picture(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    add_image(conn, RIVAL_NEWGEN, "b", RIVAL, band="W04")
    moved = [replace(p, club_id=ABROAD) if p.unique_id == RIVAL_NEWGEN else p for p in world()]
    sync(conn, snapshot(moved), settings)
    assert RIVAL_NEWGEN not in queued(conn)
    # Abroad, PA 152 meets no rule, so the player drops out; the picture stays.
    assert player_row(conn, RIVAL_NEWGEN)["group_name"] is None
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_tracked_player_is_redrawn_on_age_band_change(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    add_image(conn, MINE_OLD, "a", MY_CLUB, band="W08")  # group A at my club, 29: W08
    older = [replace(p, age=34) if p.unique_id == MINE_OLD else p for p in world()]
    sync(conn, snapshot(older), settings)
    assert queued(conn)[MINE_OLD] == ("age", "a")


def test_pre_existing_player_who_leaves_my_club_keeps_the_picture(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    add_image(conn, MINE_OLD, "a", MY_CLUB, band="W08")
    sold = [replace(p, club_id=RIVAL) if p.unique_id == MINE_OLD else p for p in world()]
    sync(conn, snapshot(sold), settings)
    assert player_row(conn, MINE_OLD)["group_name"] is None
    assert MINE_OLD not in queued(conn)


def test_loaned_out_player_counts_as_at_my_club(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    loanee = player(1500, club_id=ABROAD, loan_parent=MY_CLUB)
    sync(conn, snapshot([loanee]), settings)
    assert player_row(conn, 1500)["at_my_club"] == 1
    assert player_row(conn, 1500)["group_name"] == "a"


def test_loaned_out_setting_off(conn: sqlite3.Connection) -> None:
    set_override(conn, "eligibility.loaned_out_at_my_club", False)
    s = load_settings(conn)
    sync(conn, snapshot([player(1500, club_id=ABROAD, loan_parent=MY_CLUB)]), s)
    assert player_row(conn, 1500)["group_name"] is None


# --- free agents and men's football -------------------------------------------------------------


def test_free_agents_count_only_if_seen_at_a_mens_club(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    star = player(NEWGEN + 70, ca=170, club_id=ABROAD)
    unknown = player(NEWGEN + 71, ca=170, club_id=None)
    sync(conn, snapshot([star, unknown]), settings)
    assert player_row(conn, NEWGEN + 71)["in_scope"] == 0

    released = replace(star, club_id=None)
    sync(conn, snapshot([released, unknown]), settings)
    row = player_row(conn, NEWGEN + 70)
    assert (row["in_scope"], row["group_name"]) == (1, "a")
    assert player_row(conn, NEWGEN + 71)["group_name"] is None


def test_tracked_free_agent_waits_for_a_club(conn: sqlite3.Connection, settings: Settings) -> None:
    sync(conn, snapshot(world()), settings)
    add_image(conn, ABROAD_A, "a", ABROAD, band="W03")
    free = [replace(p, club_id=None) if p.unique_id == ABROAD_A else p for p in world()]
    sync(conn, snapshot(free), settings)
    assert ABROAD_A not in queued(conn)
    signed = [replace(p, club_id=RIVAL) if p.unique_id == ABROAD_A else p for p in world()]
    sync(conn, snapshot(signed), settings)
    assert queued(conn)[ABROAD_A] == ("kit", "a")


def test_womens_managed_club_gives_a_warning(conn: sqlite3.Connection, settings: Settings) -> None:
    clubs = DEFAULT_CLUBS | {MY_CLUB: club(MY_CLUB, Gender.WOMEN)}
    report = sync(conn, snapshot(world(), clubs=clubs), settings)
    assert any("not a men's club" in w for w in report.warnings)
    assert MINE_OLD not in groups(conn)


def test_all_genders_when_mens_only_is_off(conn: sqlite3.Connection) -> None:
    set_override(conn, "eligibility.mens_football_only", False)
    sync(conn, snapshot(world()), load_settings(conn))
    assert groups(conn)[WOMAN] == "a"  # PA 190 > 170


# --- continental competitions ---------------------------------------------------------------


def test_continental_competition_ids_from_settings(conn: sqlite3.Connection) -> None:
    sync(conn, snapshot(world()), load_settings(conn))
    assert CONT_NEWGEN not in groups(conn)
    set_override(conn, "eligibility.continental_competition_ids", [CONTINENTAL_COMP])
    report = sync(conn, snapshot(world()), load_settings(conn))
    assert groups(conn)[CONT_NEWGEN] == "b"
    assert report.continental_clubs == 1


# --- threshold changes and departures ---------------------------------------------------------


def test_threshold_change_reclassifies_without_a_new_save(conn: sqlite3.Connection) -> None:
    report = sync(conn, snapshot(world()), load_settings(conn))
    cid = report.career.id

    # Raise the C bar: the queued C job is cancelled.
    s = set_override(conn, "groups.c.league_ca_above", 125)
    result = reclassify(conn, cid, s)
    assert [a.unique_id for a in result.plan.cancel] == [RIVAL_C]
    assert RIVAL_C not in queued(conn)

    # Lower the A bar: the queued B job becomes an A job.
    s = set_override(conn, "groups.a.pa_above", 150)
    result = reclassify(conn, cid, s)
    assert [a.unique_id for a in result.plan.update] == [RIVAL_NEWGEN]
    assert queued(conn)[RIVAL_NEWGEN] == ("initial", "a")


def test_reclassify_dry_run_previews_without_storing(conn: sqlite3.Connection) -> None:
    report = sync(conn, snapshot(world()), load_settings(conn))
    s = set_override(conn, "groups.c.league_ca_above", 125)
    preview = reclassify(conn, report.career.id, s, dry_run=True)
    assert [a.unique_id for a in preview.plan.cancel] == [RIVAL_C]
    assert RIVAL_C in queued(conn)


def test_threshold_change_never_deletes_pictures(conn: sqlite3.Connection) -> None:
    report = sync(conn, snapshot(world()), load_settings(conn))
    add_image(conn, RIVAL_C, "c", RIVAL, band="W04")
    s = set_override(conn, "groups.c.league_ca_above", 125)
    reclassify(conn, report.career.id, s)
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_promotion_to_a_after_a_picture_queues_an_upgrade(conn: sqlite3.Connection) -> None:
    report = sync(conn, snapshot(world()), load_settings(conn))
    add_image(conn, RIVAL_NEWGEN, "b", RIVAL, band="W04")
    s = set_override(conn, "groups.a.pa_above", 150)
    reclassify(conn, report.career.id, s)
    assert queued(conn)[RIVAL_NEWGEN] == ("upgrade", "a")


def test_player_gone_from_the_save_is_kept_but_unqueued(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    without = [p for p in world() if p.unique_id != ABROAD_A]
    sync(conn, snapshot(without), settings)
    row = player_row(conn, ABROAD_A)
    assert row["present"] == 0 and row["group_name"] is None
    assert ABROAD_A not in queued(conn)


def test_pin_never_cancels_and_pin_brings_a_player_in(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    report = sync(conn, snapshot(world()), settings)
    conn.execute("UPDATE players SET pin = 'never' WHERE unique_id = ?", (MINE_OLD,))
    conn.execute("UPDATE players SET pin = 'b' WHERE unique_id = ?", (STAR_ELSEWHERE,))
    reclassify(conn, report.career.id, settings)
    assert MINE_OLD not in queued(conn)
    assert queued(conn)[STAR_ELSEWHERE] == ("initial", "b")
    # The pin survives the next sync.
    sync(conn, snapshot(world()), settings)
    assert player_row(conn, STAR_ELSEWHERE)["pin"] == "b"
    assert queued(conn)[STAR_ELSEWHERE] == ("initial", "b")


def test_failed_job_is_not_requeued(conn: sqlite3.Connection, settings: Settings) -> None:
    sync(conn, snapshot(world()), settings)
    conn.execute("UPDATE jobs SET status = 'failed' WHERE unique_id = ?", (ABROAD_A,))
    report = sync(conn, snapshot(world()), settings)
    assert not report.enqueued
    assert ABROAD_A not in queued(conn)


def test_club_since_is_kept_while_the_player_stays(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    youth = player(4243, join=None)
    sync(conn, snapshot([*world(), youth], game_date=date(2026, 6, 1)), settings)
    assert player_row(conn, 4243)["club_since"] == "2026-06-01"
    sync(conn, snapshot([*world(), youth], game_date=date(2026, 9, 1)), settings)
    assert player_row(conn, 4243)["club_since"] == "2026-06-01"


def test_arrival_date_after_the_in_game_date_is_clamped(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    signing = player(4244, join=date(2026, 9, 25))  # the save says September; it is March
    report = sync(conn, snapshot([*world(), signing]), settings)
    assert player_row(conn, 4244)["club_since"] == "2026-03-01"
    # Clamped to the in-game date, it is still the newest arrival (MINE_NEW joined 1 Feb).
    assert [e.unique_id for e in report.queue][:3] == [4244, MINE_NEW, MINE_OLD]


def test_record_failed_sync(conn: sqlite3.Connection) -> None:
    record_failed_sync(conn, "x.fm", "unsupported build", build="99.0.0")
    row = conn.execute("SELECT * FROM sync_runs").fetchone()
    assert (row["save_file"], row["error"], row["career_id"]) == ("x.fm", "unsupported build", None)
    assert (row["save_mtime"], row["unsupported"]) == (None, 0)
    # Without a save time the failure is not "dealt with": the watcher tries again.
    assert sync_of_save(conn, "x.fm", 100.0) is None

    record_failed_sync(
        conn,
        "x.fm",
        "FM27 build 27.9",
        "27.9",
        save_mtime=100.0,
        unsupported=True,
        trigger="poller",
    )
    done = sync_of_save(conn, "x.fm", 100.0)
    assert done is not None and done["unsupported"] == 1 and done["error"] == "FM27 build 27.9"
    assert sync_of_save(conn, "x.fm", 100.5) is None  # a newer version of the save
    assert sync_of_save(conn, "y.fm", 100.0) is None


def test_sync_records_the_save_time_and_who_started_it(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings, save_mtime=1_000.5, trigger="poller")
    run = conn.execute("SELECT save_mtime, trigger, error FROM sync_runs").fetchone()
    assert tuple(run) == (1_000.5, "poller", None)
    assert sync_of_save(conn, "career.fm", 1_000.5) is not None
    # An older copy of the same career does not make its newest save look older.
    sync(conn, snapshot(world()), settings, save_mtime=500.0)
    assert conn.execute("SELECT last_save_mtime FROM careers").fetchone()[0] == 1_000.5


def test_the_active_career_is_the_one_with_the_newest_save(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    from facegen.db import active_career_id
    from facegen.worker import load_career

    one = sync(conn, snapshot(world(), career="one"), settings, save_mtime=2_000.0).career
    # Career two is synced later, from an older save (say `facegen sync --save`).
    two = sync(
        conn, snapshot(world(), career="two", file_name="two.fm"), settings, save_mtime=1_000.0
    )
    assert two.career.id != one.id
    assert active_career_id(conn) == one.id
    assert load_career(conn).key == one.key  # type: ignore[union-attr]
    # Then career two is saved again: now it is the newest.
    sync(conn, snapshot(world(), career="two", file_name="two.fm"), settings, save_mtime=3_000.0)
    assert active_career_id(conn) == two.career.id


def test_youth_intake_goes_to_the_front_of_the_queue(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """Spec section 6: new players at my club are drawn first at the first save after
    youth intake day, ahead of every job already waiting, groups B and C included."""
    signing = 1003  # pre-existing, abroad at first, signed on intake day
    before = [*world(), player(signing, ca=100, pa=110, age=27, club_id=ABROAD)]
    first = sync(conn, snapshot(before), settings)
    assert first.new_at_my_club == 0  # on a first sync everyone is new; nothing to report

    intake_day = date(2026, 3, 15)
    youth = [NEWGEN + 100 + i for i in range(3)]  # no join date, like created players
    after = [
        *world(),
        player(signing, ca=100, pa=110, age=27, join=intake_day),
        *(player(uid, ca=45, pa=120, age=15) for uid in youth),
    ]
    report = sync(conn, snapshot(after, game_date=intake_day), settings)

    assert report.new_at_my_club == 4
    rows = conn.execute(
        "SELECT unique_id, tier FROM jobs WHERE status = 'queued' ORDER BY tier, rank, id"
    ).fetchall()
    order = [r["unique_id"] for r in rows]
    assert set(order[:4]) == {*youth, signing}
    assert order[4:6] == [MINE_NEW, MINE_OLD]  # waiting since the first sync
    assert {r["tier"] for r in rows[6:]} >= {TIER_B, TIER_C}  # B and C come after


@pytest.mark.parametrize("managed", [None, 999])
def test_no_managed_club(conn: sqlite3.Connection, settings: Settings, managed: int | None) -> None:
    report = sync(conn, snapshot(world(), managed=managed), settings)
    assert report.managed_club is None
    assert any("no managed club" in w for w in report.warnings)
    assert groups(conn) == {ABROAD_A: "a"}


def test_reclassify_with_report_says_what_moved(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    from facegen.db import transaction
    from facegen.settings import set_overrides
    from facegen.sync import ChangeReport, reclassify_with_report

    report = sync(conn, snapshot(world()), settings)
    with transaction(conn):
        applied = set_overrides(conn, {"groups.c.league_ca_above": 125, "groups.a.ca_above": 150})
        r = reclassify_with_report(conn, report.career.id, applied)
    # RIVAL_C (CA 120) drops out of C; nobody new reaches A at 150 in this world.
    assert r.before["c"] == 1 and r.after["c"] == 0
    assert r.before["a"] == r.after["a"]
    assert r.cancelled == 1 and not r.enqueue and r.changed_total == 1
    assert [(c.unique_id, c.before, c.after) for c in r.changes] == [(RIVAL_C, "c", None)]
    assert not r.empty
    assert r.summary() == "C 1 → 0; queue: 1 cancelled"
    assert ChangeReport.from_json(r.to_json()) == r
    # It was applied, so doing it again moves nobody.
    with transaction(conn):
        again = reclassify_with_report(conn, report.career.id, applied)
    assert again.empty and again.summary() == "no player moved; the queue is unchanged"


def test_reclassify_with_report_lists_newly_queued_players(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    from facegen.db import transaction
    from facegen.settings import set_overrides
    from facegen.sync import reclassify_with_report

    report = sync(conn, snapshot(world()), settings)
    with transaction(conn):
        applied = set_overrides(conn, {"groups.b.any_ca_above": 59})
        r = reclassify_with_report(conn, report.career.id, applied)
    # ABROAD_LOW (CA 60) and CONT_NEWGEN (CA 131) now reach B anywhere; RIVAL_C's queued C
    # job becomes a B job.
    assert r.enqueue == {("initial", "b"): 2} and r.updated == 1
    assert {ABROAD_LOW, CONT_NEWGEN, RIVAL_C} <= {c.unique_id for c in r.changes}
    assert r.before["b"] + 3 == r.after["b"] and r.before["c"] - 1 == r.after["c"]


# --- my club's staff (user, 2026-11-12) ----------------------------------------------------------

COACH = NEWGEN + 100  # a created coach: the newgen rules must not apply to staff
ME = NEWGEN + 101  # the save's human manager


def staff_world() -> list:
    from factories import staff_member

    return [
        staff_member(COACH, age=45, job="assistant manager", ca=180),
        staff_member(ME, age=49, job="manager", ca=None, human=True),
    ]


def jobs_of(conn: sqlite3.Connection) -> dict[int, tuple[str, str, str | None]]:
    rows = conn.execute(
        "SELECT unique_id, kind, group_name, target_band FROM jobs WHERE status = 'queued'"
    )
    return {r["unique_id"]: (r["kind"], r["group_name"], r["target_band"]) for r in rows}


def test_my_clubs_staff_are_group_a_tracked_and_aged_by_the_staff_table(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world(), staff=staff_world()), settings)
    coach, me = player_row(conn, COACH), player_row(conn, ME)
    assert (coach["is_staff"], coach["staff_job"], coach["group_name"]) == (
        1,
        "assistant manager",
        "a",
    )
    assert coach["tracked"] and coach["at_my_club"] and coach["present"]
    assert coach["reasons"] == '["on my club\'s staff"]'  # not "newgen, CA 180 > 155"
    assert (me["staff_job"], me["ca"], me["height_cm"]) == ("manager (you)", 0, None)
    # Wide staff bands: 45 is W05 and 49 is W06 (group A at my club would say W10).
    assert jobs_of(conn)[COACH] == ("initial", "a", "W05")
    assert jobs_of(conn)[ME] == ("initial", "a", "W06")


def test_a_staff_age_band_change_is_redrawn(conn: sqlite3.Connection, settings: Settings) -> None:
    from factories import staff_member

    sync(conn, snapshot(world(), staff=[staff_member(COACH, age=52)]), settings)
    conn.execute("DELETE FROM jobs")
    add_image(conn, COACH, "a", MY_CLUB, "W06")
    sync(conn, snapshot(world(), staff=[staff_member(COACH, age=52)]), settings)
    assert COACH not in jobs_of(conn)
    sync(conn, snapshot(world(), staff=[staff_member(COACH, age=53)]), settings)
    assert jobs_of(conn)[COACH] == ("age", "a", "W07")


def test_a_player_coach_is_stored_once_as_a_player(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    from factories import staff_member

    sync(conn, snapshot(world(), staff=[staff_member(MINE_OLD, job="coach")]), settings)
    row = player_row(conn, MINE_OLD)
    assert (row["is_staff"], row["staff_job"], row["ca"]) == (0, None, 120)
    assert (
        conn.execute("SELECT COUNT(*) FROM players WHERE unique_id = ?", (MINE_OLD,)).fetchone()[0]
        == 1
    )


def test_staff_who_leave_keep_their_picture_and_are_no_longer_drawn(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world(), staff=staff_world()), settings)
    sync(conn, snapshot(world(), staff=staff_world()[1:]), settings)  # the coach left
    coach = player_row(conn, COACH)
    assert not coach["present"] and coach["group_name"] is None and not coach["tracked"]
    assert coach["reasons"] == '["no longer on my club\'s staff"]'
    assert COACH not in jobs_of(conn) and ME in jobs_of(conn)


def test_staff_stay_as_they_were_when_the_staff_reader_failed(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world(), staff=staff_world()), settings)
    sync(conn, snapshot(world(), staff=(), staff_read=False), settings)
    assert player_row(conn, COACH)["present"] and player_row(conn, COACH)["group_name"] == "a"
    assert COACH in jobs_of(conn)


def test_staff_pictures_can_be_turned_off(conn: sqlite3.Connection, settings: Settings) -> None:
    report = sync(conn, snapshot(world(), staff=staff_world()), settings)
    s = set_override(conn, "staff.enabled", False)
    reclassify(conn, report.career.id, s)
    coach = player_row(conn, COACH)
    assert coach["group_name"] is None and not coach["tracked"]
    assert "staff pictures are off" in coach["reasons"]
    assert COACH not in jobs_of(conn) and ME not in jobs_of(conn)
    # The players are unchanged; a pin still hand-picks a member of staff.
    assert groups(conn)[MINE_OLD] == "a"
    conn.execute("UPDATE players SET pin = 'b' WHERE unique_id = ?", (COACH,))
    reclassify(conn, report.career.id, s)
    assert player_row(conn, COACH)["group_name"] == "b"
    assert jobs_of(conn)[COACH] == ("initial", "b", "W05")  # still the staff table


def test_new_staff_are_not_counted_as_arrivals(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(world()), settings)
    report = sync(conn, snapshot(world(), staff=staff_world()), settings)
    assert report.new_at_my_club == 0
    assert report.counts["a"] == 2 + 2 + 1  # my two players, the two staff, ABROAD_A


# --- pre-existing players without a photo (user, 2026-11-12) ---------------------------------

WITH_PHOTO = 1003  # pre-existing, my league, CA 150: FM27 ships a photo
WITH_FACEPACK = 1004  # the same, with a facepack picture
NO_PHOTO_B = 1005  # the same, with neither: B like a newgen in my league


def photo_scan(
    game: set[int] | None,
    facepack: set[int] | None = None,
    *,
    version: str = "v1",
    warnings: tuple[str, ...] = (),
) -> PhotoScan:
    return PhotoScan(
        state={"catalog": version, "configs": [], "folders": {}},
        index=PhotoIndex(
            None if game is None else frozenset(game), frozenset(facepack or ()), warnings
        ),
    )


def photo_world() -> list[PlayerRecord]:
    return world() + [
        player(uid, ca=150, pa=150, club_id=RIVAL)
        for uid in (WITH_PHOTO, WITH_FACEPACK, NO_PHOTO_B)
    ]


def test_pre_existing_players_without_a_photo_are_picked_like_newgens(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    scan = photo_scan({WITH_PHOTO}, {WITH_FACEPACK})
    report = sync(conn, snapshot(photo_world()), settings, photos=scan)
    g = groups(conn)
    assert (g[STAR_ELSEWHERE], g[NO_PHOTO_B]) == ("a", "b")
    assert WITH_PHOTO not in g and WITH_FACEPACK not in g
    assert g[MINE_OLD] == "a"  # my club, as before
    assert queued(conn)[STAR_ELSEWHERE] == ("initial", "a")
    assert queued(conn)[NO_PHOTO_B] == ("initial", "b")
    star = player_row(conn, STAR_ELSEWHERE)
    assert json.loads(star["reasons"]) == ["no photo, CA 190 > 155"]
    assert star["tracked"] == 1
    assert json.loads(player_row(conn, NO_PHOTO_B)["reasons"]) == [
        "no photo, in my league, CA 150 > 130",
        "no photo, CA 150 > 140",
    ]
    c = report.counts
    # Without a photo: MINE_OLD (at my club), STAR_ELSEWHERE and NO_PHOTO_B.
    assert (c["with_photo"], c["without_photo"], c["without_photo_picked"]) == (2, 3, 2)


def test_without_a_photo_list_no_pre_existing_player_is_picked(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    scan = photo_scan(None, warnings=("FM27's photo list (x) could not be read",))
    report = sync(conn, snapshot(photo_world()), settings, photos=scan)
    g = groups(conn)
    assert STAR_ELSEWHERE not in g and NO_PHOTO_B not in g
    assert any("could not be read" in w for w in report.warnings)
    assert report.counts["without_photo"] == 0


def test_before_any_photo_scan_no_pre_existing_player_is_picked(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(photo_world()), settings)
    assert STAR_ELSEWHERE not in groups(conn) and NO_PHOTO_B not in groups(conn)


def test_the_no_photo_setting_reclassifies_without_a_new_save(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    report = sync(conn, snapshot(photo_world()), settings, photos=photo_scan(set()))
    assert STAR_ELSEWHERE in queued(conn)
    off = set_override(conn, "no_photo.enabled", False)
    reclassify(conn, report.career.id, off)
    assert STAR_ELSEWHERE not in groups(conn) and STAR_ELSEWHERE not in queued(conn)
    assert groups(conn)[RIVAL_NEWGEN] == "b"  # newgens are unaffected
    on = set_override(conn, "no_photo.enabled", True)
    reclassify(conn, report.career.id, on)
    assert queued(conn)[STAR_ELSEWHERE] == ("initial", "a")


def test_a_player_who_gains_a_photo_keeps_the_drawn_picture(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(photo_world()), settings, photos=photo_scan(set()))
    add_image(conn, STAR_ELSEWHERE, "a", ABROAD, "W03")
    conn.execute("UPDATE jobs SET status = 'cancelled' WHERE unique_id = ?", (NO_PHOTO_B,))
    later = photo_scan(set(), {STAR_ELSEWHERE, NO_PHOTO_B}, version="v2")
    sync(conn, snapshot(photo_world()), settings, photos=later)
    assert STAR_ELSEWHERE not in groups(conn) and NO_PHOTO_B not in groups(conn)
    assert NO_PHOTO_B not in queued(conn)
    current = conn.execute(
        "SELECT COUNT(*) FROM images WHERE unique_id = ? AND is_current = 1", (STAR_ELSEWHERE,)
    ).fetchone()[0]
    assert current == 1


def test_an_unchanged_photo_scan_keeps_the_stored_ids(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    sync(conn, snapshot(photo_world()), settings, photos=photo_scan(set()))
    unchanged = PhotoScan(state={"catalog": "v1", "configs": [], "folders": {}}, index=None)
    sync(conn, snapshot(photo_world()), settings, photos=unchanged)
    assert groups(conn)[STAR_ELSEWHERE] == "a"


def test_a_dry_run_stores_no_photo_scan(conn: sqlite3.Connection, settings: Settings) -> None:
    report = sync(conn, snapshot(photo_world()), settings, photos=photo_scan(set()), dry_run=True)
    assert report.enqueued[("initial", "a")] == 4  # my two, ABROAD_A and STAR_ELSEWHERE
    for table in ("photos", "photo_scan"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
