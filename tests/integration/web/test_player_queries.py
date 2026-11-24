"""Check roster pagination, indexed lookups and agreement with the web Review count."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from itertools import count

import pytest

from facegen.persistence import db, player_queries
from facegen.persistence.player_queries import PlayerFilter, players_page
from facegen.web.queries import review_count

CAREER = 1
_ids = count(1)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = db.connect(":memory:")
    now = "2026-11-17T00:00:00"
    c.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'k', 'Career', '[]', ?, ?), (2, 'k2', 'Other', '[]', ?, ?)",
        (now, now, now, now),
    )
    c.executemany(
        "INSERT INTO clubs (career_id, club_id, name, gender) VALUES (?, ?, ?, 'men')",
        [(1, 10, "Telford United"), (1, 11, "telford town"), (1, 12, "AFC Wrexham"),
         (2, 10, "Telford United")],
    )  # fmt: skip
    yield c
    c.close()


def add(
    c: sqlite3.Connection, ca: int, pa: int, club: int | None, *, career: int = CAREER, **kw: object
) -> int:
    uid = next(_ids)
    c.execute(
        "INSERT INTO players (career_id, unique_id, name, ca, pa, club_id, is_newgen, in_scope,"
        " at_my_club, in_my_league, in_continental, academy, present)"
        " VALUES (?, ?, ?, ?, ?, ?, 1, 1, 0, 0, 0, 0, ?)",
        (career, uid, kw.get("name", f"P{uid}"), ca, pa, club, kw.get("present", 1)),
    )
    return uid


def job(c: sqlite3.Connection, uid: int, status: str, career: int = CAREER) -> None:
    now = "2026-11-17T00:00:00"
    c.execute(
        "INSERT INTO jobs (career_id, unique_id, kind, group_name, tier, rank, status, reason,"
        " created_at, updated_at) VALUES (?, ?, 'initial', 'b', 3, 0, ?, 'test', ?, ?)",
        (career, uid, status, now, now),
    )


def all_pages(c: sqlite3.Connection, f: PlayerFilter, per_page: int) -> list[int]:
    seen: list[int] = []
    page = 1
    while True:
        rows, total = players_page(c, CAREER, replace(f, page=page), per_page)
        assert len(rows) == min(per_page, total - len(seen))
        if not rows:
            break
        seen += [r["unique_id"] for r in rows]
        page += 1
    assert len(seen) == total
    return seen


def test_pages_agree_on_the_order_through_ties(conn: sqlite3.Connection) -> None:
    uids = [add(conn, 120, 150, 10) for _ in range(7)] + [add(conn, 130, 140, 11) for _ in range(5)]
    add(conn, 200, 200, 10, present=0)  # gone: never listed
    add(conn, 200, 200, 10, career=2)  # another career
    seen = all_pages(conn, PlayerFilter(), per_page=3)
    assert seen == sorted(uids[7:]) + sorted(uids[:7])  # CA first, then the unique ID
    assert all_pages(conn, PlayerFilter(sort="pa"), per_page=4) == sorted(uids[:7]) + sorted(
        uids[7:]
    )


def test_sorting_by_club_is_by_its_name_without_case(conn: sqlite3.Connection) -> None:
    a = add(conn, 100, 100, 12)  # AFC Wrexham
    b = add(conn, 150, 150, 11)  # telford town
    c = add(conn, 90, 90, 10)  # Telford United
    d = add(conn, 95, 95, None)  # no club: first, as NULL sorts first
    rows, _ = players_page(conn, CAREER, PlayerFilter(sort="club"))
    assert [r["unique_id"] for r in rows] == [d, a, b, c]
    assert [r["club"] for r in rows] == [None, "AFC Wrexham", "telford town", "Telford United"]


def test_the_club_name_filter_matches_without_case(conn: sqlite3.Connection) -> None:
    united = add(conn, 100, 100, 10)
    town = add(conn, 100, 100, 11)
    add(conn, 100, 100, 12)
    add(conn, 100, 100, None)
    add(conn, 100, 100, 99)  # a club that is not stored
    add(conn, 100, 100, 10, career=2)
    rows, total = players_page(conn, CAREER, PlayerFilter(club="TELFORD"))
    assert total == 2 and {r["unique_id"] for r in rows} == {united, town}
    rows, total = players_page(conn, CAREER, PlayerFilter(club="town"))
    assert total == 1 and rows[0]["unique_id"] == town and rows[0]["club"] == "telford town"
    rows, total = players_page(conn, CAREER, PlayerFilter(club="town", club_id=10))
    assert total == 1 and rows[0]["unique_id"] == united


@pytest.mark.parametrize(("search_field", "row_field"), [("q", "name"), ("club", "club")])
@pytest.mark.parametrize(
    ("term", "expected"), [("a_", "A_B"), ("%", "A%B"), ("!", "A!B"), ("Xb", "AxB")]
)
def test_literal_name_substrings(
    conn: sqlite3.Connection, search_field: str, row_field: str, term: str, expected: str
) -> None:
    names = ["A_B", "A%B", "AxB", "A!B"]
    for club_id, (player, club) in enumerate(zip(names, reversed(names), strict=True), start=20):
        conn.execute(
            "INSERT INTO clubs (career_id, club_id, name, gender) VALUES (?, ?, ?, 'men')",
            (CAREER, club_id, club),
        )
        add(conn, 100, 100, club_id, name=player)

    rows, total = players_page(conn, CAREER, PlayerFilter(**{search_field: term}))
    assert total == 1
    assert [row[row_field] for row in rows] == [expected]


def test_numeric_player_search_matches_id_or_name(conn: sqlite3.Connection) -> None:
    exact_id = add(conn, 100, 100, None, name="Captain")
    padded = f"00{exact_id}"
    arabic_digits = str(exact_id).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    named = add(conn, 100, 100, None, name=f"Player {padded} ({arabic_digits})")
    add(conn, 100, 100, None, name="Teammate")
    for term in (str(exact_id), padded, arabic_digits):
        rows, total = players_page(conn, CAREER, PlayerFilter(q=term))
        assert total == 2
        assert {row["unique_id"] for row in rows} == {exact_id, named}


@pytest.mark.parametrize("term", ["9223372036854775808", "²"])
def test_digits_outside_the_id_format_match_names(conn: sqlite3.Connection, term: str) -> None:
    named = add(conn, 100, 100, None, name=f"Player {term}")
    add(conn, 100, 100, None, name="Teammate")
    rows, total = players_page(conn, CAREER, PlayerFilter(q=term))
    assert total == 1
    assert [row["unique_id"] for row in rows] == [named]


def test_the_review_filter_is_the_players_whose_latest_job_failed(
    conn: sqlite3.Connection,
) -> None:
    failed = add(conn, 100, 100, 10)
    job(conn, failed, "failed")
    failed_then_cancelled = add(conn, 100, 100, 10)  # a cancelled job does not count
    job(conn, failed_then_cancelled, "failed")
    job(conn, failed_then_cancelled, "cancelled")
    retried = add(conn, 100, 100, 10)
    job(conn, retried, "failed")
    job(conn, retried, "done")
    queued_again = add(conn, 100, 100, 10)
    job(conn, queued_again, "failed")
    job(conn, queued_again, "queued")
    elsewhere = add(conn, 100, 100, 10, career=2)
    job(conn, elsewhere, "failed", career=2)
    add(conn, 100, 100, 10)
    rows, total = players_page(conn, CAREER, PlayerFilter(picture="review"))
    assert {r["unique_id"] for r in rows} == {failed, failed_then_cancelled}
    assert total == 2 == review_count(conn, CAREER)
    assert {r["latest_job"] for r in rows} == {"failed"}
    rows, total = players_page(conn, CAREER, PlayerFilter(picture="queued"))
    assert [r["unique_id"] for r in rows] == [queued_again] and rows[0]["job_status"] == "queued"


def test_the_auto_update_filter_is_the_tracked_players(conn: sqlite3.Connection) -> None:
    """The dashboard's Auto-update count links to this list (Phase 17e)."""
    tracked = [add(conn, 100, 100, 10), add(conn, 90, 90, None)]
    add(conn, 120, 120, 10)  # not tracked
    gone = add(conn, 100, 100, 10, present=0)  # left the save: counted nowhere
    elsewhere = add(conn, 100, 100, 10, career=2)
    conn.execute(
        "UPDATE players SET tracked = 1 WHERE unique_id IN (?, ?, ?, ?)",
        (*tracked, gone, elsewhere),
    )
    rows, total = players_page(conn, CAREER, PlayerFilter(group="tracked"))
    assert {r["unique_id"] for r in rows} == set(tracked) and total == 2
    assert all(r["tracked"] == 1 for r in rows)


def test_counting_joins_nothing_for_the_lookups() -> None:
    lookups = (
        PlayerFilter(picture="review"),
        PlayerFilter(club="tel"),
        PlayerFilter(shown="targets"),
    )
    for f in lookups:
        where, _ = player_queries._where(f, CAREER)
        assert player_queries._count_from(where) == " FROM players p"
    where, _ = player_queries._where(PlayerFilter(sort="club"), CAREER)
    assert "JOIN clubs c" in player_queries._count_from(f"{where} {player_queries.SORTS['club']}")


def test_page_looks_up_selected_players_and_their_job_history_by_index(
    conn: sqlite3.Connection,
) -> None:
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    players_page(conn, CAREER, PlayerFilter())
    conn.set_trace_callback(None)
    plan = [row["detail"] for row in conn.execute("EXPLAIN QUERY PLAN " + statements[-1])]
    assert any("SEARCH p " in step and "career_id=? AND unique_id=?" in step for step in plan)
    assert any("SEARCH jj " in step and "jobs_by_player" in step for step in plan)


def picture(c: sqlite3.Connection, uid: int, *, current: bool, career: int = CAREER) -> None:
    version = (
        1
        + c.execute(
            "SELECT COUNT(*) FROM images WHERE career_id = ? AND unique_id = ?", (career, uid)
        ).fetchone()[0]
    )
    c.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model, prompt,"
        " negative_prompt, seed, is_current, created_at)"
        " VALUES (?, ?, ?, 'initial', 'b', 'm', 'p', 'n', 1, ?, 'now')",
        (career, uid, version, int(current)),
    )


def test_the_picture_targets_are_the_people_this_app_draws(conn: sqlite3.Connection) -> None:
    """Phase 17h (D6): in a group, pinned (to never as well), with a picture, a queued or
    running job, or in review; everyone in the save is the other choice."""
    targets = {}
    targets["grouped"] = add(conn, 120, 150, 10)
    conn.execute("UPDATE players SET group_name = 'c' WHERE unique_id = ?", (targets["grouped"],))
    targets["never"] = add(conn, 100, 100, 10)
    conn.execute("UPDATE players SET pin = 'never' WHERE unique_id = ?", (targets["never"],))
    targets["picture"] = add(conn, 100, 100, 11)
    picture(conn, targets["picture"], current=True)
    targets["queued"] = add(conn, 90, 90, 12)
    job(conn, targets["queued"], "queued")
    targets["running"] = add(conn, 90, 90, 12)
    job(conn, targets["running"], "running")
    targets["review"] = add(conn, 80, 80, None)
    job(conn, targets["review"], "failed")
    targets["review, then cancelled"] = add(conn, 80, 80, None)
    job(conn, targets["review, then cancelled"], "failed")
    job(conn, targets["review, then cancelled"], "cancelled")
    others = [add(conn, 150, 170, 10) for _ in range(3)]  # pre-existing stars, say
    failed_then_done = add(conn, 70, 70, 10)
    job(conn, failed_then_done, "failed")
    job(conn, failed_then_done, "done")
    old_picture = add(conn, 70, 70, 11)  # its picture was removed
    picture(conn, old_picture, current=False)
    others += [failed_then_done, old_picture]
    gone = add(conn, 200, 200, 10, present=0)
    conn.execute("UPDATE players SET group_name = 'a' WHERE unique_id = ?", (gone,))
    elsewhere = add(conn, 200, 200, 10, career=2)
    conn.execute("UPDATE players SET group_name = 'a' WHERE unique_id = ?", (elsewhere,))
    picture(conn, others[0], current=True, career=2)  # the same ID drawn in another career
    job(conn, others[1], "queued", career=2)

    rows, total = players_page(conn, CAREER, PlayerFilter(shown="targets"))
    assert total == len(targets) == 7
    assert {r["unique_id"] for r in rows} == set(targets.values())
    assert player_queries.player_count(conn, CAREER, PlayerFilter(shown="targets")) == 7
    rows, total = players_page(conn, CAREER, PlayerFilter(shown="everyone"))
    assert total == len(targets) + len(others) == 12
    assert {r["unique_id"] for r in rows} == {*targets.values(), *others}
    assert PlayerFilter().shown == "everyone"  # the query's own default narrows nothing
    # With the other filters, and across pages in the same order.
    rows, total = players_page(conn, CAREER, PlayerFilter(shown="targets", club_id=10))
    assert total == 2 and {r["unique_id"] for r in rows} == {targets["grouped"], targets["never"]}
    rows, total = players_page(conn, CAREER, PlayerFilter(shown="targets", picture="review"))
    assert {r["unique_id"] for r in rows} == {targets["review"], targets["review, then cancelled"]}
    in_order = [
        r["unique_id"] for r in players_page(conn, CAREER, PlayerFilter(shown="targets"))[0]
    ]
    assert all_pages(conn, PlayerFilter(shown="targets"), per_page=2) == in_order


def test_the_targets_are_found_by_the_players_key(conn: sqlite3.Connection) -> None:
    """The list of targets drives the count (research.md section 15b): looked up by key, the
    jobs by their status index and the pins by their partial index, never every player."""
    where, args = player_queries._where(PlayerFilter(shown="targets"), CAREER)
    plan = [
        r["detail"]
        for r in conn.execute(
            f"EXPLAIN QUERY PLAN SELECT COUNT(*) FROM players p WHERE {where}", args
        )
    ]
    assert any("SEARCH p " in step and "career_id=? AND unique_id=?" in step for step in plan)
    assert any("players_pinned" in step for step in plan)
    assert sum("jobs_queue (status=?)" in step for step in plan) == 2
    assert not any(step.startswith("SCAN") for step in plan)
