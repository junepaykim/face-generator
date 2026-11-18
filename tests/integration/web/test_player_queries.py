"""The Players page's queries after the SQLite follow-up (web/queries.py): the page chosen
first and then read in full, and the Review and club-name filters as lists looked up once.
They must give exactly what the one-step queries gave; checked here on made-up rows, and on a
copy of a 181,000-player database while measuring (docs/research.md)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from itertools import count

import pytest

from facegen.persistence import db
from facegen.web import queries
from facegen.web.queries import PlayerFilter

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
        rows, total = queries.players_page(c, CAREER, replace(f, page=page), per_page)
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
    rows, _ = queries.players_page(conn, CAREER, PlayerFilter(sort="club"))
    assert [r["unique_id"] for r in rows] == [d, a, b, c]
    assert [r["club"] for r in rows] == [None, "AFC Wrexham", "telford town", "Telford United"]


def test_the_club_name_filter_matches_without_case(conn: sqlite3.Connection) -> None:
    united = add(conn, 100, 100, 10)
    town = add(conn, 100, 100, 11)
    add(conn, 100, 100, 12)
    add(conn, 100, 100, None)
    add(conn, 100, 100, 99)  # a club that is not stored
    add(conn, 100, 100, 10, career=2)
    rows, total = queries.players_page(conn, CAREER, PlayerFilter(club="TELFORD"))
    assert total == 2 and {r["unique_id"] for r in rows} == {united, town}
    rows, total = queries.players_page(conn, CAREER, PlayerFilter(club="town"))
    assert total == 1 and rows[0]["unique_id"] == town and rows[0]["club"] == "telford town"


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
    rows, total = queries.players_page(conn, CAREER, PlayerFilter(picture="review"))
    assert {r["unique_id"] for r in rows} == {failed, failed_then_cancelled}
    assert total == 2 == queries.review_count(conn, CAREER)
    assert {r["latest_job"] for r in rows} == {"failed"}
    rows, total = queries.players_page(conn, CAREER, PlayerFilter(picture="queued"))
    assert [r["unique_id"] for r in rows] == [queued_again] and rows[0]["job_status"] == "queued"


def test_counting_joins_nothing_for_the_lookups() -> None:
    for f in (PlayerFilter(picture="review"), PlayerFilter(club="tel")):
        where, _ = queries._where(f)
        assert queries._count_from(where) == " FROM players p"
    where, _ = queries._where(PlayerFilter(sort="club"))
    assert "JOIN clubs c" in queries._count_from(f"{where} {queries.SORTS['club']}")
