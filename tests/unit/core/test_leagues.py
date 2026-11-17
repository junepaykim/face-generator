from __future__ import annotations

from factories import club

from facegen.core.leagues import clubs_in_competitions, my_league_competitions
from facegen.core.records import LeagueTableRecord as T

HOME = {i: club(i, nation=1) for i in range(1, 30)}
AWAY = {i: club(i, nation=2) for i in range(30, 60)}
CLUBS = HOME | AWAY
LEAGUE, CUP, CUP_GROUP, CONTINENTAL, CONFERENCE_LEAGUE = 10, 11, 12, 13, 14


def test_my_league_from_the_largest_domestic_table() -> None:
    tables = (
        T(LEAGUE, tuple(range(1, 21))),
        T(CUP_GROUP, (1, 2, 3, 4)),  # a domestic cup group holding my club
        T(CONTINENTAL, (1, 30, 31, 32)),  # not domestic
    )
    assert my_league_competitions(1, CLUBS, tables, {}) == {LEAGUE}


def test_conferences_of_one_league_count_as_one() -> None:
    tables = (T(LEAGUE, tuple(range(1, 16))), T(LEAGUE, tuple(range(16, 30))))
    league = my_league_competitions(1, CLUBS, tables, {})
    assert league == {LEAGUE}
    assert clubs_in_competitions(league, tables, {}) == frozenset(range(1, 30))


def test_my_league_from_fixtures_before_any_table_exists() -> None:
    fixtures = {
        LEAGUE: {c: 30 for c in range(1, 17)},  # 30 scheduled league fixtures each
        CUP: {1: 8} | {c: 1 for c in range(2, 29)},  # pre-season event, more clubs
        CONTINENTAL: {1: 40, 30: 40, 31: 40, 32: 40, 33: 40},  # more fixtures, but abroad
    }
    assert my_league_competitions(1, CLUBS, (), fixtures) == {LEAGUE}


def test_fixture_count_beats_table_size() -> None:
    tables = (T(CUP_GROUP, tuple(range(1, 25))),)
    fixtures = {LEAGUE: {c: 30 for c in range(1, 17)}, CUP_GROUP: {1: 6}}
    assert my_league_competitions(1, CLUBS, tables, fixtures) == {LEAGUE}


def test_mostly_domestic_league_with_a_foreign_club_still_counts() -> None:
    # A league with one club from another nation's FA still has 19 of 20 home clubs.
    tables = (T(LEAGUE, (*range(1, 20), 30)),)
    assert my_league_competitions(1, CLUBS, tables, {}) == {LEAGUE}


def test_no_league_without_a_managed_club_or_data() -> None:
    tables = (T(LEAGUE, tuple(range(1, 21))),)
    assert my_league_competitions(None, CLUBS, tables, {}) == frozenset()
    assert my_league_competitions(999, CLUBS, tables, {}) == frozenset()  # unknown club
    assert my_league_competitions(1, CLUBS, (), {}) == frozenset()
    assert my_league_competitions(25, CLUBS, tables, {}) == frozenset()  # not in any table


def test_table_without_a_competition_is_ignored() -> None:
    assert my_league_competitions(1, CLUBS, (T(None, (1, 2, 3)),), {}) == frozenset()


def test_clubs_in_competitions_unions_tables_and_fixtures() -> None:
    tables = (T(CONTINENTAL, (30, 31)), T(LEAGUE, (1, 2)))
    fixtures = {CONTINENTAL: {32: 2}, CONFERENCE_LEAGUE: {33: 1}, CUP: {3: 1}}
    got = clubs_in_competitions([CONTINENTAL, CONFERENCE_LEAGUE], tables, fixtures)
    assert got == {30, 31, 32, 33}
    assert clubs_in_competitions([], tables, fixtures) == frozenset()
