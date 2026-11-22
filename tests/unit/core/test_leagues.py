from __future__ import annotations

from datetime import date

from factories import club

from facegen.core.leagues import (
    UEFA_CLUB_COMPETITIONS,
    Ladder,
    clubs_in_competitions,
    competition_name,
    continental_competitions,
    league_ladder,
    my_league,
    my_league_competitions,
    pyramids,
    uefa_season,
)
from facegen.core.records import ClubRecord, CompetitionRules, DivisionList, Gender
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


# --- the league ladder (Phase 16) -----------------------------------------------------------------


def rated(first: int, count: int, reputation: int, nation: int = 1, step: int = 10) -> dict:
    """`count` clubs from `first`, reputations spread a little around `reputation`."""
    return {
        c: ClubRecord(c, f"Club {c}", Gender.MEN, nation, reputation + step * (i - count // 2))
        for i, c in enumerate(range(first, first + count))
    }


def league_fixtures(first: int, count: int, games: int | None = None) -> dict[int, int]:
    return {c: games if games is not None else 2 * (count - 1) for c in range(first, first + count)}


TOP, SECOND, NORTH, SOUTH, CUP, YOUTH, WOMEN = 101, 102, 103, 104, 105, 106, 107
PYRAMID = (
    rated(1, 20, 7000)  # the top level
    | rated(101, 24, 5000)  # level 2
    | rated(201, 24, 2600, step=30)  # level 3, north
    | rated(301, 24, 2700, step=30)  # level 3, south: parallel
    | {c: ClubRecord(c, f"W{c}", Gender.WOMEN, 1, 6000) for c in range(401, 413)}
)
FIXTURES = {
    TOP: league_fixtures(1, 20),
    SECOND: league_fixtures(101, 24),
    NORTH: league_fixtures(201, 24),
    SOUTH: league_fixtures(301, 24),
    CUP: {c: 3 for c in (*range(1, 21), *range(101, 125), *range(201, 225))},
    YOUTH: league_fixtures(1, 14, 26),  # the top clubs' reserve league: they play more above
    WOMEN: league_fixtures(401, 12),
}


def test_the_ladder_orders_my_nations_leagues_by_reputation() -> None:
    ladder = league_ladder(110, PYRAMID, (), FIXTURES)
    assert [lg.competition_id for lg in ladder.leagues] == [TOP, SECOND, SOUTH, NORTH]
    assert ladder.mine is not None and ladder.mine.competition_id == SECOND
    assert [lg.competition_id for lg in ladder.at(1)] == [TOP]
    assert sorted(lg.competition_id for lg in ladder.at(-1)) == [NORTH, SOUTH]
    assert ladder.at(0) == (ladder.mine,)


def test_parallel_divisions_share_a_level_and_neither_is_above_the_other() -> None:
    ladder = league_ladder(210, PYRAMID, (), FIXTURES)  # in the north division
    assert ladder.mine is not None and ladder.mine.competition_id == NORTH
    assert [lg.competition_id for lg in ladder.at(1)] == [SECOND]
    assert ladder.at(-1) == ()  # the lowest level loaded
    south = next(lg for lg in ladder.leagues if lg.competition_id == SOUTH)
    assert ladder.offset(south) == 0
    offsets = ladder.club_leagues()
    assert offsets[310] == (SOUTH, 0) and offsets[105] == (SECOND, 1)


def test_cups_reserve_and_womens_leagues_are_not_on_the_ladder() -> None:
    ids = {lg.competition_id for lg in league_ladder(110, PYRAMID, (), FIXTURES).leagues}
    assert not ids & {CUP, YOUTH, WOMEN}


def test_a_top_division_has_no_league_above() -> None:
    ladder = league_ladder(5, PYRAMID, (), FIXTURES)
    assert ladder.mine is not None and ladder.mine.competition_id == TOP
    assert ladder.at(1) == ()
    assert [lg.competition_id for lg in ladder.at(-1)] == [SECOND]


def test_no_ladder_without_my_club_its_league_or_reputations() -> None:
    assert league_ladder(None, PYRAMID, (), FIXTURES).mine is None
    assert league_ladder(999, PYRAMID, (), FIXTURES).leagues == ()
    assert league_ladder(110, PYRAMID, (), {}).mine is None
    unrated = {c: ClubRecord(c, "x", Gender.MEN, 1) for c in PYRAMID}
    assert league_ladder(110, unrated, (), FIXTURES).leagues == ()


def test_a_league_from_its_table_alone() -> None:
    tables = (T(SECOND, tuple(range(101, 125))), T(CUP_GROUP, (101, 102, 103, 104)))
    ladder = league_ladder(110, PYRAMID, tables, {})
    assert ladder.mine is not None and ladder.mine.competition_id == SECOND


def test_continental_competitions_my_club_plays_in() -> None:
    clubs = PYRAMID | {c: ClubRecord(c, "x", Gender.MEN, n, 6000) for n, c in
                       ((2 + i // 3, 500 + i) for i in range(30))}  # fmt: skip
    europe = {110: 6, 1: 6, **{500 + i: 6 for i in range(30)}}
    friendlies = {110: 2, **{c: 1 for c in range(201, 225)}, **{500 + i: 1 for i in range(30)}}
    tour = {110: 2, 500: 2, 503: 2, 506: 2}  # a pre-season event: too few clubs
    fixtures = {**FIXTURES, 900: europe, 901: friendlies, 902: tour, 903: {1: 6, 500: 6}}
    assert continental_competitions(110, clubs, (), fixtures) == {900}
    assert continental_competitions(110, clubs, (), fixtures, frozenset({902, 903})) == {902}
    assert continental_competitions(None, clubs, (), fixtures) == frozenset()


# --- the ladder from the save's own rules (docs/research.md sections 23-25) ------------------

PL, CH, L1, L2, NL, NLNS, NLN, NLS, WSL, WSL2 = (
    11,
    12,
    13,
    14,
    109201,
    150990,
    5123054,
    5123055,
    2000101095,
    2000107193,
)
ENGLAND = CompetitionRules(
    division_lists=(
        DivisionList(765, False, (PL, CH, L1, L2, NL, NLNS)),
        DivisionList(765, True, (WSL, WSL2)),
    ),
    levels={PL: 0, CH: 1, L1: 2, L2: 3, NL: 4, NLNS: 5},
    children={NLNS: (NLN, NLS)},
)
# No reputations: the rules need none. Club 24 of League Two is Welsh, as Newport is.
ENGLISH = {
    c: ClubRecord(c, f"Club {c}", Gender.MEN, 2 if c == 24 else 1)
    for c in (*range(1, 25), *range(101, 125), *range(201, 225), *range(301, 325))
} | {c: ClubRecord(c, f"W{c}", Gender.WOMEN, 1) for c in range(401, 413)}
SEASON = {L2: 2027, NL: 2027, NLN: 2027, NLS: 2027, WSL: 2027}
GAMES = {
    L2: league_fixtures(1, 24),
    NL: league_fixtures(101, 24),
    NLN: league_fixtures(201, 24),
    NLS: league_fixtures(301, 24),
    WSL: league_fixtures(401, 12),
    CUP: {c: 3 for c in (*range(1, 25), *range(101, 125))},
}


def rules_ladder_of(
    club_id: int,
    fixtures: dict = GAMES,
    seasons: dict = SEASON,
    rules: CompetitionRules = ENGLAND,
) -> Ladder:
    return league_ladder(club_id, ENGLISH, (), fixtures, seasons=seasons, rules=rules)


def test_the_rules_place_my_league_and_the_levels_around_it() -> None:
    ladder = rules_ladder_of(110)  # the National League, tier 5
    assert ladder.from_rules and ladder.mine is not None and ladder.mine.competition_id == NL
    assert ladder.mine.level == 4
    assert [lg.competition_id for lg in ladder.at(1)] == [L2]
    assert 24 in ladder.at(1)[0].club_ids  # a Welsh club in England's pyramid
    assert sorted(lg.competition_id for lg in ladder.at(-1)) == [NLN, NLS]  # both groups
    assert ladder.at(2) == ()  # League One has no club in the save: not loaded
    assert competition_name(NL) == "National League" and competition_name(999) is None


def test_parallel_groups_share_a_level_and_only_mine_is_my_league() -> None:
    ladder = rules_ladder_of(210)  # National League North
    assert ladder.mine is not None and ladder.mine.competition_id == NLN
    south = next(lg for lg in ladder.leagues if lg.competition_id == NLS)
    assert ladder.offset(south) == 0 and ladder.at(0) == (ladder.mine,)
    assert [lg.competition_id for lg in ladder.at(1)] == [NL]
    assert ladder.at(-1) == ()  # the lowest level in the rules


def test_a_club_just_promoted_counts_in_its_new_division() -> None:
    """Telford, December 2027: last season's 48 games in the North (whose new season is not in
    the save yet) are more than this season's 46 in the National League."""
    north = league_fixtures(201, 24, 46) | {101: 48}
    fixtures = GAMES | {NLN: north, NL: league_fixtures(102, 23, 44) | {101: 46}}
    ladder = rules_ladder_of(101, fixtures, SEASON | {NLN: 2026})
    assert ladder.mine is not None and ladder.mine.competition_id == NL
    assert 101 not in next(lg for lg in ladder.leagues if lg.competition_id == NLN).club_ids


def test_my_league_from_the_rules_leaves_out_a_play_off_opponent() -> None:
    fixtures = GAMES | {NL: league_fixtures(101, 24) | {201: 2}}  # a North club's play-off
    ladder = rules_ladder_of(110, fixtures)
    competitions, clubs = my_league(ladder, 110, ENGLISH, (), fixtures)
    assert competitions == {NL} and 201 not in clubs and len(clubs) == 24
    # Without the rules: every club with a fixture in the competition, as before.
    plain = league_ladder(110, ENGLISH, (), fixtures)
    assert not plain.from_rules
    assert 201 in my_league(plain, 110, ENGLISH, (), fixtures)[1]


def test_a_league_the_rules_do_not_list_is_ordered_by_reputation() -> None:
    ladder = league_ladder(110, PYRAMID, (), FIXTURES, rules=ENGLAND)  # other IDs
    assert (
        not ladder.from_rules and ladder.mine is not None and ladder.mine.competition_id == SECOND
    )


def test_rules_that_disagree_give_no_rules_ladder() -> None:
    wrong_level = CompetitionRules(
        ENGLAND.division_lists, {**ENGLAND.levels, NL: 3}, ENGLAND.children
    )
    assert not rules_ladder_of(110, rules=wrong_level).from_rules
    unsure = CompetitionRules(
        ENGLAND.division_lists, ENGLAND.levels, ENGLAND.children, frozenset({NLS})
    )
    assert not rules_ladder_of(110, rules=unsure).from_rules
    twice = CompetitionRules((*ENGLAND.division_lists, DivisionList(None, False, (L1, NL))))
    assert not rules_ladder_of(110, rules=twice).from_rules  # my league in two lists


def test_each_gender_has_its_own_list() -> None:
    assert pyramids(ENGLAND, Gender.WOMEN) == [((WSL,), (WSL2,))]
    assert len(pyramids(ENGLAND, Gender.MEN)[0]) == 6
    assert len(pyramids(ENGLAND, Gender.UNKNOWN)) == 2
    ladder = rules_ladder_of(405)
    assert ladder.mine is not None and ladder.mine.competition_id == WSL and ladder.at(-1) == ()


# --- the UEFA club competitions (docs/research.md section 24) ---------------------------------

UCL, UEL, UECL = 1301394, 1301396, 31051584


def test_the_uefa_season_starts_in_july() -> None:
    assert uefa_season(date(2027, 6, 30)) == 2026 and uefa_season(date(2027, 7, 1)) == 2027
    assert uefa_season(None) is None
    assert set(UEFA_CLUB_COMPETITIONS) == {UCL, UEL, UECL}


def test_a_uefa_competition_counts_in_its_current_edition() -> None:
    clubs = ENGLISH | {c: ClubRecord(c, "x", Gender.MEN, 3) for c in range(600, 603)}
    fixtures = {
        **GAMES,
        UCL: {110: 2, 600: 2},  # this season's qualifier: two clubs are enough
        UEL: {110: 6, 601: 6, 602: 6},  # last season's: unconfirmed for this one
    }
    seasons = SEASON | {UCL: 2027, UEL: 2026}
    today = date(2027, 8, 1)
    found = continental_competitions(110, clubs, (), fixtures, seasons=seasons, game_date=today)
    assert found == {UCL}
    listed = frozenset({UEL, UCL})
    assert continental_competitions(
        110, clubs, (), fixtures, listed, seasons=seasons, game_date=today
    ) == {UCL}
    # A table row from an earlier edition, with this edition's fixtures not holding my club.
    stale = (T(UECL, (110, 600, 601)),)
    played = {**fixtures, UECL: {600: 2, 601: 2}}
    assert UECL not in continental_competitions(
        110, clubs, stale, played, seasons=seasons | {UECL: 2027}, game_date=today
    )
