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


def test_a_competitions_clubs_are_its_fixtures_and_its_tables_only_without_any() -> None:
    tables = (T(CONTINENTAL, (30, 31)), T(LEAGUE, (1, 2)), T(CUP_GROUP, (4, 5)))
    fixtures = {CONTINENTAL: {32: 2}, CONFERENCE_LEAGUE: {33: 1}, CUP: {3: 1}}
    got = clubs_in_competitions([CONTINENTAL, CONFERENCE_LEAGUE, CUP_GROUP], tables, fixtures)
    # CONTINENTAL's table is left out: with fixtures, it may be from an earlier edition.
    assert got == {32, 33, 4, 5}
    assert clubs_in_competitions([], tables, fixtures) == frozenset()


def test_a_uefa_competitions_clubs_leave_out_an_earlier_editions_table() -> None:
    # This edition's fixtures hold my club (1) and a French club (2); last edition's table
    # still holds another French club (3).
    clubs = {1: club(1, nation=1), 2: club(2, nation=2), 3: club(3, nation=2)}
    ucl = 1301394
    fixtures = {ucl: {1: 2, 2: 2}}
    tables = (T(ucl, (1, 3)),)
    found = continental_competitions(
        1, clubs, tables, fixtures, seasons={ucl: 2027}, game_date=date(2027, 12, 26)
    )
    assert found == {ucl}
    assert clubs_in_competitions(found, tables, fixtures) == {1, 2}


def test_a_stale_table_cannot_establish_my_clubs_membership() -> None:
    # The managed club was in both competitions last season; each now has other participants.
    tables = (T(LEAGUE, (1, 2)), T(CONTINENTAL, (1, 2)))
    clubs = {c: club(c, nation=1 if c < 10 else c) for c in range(1, 18)}
    fixtures = {LEAGUE: {3: 2, 4: 2}, CONTINENTAL: {c: 2 for c in range(10, 18)}}
    assert my_league(Ladder(), 1, clubs, tables, fixtures) == (frozenset(), frozenset())
    assert continental_competitions(1, clubs, tables, fixtures) == frozenset()
    assert (
        continental_competitions(1, clubs, tables, fixtures, frozenset({CONTINENTAL}))
        == frozenset()
    )


def test_my_league_is_off_when_its_season_is_older_than_another() -> None:
    """Just promoted, before the save's rules place my club: the old division's latest season
    (48 fixtures) against the new one's (46). The seasons conflict, so no league."""
    old, new = LEAGUE, CUP
    fixtures = {
        old: {c: 48 for c in range(1, 25)},
        new: {1: 46} | {c: 46 for c in range(2, 25)},
    }
    assert my_league_competitions(1, CLUBS, (), fixtures) == {old}  # without seasons
    seasons = {old: 2026, new: 2027}
    assert my_league_competitions(1, CLUBS, (), fixtures, seasons) == frozenset()
    competitions, members = my_league(Ladder(), 1, CLUBS, (), fixtures, seasons=seasons)
    assert competitions == frozenset() and members == frozenset()
    # Once the new season's fixtures are the most, it is my league.
    fixtures[new] = {c: 50 for c in range(1, 25)}
    assert my_league_competitions(1, CLUBS, (), fixtures, seasons) == {new}


def test_seasons_that_agree_or_are_unknown_leave_my_league_alone() -> None:
    fixtures = {LEAGUE: {c: 46 for c in range(1, 25)}, CUP: {1: 3, 2: 3}}
    assert my_league_competitions(1, CLUBS, (), fixtures, {LEAGUE: 2027, CUP: 2027}) == {LEAGUE}
    assert my_league_competitions(1, CLUBS, (), fixtures, {LEAGUE: 2027}) == {LEAGUE}
    assert my_league_competitions(1, CLUBS, (), fixtures, {}) == {LEAGUE}
    # A season known for a cup but not for the league: they cannot be compared, so off.
    assert my_league_competitions(1, CLUBS, (), fixtures, {CUP: 2027}) == frozenset()
    # A competition my club has no fixture in does not count.
    fixtures[CUP_GROUP] = {5: 2, 6: 2}
    tables = (T(CUP_GROUP, (1, 5, 6)),)
    seasons = {LEAGUE: 2026, CUP_GROUP: 2027}
    assert my_league_competitions(1, CLUBS, tables, fixtures, seasons) == {LEAGUE}


# --- continental competitions ------------------------------------------------------------------


def league_fixtures(first: int, count: int, games: int | None = None) -> dict[int, int]:
    return {c: games if games is not None else 2 * (count - 1) for c in range(first, first + count)}


SECOND = 102
DOMESTIC = {c: club(c) for c in (*range(1, 21), *range(101, 125), *range(201, 225))}
SEASON_GAMES = {
    SECOND: league_fixtures(101, 24),
    CUP: {c: 3 for c in (*range(1, 21), *range(101, 125), *range(201, 225))},  # a domestic cup
}


def test_continental_competitions_my_club_plays_in() -> None:
    clubs = DOMESTIC | {c: ClubRecord(c, "x", Gender.MEN, n, 6000) for n, c in
                        ((2 + i // 3, 500 + i) for i in range(30))}  # fmt: skip
    europe = {110: 6, 1: 6, **{500 + i: 6 for i in range(30)}}
    friendlies = {110: 2, **{c: 1 for c in range(201, 225)}, **{500 + i: 1 for i in range(30)}}
    tour = {110: 2, 500: 2, 503: 2, 506: 2}  # a pre-season event: too few clubs
    fixtures = {**SEASON_GAMES, 900: europe, 901: friendlies, 902: tour, 903: {1: 6, 500: 6}}
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
# Club 24 of League Two is Welsh, as Newport is.
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
    assert ladder.mine is not None and ladder.mine.competition_id == NL
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
    # Without the rules: no ladder, and my league is every club with a fixture in it.
    plain = league_ladder(110, ENGLISH, (), fixtures)
    assert plain == Ladder()
    competitions, clubs = my_league(plain, 110, ENGLISH, (), fixtures)
    assert competitions == {NL} and 201 in clubs


def test_no_ladder_without_my_club_the_rules_or_my_league_in_them() -> None:
    """Nobody is then in the league above or below: no guess from the clubs' reputations."""
    assert league_ladder(None, ENGLISH, (), GAMES, rules=ENGLAND) == Ladder()
    assert league_ladder(999, ENGLISH, (), GAMES, rules=ENGLAND) == Ladder()  # unknown club
    assert league_ladder(110, ENGLISH, (), GAMES) == Ladder()  # the rules unread
    assert league_ladder(110, ENGLISH, (), {}, rules=ENGLAND) == Ladder()  # no fixture
    other = {SECOND: league_fixtures(101, 24)}  # a league the rules do not list
    assert league_ladder(110, ENGLISH, (), other, rules=ENGLAND) == Ladder()
    assert Ladder().at(1) == () and Ladder().at(-1) == () and Ladder().club_leagues() == {}


def test_rules_that_disagree_give_no_ladder() -> None:
    wrong_level = CompetitionRules(
        ENGLAND.division_lists, {**ENGLAND.levels, NL: 3}, ENGLAND.children
    )
    assert rules_ladder_of(110, rules=wrong_level) == Ladder()
    unsure = CompetitionRules(
        ENGLAND.division_lists, ENGLAND.levels, ENGLAND.children, frozenset({NLS})
    )
    assert rules_ladder_of(110, rules=unsure) == Ladder()
    twice = CompetitionRules((*ENGLAND.division_lists, DivisionList(None, False, (L1, NL))))
    assert rules_ladder_of(110, rules=twice) == Ladder()  # my league in two lists


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
    # A table row and no fixture at all: no edition confirmed either.
    assert UECL not in continental_competitions(
        110, clubs, stale, fixtures, seasons=seasons, game_date=today
    )


def test_a_uefa_edition_must_be_known_and_match_the_save_date() -> None:
    fixtures = {UCL: {1: 1, 2: 1}}
    today = date(2027, 6, 30)  # the 2026/27 edition is current
    for years, game_date in (({}, today), ({UCL: 2027}, today), ({UCL: 2026}, None)):
        for listed in (frozenset(), frozenset({UCL})):
            assert not continental_competitions(
                1, CLUBS, (), fixtures, listed, seasons=years, game_date=game_date
            )
    # The qualifying clubs on the available fixture can both come from one nation.
    assert not my_league_competitions(1, CLUBS, (), fixtures)
