"""The fmsave adapter's conversions, with stand-in records shaped like fmsave's.

The last test reads a real save copy when FACEGEN_TEST_SAVE points at one; it is skipped
otherwise, so the suite needs no save file.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from facegen.datasource.fmsave_source import (
    FmsaveDataSource,
    club_gender,
    competition_key,
    convert_clubs,
    convert_fixtures,
    convert_league_tables,
    convert_player,
    convert_players,
)
from facegen.datasource.records import Gender


def coded(name: str) -> NS:
    return NS(label=NS(name=name), raw=0)


def fm_club(uid: int, unique_id: int | None, gender: str | None = "MEN", nation: int = 1) -> NS:
    return NS(
        uid=uid,
        unique_id=unique_id,
        name=f"Club {uid}",
        gender=coded(gender) if gender else None,
        nation_id=nation,
    )


def fm_player(unique_id: int | None, **kw: object) -> NS:
    base: dict[str, object] = {
        "uid": 1,
        "unique_id": unique_id,
        "name": "Ana Lima",
        "full_name": "Ana Lima",
        "legal_name": None,
        "birth_date": date(2008, 5, 20),
        "age": 17,
        "height_cm": 172,
        "ability": NS(current=95, potential=160, potential_range_code=None),
        "club_uid": 10,
        "on_loan": False,
        "loan_parent_club_uid": None,
        "loan_start": None,
        "home_grown_club_uids": (10, None, 99),
        "club_join_date": date(2025, 7, 1),
        "natural_positions": ("MC", "DM"),
        "attributes": NS(strength=11),
    }
    base.update(kw)
    return NS(**base)


UID_MAP = {10: 1010, 11: 1011, 12: 1012}


def test_competition_key_prefers_the_database_id() -> None:
    assert competition_key(1301385, 7) == 1301385
    assert competition_key(None, 7) == -8
    assert competition_key(None, 0) == -1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (coded("MEN"), Gender.MEN),
        (coded("WOMEN"), Gender.WOMEN),
        (coded("UNKNOWN"), Gender.UNKNOWN),
        (None, Gender.UNKNOWN),
    ],
)
def test_club_gender(value: NS | None, expected: Gender) -> None:
    assert club_gender(value) is expected


def test_clubs_are_keyed_on_their_stable_unique_id() -> None:
    records, uid_map = convert_clubs([fm_club(10, 1010), fm_club(11, None, "WOMEN")])
    assert uid_map == {10: 1010, 11: 11}  # falls back to the uid without a unique ID
    assert records[1010].gender is Gender.MEN
    assert records[11].gender is Gender.WOMEN
    assert records[1010].nation_id == 1


def test_player_conversion() -> None:
    p = convert_player(fm_player(2_002_070_000), UID_MAP, date(2026, 3, 1))
    assert p.unique_id == 2_002_070_000
    assert (p.ca, p.pa, p.height_cm, p.strength) == (95, 160, 172, 11)
    assert p.club_id == 1010
    assert p.loan_parent_club_id is None
    assert p.home_grown_club_ids == (1010,)  # None and unknown clubs dropped
    assert not p.is_goalkeeper


def test_arrival_is_the_loan_start_while_on_loan() -> None:
    loanee = fm_player(
        1,
        on_loan=True,
        club_uid=11,
        loan_parent_club_uid=10,
        loan_start=date(2026, 7, 13),
        club_join_date=date(2022, 8, 5),  # when they joined the parent club
    )
    assert convert_player(loanee, UID_MAP, None).club_join_date == date(2026, 7, 13)
    permanent = fm_player(1, loan_start=date(2026, 7, 13))  # not on loan: ignored
    assert convert_player(permanent, UID_MAP, None).club_join_date == date(2025, 7, 1)


def test_loan_parent_only_while_on_loan() -> None:
    on = convert_player(
        fm_player(1, club_uid=11, on_loan=True, loan_parent_club_uid=10), UID_MAP, None
    )
    assert (on.club_id, on.loan_parent_club_id) == (1011, 1010)
    off = convert_player(fm_player(1, on_loan=None, loan_parent_club_uid=10), UID_MAP, None)
    assert off.loan_parent_club_id is None


def test_free_agent_and_unknown_club() -> None:
    assert convert_player(fm_player(1, club_uid=None), UID_MAP, None).club_id is None
    assert convert_player(fm_player(1, club_uid=555), UID_MAP, None).club_id is None


def test_goalkeeper_from_best_natural_position() -> None:
    assert convert_player(fm_player(1, natural_positions=("GK",)), UID_MAP, None).is_goalkeeper
    assert not convert_player(fm_player(1, natural_positions=()), UID_MAP, None).is_goalkeeper


def test_age_falls_back_to_the_birth_date() -> None:
    p = fm_player(1, age=None, birth_date=date(2008, 5, 20))
    assert convert_player(p, UID_MAP, date(2026, 5, 19)).age == 17
    assert convert_player(p, UID_MAP, date(2026, 5, 20)).age == 18
    assert convert_player(p, UID_MAP, None).age is None


def test_name_fallbacks() -> None:
    assert convert_player(fm_player(1, name=None), UID_MAP, None).name == "Ana Lima"
    nameless = fm_player(7, name=None, full_name=None, legal_name=None)
    assert convert_player(nameless, UID_MAP, None).name == "Player 7"


def test_unknown_potential_stays_unknown() -> None:
    p = fm_player(1, ability=NS(current=80, potential=None, potential_range_code=-9))
    assert convert_player(p, UID_MAP, None).pa is None


def test_players_without_or_with_repeated_unique_ids_are_skipped() -> None:
    players, skipped = convert_players(
        [fm_player(1), fm_player(None), fm_player(2), fm_player(1, name="Copy")], UID_MAP, None
    )
    assert [p.unique_id for p in players] == [1, 2]
    assert players[0].name == "Ana Lima"
    assert skipped == 2


def test_league_tables_keep_first_team_rows() -> None:
    table = NS(
        competition_id=30,
        rows=(
            NS(club_uid=10, team_slot=0),
            NS(club_uid=11, team_slot=None),
            NS(club_uid=12, team_slot=1),  # a reserve team
            NS(club_uid=None, team_slot=0),
            NS(club_uid=555, team_slot=0),  # unknown club
        ),
    )
    unlinked = NS(competition_id=None, rows=(NS(club_uid=10, team_slot=0),))
    out = convert_league_tables([table, unlinked], UID_MAP, {30: 40})
    assert out[0].competition_id == 40
    assert out[0].club_ids == (1010, 1011)
    assert out[1].competition_id is None


def fixture(comp: int | None, season: int | None, home: int, away: int, slots=(0, 0)) -> NS:
    return NS(
        competition_id=comp,
        season_start_year=season,
        home_club_uid=home,
        away_club_uid=away,
        home_team_slot=slots[0],
        away_team_slot=slots[1],
    )


def test_fixtures_count_first_team_games_in_the_latest_season() -> None:
    fixtures = [
        fixture(6601, 2025, 10, 11),  # last season: ignored
        fixture(6601, 2026, 10, 11),
        fixture(6601, 2026, 11, 10),
        fixture(6601, 2026, 10, 12, slots=(0, 1)),  # reserve side does not count
        fixture(7233, None, 10, 12),  # no season stored
        fixture(None, 2026, 10, 11),  # no competition
        fixture(9999, 2026, 10, 11),  # competition not in the table
    ]
    out = convert_fixtures(fixtures, UID_MAP, {6601: 2000132563, 7233: 2000315308})
    assert out == {2000132563: {1010: 3, 1011: 2}, 2000315308: {1010: 1, 1012: 1}}


@pytest.mark.skipif(
    not os.environ.get("FACEGEN_TEST_SAVE"), reason="set FACEGEN_TEST_SAVE to a save copy"
)
def test_reads_a_real_save_copy() -> None:
    snap = FmsaveDataSource().load(Path(os.environ["FACEGEN_TEST_SAVE"]))
    assert snap.meta.game == "FM27"
    assert len(snap.players) > 1000
    assert snap.managed_club_id in snap.clubs
    assert {c.gender for c in snap.clubs.values()} >= {Gender.MEN}
    assert len({p.unique_id for p in snap.players}) == len(snap.players)
