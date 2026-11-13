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
    convert_staff,
    staff_job,
)
from facegen.datasource.records import Gender, Personality


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


def fm_staff(unique_id: int | None, **kw: object) -> NS:
    base: dict[str, object] = {
        "uid": 5,
        "unique_id": unique_id,
        "is_human_manager": False,
        "name": "Rhys Morris",
        "full_name": "Rhys Morris",
        "legal_name": None,
        "birth_date": date(1980, 2, 1),
        "age": 46,
        "club_uid": 10,
        "job": NS(label=NS(name="ASSISTANT_MANAGER"), raw=20),
        "contract_start": date(2025, 6, 1),
        "ability": NS(current=55, potential=70),
        "personality": None,
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


def test_personality_is_read_when_the_save_has_one() -> None:
    values = {
        "adaptability": 14,
        "ambition": 15,
        "loyalty": 9,
        "pressure": 7,
        "professionalism": 18,
        "sportsmanship": 12,
        "temperament": 3,
        "controversy": 11,
    }
    p = convert_player(fm_player(1, personality=NS(**values)), UID_MAP, None)
    assert p.personality == Personality(**values)
    assert convert_player(fm_player(1, personality=None), UID_MAP, None).personality is None
    assert convert_player(fm_player(1), UID_MAP, None).personality is None  # no field at all


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
    with_personality = sum(p.personality is not None for p in snap.players)
    assert with_personality > 0.9 * len(snap.players)


def test_staff_job_in_words() -> None:
    assert staff_job(NS(label=NS(name="DIRECTOR_OF_FOOTBALL"), raw=10)) == "director of football"
    assert staff_job(NS(label=NS(name="UNKNOWN"), raw=99)) is None
    assert staff_job(None) is None


def test_convert_staff_keeps_one_club_and_each_person_once() -> None:
    staff = convert_staff(
        [
            fm_staff(3001),
            fm_staff(3002, club_uid=11),  # another club: not read
            fm_staff(3001, name="Repeat"),  # a second row for the same person
            fm_staff(None),  # no unique ID
            fm_staff(
                3003,
                is_human_manager=True,
                ability=None,
                age=None,
                birth_date=date(1977, 8, 1),
                job=NS(label=NS(name="MANAGER"), raw=16),
            ),
        ],
        UID_MAP,
        10,
        date(2026, 7, 13),
    )
    assert [s.unique_id for s in staff] == [3001, 3003]
    coach, me = staff
    assert (coach.club_id, coach.job, coach.ca, coach.pa) == (1010, "assistant manager", 55, 70)
    assert coach.club_join_date == date(2025, 6, 1) and coach.name == "Rhys Morris"
    assert me.is_human_manager and me.ca is None and me.pa is None
    assert me.age == 48  # from the birth date when the save has no age


# --- load(): how fmsave's errors and warnings reach the report ---------------------------------


class FakeSave:
    """Stands in for fmsave.Save: one club, one player, and readers that can fail."""

    def __init__(
        self,
        *,
        known_build: bool = True,
        fail: Exception | None = None,
        staff_fail: Exception | None = None,
    ) -> None:
        self.info = NS(
            game="FM27",
            build="27.0.0+2486695" if known_build else "27.1.0+2600000",
            known_build=known_build,
            game_date=date(2026, 7, 13),
            save_name="Career",
            file_name="Career.fm",
        )
        self._fail = fail
        self._staff_fail = staff_fail
        self.closed = False

    def clubs(self) -> list[NS]:
        return [fm_club(10, 1010)]

    def players(self) -> list[NS]:
        if self._fail is not None:
            raise self._fail
        return [fm_player(2_002_100_000)]

    def managed_clubs(self) -> list[NS]:
        return [NS(club_uid=10)]

    def competitions(self) -> list[NS]:
        return []

    def league_tables(self) -> list[NS]:
        return []

    def fixtures(self) -> list[NS]:
        return []

    def staff(self) -> list[NS]:
        if self._staff_fail is not None:
            raise self._staff_fail
        return [fm_staff(3001), fm_staff(3002, club_uid=11)]

    def close(self) -> None:
        self.closed = True


def _patch_open(monkeypatch: pytest.MonkeyPatch, result: FakeSave | Exception) -> None:
    import fmsave

    def fake_open(path: Path) -> FakeSave:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(fmsave, "open", fake_open)


def test_load_reads_and_closes_the_save(monkeypatch: pytest.MonkeyPatch) -> None:
    save = FakeSave()
    _patch_open(monkeypatch, save)
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert [p.unique_id for p in snap.players] == [2_002_100_000]
    assert [s.unique_id for s in snap.staff] == [3001]  # my club's staff only
    assert snap.staff_read
    assert snap.meta.known_build and save.closed
    assert not any("not one fmsave" in w for w in snap.meta.warnings)


def test_a_failing_staff_reader_leaves_the_staff_as_they_were(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fmsave

    save = FakeSave(staff_fail=fmsave.ReaderCheckError("staff: checks failed"))
    _patch_open(monkeypatch, save)
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.players and snap.staff == () and not snap.staff_read
    assert any("staff data unavailable" in w for w in snap.meta.warnings)


def test_an_unknown_build_is_read_with_a_clear_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    import fmsave

    _patch_open(monkeypatch, FakeSave(known_build=False))
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.meta.warnings[0].startswith(
        f"FM27 build 27.1.0+2600000 is not one fmsave {fmsave.__version__} was made for"
    )


def test_a_reader_refusing_the_build_makes_the_save_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fmsave

    from facegen.datasource.base import UnsupportedSaveError

    save = FakeSave(
        known_build=False, fail=fmsave.UnsupportedGameError("players: section schema differs")
    )
    _patch_open(monkeypatch, save)
    with pytest.raises(UnsupportedSaveError) as info:
        FmsaveDataSource().load(Path("Career.fm"))
    assert info.value.build == "27.1.0+2600000"
    assert info.value.reader == f"fmsave {fmsave.__version__}"
    assert "FM27 build 27.1.0+2600000 cannot be read" in str(info.value)
    assert "section schema differs" in str(info.value)
    assert save.closed


def test_a_save_from_another_game_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    import fmsave

    from facegen.datasource.base import SaveReadError, UnsupportedSaveError

    _patch_open(monkeypatch, fmsave.UnsupportedGameError("Career.fm is an FM25 save"))
    with pytest.raises(UnsupportedSaveError, match="FM25"):
        FmsaveDataSource().load(Path("Career.fm"))
    _patch_open(monkeypatch, fmsave.CorruptSaveError("truncated"))
    with pytest.raises(SaveReadError, match="truncated"):
        FmsaveDataSource().load(Path("Career.fm"))
    damaged = FakeSave(fail=fmsave.ReaderCheckError("players: checks failed"))
    _patch_open(monkeypatch, damaged)
    with pytest.raises(SaveReadError, match="checks failed"):
        FmsaveDataSource().load(Path("Career.fm"))
    assert damaged.closed
