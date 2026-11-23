"""The fmsave adapter's conversions, with stand-in records shaped like fmsave's.

The real-save check lives in tests/opt_in/test_real_save.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from support.fmsave import UID_MAP, coded, fm_staff

from facegen.core.records import Gender, Personality
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
    out, seasons = convert_fixtures(fixtures, UID_MAP, {6601: 2000132563, 7233: 2000315308})
    assert out == {2000132563: {1010: 3, 1011: 2}, 2000315308: {1010: 1, 1012: 1}}
    # The season the counts are from; none where the fixtures store none.
    assert seasons == {2000132563: 2026}


@pytest.mark.parametrize(
    ("game_date", "edition", "uefa_counts"),
    [
        (date(2027, 6, 30), 2026, {1010: 1, 1011: 1}),
        (date(2027, 7, 1), 2027, {1011: 1, 1012: 1}),
        (None, None, {}),
    ],
)
def test_uefa_fixtures_use_the_save_edition_before_discarding_other_seasons(
    game_date: date | None, edition: int | None, uefa_counts: dict[int, int]
) -> None:
    ucl, league = 1301394, 11
    fixtures = [
        fixture(7, 2026, 10, 11),
        fixture(7, 2027, 11, 12),
        fixture(7, None, 10, 12),
        fixture(8, 2026, 10, 11),
        fixture(8, 2027, 11, 12),
    ]
    counts, years = convert_fixtures(fixtures, UID_MAP, {7: ucl, 8: league}, game_date)
    assert counts.get(ucl, {}) == uefa_counts
    assert years.get(ucl) == edition
    assert counts[league] == {1011: 1, 1012: 1} and years[league] == 2027


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


class FakePool:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.entry_count = len(names)

    def name_at(self, buffer: bytes, index: int) -> str | None:
        return self.names[index] if 0 <= index < len(self.names) else None


class FakeContext:
    """fmsave's internals that datasource/people.py borrows, over support.people's bytes."""

    def name_pools(self) -> NS:
        from support.people import FIRST, SURNAMES

        return NS(
            first_names=FakePool(FIRST),
            surnames=FakePool(SURNAMES),
            common_names=FakePool([]),
            end_offset=0,
        )

    @contextmanager
    def section(self, name: str) -> Iterator[bytes]:
        from support.people import db

        assert name == "game_db"
        yield db()

    def club_index(self) -> NS:
        return NS(uid_by_club_index={12666: 10, 99: 11})


class FakeSave:
    """Stands in for fmsave.Save: one club, one player, and readers that can fail."""

    def __init__(
        self,
        *,
        known_build: bool = True,
        fail: Exception | None = None,
        staff_fail: Exception | None = None,
        people: bool = False,
        nations_fail: Exception | None = None,
    ) -> None:
        if people:
            self._context = FakeContext()
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
        self._nations_fail = nations_fail
        self.closed = False

    def nations(self) -> list[NS]:
        if self._nations_fail is not None:
            raise self._nations_fail
        return [NS(nation_id=i, database_id=d) for i, d in ((7, 1651), (139, 765), (140, 769))]

    def clubs(self) -> list[NS]:
        return [fm_club(10, 1010)]

    def players(self) -> list[NS]:
        if self._fail is not None:
            raise self._fail
        return [fm_player(2_002_100_000, nation_id=7)]

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
        return [
            fm_staff(3001, first_name="Emma", nation_id=139),
            fm_staff(3002, club_uid=11),
            fm_staff(3003, first_name="Jake", nation_id=99),  # a nation the table lacks
        ]

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
    assert [s.unique_id for s in snap.staff] == [3001, 3003]  # my club's staff only
    assert snap.staff_read and snap.competitions_read
    assert snap.meta.known_build and save.closed
    assert not any("not one fmsave" in w for w in snap.meta.warnings)


def test_reader_selects_managers_from_the_snapshots_domestic_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = FakeSave()
    monkeypatch.setattr(save, "clubs", lambda: [fm_club(uid, cid) for uid, cid in UID_MAP.items()])
    monkeypatch.setattr(save, "competitions", lambda: [NS(id=7, database_id=11)])
    monkeypatch.setattr(save, "fixtures", lambda: [fixture(7, 2026, 10, 11)])
    monkeypatch.setattr(
        save,
        "staff",
        lambda: [fm_staff(uid, club_uid=uid, job=coded("MANAGER"), team_slot=0) for uid in UID_MAP],
    )
    _patch_open(monkeypatch, save)
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.domestic.competition_ids == {11}
    assert snap.domestic.club_ids == {1010, 1011} == {s.club_id for s in snap.staff}
    # Callers building a different snapshot from these records get its own membership.
    changed = replace(snap, fixture_counts={11: {1010: 1, 1012: 1}})
    assert changed.domestic.club_ids == {1010, 1012}


def _league_save(monkeypatch: pytest.MonkeyPatch) -> FakeSave:
    """My club (1010) and 1011 in competition 7 (editor ID 11), with a manager at each club."""
    save = FakeSave()
    monkeypatch.setattr(save, "clubs", lambda: [fm_club(uid, cid) for uid, cid in UID_MAP.items()])
    monkeypatch.setattr(save, "competitions", lambda: [NS(id=7, database_id=11)])
    monkeypatch.setattr(save, "fixtures", lambda: [fixture(7, 2026, 10, 11)])
    rows = (NS(club_uid=10, team_slot=0), NS(club_uid=11, team_slot=0))
    monkeypatch.setattr(save, "league_tables", lambda: [NS(competition_id=7, rows=rows)])
    monkeypatch.setattr(
        save,
        "staff",
        lambda: [fm_staff(uid, club_uid=uid, job=coded("MANAGER"), team_slot=0) for uid in UID_MAP],
    )
    return save


def _fail(message: str):  # type: ignore[no-untyped-def]
    import fmsave

    def reader() -> list[NS]:
        raise fmsave.ReaderCheckError(message)

    return reader


def test_failing_league_tables_still_leave_the_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    save = _league_save(monkeypatch)
    monkeypatch.setattr(save, "league_tables", _fail("league tables: checks failed"))
    _patch_open(monkeypatch, save)
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.competitions_read and snap.league_tables == ()
    assert snap.fixture_counts == {11: {1010: 1, 1011: 1}}
    assert snap.domestic.club_ids == {1010, 1011} == {s.club_id for s in snap.staff}
    assert any("league tables unavailable" in w for w in snap.meta.warnings)


@pytest.mark.parametrize("reader", ["competitions", "fixtures"])
def test_unread_competitions_leave_no_league_and_say_so(
    monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    save = _league_save(monkeypatch)
    monkeypatch.setattr(save, reader, _fail(f"{reader}: checks failed"))
    _patch_open(monkeypatch, save)
    snap = FmsaveDataSource().load(Path("Career.fm"))
    # Neither half of what was read stands in for the rest: the sync keeps what it stored.
    assert not snap.competitions_read
    assert (snap.league_tables, snap.fixture_counts, snap.fixture_seasons) == ((), {}, {})
    assert snap.domestic.club_ids == frozenset()
    assert [s.club_id for s in snap.staff] == [1010]  # my club's staff only
    assert snap.staff_read and snap.players
    assert any("stay as the last sync read them" in w for w in snap.meta.warnings)


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


def test_a_failed_layout_check_on_an_unknown_build_makes_the_save_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FM27 27.1.0 with fmsave 0.11.0 (research.md section 17): the reader's checks refuse the
    club table. The page should say the build cannot be read yet, not "unexpected error"."""
    import fmsave

    from facegen.datasource.base import UnsupportedSaveError

    message = "section 'game_db': club table framing is invalid or ambiguous"
    save = FakeSave(known_build=False, fail=fmsave.ReaderCheckError(message))
    _patch_open(monkeypatch, save)
    with pytest.raises(UnsupportedSaveError) as info:
        FmsaveDataSource().load(Path("Career.fm"))
    assert info.value.build == "27.1.0+2600000"
    assert "FM27 build 27.1.0+2600000 cannot be read" in str(info.value)
    assert "club table framing" in str(info.value) and save.closed


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


def test_team_slot_is_read() -> None:
    assert convert_player(fm_player(1, team_slot=2), UID_MAP, None).team_slot == 2
    assert convert_player(fm_player(1), UID_MAP, None).team_slot is None


def test_journalists_press_officers_and_genders(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(monkeypatch, FakeSave(people=True))
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.media_read
    assert [(m.unique_id, m.role, m.nation_id, m.club_id, m.female) for m in snap.media] == [
        (2002097912, "journalist", 139, None, False),
        (2002097939, "journalist", 140, None, True),
        (2002076347, "press officer", 139, 1010, True),
    ]
    jake = snap.media[0]
    assert (jake.name, jake.age, jake.birth_date) == ("Jake Diaz", 55, date(1970, 10, 24))
    # My club's staff: Emma is a woman's name, Jake a man's.
    assert [(s.unique_id, s.female) for s in snap.staff] == [(3001, True), (3003, False)]
    # Nations as the game's database IDs (Phase 15f): 139 is England's here, 140 France's.
    assert [m.nation for m in snap.media] == [765, 769, 765]
    assert [s.nation for s in snap.staff] == [765, None]
    assert [p.nation for p in snap.players] == [1651]


def test_without_the_nation_table_nobody_has_a_nation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Men's looks then come from the world's mix; the sync goes on."""
    import fmsave

    _patch_open(monkeypatch, FakeSave(people=True, nations_fail=fmsave.FmsaveError("no table")))
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.players and snap.staff and snap.media
    assert {r.nation for r in (*snap.players, *snap.staff, *snap.media)} == {None}
    assert any("nations unavailable" in w for w in snap.meta.warnings)


def test_a_nation_is_its_database_id() -> None:
    nations = {7: 1651}
    assert convert_player(fm_player(1, nation_id=7), UID_MAP, None, nations).nation == 1651
    assert convert_player(fm_player(1, nation_id=8), UID_MAP, None, nations).nation is None
    assert convert_player(fm_player(1, nation_id=None), UID_MAP, None, nations).nation is None
    assert convert_player(fm_player(1), UID_MAP, None).nation is None  # no table, no field


def test_the_saves_league_rules_are_read_or_their_absence_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_open(monkeypatch, FakeSave(people=True))
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.competition_rules is not None  # these bytes hold no division list
    assert snap.competition_rules.division_lists == ()
    _patch_open(monkeypatch, FakeSave())  # no fmsave internals to borrow
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.competition_rules is None and snap.players
    assert any("league levels unavailable" in w for w in snap.meta.warnings)


def test_without_the_people_reader_the_sync_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(monkeypatch, FakeSave())  # no fmsave internals to borrow
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert snap.players and snap.staff and snap.media == () and not snap.media_read
    assert all(s.female is None for s in snap.staff)
    assert any(
        "journalists, press officers and staff genders unavailable" in w for w in snap.meta.warnings
    )


def test_without_the_gender_table_the_people_stay_as_they_were(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = FakeSave(people=True)
    real = save._context.name_pools

    def pools() -> NS:
        found = real()
        found.first_names = FakePool([*found.first_names.names, "Extra"])  # count no longer matches
        return found

    save._context.name_pools = pools  # type: ignore[method-assign]
    _patch_open(monkeypatch, save)
    snap = FmsaveDataSource().load(Path("Career.fm"))
    assert not snap.media_read and snap.media == ()
    assert any("gender table was not found" in w for w in snap.meta.warnings)
