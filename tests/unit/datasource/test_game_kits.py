"""FM27's database read from synthetic bytes (support/game_db.py): the two club tables, the
one home shirt per club, and every way the reader refuses a file it does not understand."""

from __future__ import annotations

import pytest
from support import game_db
from support.game_db import BARCELONA, CLUBS, Club, Entry

from kitshot.core.kit_values import GameShirt
from kitshot.datasource.game_kits import (
    GameKitsError,
    club_ids,
    home_shirts,
    parse_database,
    read_tables,
)


def rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def test_each_club_gets_its_home_shirt_and_goalkeeper_shirt() -> None:
    shirts = parse_database(game_db.database(CLUBS))
    home, keeper = shirts[1708]
    assert home == GameShirt(24, (rgb("BD1952"), rgb("19316B"), rgb("0042BD")))
    assert keeper == GameShirt(12, (rgb("FFBD52"), rgb("FF4A00"), rgb("FF4A00")))
    real, no_keeper = shirts[1736]
    assert real.style == 25 and real.colours[1] == (255, 255, 255) and no_keeper is None


def test_a_club_without_exactly_one_home_shirt_has_none() -> None:
    """Two home shirts for any competition and season: which one the game shows is unknown."""
    shirts = parse_database(game_db.database(CLUBS))
    assert set(shirts) == {1708, 1736}  # not 5000 (two), 5001 (no kits) or the empty slot


def test_cup_season_away_and_shorts_entries_are_not_the_home_shirt() -> None:
    only_others = Club(
        7,
        kits=tuple(e for e in BARCELONA.kits if e.style != 24 and e.wearer == 1),
    )
    assert parse_database(game_db.database([only_others])) == {}


def test_rows_are_joined_by_position_not_by_name() -> None:
    clubs = [Club(11, "Same Name", kits=(Entry(("FFFFFF", "C50021", "FFFFFF")),)),
             Club(12, "Same Name", kits=(Entry(("FFFFFF", "0042BD", "FFFFFF")),))]  # fmt: skip
    shirts = parse_database(game_db.database(clubs))
    assert shirts[11][0].colours[1] == rgb("C50021")
    assert shirts[12][0].colours[1] == rgb("0042BD")


def test_club_ids_walks_both_row_kinds() -> None:
    assert club_ids(game_db.clubs_table(CLUBS)) == [1708, 5001, 1736, 2000000000, 5000]


@pytest.mark.parametrize(
    "damage",
    [
        lambda b: b"\x00" + b[1:],  # no header
        lambda b: b[:12] + b"\x00" * 6 + b[18:],  # no container
        lambda b: b[:-40],  # the directory cut short
    ],
)
def test_a_file_that_is_not_a_database_is_refused(damage) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(GameKitsError):
        parse_database(damage(game_db.database(CLUBS)))


def test_a_table_of_another_size_than_listed_is_refused() -> None:
    data = game_db.database(CLUBS, sizes={"table_extra_3": 99})
    with pytest.raises(GameKitsError, match="not the 99 listed"):
        read_tables(data, ("table_extra_3",))


def test_a_missing_table_is_refused() -> None:
    t = game_db.tables(CLUBS)
    t.others = {"table_0": b"x"}
    with pytest.raises(GameKitsError, match="no table_4"):
        read_tables(game_db.database(t), ("table_4",))


def test_club_rows_must_close_with_their_position_and_id() -> None:
    table = bytearray(game_db.clubs_table(CLUBS))
    footer = table.find(bytes.fromhex("00000000") + (1708).to_bytes(4, "little") * 2)
    table[footer + 8] ^= 1  # the second copy of Barcelona's ID
    with pytest.raises(GameKitsError, match="row 0"):
        club_ids(bytes(table))


def test_an_unknown_club_row_kind_is_refused() -> None:
    table = bytearray(game_db.clubs_table(CLUBS))
    table[13] = 5
    with pytest.raises(GameKitsError, match="unknown kind"):
        club_ids(bytes(table))


def test_bytes_after_the_rows_are_refused() -> None:
    with pytest.raises(GameKitsError, match="after the rows"):
        club_ids(game_db.clubs_table(CLUBS) + b"\x00")
    ids = club_ids(game_db.clubs_table(CLUBS))
    with pytest.raises(GameKitsError, match="after the rows"):
        home_shirts(game_db.extra_table(CLUBS) + b"\x00", ids)


def test_the_extra_table_must_have_a_row_per_club() -> None:
    ids = club_ids(game_db.clubs_table(CLUBS))
    with pytest.raises(GameKitsError, match="rows"):
        home_shirts(game_db.extra_table(CLUBS[:-1]), ids)


def test_extra_rows_must_open_with_their_position() -> None:
    extra = bytearray(game_db.extra_table(CLUBS))
    extra[12] = 3  # row 0 says it is row 3
    with pytest.raises(GameKitsError, match="row 0"):
        home_shirts(bytes(extra), club_ids(game_db.clubs_table(CLUBS)))
