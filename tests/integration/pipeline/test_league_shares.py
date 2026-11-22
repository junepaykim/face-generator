"""League-share cutoffs through the sync and the settings history (Phase 16): the ladder (from the
save's own rules) and the candidates' stats are stored by the sync, a settings change
reclassifies from them without a new save, and switching Auto <-> Manual is one change that
picks the same people."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date

import pytest
from factories import NEWGEN, player, snapshot

from facegen.core.records import ClubRecord, CompetitionRules, DivisionList, Gender
from facegen.persistence import league_stats
from facegen.persistence.settings import load_settings, read_overrides
from facegen.pipeline import settings_history as history
from facegen.pipeline.sync import run_sync

MY_CLUB = 100
MINE = range(100, 110)  # my league: 10 clubs, reputation about 3000
ABOVE = range(200, 210)  # the league above: about 4500
BELOW_N, BELOW_S = range(400, 410), range(410, 420)  # two parallel leagues below: about 2000
LEAGUE, UP, NORTH, SOUTH, REGIONAL = 50, 51, 52, 53, 54
RULES = CompetitionRules(
    division_lists=(DivisionList(1, False, (UP, LEAGUE, REGIONAL)),),
    levels={UP: 0, LEAGUE: 1, REGIONAL: 2},
    children={REGIONAL: (NORTH, SOUTH)},
)
"""The save's own division list: the league above, mine, and a level of two regional groups."""


def clubs() -> dict[int, ClubRecord]:
    out = {}
    for ids, rep in ((MINE, 3000), (ABOVE, 4500), (BELOW_N, 2000), (BELOW_S, 2005)):
        for i, c in enumerate(ids):
            out[c] = ClubRecord(c, f"Club {c}", Gender.MEN, 1, rep + 10 * i)
    return out


def fixtures() -> dict[int, dict[int, int]]:
    return {
        comp: dict.fromkeys(ids, 18)
        for comp, ids in ((LEAGUE, MINE), (UP, ABOVE), (NORTH, BELOW_N), (SOUTH, BELOW_S))
    }


def newgens() -> list:
    """Ten newgens per league, CA 50-95 by 5 and PA 100-145, one per club."""
    people = []
    for k, ids in enumerate((MINE, ABOVE, BELOW_N, BELOW_S)):
        for i, c in enumerate(ids):
            uid = NEWGEN + 100 * k + i
            if c == MY_CLUB:
                continue  # nobody at my club: the tests are about other clubs
            people.append(player(uid, ca=50 + 5 * i, pa=100 + 5 * i, club_id=c))
    return people


def world(conn: sqlite3.Connection, rules: CompetitionRules | None = RULES) -> None:
    snap = snapshot(
        newgens(), clubs=clubs(), tables=(), fixture_counts=fixtures(), competition_rules=rules
    )
    run_sync(conn, snap, load_settings(conn), save_file="career.fm")


def group(conn: sqlite3.Connection, uid: int) -> str | None:
    return conn.execute("SELECT group_name FROM players WHERE unique_id = ?", (uid,)).fetchone()[0]


def groups(conn: sqlite3.Connection) -> dict[int, str | None]:
    return dict(conn.execute("SELECT unique_id, group_name FROM players WHERE is_newgen = 1"))


def test_the_sync_stores_the_ladder_and_each_scopes_candidates(conn: sqlite3.Connection) -> None:
    world(conn)
    rows = {
        r["club_id"]: (r["reputation"], r["league_id"], r["league_offset"])
        for r in conn.execute("SELECT club_id, reputation, league_id, league_offset FROM clubs")
    }
    assert rows[101] == (3010, LEAGUE, 0)
    assert rows[205] == (4550, UP, 1)
    assert rows[401][1:] == (NORTH, -1) and rows[411][1:] == (SOUTH, -1)
    stats = league_stats.load(conn, 1)
    assert set(stats) == {"league", "above", "below"}  # no continental competition
    assert stats["league"].competitions == (LEAGUE,)
    assert stats["below"].competitions == (NORTH, SOUTH)  # parallel leagues, measured together
    assert stats["league"].newgen_ca.total == 9 and stats["below"].newgen_ca.total == 20
    found = {f.scope: f for f in league_stats.found_leagues(conn, 1)}
    assert found["above"].clubs[0] == "Club 209" and found["above"].club_count == 10


def test_the_saves_own_rules_set_the_levels_and_name_the_leagues(
    conn: sqlite3.Connection,
) -> None:
    """England's men's list (docs/research.md section 23) decides, whatever the reputations
    say: here they are upside down. The UEFA competitions count by their editor IDs."""
    real = {LEAGUE: 109201, UP: 14, NORTH: 5123054, SOUTH: 5123055}
    rules = CompetitionRules(
        division_lists=(DivisionList(765, False, (14, 109201, 150990)),),
        levels={14: 0, 109201: 1, 150990: 2},
        children={150990: (5123054, 5123055)},
    )
    upside_down = {c: replace(r, reputation=9000 - (r.reputation or 0)) for c, r in clubs().items()}
    games = {real[comp]: counts for comp, counts in fixtures().items()}
    games[1301394] = {MY_CLUB: 2, 900: 2}  # a Champions League qualifier this season
    abroad = {900: ClubRecord(900, "Abroad", Gender.MEN, 2, 5000)}
    snap = snapshot(
        newgens(),
        clubs=upside_down | abroad,
        tables=(),
        fixture_counts=games,
        fixture_seasons=dict.fromkeys(games, 2027),
        competition_rules=rules,
        game_date=date(2027, 8, 1),
    )
    run_sync(conn, snap, load_settings(conn), save_file="career.fm")
    offsets = dict(conn.execute("SELECT club_id, league_offset FROM clubs"))
    assert (offsets[101], offsets[205], offsets[401], offsets[411]) == (0, 1, -1, -1)
    names = {(f.scope, f.name) for f in league_stats.found_leagues(conn, 1)}
    assert names == {
        ("league", "National League"),
        ("above", "League Two"),
        ("below", "National League North"),
        ("below", "National League South"),
        ("continental", "UEFA Champions League"),
    }


def test_the_default_shares_pick_by_league(conn: sqlite3.Connection) -> None:
    world(conn)
    # My league's 9 candidates, CA 55-95: top 2.5% is rank 1 (CA 95), top 12% rank 2 (CA 90),
    # top 80% rank 8 (CA 60); PA 105-145: top 2.5% PA 145, top 12% rank 2 (PA 140).
    g = groups(conn)
    assert [g[NEWGEN + i] for i in range(1, 10)] == [
        None, "c", "c", "c", "c", "c", "c", "b", "a",
    ]  # fmt: skip
    # The league above: top 15% of 10 is rank 2 (CA 90), so CA 90 and 95 are B; Light for the
    # league above is on by default: top 50% is rank 5 (CA 75), so CA 75-85 are C.
    assert [g[NEWGEN + 100 + i] for i in range(10)] == [None] * 5 + ["c"] * 3 + ["b", "b"]
    assert all(g[NEWGEN + 200 + i] is None for i in range(10))  # Light below: off by default
    reasons = json.loads(
        conn.execute("SELECT reasons FROM players WHERE unique_id = ?", (NEWGEN + 9,)).fetchone()[0]
    )
    assert reasons == ["newgen in my league, CA 95 (top 2.5%: 95+)"]  # CA's rule, then PA's


def test_a_share_change_reclassifies_without_a_new_save_and_one_undo_restores(
    conn: sqlite3.Connection,
) -> None:
    world(conn)
    before = groups(conn)
    out = history.save(conn, {"cutoffs.light_below": True})
    assert out.report is not None and out.report.after["c"] > out.report.before["c"]
    # The league below now draws Light too: top 50% of 20 below is rank 10 (CA 75).
    assert group(conn, NEWGEN + 200 + 5) == "c" and group(conn, NEWGEN + 200 + 4) is None
    assert group(conn, NEWGEN + 300 + 5) == "c"  # the parallel league, measured with it
    out = history.save(conn, {"cutoffs.share.b_ca": 40.0})
    assert group(conn, NEWGEN + 6) == "b"  # CA 80 is now in the top 40% (rank 4)
    history.undo(conn)
    history.undo(conn)
    assert groups(conn) == before and read_overrides(conn) == {}


def test_auto_manual_auto_is_one_change_each_and_picks_the_same_people(
    conn: sqlite3.Connection,
) -> None:
    world(conn)
    auto = groups(conn)
    out = history.save(conn, {"cutoffs.mode": "manual"}, convert=True)
    (entry,) = out.entries
    assert entry.after == {
        "cutoffs.mode": "manual",
        "cutoffs.manual.a_ca": 95,
        "cutoffs.manual.a_pa": 145,
        "cutoffs.manual.b_ca": 90,
        "cutoffs.manual.b_pa": 140,
        "cutoffs.manual.c_ca": 60,
        "cutoffs.manual.c_pa": "-",
    }
    assert groups(conn) == auto and out.report is not None and out.report.changed_total == 0
    out = history.save(conn, {"cutoffs.mode": "auto"}, convert=True)
    assert out.entries[0].after == {"cutoffs.mode": None}  # every share already gives its number
    assert groups(conn) == auto
    history.undo(conn)
    history.undo(conn)  # one Undo per switch
    assert read_overrides(conn) == {} and groups(conn) == auto


def test_a_share_sent_with_the_switch_to_manual_is_the_one_converted(
    conn: sqlite3.Connection,
) -> None:
    """A share edited while a save was on its way reaches the server with the switch: the
    number comes from that share, not the one stored before it."""
    world(conn)
    out = history.save(conn, {"cutoffs.share.b_ca": 40.0, "cutoffs.mode": "manual"}, convert=True)
    (entry,) = out.entries
    assert entry.after["cutoffs.share.b_ca"] == 40.0
    assert entry.after["cutoffs.manual.b_ca"] == 80  # top 40%: rank 4 is CA 80, not 90
    assert group(conn, NEWGEN + 6) == "b"


def test_a_number_set_by_hand_becomes_its_share(conn: sqlite3.Connection) -> None:
    world(conn)
    history.save(conn, {"cutoffs.mode": "manual"}, convert=True)
    history.save(conn, {"cutoffs.manual.b_ca": 70})  # CA 70-95: 6 of 9
    manual = groups(conn)
    out = history.save(conn, {"cutoffs.mode": "auto"}, convert=True)
    assert out.entries[0].after["cutoffs.share.b_ca"] == 66.6  # rank ceil(5.99) = 6
    assert groups(conn) == manual


def test_without_the_saves_rules_there_is_no_league_above_or_below(
    conn: sqlite3.Connection,
) -> None:
    """No guess from the clubs' reputations: my league still comes from my club's fixtures, and
    only the everywhere cuts pick players in other leagues (the Cutoffs tab turns the league
    above's and below's settings off)."""
    world(conn, rules=None)
    offsets = dict(conn.execute("SELECT club_id, league_offset FROM clubs"))
    assert offsets[101] is None and offsets[205] is None and offsets[401] is None
    stats = league_stats.load(conn, 1)
    assert set(stats) == {"league"} and stats["league"].competitions == (LEAGUE,)
    g = groups(conn)
    assert g[NEWGEN + 9] == "a"  # my league's shares still pick
    assert all(g[NEWGEN + 100 * k + i] is None for k in (1, 2, 3) for i in range(10))


def test_without_a_league_only_the_global_cuts_pick(conn: sqlite3.Connection) -> None:
    people = [*newgens(), player(NEWGEN + 900, ca=170, club_id=205)]
    run_sync(conn, snapshot(people, clubs=clubs(), tables=(), fixture_counts={}), load_settings(conn), save_file="career.fm")  # fmt: skip
    assert league_stats.load(conn, 1) == {}
    assert set(groups(conn).values()) == {None, "a"}
    assert group(conn, NEWGEN + 900) == "a"


@pytest.mark.parametrize("value", [True, False])
def test_the_no_photo_setting_changes_the_candidates_without_a_new_save(
    conn: sqlite3.Connection, value: bool
) -> None:
    world(conn)
    out = history.save(conn, {"no_photo.enabled": value})
    assert out.changed is (not value)  # on by default
