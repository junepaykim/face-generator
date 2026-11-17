"""Which clubs are in my league and in the listed continental competitions. Pure.

"My league" is the division my club's first team plays in (spec section 2). Among the
domestic competitions holding my club's first team, in a league table or a fixture, it is
the one where my club plays the most first-team fixtures this season (a league schedules
every round up front; cups and pre-season events have few), with the larger table as the
tie-break. Every club in any table or fixture of that competition is in my league, so a
league split into conferences counts as one.

League tables alone are not enough: before a season starts the save may hold no table yet
(the Phase 0.1 save, 2026-03-01, had none for its NWSL club; docs/research.md).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from facegen.core.records import ClubRecord, LeagueTableRecord

# A competition counts as domestic when at least this share of its clubs play in my nation.
DOMESTIC_SHARE = 0.75


def _competition_clubs(
    tables: Sequence[LeagueTableRecord], fixture_counts: Mapping[int, Mapping[int, int]]
) -> dict[int, set[int]]:
    clubs: dict[int, set[int]] = {}
    for t in tables:
        if t.competition_id is not None:
            clubs.setdefault(t.competition_id, set()).update(t.club_ids)
    for cid, counts in fixture_counts.items():
        clubs.setdefault(cid, set()).update(counts)
    return clubs


def my_league_competitions(
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
) -> frozenset[int]:
    """The competition ID of my club's league, as a set (empty when not found)."""
    if managed_club_id is None or managed_club_id not in clubs:
        return frozenset()
    nation = clubs[managed_club_id].nation_id

    def domestic(members: set[int]) -> bool:
        same = sum(1 for c in members if c in clubs and clubs[c].nation_id == nation)
        return same >= DOMESTIC_SHARE * len(members)

    table_size: dict[int, int] = {}
    for t in tables:
        if t.competition_id is not None and managed_club_id in t.club_ids:
            size = len(t.club_ids)
            table_size[t.competition_id] = max(size, table_size.get(t.competition_id, 0))

    candidates = [
        cid
        for cid, members in _competition_clubs(tables, fixture_counts).items()
        if managed_club_id in members and domestic(members)
    ]
    if not candidates:
        return frozenset()

    def score(cid: int) -> tuple[int, int]:
        return fixture_counts.get(cid, {}).get(managed_club_id, 0), table_size.get(cid, 0)

    return frozenset({max(candidates, key=score)})


def clubs_in_competitions(
    competition_ids: Iterable[int],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
) -> frozenset[int]:
    """Every club with a first-team table row or fixture in any of the competitions."""
    wanted = frozenset(competition_ids)
    found: set[int] = set()
    for t in tables:
        if t.competition_id in wanted:
            found.update(t.club_ids)
    for cid in wanted:
        found.update(fixture_counts.get(cid, {}))
    return frozenset(found)
