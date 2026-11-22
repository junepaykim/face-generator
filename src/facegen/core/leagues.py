"""Which clubs are in my league, the leagues above and below it, and my continental competitions.
Pure.

"My league" is the division my club's first team plays in (spec section 2). The save's own
competition rules list each nation's divisions from the top (docs/research.md sections 23-25):
when they place my club's league, the ladder comes from them (`rules_ladder`). Otherwise, among
the domestic competitions holding my club's first team, in a league table or a fixture, my
league is the one where my club plays the most first-team fixtures this season (a league
schedules every round up front; cups and pre-season events have few), with the larger table as
the tie-break, and the leagues are ordered by their clubs' reputation (`league_ladder`).

League tables alone are not enough: before a season starts the save may hold no table yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from facegen.core.records import ClubRecord, CompetitionRules, Gender, LeagueTableRecord

UEFA_CLUB_COMPETITIONS: Mapping[int, str] = {
    1301394: "UEFA Champions League",
    1301396: "UEFA Europa League",
    31051584: "UEFA Conference League",
}
"""The men's UEFA club competitions by editor ID (docs/research.md section 24): continental
whenever my club's first team plays in their current edition."""

COMPETITION_NAMES: Mapping[int, str] = {
    11: "Premier League",
    12: "Championship",
    13: "League One",
    14: "League Two",
    109201: "National League",
    150990: "National League North/South",
    5123054: "National League North",
    5123055: "National League South",
    67: "La Liga",
    68: "Segunda División",
    2000048840: "Primera Federación",
    2000099902: "Segunda Federación",
    32: "Serie A",
    33: "Serie B",
    43127171: "Serie C",
    22: "Bundesliga",
    23: "2. Bundesliga",
    35010926: "3. Liga",
    **UEFA_CLUB_COMPETITIONS,
}
"""Names for the competitions checked by editor ID (docs/research.md sections 23-25). The save
stores no competition names; any other competition is named by its clubs."""


def competition_name(competition_id: int) -> str | None:
    return COMPETITION_NAMES.get(competition_id)


def uefa_season(game_date: date | None) -> int | None:
    """The UEFA club season under way on `game_date`, as the year it starts (a new one from
    July, when the first qualifiers are played)."""
    if game_date is None:
        return None
    return game_date.year if game_date.month >= 7 else game_date.year - 1


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


# --- the league ladder (Phase 16) -------------------------------------------------------------
#
# fmsave reads no league level or competition name, only each club's reputation (a verified
# field). My nation's leagues are ordered by their clubs' reputation; Phase 16a measured the
# method on the Telford and Colchester careers (docs/research.md section 22).

LEVEL_SEPARATION = 0.8
"""A league starts a new level below another when a club of the upper one has the higher
reputation in at least this share of the pairs (one club from each). England's levels measured
0.93 to 1.00 against the one above; National League North and South, one level, 0.61 to 0.69."""
MIN_TABLE_LEAGUE = 10
"""With no fixtures at all, a table this long marks a league (a cup's group tables are short)."""
MIN_CONTINENTAL_CLUBS = 8
MAX_PER_NATION = 8
"""A continental competition holds a few clubs from each nation. A friendlies "competition"
holds hundreds of one nation's clubs, and my club appears in it."""


@dataclass(frozen=True, slots=True)
class League:
    """One of my nation's leagues on the ladder."""

    competition_id: int
    club_ids: frozenset[int]
    """The clubs whose league it is: where they play the most first-team fixtures."""
    reputation: float
    """The median reputation of those clubs."""
    level: int
    """0 for the top level: the save's top division, or the top level loaded when the ladder
    comes from reputations. Parallel divisions share a level."""


@dataclass(frozen=True, slots=True)
class Ladder:
    """My nation's leagues by level, and mine among them."""

    leagues: tuple[League, ...] = ()
    mine: League | None = None
    from_rules: bool = False
    """True when the levels come from the save's own division lists (`rules_ladder`), False
    when from the clubs' reputations."""

    def offset(self, league: League) -> int | None:
        """How many levels the league is above mine (below: negative); None without mine."""
        return None if self.mine is None else self.mine.level - league.level

    def at(self, offset: int) -> tuple[League, ...]:
        """The leagues `offset` levels above mine (-1: the level below). My league itself is
        the only one at offset 0: a parallel division is not my league."""
        if self.mine is None:
            return ()
        if offset == 0:
            return (self.mine,)
        return tuple(lg for lg in self.leagues if self.offset(lg) == offset)

    def club_leagues(self) -> dict[int, tuple[int, int | None]]:
        """Club ID to (its league's competition ID, that league's offset from mine)."""
        return {c: (lg.competition_id, self.offset(lg)) for lg in self.leagues for c in lg.club_ids}


def _separation(upper: list[int], lower: list[int]) -> float:
    """The share of pairs (one club from each) in which the upper league's club has the higher
    reputation; a tie counts half."""
    wins = 0.0
    for u in upper:
        for low in lower:
            wins += 1.0 if u > low else 0.5 if u == low else 0.0
    return wins / (len(upper) * len(lower))


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def league_ladder(
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
    *,
    seasons: Mapping[int, int] | None = None,
    rules: CompetitionRules | None = None,
) -> Ladder:
    """My nation's leagues by level: from the save's own rules where they place my club's
    league (`rules_ladder`), otherwise ordered by their clubs' reputation.

    The reputation method:

    - A league is a competition of my nation and my club's gender (at least DOMESTIC_SHARE of
      its clubs) in which some club plays at least half a round of fixtures (cups and their
      groups have few), or, with no fixtures for it, a table of MIN_TABLE_LEAGUE clubs or more.
    - Each club belongs to the league where it plays the most first-team fixtures (then the
      larger table); a league no club belongs to (a youth or reserve league) drops out.
    - Leagues are ordered by their clubs' median reputation. A league joins the level of the
      one above it unless that level's top league stands clearly above it (LEVEL_SEPARATION):
      parallel divisions share a level.
    - Mine is the league my club belongs to. No managed club, no league found, or none with
      reputations: an empty ladder, so the rules for other leagues pick nobody.
    """
    if managed_club_id is None or managed_club_id not in clubs:
        return Ladder()
    if rules is not None:
        found = rules_ladder(managed_club_id, clubs, tables, fixture_counts, seasons or {}, rules)
        if found is not None:
            return found
    me = clubs[managed_club_id]

    def ours(c: int) -> bool:
        club = clubs.get(c)
        if club is None or club.nation_id != me.nation_id:
            return False
        return me.gender is Gender.UNKNOWN or club.gender is me.gender

    table_size: dict[int, int] = {}
    for t in tables:
        if t.competition_id is not None:
            size = max(len(t.club_ids), table_size.get(t.competition_id, 0))
            table_size[t.competition_id] = size

    leagues: dict[int, set[int]] = {}
    for cid, members in _competition_clubs(tables, fixture_counts).items():
        if not members or sum(1 for c in members if ours(c)) < DOMESTIC_SHARE * len(members):
            continue
        counts = fixture_counts.get(cid, {})
        most = max((counts.get(c, 0) for c in members), default=0)
        if counts:
            if most < max(2, (len(members) - 1) / 2):
                continue
        elif table_size.get(cid, 0) < MIN_TABLE_LEAGUE:
            continue
        leagues[cid] = members

    home: dict[int, tuple[tuple[int, int, int], int]] = {}
    for cid, members in leagues.items():
        for c in members:
            if not ours(c):
                continue
            rank = (fixture_counts.get(cid, {}).get(c, 0), table_size.get(cid, 0), -cid)
            if c not in home or rank > home[c][0]:
                home[c] = (rank, cid)
    members_of: dict[int, set[int]] = {}
    for c, (_, cid) in home.items():
        members_of.setdefault(cid, set()).add(c)

    rated = []
    for cid, members in members_of.items():
        reps = [r for c in members if (r := clubs[c].reputation) is not None]
        if reps:
            rated.append((cid, frozenset(members), reps))
    rated.sort(key=lambda x: (-_median(x[2]), x[0]))

    out: list[League] = []
    top_reps: list[int] = []
    level = -1
    for cid, members, reps in rated:
        if level < 0 or _separation(top_reps, reps) >= LEVEL_SEPARATION:
            level += 1
            top_reps = reps
        out.append(League(cid, members, _median(reps), level))
    mine = next((lg for lg in out if managed_club_id in lg.club_ids), None)
    return Ladder(tuple(out), mine)


def continental_competitions(
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
    listed: frozenset[int] = frozenset(),
    *,
    seasons: Mapping[int, int] | None = None,
    game_date: date | None = None,
) -> frozenset[int]:
    """The continental competitions my club's first team plays in.

    - The UEFA club competitions (UEFA_CLUB_COMPETITIONS) count when my club's first team has
      a fixture in their current edition (`uefa_season` of the game date; `seasons` gives the
      edition the fixtures are from). An earlier edition leaves this one unconfirmed: nothing.
    - `listed` (settings `eligibility.continental_competition_ids`) overrides the search: then
      the listed competitions my club plays in, with the same edition check.
    - Otherwise any other competition counts when my club has a table row or fixture in it, it
      has at least MIN_CONTINENTAL_CLUBS clubs, at least DOMESTIC_SHARE of them are from other
      nations (the domestic test reversed), and no nation has more than MAX_PER_NATION of them.
    """
    if managed_club_id is None or managed_club_id not in clubs:
        return frozenset()
    nation = clubs[managed_club_id].nation_id
    edition = uefa_season(game_date)
    found = set()
    for cid, members in _competition_clubs(tables, fixture_counts).items():
        if managed_club_id not in members:
            continue
        if cid in UEFA_CLUB_COMPETITIONS:
            played = fixture_counts.get(cid, {})
            season = (seasons or {}).get(cid)
            if played and managed_club_id not in played:
                continue  # only a table row, from an earlier edition
            if edition is not None and season is not None and season < edition:
                continue
        if listed:
            if cid in listed:
                found.add(cid)
            continue
        if cid in UEFA_CLUB_COMPETITIONS:
            found.add(cid)
            continue
        if len(members) < MIN_CONTINENTAL_CLUBS:
            continue
        nations: dict[int | None, int] = {}
        for c in members:
            n = clubs[c].nation_id if c in clubs else None
            nations[n] = nations.get(n, 0) + 1
        foreign = len(members) - nations.get(nation, 0)
        if foreign >= DOMESTIC_SHARE * len(members) and max(nations.values()) <= MAX_PER_NATION:
            found.add(cid)
    return frozenset(found)


# --- the league ladder from the save's own rules (docs/research.md sections 23-25) ------------

_MAX_NESTING = 3


def _division_members(
    division: int, rules: CompetitionRules, depth: int = 0
) -> tuple[int, ...] | None:
    """The competitions clubs play in at a division: the division itself and, for a regional
    parent, its groups. None when the rules disagree about it."""
    if division in rules.unknown or depth > _MAX_NESTING:
        return None
    members = [division]
    for child in rules.children.get(division, ()):
        below = _division_members(child, rules, depth + 1)
        if below is None:
            return None
        members.extend(below)
    return tuple(members)


def pyramids(rules: CompetitionRules, gender: Gender) -> list[tuple[tuple[int, ...], ...]]:
    """Each division list of my club's gender as its levels' competitions, top first.

    A list is left out when a division's stored level is not its place in the list, or the
    rules disagree about one of its divisions or groups (an unknown level, not a guess)."""
    out = []
    for listed in rules.division_lists:
        known = gender is not Gender.UNKNOWN and listed.women is not None
        if known and listed.women != (gender is Gender.WOMEN):
            continue
        levels: list[tuple[int, ...]] = []
        for place, division in enumerate(listed.divisions):
            stored = rules.levels.get(division)
            members = _division_members(division, rules)
            if members is None or (stored is not None and stored != place):
                levels = []
                break
            levels.append(members)
        if levels:
            out.append(tuple(levels))
    return out


def rules_ladder(
    managed_club_id: int,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
    seasons: Mapping[int, int],
    rules: CompetitionRules,
) -> Ladder | None:
    """My pyramid's leagues by level, as the save's rules list them; None where they do not place
    my club's league (the reputation method then decides).

    - My league is the division (a group, for a regional level) where my club's first team plays
      in the latest season, then the most fixtures; a table counts only for a competition with
      no fixtures. `seasons` gives each competition's season (`Snapshot.fixture_seasons`), so a
      club just promoted counts in its new division, not the old one it played more in.
    - Every club of my club's gender, whatever its nation (Welsh clubs play in England's
      pyramid), belongs to the division of my pyramid it plays in the same way.
    - A level is the division's place in the list; parallel groups share it. A level with no
      club playing in it is left out, so "the league below" is only ever the next level.
    - My league in two lists, or a list the rules disagree about, gives None.
    """
    me = clubs.get(managed_club_id)
    if me is None:
        return None
    found = pyramids(rules, me.gender)
    where: dict[int, list[int]] = {}
    for index, levels in enumerate(found):
        for members in levels:
            for c in members:
                where.setdefault(c, []).append(index)

    table_clubs: dict[int, set[int]] = {}
    for t in tables:
        if t.competition_id is not None:
            table_clubs.setdefault(t.competition_id, set()).update(t.club_ids)

    def rank(c: int, club: int) -> tuple[int, int, int, int] | None:
        played = fixture_counts.get(c, {})
        size = len(table_clubs.get(c, ()))
        if played.get(club):
            return seasons.get(c, -1), played[club], size, -c
        if not played and club in table_clubs.get(c, ()):
            return -1, 0, size, -c
        return None

    ranked = [(r, c) for c in where if (r := rank(c, managed_club_id)) is not None]
    if not ranked:
        return None
    mine_id = max(ranked)[1]
    if len(where[mine_id]) != 1:
        return None
    levels = found[where[mine_id][0]]
    level_of = {c: level for level, members in enumerate(levels) for c in members}

    def same_gender(club: ClubRecord) -> bool:
        return me.gender is Gender.UNKNOWN or club.gender in (me.gender, Gender.UNKNOWN)

    home: dict[int, tuple[tuple[int, int, int, int], int]] = {}
    for c in level_of:
        played = fixture_counts.get(c, {})
        for club in set(played) | (table_clubs.get(c, set()) if not played else set()):
            record = clubs.get(club)
            if record is None or not same_gender(record):
                continue
            r = rank(c, club)
            if r is not None and (club not in home or r > home[club][0]):
                home[club] = (r, c)
    members_of: dict[int, set[int]] = {}
    for club, (_, c) in home.items():
        members_of.setdefault(c, set()).add(club)

    leagues = []
    for c in sorted(members_of, key=lambda c: (level_of[c], c)):
        reps = [r for club in members_of[c] if (r := clubs[club].reputation) is not None]
        leagues.append(
            League(c, frozenset(members_of[c]), _median(reps) if reps else 0.0, level_of[c])
        )
    mine = next((lg for lg in leagues if managed_club_id in lg.club_ids), None)
    return Ladder(tuple(leagues), mine, from_rules=True)


def my_league(
    ladder: Ladder,
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
) -> tuple[frozenset[int], frozenset[int]]:
    """My league's competitions and clubs: the division the save's rules place my club in, with
    the clubs whose division it is; otherwise (`my_league_competitions`) the domestic competition
    my club plays the most in, with every club in its tables and fixtures."""
    if ladder.from_rules and ladder.mine is not None:
        return frozenset({ladder.mine.competition_id}), ladder.mine.club_ids
    competitions = my_league_competitions(managed_club_id, clubs, tables, fixture_counts)
    return competitions, clubs_in_competitions(competitions, tables, fixture_counts)
