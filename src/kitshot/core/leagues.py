"""Which clubs are in my league, the leagues above and below it, and my continental competitions.
Pure.

"My league" is the division my club's first team plays in (spec section 2). The save's own
competition rules list each nation's divisions from the top (docs/research.md sections 23-25):
when they place my club's league, the ladder comes from them (`league_ladder`) and my league is
its division. Otherwise there is no ladder, so nobody is in the league above or below (user,
2026-11-22: no guess from the clubs' reputations), and my league is the domestic competition
holding my club's first team, in a league table or a fixture, where my club plays the most
first-team fixtures this season (a league schedules every round up front; cups and pre-season
events have few), with the larger table as the tie-break (`my_league_competitions`). When that
competition's fixtures are from an older season than another one my club plays in (just
promoted or relegated, before the new division's season is read), my league is not known and
its rules pick nobody (user, 2026-11-22: turn it off when the seasons conflict).

A competition's clubs are those in its fixtures, which are its latest season's; its tables
count only when it has no fixtures, as a table can be from an earlier season or edition.

League tables alone are not enough: before a season starts the save may hold no table yet.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from kitshot.core.records import ClubRecord, CompetitionRules, Gender, LeagueTableRecord

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
) -> dict[int, Collection[int]]:
    """Use a competition's tables only when it has no first-team fixture participants."""
    table_clubs: dict[int, set[int]] = {}
    for t in tables:
        if t.competition_id is not None:
            members = table_clubs.setdefault(t.competition_id, set())
            if not fixture_counts.get(t.competition_id):
                members.update(t.club_ids)
    return {
        **table_clubs,
        **{cid: counts.keys() for cid, counts in fixture_counts.items() if counts},
    }


def my_league_competitions(
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
    seasons: Mapping[int, int] | None = None,
) -> frozenset[int]:
    """The competition ID of my club's league, as a set: empty when not found, or when its
    fixtures' season (`seasons`, `Snapshot.fixture_seasons`) is older than that of another
    domestic competition my club has fixtures in, or unknown while another's is known."""
    if managed_club_id is None or managed_club_id not in clubs:
        return frozenset()
    nation = clubs[managed_club_id].nation_id

    def domestic(members: Collection[int]) -> bool:
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
        if cid not in UEFA_CLUB_COMPETITIONS and managed_club_id in members and domestic(members)
    ]
    if not candidates:
        return frozenset()

    def played(cid: int) -> int:
        return fixture_counts.get(cid, {}).get(managed_club_id, 0)

    def score(cid: int) -> tuple[int, int]:
        return played(cid), table_size.get(cid, 0)

    mine = max(candidates, key=score)
    season_of = seasons or {}
    newest = max((season_of.get(c, -1) for c in candidates if played(c)), default=-1)
    if played(mine) and season_of.get(mine, -1) < newest:
        return frozenset()  # the seasons conflict: an old division, or a guess
    return frozenset({mine})


def clubs_in_competitions(
    competition_ids: Iterable[int],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
) -> frozenset[int]:
    """Every club with a first-team fixture in any of the competitions, their latest season's;
    for a competition with no fixtures, every club with a first-team row in its tables (which
    can be from an earlier season or edition, so they count only then)."""
    by_competition = _competition_clubs(tables, fixture_counts)
    return frozenset(club for cid in competition_ids for club in by_competition.get(cid, ()))


# --- the league ladder from the save's own rules (docs/research.md sections 23-25) ------------


@dataclass(frozen=True, slots=True)
class League:
    """One division (or regional group) of my pyramid on the ladder."""

    competition_id: int
    club_ids: frozenset[int]
    """The clubs whose league it is: where they play in the latest season, then the most."""
    level: int
    """The division's place in the save's list, 0 for the top; parallel groups share it."""


@dataclass(frozen=True, slots=True)
class Ladder:
    """My pyramid's leagues by level, and mine among them; empty where the save's rules do not
    place my club's league."""

    leagues: tuple[League, ...] = ()
    mine: League | None = None

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


def league_ladder(
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
    *,
    seasons: Mapping[int, int] | None = None,
    rules: CompetitionRules | None = None,
) -> Ladder:
    """My pyramid's leagues by level, as the save's rules list them; an empty ladder where they
    do not place my club's league, so the rules for the leagues above and below pick nobody.

    - My league is the division (a group, for a regional level) where my club's first team plays
      in the latest season, then the most fixtures; a table counts only for a competition with
      no fixtures. `seasons` gives each competition's season (`Snapshot.fixture_seasons`), so a
      club just promoted counts in its new division, not the old one it played more in.
    - Every club of my club's gender, whatever its nation (Welsh clubs play in England's
      pyramid), belongs to the division of my pyramid it plays in the same way.
    - A level is the division's place in the list; parallel groups share it. A level with no
      club playing in it is left out, so "the league below" is only ever the next level.
    - No managed club, no rules read, my league in no list or in two, or a list the rules
      disagree about: an empty ladder.
    """
    if managed_club_id is None or managed_club_id not in clubs or rules is None:
        return Ladder()
    me = clubs[managed_club_id]
    season_of = seasons or {}
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
            return season_of.get(c, -1), played[club], size, -c
        if not played and club in table_clubs.get(c, ()):
            return -1, 0, size, -c
        return None

    ranked = [(r, c) for c in where if (r := rank(c, managed_club_id)) is not None]
    if not ranked:
        return Ladder()
    mine_id = max(ranked)[1]
    if len(where[mine_id]) != 1:
        return Ladder()
    levels = found[where[mine_id][0]]
    level_of = {c: level for level, members in enumerate(levels) for c in members}

    def same_gender(club: ClubRecord) -> bool:
        return me.gender is Gender.UNKNOWN or club.gender in (me.gender, Gender.UNKNOWN)

    participants = _competition_clubs(tables, fixture_counts)
    home: dict[int, tuple[tuple[int, int, int, int], int]] = {}
    for c in level_of:
        for club in participants.get(c, ()):
            record = clubs.get(club)
            if record is None or not same_gender(record):
                continue
            r = rank(c, club)
            if r is not None and (club not in home or r > home[club][0]):
                home[club] = (r, c)
    members_of: dict[int, set[int]] = {}
    for club, (_, c) in home.items():
        members_of.setdefault(c, set()).add(club)

    leagues = tuple(
        League(c, frozenset(members_of[c]), level_of[c])
        for c in sorted(members_of, key=lambda c: (level_of[c], c))
    )
    mine = next((lg for lg in leagues if managed_club_id in lg.club_ids), None)
    return Ladder(leagues, mine)


def my_league(
    ladder: Ladder,
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
    *,
    seasons: Mapping[int, int] | None = None,
) -> tuple[frozenset[int], frozenset[int]]:
    """My league's competitions and clubs: the ladder's division for my club, with the clubs whose
    division it is; without a ladder (`my_league_competitions`) the domestic competition my club
    plays the most in, with its clubs (`clubs_in_competitions`); nothing when the seasons
    conflict."""
    if ladder.mine is not None:
        return frozenset({ladder.mine.competition_id}), ladder.mine.club_ids
    competitions = my_league_competitions(managed_club_id, clubs, tables, fixture_counts, seasons)
    return competitions, clubs_in_competitions(competitions, tables, fixture_counts)


@dataclass(frozen=True, slots=True)
class DomesticMembership:
    """The managed club's division and pyramid, shared by staff selection and sync."""

    ladder: Ladder
    competition_ids: frozenset[int]
    club_ids: frozenset[int]


def domestic_membership(
    managed_club_id: int | None,
    clubs: Mapping[int, ClubRecord],
    tables: Sequence[LeagueTableRecord],
    fixture_counts: Mapping[int, Mapping[int, int]],
    seasons: Mapping[int, int],
    rules: CompetitionRules | None,
) -> DomesticMembership:
    ladder = league_ladder(
        managed_club_id, clubs, tables, fixture_counts, seasons=seasons, rules=rules
    )
    ids, members = my_league(
        ladder, managed_club_id, clubs, tables, fixture_counts, seasons=seasons
    )
    return DomesticMembership(ladder, ids, members)


# --- continental competitions ----------------------------------------------------------------

MIN_CONTINENTAL_CLUBS = 8
MAX_PER_NATION = 8
"""A continental competition holds a few clubs from each nation. A friendlies "competition"
holds hundreds of one nation's clubs, and my club appears in it."""


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
      edition the fixtures are from). Both the save date and a matching season are required.
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
            season = (seasons or {}).get(cid)
            if managed_club_id not in fixture_counts.get(cid, {}):
                continue  # only a table row, maybe of an earlier edition: unconfirmed
            if edition is None or season != edition:
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
