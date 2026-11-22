"""The candidates' CA and PA histograms per scope, as the last sync read them (Phase 16).

A sync writes them after storing the clubs and people and before classifying; a settings
change, Undo and Redo read them, so new shares apply without a new save. Candidates are counted
from the stored people: present players in men's football (not staff or media), newgens and
pre-existing players without a photo apart (core/cutoffs.py `ScopeStats`).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from facegen.core.cutoffs import SCOPES, Histogram, ScopeStats
from facegen.core.leagues import competition_name
from facegen.persistence import photo_index

_OFFSET_SCOPES = {"above": 1, "below": -1}


def collect(
    conn: sqlite3.Connection, career_id: int, competitions: Mapping[str, tuple[int, ...]]
) -> dict[str, ScopeStats]:
    """Count each found scope's candidates from the stored people and clubs.

    `competitions` names the scopes the sync found ("league", "above", "below", "continental")
    and their competitions; a scope left out was not found and gets no stats. Clubs must already
    carry `in_my_league`, `continental` and `league_offset` for this sync.
    """
    status = photo_index.status(conn)
    photos_known = status is not None and status.game_known
    values: dict[str, dict[str, list[int | None]]] = {
        scope: {"newgen_ca": [], "newgen_pa": [], "no_photo_ca": [], "no_photo_pa": []}
        for scope in competitions
    }
    rows = conn.execute(
        "SELECT p.ca, p.pa, p.is_newgen, ph.unique_id IS NOT NULL AS has_photo,"
        " c.in_my_league, c.continental, c.league_offset FROM players p"
        " JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        " LEFT JOIN photos ph ON ph.unique_id = p.unique_id"
        " WHERE p.career_id = ? AND p.present = 1 AND p.in_scope = 1 AND p.is_staff = 0"
        " AND p.media_role IS NULL"
        " AND (c.in_my_league = 1 OR c.continental = 1 OR c.league_offset IN (1, -1))",
        (career_id,),
    )
    for r in rows:
        if r["is_newgen"]:
            kind = "newgen"
        elif photos_known and not r["has_photo"]:
            kind = "no_photo"
        else:
            continue
        scopes = []
        if r["in_my_league"]:
            scopes.append("league")
        if r["continental"]:
            scopes.append("continental")
        for scope, offset in _OFFSET_SCOPES.items():
            if r["league_offset"] == offset:
                scopes.append(scope)
        for scope in scopes:
            if scope in values:
                values[scope][f"{kind}_ca"].append(r["ca"])
                values[scope][f"{kind}_pa"].append(r["pa"])
    return {
        scope: ScopeStats(
            competitions=tuple(sorted(competitions[scope])),
            **{name: Histogram.of(v) for name, v in kinds.items()},
        )
        for scope, kinds in values.items()
    }


def store(
    conn: sqlite3.Connection,
    career_id: int,
    stats: Mapping[str, ScopeStats],
    game_date: date | None,
) -> None:
    """Replace the career's stats with these; scopes left out are removed (not found)."""
    conn.execute("DELETE FROM league_stats WHERE career_id = ?", (career_id,))
    conn.executemany(
        "INSERT INTO league_stats (career_id, scope, competitions, newgen_ca, newgen_pa,"
        " no_photo_ca, no_photo_pa, game_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                career_id,
                scope,
                json.dumps(list(s.competitions)),
                json.dumps(list(s.newgen_ca.counts)),
                json.dumps(list(s.newgen_pa.counts)),
                json.dumps(list(s.no_photo_ca.counts)),
                json.dumps(list(s.no_photo_pa.counts)),
                game_date.isoformat() if game_date else None,
            )
            for scope, s in stats.items()
            if scope in SCOPES
        ],
    )


def load(conn: sqlite3.Connection, career_id: int) -> dict[str, ScopeStats]:
    """The career's stats by scope; empty before its first sync with Phase 16."""
    return {
        r["scope"]: ScopeStats(
            competitions=tuple(json.loads(r["competitions"])),
            newgen_ca=Histogram.from_list(json.loads(r["newgen_ca"])),
            newgen_pa=Histogram.from_list(json.loads(r["newgen_pa"])),
            no_photo_ca=Histogram.from_list(json.loads(r["no_photo_ca"])),
            no_photo_pa=Histogram.from_list(json.loads(r["no_photo_pa"])),
        )
        for r in conn.execute("SELECT * FROM league_stats WHERE career_id = ?", (career_id,))
    }


@dataclass(frozen=True, slots=True)
class FoundLeague:
    """A league (or continental competition) as the page names it: by name where its editor ID
    is one checked (core/leagues.py COMPETITION_NAMES), else by its best-known clubs, since the
    save stores no competition names."""

    scope: str
    competition_id: int
    clubs: tuple[str, ...]
    """Up to three club names, the best known first."""
    club_count: int
    name: str | None = None
    """The competition's name, or the continental competitions' names joined; None if unknown."""


def found_leagues(conn: sqlite3.Connection, career_id: int) -> list[FoundLeague]:
    """Each found scope's leagues, named by their clubs of highest reputation: my league, the
    league(s) above and below, and the continental competitions' clubs as one."""
    out: list[FoundLeague] = []
    scopes = {
        r["scope"]: json.loads(r["competitions"])
        for r in conn.execute(
            "SELECT scope, competitions FROM league_stats WHERE career_id = ?", (career_id,)
        )
    }
    joined: str | None = None
    for scope in SCOPES:
        if scope not in scopes:
            continue
        if scope == "continental":
            where, args = "continental = 1", ()
            groups = [(scopes[scope][0] if scopes[scope] else 0, where, args)]
            names = [competition_name(c) for c in scopes[scope]]
            joined = " · ".join(n for n in names if n) if all(names) and names else None
        elif scope == "league":
            groups = [(c, "in_my_league = 1", ()) for c in scopes[scope][:1]]
        else:
            groups = [(c, "league_id = ?", (c,)) for c in scopes[scope]]
        for competition, where, args in groups:
            rows = conn.execute(
                f"SELECT name FROM clubs WHERE career_id = ? AND {where}"
                " ORDER BY reputation IS NULL, reputation DESC, name",
                (career_id, *args),
            ).fetchall()
            name = joined if scope == "continental" else competition_name(competition)
            out.append(
                FoundLeague(scope, competition, tuple(r[0] for r in rows[:3]), len(rows), name)
            )
    return out
