"""Phase 0.1 spike: group counts computed straight from fmsave, sharing no code with facegen.

Used to cross-check `facegen sync --dry-run` (Phase 1 exit test). Applies the default
thresholds from docs/spec.md. Throwaway.

    uv run python scripts/spikes/dry_run_counts.py SNAPSHOT [--all-genders]
"""

import sys
import warnings
from collections import Counter

import fmsave

warnings.simplefilter("ignore")
path = sys.argv[1]
mens_only = "--all-genders" not in sys.argv
save = fmsave.open(path)
clubs = {c.uid: c for c in save.clubs()}
players = [p for p in save.players() if p.unique_id is not None]
managed = list(save.managed_clubs())
my = managed[0].club_uid if managed else None

# Newgen boundary: the largest gap in IDs above 2e9.
ids = sorted({p.unique_id for p in players if p.unique_id >= 2_000_000_000})
gap, i = max((ids[k + 1] - ids[k], k) for k in range(len(ids) - 1))
boundary = ids[i + 1]

# My league: the domestic competition holding my first team with the most first-team
# fixtures this season (table size breaks ties); all its tables' and fixtures' clubs.
tables = list(save.league_tables())
fixtures = [f for f in save.fixtures() if f.competition_id is not None]
latest = {}
for f in fixtures:
    latest[f.competition_id] = max(latest.get(f.competition_id, -1), f.season_start_year or -1)
members, my_fx, size = {}, Counter(), Counter()
for t in tables:
    if t.competition_id is None:
        continue
    rows = {r.club_uid for r in t.rows if r.team_slot in (0, None) and r.club_uid is not None}
    members.setdefault(t.competition_id, set()).update(rows)
    if my in rows:
        size[t.competition_id] = max(size[t.competition_id], len(rows))
for f in fixtures:
    if (f.season_start_year or -1) != latest[f.competition_id]:
        continue
    for uid, slot in ((f.home_club_uid, f.home_team_slot), (f.away_club_uid, f.away_team_slot)):
        if uid is not None and slot in (0, None):
            members.setdefault(f.competition_id, set()).add(uid)
            if uid == my:
                my_fx[f.competition_id] += 1
nation = clubs[my].nation_id if my in clubs else None
cands = [
    c
    for c, m in members.items()
    if my in m and sum(1 for u in m if u in clubs and clubs[u].nation_id == nation) >= 0.75 * len(m)
]
league = set()
if cands:
    comp = max(cands, key=lambda c: (my_fx[c], size[c]))
    league = members[comp]


def gender(p):
    c = clubs.get(p.club_uid) if p.club_uid is not None else None
    return c.gender.label.name if c and c.gender else ("NONE" if c is None else "UNKNOWN")


groups = Counter()
tracked = 0
split = Counter(gender(p) for p in players)
for p in players:
    g = gender(p)
    if g == "NONE" or (mens_only and g != "MEN"):
        continue  # first sync: no free agent has been seen at a club before
    newgen = p.unique_id >= boundary
    ca, pa = p.ability.current, p.ability.potential or 0
    at_my = my is not None and (p.club_uid == my or (p.on_loan and p.loan_parent_club_uid == my))
    academy = newgen and my is not None and my in p.home_grown_club_uids
    in_league = p.club_uid in league
    group = None
    if at_my or (newgen and (ca > 155 or pa > 170)):
        group = "A"
    elif newgen and (academy or (in_league and (ca > 130 or pa > 150)) or ca > 140 or pa > 155):
        group = "B"
    elif newgen and in_league and (ca > 115 or pa > 140):
        group = "C"
    if group:
        groups[group] += 1
        if group == "A" or academy:
            tracked += 1

print("mens only" if mens_only else "all genders", "| boundary", boundary)
print("split", dict(split))
print("my league clubs", len(league))
print("groups", dict(sorted(groups.items())), "tracked", tracked)
