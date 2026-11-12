"""Phase 0.1 spike: club key stability, league-table team slots, generated-name uniqueness.

Throwaway. uv run python scripts/spikes/club_keys.py SNAPSHOT
"""

import sys
import warnings
from collections import Counter

import fmsave

warnings.simplefilter("ignore")
save = fmsave.open(sys.argv[1])
clubs = list(save.clubs())
same = sum(1 for c in clubs if c.unique_id == c.uid)
print(
    "clubs uid == unique_id:",
    same,
    "/",
    len(clubs),
    "unique_id None:",
    sum(1 for c in clubs if c.unique_id is None),
)
print("example", [(c.uid, c.unique_id, c.name) for c in clubs[:3]])
by_uid = {c.uid: c for c in clubs}
for t in save.league_tables():
    if t.competition_id == 30:
        print(
            "MLS slots",
            Counter(r.team_slot for r in t.rows),
            "nations",
            Counter(by_uid[r.club_uid].nation_id for r in t.rows),
        )
        print("   fa nations", Counter(by_uid[r.club_uid].fa_nation_id for r in t.rows))
fx = [f for f in save.fixtures() if f.competition_id == 30]
print("MLS fixtures", len(fx), "home slots", Counter(f.home_team_slot for f in fx))
players = save.players()
gen = [p for p in players if p.unique_id and p.unique_id > 2_001_500_000]
keys = Counter((p.name, p.birth_date) for p in gen)
print("generated (name, birth) duplicates:", sum(v for v in keys.values() if v > 1))
pl = [p for p in players if p.unique_id]
print("player unique_id duplicates:", len(pl) - len({p.unique_id for p in pl}))
print(
    "players at managed club:",
    sum(1 for p in players if p.club_uid == 2000204569),
    "loaned out:",
    sum(1 for p in players if p.loan_parent_club_uid == 2000204569),
)
print(
    "team_slot of managed club players",
    Counter(p.team_slot for p in players if p.club_uid == 2000204569),
)
gk = Counter(p.natural_positions[:1] for p in players[:2000])
print("first natural positions", gk.most_common(8))
print("empty natural", sum(1 for p in players if not p.natural_positions))
