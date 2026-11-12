"""Phase 0.1 spike: explore player IDs, clubs, managed club and competitions.

Throwaway. Reads a snapshot copy only: uv run python scripts/spikes/explore_save.py SNAPSHOT
"""

import sys
import warnings
from collections import Counter

import fmsave

warnings.simplefilter("ignore")
save = fmsave.open(sys.argv[1])
info = save.info
print("build", info.build, "date", info.game_date, "save_name", repr(info.save_name))
print("summary strings:", info.summary_strings[:12])

clubs = save.clubs()
by_uid = {c.uid: c for c in clubs}
genders = Counter(c.gender.label.name if c.gender else None for c in clubs)
print("clubs", len(clubs), genders)

managed = list(save.managed_clubs())
print("managed", managed)

players = save.players()
print("players", len(players))
pg = Counter()
for p in players:
    c = by_uid.get(p.club_uid) if p.club_uid is not None else None
    pg[(c.gender.label.name if c and c.gender else "none")] += 1
print("player club gender", pg)

uids = [p.unique_id for p in players if p.unique_id is not None]
print("unique_id None:", sum(1 for p in players if p.unique_id is None))
print("unique_id min/max", min(uids), max(uids))
digits = Counter(len(str(u)) for u in uids)
print("digits", sorted(digits.items()))
# Bucket by leading magnitude
buckets = Counter()
for u in uids:
    buckets[u // 10_000_000 * 10_000_000] += 1
for k in sorted(buckets):
    print(f"  >= {k:>12,}: {buckets[k]}")

# Join dates in the career (after the game start) are a hint for generated players.
start = min((p.club_join_date for p in players if p.club_join_date), default=None)
print("earliest join", start)
