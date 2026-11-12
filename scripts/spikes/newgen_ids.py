"""Phase 0.1 spike: look for the ID boundary between database players and in-game-created ones.

Throwaway. uv run python scripts/spikes/newgen_ids.py SNAPSHOT
"""

import sys
import warnings
from collections import Counter

import fmsave

warnings.simplefilter("ignore")
save = fmsave.open(sys.argv[1])
players = save.players()
staff = save.staff()
high = sorted(p.unique_id for p in players if p.unique_id and p.unique_id >= 2_000_000_000)
print("players >= 2e9:", len(high), "min", high[0], "max", high[-1])
b = Counter((u - 2_000_000_000) // 20_000 for u in high)
for k in sorted(b):
    print(f"  {2_000_000_000 + k * 20_000:>13,}: {b[k]}")
# Largest gaps in the sorted ID list.
gaps = sorted(
    ((high[i + 1] - high[i], high[i], high[i + 1]) for i in range(len(high) - 1)), reverse=True
)[:8]
print("largest gaps:", gaps)

staff_ids = sorted(s.unique_id for s in staff if getattr(s, "unique_id", None))
print("staff max ids:", staff_ids[-5:])
