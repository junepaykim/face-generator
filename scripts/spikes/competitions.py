"""Phase 0.1 spike: generated-player block, league tables and multi-nation competitions.

Throwaway. uv run python scripts/spikes/competitions.py SNAPSHOT
"""

import sys
import warnings
from collections import Counter, defaultdict

import fmsave

warnings.simplefilter("ignore")
save = fmsave.open(sys.argv[1])
date = save.info.game_date
clubs = {c.uid: c for c in save.clubs()}

players = save.players()

BOUND = 2_001_500_000
gen = [p for p in players if p.unique_id and p.unique_id > BOUND]


def g(p):
    c = clubs.get(p.club_uid) if p.club_uid is not None else None
    return c.gender.label.name if c and c.gender else "none"


print("generated block:", len(gen), Counter(g(p) for p in gen))
print("  ages:", sorted(Counter(p.age for p in gen).items()))
print("  join dates:", Counter(str(p.club_join_date)[:7] for p in gen).most_common(6))
print(
    "  CA max",
    max(p.ability.current for p in gen),
    "PA max",
    max((p.ability.potential or 0) for p in gen),
)
print("  PA None:", sum(1 for p in gen if p.ability.potential is None))
pre = [p for p in players if p.unique_id and p.unique_id <= BOUND]
print(
    "pre-existing join dates after 2025-12:",
    sum(1 for p in pre if p.club_join_date and p.club_join_date.year >= 2026),
)
print(
    "PA None overall:",
    sum(1 for p in players if p.ability.potential is None),
    Counter(
        p.ability.potential_range_code for p in players if p.ability.potential is None
    ).most_common(5),
)
print("height range", min(p.height_cm for p in players), max(p.height_cm for p in players))
print(
    "age None",
    sum(1 for p in players if p.age is None),
    "birth None",
    sum(1 for p in players if p.birth_date is None),
)
print("on_loan True", sum(1 for p in players if p.on_loan))

comps = {c.id: c for c in save.competitions()}
print("\nleague tables:")
for t in save.league_tables():
    cs = [clubs.get(r.club_uid) for r in t.rows if r.club_uid is not None]
    nat = Counter(c.nation_id for c in cs if c)
    gen_ = Counter(c.gender.label.name for c in cs if c and c.gender)
    rep = sum(c.reputation or 0 for c in cs if c) / max(1, len(cs))
    comp = comps.get(t.competition_id)
    print(
        f"  comp {t.competition_id} db {comp.database_id if comp else None} rows {t.club_count} nations {len(nat)} {dict(gen_)} rep {rep:.0f}"
    )

print("\nfixture competitions spanning many nations:")
clubs_by_comp = defaultdict(set)
for f in save.fixtures():
    for uid in (f.home_club_uid, f.away_club_uid):
        if uid is not None and f.competition_id is not None:
            clubs_by_comp[f.competition_id].add(uid)
rows = []
for cid, uids in clubs_by_comp.items():
    cs = [clubs[u] for u in uids if u in clubs]
    nat = {c.nation_id for c in cs}
    gen_ = Counter(c.gender.label.name for c in cs if c.gender)
    rep = sum(c.reputation or 0 for c in cs) / max(1, len(cs))
    rows.append(
        (
            len(nat),
            cid,
            comps[cid].database_id if cid in comps else None,
            len(cs),
            dict(gen_),
            round(rep),
        )
    )
for r in sorted(rows, reverse=True)[:15]:
    print("  nations {} comp {} db {} clubs {} {} rep {}".format(*r))
