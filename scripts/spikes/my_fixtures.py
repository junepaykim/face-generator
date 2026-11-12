"""Phase 0.1 spike: the managed club's fixtures by competition. Throwaway."""

import sys
import warnings
from collections import Counter, defaultdict

import fmsave

warnings.simplefilter("ignore")
save = fmsave.open(sys.argv[1])
clubs = {c.uid: c for c in save.clubs()}
comps = {c.id: c for c in save.competitions()}
my = next(iter(save.managed_clubs())).club_uid
per = defaultdict(list)
for f in save.fixtures():
    if my in (f.home_club_uid, f.away_club_uid):
        per[f.competition_id].append(f)
for cid, fx in per.items():
    opp = {f.away_club_uid if f.home_club_uid == my else f.home_club_uid for f in fx}
    nat = Counter(clubs[o].nation_id for o in opp if o in clubs)
    allc = defaultdict(set)
    for f in save.fixtures():
        if f.competition_id == cid:
            allc[f.season_start_year].update({f.home_club_uid, f.away_club_uid})
    print(
        cid,
        comps[cid].database_id if cid in comps else None,
        "my fixtures",
        len(fx),
        "seasons",
        Counter(f.season_start_year for f in fx),
        "played",
        sum(f.played for f in fx),
        "opponents",
        len(opp),
        "nations",
        dict(nat),
        "clubs in comp",
        {k: len(v) for k, v in allc.items()},
        "slots",
        Counter(f.home_team_slot for f in fx),
    )
print("my nation", clubs[my].nation_id)
