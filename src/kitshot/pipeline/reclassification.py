"""Reclassify stored people, plan their jobs and report group and queue changes."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Collection
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from kitshot.core.age import age_wording
from kitshot.core.classify import (
    Classification,
    Group,
    PlayerFacts,
    Rules,
    Why,
    classified,
    classify,
    whys_to_json,
)
from kitshot.core.cutoffs import Cutoffs, effective_cutoffs
from kitshot.core.domain import ACTIVE_JOB_STATUSES
from kitshot.core.planner import ImageState, JobState, Plan, PlayerPlanState, plan
from kitshot.core.records import HUMAN_MANAGER_JOB
from kitshot.core.settings_values import Settings
from kitshot.persistence import league_stats, photo_index
from kitshot.persistence.db import transaction
from kitshot.persistence.jobs import apply_plan


def _date(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


@dataclass(frozen=True, slots=True)
class Move:
    """A player whose group or tracking a classification changes, from the stored values."""

    unique_id: int
    group_before: str | None
    tracked_before: bool
    group_after: str | None
    tracked_after: bool


@dataclass(frozen=True, slots=True)
class PlanResult:
    plan: Plan
    classifications: dict[int, Classification]
    moved: tuple[Move, ...] = ()


def _job_state(r: sqlite3.Row) -> JobState:
    return JobState(
        job_id=r["id"],
        kind=r["kind"],
        group=Group(r["group_name"]),
        status=r["status"],
        manual=bool(r["manual"]),
        tier=r["tier"],
        rank=r["rank"],
        target_club_id=r["target_club_id"],
        target_band=r["target_band"],
        reason=r["reason"],
    )


def image_state(r: sqlite3.Row) -> ImageState:
    """What a current picture stands for, to the planner.

    A drawn picture shows the group, club and age band it was drawn for. A picture restored
    from the history stands for the player as they were when it was restored (manual.py), so
    it is re-drawn only after a later change.
    """
    if r["restored_at"]:
        return ImageState(
            Group(r["restored_group"] or r["group_name"]), r["restored_club_id"], r["restored_band"]
        )
    return ImageState(Group(r["group_name"]), r["club_id"], r["age_band"])


def cutoffs_for(conn: sqlite3.Connection, career_id: int | None, settings: Settings) -> Cutoffs:
    """The numbers in force for a career: the settings over its last sync's candidates."""
    stats = league_stats.load(conn, career_id) if career_id is not None else {}
    return effective_cutoffs(settings.cutoffs, stats, no_photo=settings.no_photo.enabled)


def _selection(only: Collection[int] | None, column: str) -> tuple[str, tuple[Any, ...]]:
    """A WHERE clause narrowing `column` to `only` (none: every player), and its argument."""
    if only is None:
        return "", ()
    return f" AND {column} IN (SELECT value FROM json_each(?))", (json.dumps(sorted(only)),)


def _classify(
    conn: sqlite3.Connection,
    career_id: int,
    settings: Settings,
    only: Collection[int] | None = None,
) -> tuple[PlanResult, list[tuple[Any, ...]]]:
    """Classify the stored players (`only` these, if given) and plan their jobs, reading only.

    Returns the plan and the player rows whose group, tracking or reasons (as text or as
    records, `players.why`) would change.
    """
    where, args = _selection(only, "unique_id")
    images = {
        r["unique_id"]: image_state(r)
        for r in conn.execute(
            "SELECT unique_id, group_name, club_id, age_band, restored_at, restored_group,"
            " restored_club_id, restored_band FROM images WHERE career_id = ? AND is_current = 1"
            + where,
            (career_id, *args),
        )
    }
    active_in = ", ".join("?" * len(ACTIVE_JOB_STATUSES))
    active = {
        r["unique_id"]: _job_state(r)
        for r in conn.execute(
            f"SELECT * FROM jobs WHERE career_id = ? AND status IN ({active_in})" + where,
            (career_id, *ACTIVE_JOB_STATUSES, *args),
        )
    }
    failed = {
        r["unique_id"]: _job_state(r)
        for r in conn.execute(
            "SELECT * FROM jobs WHERE career_id = ? AND status = 'failed'" + where + " ORDER BY id",
            (career_id, *args),
        )
    }

    # Who FM27 shows a photo for is known only once a sync has read its photo list.
    photo_status = photo_index.status(conn)
    photos_known = photo_status is not None and photo_status.game_known

    rules = Rules.of(settings)
    cuts = cutoffs_for(conn, career_id, settings)
    managed = conn.execute(
        "SELECT managed_club_id FROM careers WHERE id = ?", (career_id,)
    ).fetchone()
    my_club = managed[0] if managed is not None else None
    classifications: dict[int, Classification] = {}
    changed: list[tuple[Any, ...]] = []
    moved: list[Move] = []
    states: list[PlayerPlanState] = []
    player_where, player_args = _selection(only, "p.unique_id")
    for r in conn.execute(
        "SELECT p.*, ph.unique_id IS NOT NULL AS has_photo, c.league_offset FROM players p"
        " LEFT JOIN photos ph ON ph.unique_id = p.unique_id"
        " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
        " WHERE p.career_id = ?" + player_where,
        (career_id, *player_args),
    ):
        uid = r["unique_id"]
        staff = bool(r["is_staff"])
        if r["present"]:
            facts = PlayerFacts(
                unique_id=uid,
                is_newgen=bool(r["is_newgen"]),
                ca=r["ca"],
                pa=r["pa"],
                in_scope=bool(r["in_scope"]),
                at_my_club=bool(r["at_my_club"]),
                in_my_league=bool(r["in_my_league"]),
                in_continental=bool(r["in_continental"]),
                academy=bool(r["academy"]),
                pin=r["pin"],
                is_staff=staff,
                staff_job=r["staff_job"],
                photo=bool(r["has_photo"]) if photos_known else None,
                youth=bool(r["youth_team"]),
                media=r["media_role"],
                local_media=bool(r["local_media"]),
                league_offset=r["league_offset"],
                loaned_out=(
                    my_club is not None
                    and r["loan_parent_club_id"] == my_club
                    and r["club_id"] != my_club
                ),
            )
            c = classify(facts, cuts, rules)
        else:
            c = classified(None, False, [Why("left_staff" if staff else "gone")])
        classifications[uid] = c
        group = c.group.value if c.group else None
        reasons = json.dumps(list(c.reasons))
        why = whys_to_json(c.why)
        stored = (r["group_name"], r["tracked"], r["reasons"], r["why"])
        if (group, int(c.tracked), reasons, why) != stored:
            changed.append((group, int(c.tracked), reasons, why, career_id, uid))
            if (group, c.tracked) != (r["group_name"], bool(r["tracked"])):
                moved.append(Move(uid, r["group_name"], bool(r["tracked"]), group, c.tracked))
        if c.group is None and uid not in active:
            continue
        who = {"staff": staff, "media": r["media_role"] is not None}
        states.append(
            PlayerPlanState(
                unique_id=uid,
                classification=c,
                club_id=r["club_id"],
                age_band=age_wording(r["age"], c.group, bool(r["at_my_club"]), settings.age, **who),
                at_my_club=bool(r["at_my_club"]),
                club_since=_date(r["club_since"]),
                pa=r["pa"],
                image=images.get(uid),
                active_job=active.get(uid),
                last_failed_job=failed.get(uid),
                manual_allowed=bool(r["present"] and r["in_scope"] and r["pin"] != "never"),
                manual_band=(
                    age_wording(
                        r["age"], active[uid].group, bool(r["at_my_club"]), settings.age, **who
                    )
                    if uid in active
                    else None
                ),
                media=who["media"],
            )
        )
    return PlanResult(plan(states, settings.style.gender), classifications, tuple(moved)), changed


def classify_and_plan(
    conn: sqlite3.Connection,
    career_id: int,
    settings: Settings,
    *,
    only: Collection[int] | None = None,
) -> PlanResult:
    """Classify every stored player of a career, or `only` these, and bring the queue in line.

    Uses stored facts and applies the plan inside the caller's transaction. One player's group
    and jobs never depend on another's (the cutoffs come from the last sync's stored
    candidates), so a manual request, which changes only its own players, re-plans just them.
    """
    result, changed = _classify(conn, career_id, settings, only)
    conn.executemany(
        "UPDATE players SET group_name = ?, tracked = ?, reasons = ?, why = ?"
        " WHERE career_id = ? AND unique_id = ?",
        changed,
    )
    apply_plan(conn, career_id, result.plan)
    return result


def reconcile_human_managers(conn: sqlite3.Connection, settings: Settings) -> None:
    """Bring existing saves' manager classifications and queued jobs up to date at startup.

    Older versions included the human manager in automatic staff selection. The caller holds
    the write transaction so a concurrent settings edit cannot be overwritten.
    """
    for career_id, ids in conn.execute(
        "SELECT career_id, json_group_array(unique_id) FROM players"
        " WHERE is_staff = 1 AND staff_job = ? GROUP BY career_id",
        (HUMAN_MANAGER_JOB,),
    ).fetchall():
        classify_and_plan(conn, career_id, settings, only=json.loads(ids))


def reclassify(
    conn: sqlite3.Connection, career_id: int, settings: Settings, *, dry_run: bool = False
) -> PlanResult:
    """Re-classify from the stored snapshot after a settings change (spec section 7)."""
    with transaction(conn, rollback=dry_run):
        return classify_and_plan(conn, career_id, settings)


@dataclass(frozen=True, slots=True)
class GroupChange:
    unique_id: int
    name: str
    club: str | None
    ca: int
    pa: int | None
    before: str | None
    after: str | None
    tracked_before: bool
    tracked_after: bool
    reasons: tuple[str, ...]


GROUP_COUNT_KEYS = ("a", "b", "c", "tracked")


@dataclass
class ChangeReport:
    """What re-classifying after a settings change did to the groups and the queue."""

    before: dict[str, int]
    after: dict[str, int]
    enqueue: Counter[tuple[str, str]]
    updated: int
    cancelled: int
    changes: list[GroupChange]
    """Players whose group or tracking changed, highest PA first, at most `limit`."""
    changed_total: int

    @property
    def empty(self) -> bool:
        return not (self.changed_total or self.enqueue or self.updated or self.cancelled)

    @property
    def age_queued(self) -> int:
        """Pictures queued to re-draw people whose age look changed (an age table's edit)."""
        return sum(n for (kind, _), n in self.enqueue.items() if kind == "age")

    def summary(self) -> str:
        """One line: the group counts that changed and what happened to the queue."""
        parts = []
        counts = ", ".join(
            f"{'tracked' if k == 'tracked' else k.upper()} {self.before[k]} → {self.after[k]}"
            for k in GROUP_COUNT_KEYS
            if self.before[k] != self.after[k]
        )
        if counts:
            parts.append(counts)
        queue = [
            f"{n} {what}"
            for n, what in (
                (sum(self.enqueue.values()), "queued"),
                (self.updated, "updated"),
                (self.cancelled, "cancelled"),
            )
            if n
        ]
        if queue:
            parts.append("queue: " + ", ".join(queue))
        return "; ".join(parts) or "no player moved; the queue is unchanged"

    def to_json(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "enqueue": [[kind, group, n] for (kind, group), n in sorted(self.enqueue.items())],
            "updated": self.updated,
            "cancelled": self.cancelled,
            "changes": [{**asdict(c), "reasons": list(c.reasons)} for c in self.changes],
            "changed_total": self.changed_total,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ChangeReport:
        return cls(
            before=data["before"],
            after=data["after"],
            enqueue=Counter({(kind, group): n for kind, group, n in data["enqueue"]}),
            updated=data["updated"],
            cancelled=data["cancelled"],
            changes=[GroupChange(**{**c, "reasons": tuple(c["reasons"])}) for c in data["changes"]],
            changed_total=data["changed_total"],
        )


def _group_counts(classifications: dict[int, Classification]) -> dict[str, int]:
    counts = Counter(c.group.value for c in classifications.values() if c.group is not None)
    return {
        "a": counts["a"],
        "b": counts["b"],
        "c": counts["c"],
        "tracked": sum(1 for c in classifications.values() if c.tracked),
    }


def reclassify_with_report(
    conn: sqlite3.Connection, career_id: int, settings: Settings, *, limit: int = 50
) -> ChangeReport:
    """Re-classify and re-plan inside the caller's transaction, and report what moved.

    "Before" is the groups as stored, so one classification is enough: the counts before are
    the counts after with each move taken back.
    """
    result = classify_and_plan(conn, career_id, settings)
    after = _group_counts(result.classifications)
    before = dict(after)
    for m in result.moved:
        if m.group_after:
            before[m.group_after] -= 1
        if m.group_before:
            before[m.group_before] += 1
        before["tracked"] += int(m.tracked_before) - int(m.tracked_after)

    moves = {m.unique_id: m for m in result.moved}
    rows: dict[int, sqlite3.Row] = {}
    uids = list(moves)
    for i in range(0, len(uids), 500):
        chunk = uids[i : i + 500]
        rows.update(
            (r["unique_id"], r)
            for r in conn.execute(
                "SELECT p.unique_id, p.name, p.ca, p.pa, c.name AS club FROM players p"
                " LEFT JOIN clubs c ON c.career_id = p.career_id AND c.club_id = p.club_id"
                f" WHERE p.career_id = ? AND p.unique_id IN ({', '.join('?' * len(chunk))})",
                (career_id, *chunk),
            )
        )

    def sort_key(uid: int) -> tuple[int, int, int]:
        r = rows[uid]
        return -(r["pa"] or 0), -r["ca"], uid

    changes = []
    for uid in sorted(moves, key=sort_key)[:limit]:
        m, r = moves[uid], rows[uid]
        changes.append(
            GroupChange(
                unique_id=uid,
                name=r["name"],
                club=r["club"],
                ca=r["ca"],
                pa=r["pa"],
                before=m.group_before,
                after=m.group_after,
                tracked_before=m.tracked_before,
                tracked_after=m.tracked_after,
                reasons=result.classifications[uid].reasons,
            )
        )
    p = result.plan
    return ChangeReport(
        before=before,
        after=after,
        enqueue=Counter((e.job.kind, e.job.group.value) for e in p.enqueue),
        updated=len(p.update),
        cancelled=len(p.cancel),
        changes=changes,
        changed_total=len(moves),
    )
