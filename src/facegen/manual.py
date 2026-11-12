"""Manual requests from the UI (docs/spec.md section 7). No GPU, no web.

Each action runs in one transaction: it writes pins and manual jobs, then re-plans the
queue from the stored snapshot, as a settings change does. The worker picks the jobs up.

- **Generate** pins the ticked players to the chosen group (group B unless another is
  chosen) and queues a manual job: a first picture, or an upgrade when the chosen group is
  above the current picture's. Hand-picking is a pin (decisions.md).
- **New face** re-rolls the character sheet and draws from it ("reroll").
- **New picture** keeps the sheet and draws again ("redraw"), for one player or many. At my
  club it uses the current picture as a reference (spec section 5).
- **Pin** to a group is a hand-pick, exactly as Generate. Pin to "never" cancels every
  queued job of the player, requests included; clearing a pin cancels a queued Generate
  request (the hand-pick is undone) and lets the rules decide.
- **Remove picture** takes the picture out of FM27's pack (the game's own photo comes back),
  cancels the player's queued jobs and pins them to "never", so nothing re-queues them.
- **Retry** queues a review-list job again, with new seeds.
- **Cancel** withdraws a queued request; for a Generate request it also clears the pin.
- **Restore** makes an earlier picture from the player's history current again and
  publishes it. It then stands for the player as they are now, so it is not re-drawn until
  their club, age band or group changes again (sync.image_state).

Manual jobs run before everything else (spec section 2). Among them the newest request runs
first, so one "new face" click is drawn next even behind a large selection; within one
request, players run in the order given.

Players outside men's football are refused: the rules ignore them even when pinned.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from facegen.age import age_wording
from facegen.classify import GROUP_RANK, Group
from facegen.db import active_career_id, transaction
from facegen.planner import TIER_MANUAL
from facegen.publish import PublishReport, publish_lock, publish_pack, unpublish
from facegen.settings import Settings
from facegen.sync import classify_and_plan

PINS = ("a", "b", "c", "never")
DEFAULT_GROUP = "b"
"""A hand-picked player's group unless the user picks another (decisions.md, proposed)."""


class ManualError(ValueError):
    """A request that cannot be carried out; the message is for the user."""


@dataclass
class ActionResult:
    queued: list[int] = field(default_factory=list)
    """Players with a manual job queued by this request."""
    pinned: list[int] = field(default_factory=list)
    """Players pinned without a new job (they already have a good enough picture)."""
    skipped: list[tuple[int, str]] = field(default_factory=list)
    """Players left alone, with the reason."""

    def summary(self) -> str:
        parts = []
        if self.queued:
            parts.append(f"{len(self.queued)} queued")
        if self.pinned:
            parts.append(f"{len(self.pinned)} pinned (already have a picture)")
        if self.skipped:
            why = sorted({reason for _, reason in self.skipped})
            parts.append(f"{len(self.skipped)} skipped ({'; '.join(why)})")
        return ", ".join(parts) or "nothing to do"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def request_rank(now: float | None = None, position: int = 0) -> float:
    """Queue rank of a manual request: the newest request runs first.

    The request time in microseconds, negated, plus the player's position in the request,
    so players ticked together run in the order given. Whole numbers below 2**53, so exact.
    """
    micros = int((time.time() if now is None else now) * 1_000_000)
    return -float(micros) + position


def _player(conn: sqlite3.Connection, career_id: int, uid: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT p.*, i.group_name AS image_group,"
        " (SELECT status FROM jobs j WHERE j.career_id = p.career_id"
        "   AND j.unique_id = p.unique_id AND j.status IN ('queued', 'running')) AS job_status"
        " FROM players p LEFT JOIN images i ON i.career_id = p.career_id"
        "   AND i.unique_id = p.unique_id AND i.is_current = 1"
        " WHERE p.career_id = ? AND p.unique_id = ?",
        (career_id, uid),
    ).fetchone()
    if row is None:
        raise ManualError(f"no player {uid} in this career")
    return row


def _refusal(p: sqlite3.Row) -> str | None:
    """Why no job can be queued for this player now, or None."""
    if not p["present"]:
        return "not in the latest save"
    if not p["in_scope"]:
        return "not in men's football"
    if p["job_status"] == "running":
        return "being drawn right now"
    return None


def _queue(
    conn: sqlite3.Connection,
    career_id: int,
    p: sqlite3.Row,
    kind: str,
    group: str,
    rank: float,
    reason: str,
    settings: Settings,
) -> None:
    """Queue a manual job for the player, taking over their queued automatic job if any."""
    band = age_wording(p["age"], group, bool(p["at_my_club"]), settings.age)
    now = _now()
    values = (kind, group, TIER_MANUAL, rank, p["club_id"], band, reason)
    cur = conn.execute(
        "UPDATE jobs SET kind = ?, group_name = ?, manual = 1, tier = ?, rank = ?,"
        " target_club_id = ?, target_band = ?, reason = ?, error = NULL, updated_at = ?"
        " WHERE career_id = ? AND unique_id = ? AND status = 'queued'",
        (*values, now, career_id, p["unique_id"]),
    )
    if cur.rowcount == 0:
        conn.execute(
            "INSERT INTO jobs (career_id, unique_id, kind, group_name, manual, tier, rank,"
            " status, target_club_id, target_band, reason, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 1, ?, ?, 'queued', ?, ?, ?, ?, ?)",
            (career_id, p["unique_id"], kind, group, TIER_MANUAL, rank, *values[4:], now, now),
        )


def _set_pin(conn: sqlite3.Connection, career_id: int, uid: int, pin: str | None) -> None:
    conn.execute(
        "UPDATE players SET pin = ? WHERE career_id = ? AND unique_id = ?", (pin, career_id, uid)
    )


def _generate_reason(group: str) -> str:
    return f"requested: generate as {group.upper()}"


def generate(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    settings: Settings,
    group: str = DEFAULT_GROUP,
    *,
    now: float | None = None,
) -> ActionResult:
    """Pin the players to `group` and queue a picture for those who need one."""
    if group not in ("a", "b", "c"):
        raise ManualError(f"unknown group {group!r}")
    result = ActionResult()
    if now is None:
        now = time.time()
    with transaction(conn):
        for uid in dict.fromkeys(uids):
            p = _player(conn, career_id, uid)
            if why := _refusal(p):
                result.skipped.append((uid, why))
                continue
            _set_pin(conn, career_id, uid, group)
            image = p["image_group"]
            if image is None:
                kind = "initial"
            elif GROUP_RANK[Group(group)] < GROUP_RANK[Group(image)]:
                kind = "upgrade"
            else:
                result.pinned.append(uid)
                continue
            rank = request_rank(now, len(result.queued))
            _queue(conn, career_id, p, kind, group, rank, _generate_reason(group), settings)
            result.queued.append(uid)
        classify_and_plan(conn, career_id, settings)
    return result


def _redraw_group(conn: sqlite3.Connection, career_id: int, p: sqlite3.Row, kind: str) -> str:
    """The group a re-draw is drawn at. Raises ManualError(reason) if there can be none; the
    reason leaves out the player's name, so many players' reasons can be summed up."""
    if why := _refusal(p):
        raise ManualError(why)
    if p["pin"] == "never":
        raise ManualError("pinned to never; unpin first")
    if kind == "redraw" and p["image_group"] is None:
        raise ManualError("no picture to draw again; use Generate")
    last_failed = conn.execute(
        "SELECT group_name FROM jobs WHERE career_id = ? AND unique_id = ?"
        " AND status = 'failed' ORDER BY id DESC LIMIT 1",
        (career_id, p["unique_id"]),
    ).fetchone()
    group = p["group_name"] or p["image_group"] or (last_failed[0] if last_failed else None)
    if group is None:
        raise ManualError("in no group; use Generate and choose one")
    return group


def _redraw_reason(kind: str) -> str:
    return f"requested: {'new face' if kind == 'reroll' else 'new picture'}"


def _redraw(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    kind: str,
    now: float | None,
) -> ActionResult:
    with transaction(conn):
        p = _player(conn, career_id, uid)
        try:
            group = _redraw_group(conn, career_id, p, kind)
        except ManualError as exc:
            raise ManualError(f"{p['name']}: {exc}") from None
        _queue(conn, career_id, p, kind, group, request_rank(now), _redraw_reason(kind), settings)
        classify_and_plan(conn, career_id, settings)
    return ActionResult(queued=[uid])


def new_face(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    *,
    now: float | None = None,
) -> ActionResult:
    """Re-roll the character sheet and draw a picture from the new one."""
    return _redraw(conn, career_id, uid, settings, "reroll", now)


def new_picture(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    *,
    now: float | None = None,
) -> ActionResult:
    """Keep the character sheet and draw a new picture."""
    return _redraw(conn, career_id, uid, settings, "redraw", now)


def new_pictures(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    settings: Settings,
    *,
    now: float | None = None,
) -> ActionResult:
    """New picture for many players at once (the players page); those who cannot have one
    are skipped with the reason. Players run in the order given."""
    result = ActionResult()
    if now is None:
        now = time.time()
    with transaction(conn):
        for uid in dict.fromkeys(uids):
            p = _player(conn, career_id, uid)
            try:
                group = _redraw_group(conn, career_id, p, "redraw")
            except ManualError as exc:
                result.skipped.append((uid, str(exc)))
                continue
            rank = request_rank(now, len(result.queued))
            _queue(conn, career_id, p, "redraw", group, rank, _redraw_reason("redraw"), settings)
            result.queued.append(uid)
        classify_and_plan(conn, career_id, settings)
    return result


def set_pin(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    pin: str | None,
    settings: Settings,
    *,
    now: float | None = None,
) -> ActionResult:
    """Pin players to a group (a hand-pick: Generate), to "never", or clear the pin (None)."""
    if pin in ("a", "b", "c"):
        return generate(conn, career_id, uids, settings, pin, now=now)
    if pin is not None and pin not in PINS:
        raise ManualError(f"unknown pin {pin!r}")
    result = ActionResult()
    with transaction(conn):
        for uid in dict.fromkeys(uids):
            _player(conn, career_id, uid)
            _set_pin(conn, career_id, uid, pin)
            if pin == "never":
                # The rules cancel automatic jobs; a request by hand they leave alone.
                conn.execute(
                    "UPDATE jobs SET status = 'cancelled', error = 'pinned to never',"
                    " updated_at = ? WHERE career_id = ? AND unique_id = ? AND status = 'queued'",
                    (_now(), career_id, uid),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = 'cancelled', error = 'pin cleared', updated_at = ?"
                    " WHERE career_id = ? AND unique_id = ? AND status = 'queued'"
                    " AND manual = 1 AND reason LIKE ?",
                    (_now(), career_id, uid, _generate_reason("%")),
                )
            result.pinned.append(uid)
        classify_and_plan(conn, career_id, settings)
    return result


def remove_picture(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    uid: int,
    settings: Settings,
) -> bool:
    """Take the player's picture out of FM27 and keep them from being drawn again.

    The master pictures and their history stay in data/. Returns whether a published
    picture or mapping was removed.
    """
    fm_user_dir: Path = settings.paths.fm_user_dir
    # The publish lock before the transaction, as publish_pack takes them.
    with publish_lock(settings.paths.data_dir), transaction(conn):
        p = _player(conn, career_id, uid)
        if p["job_status"] == "running":
            raise ManualError(f"{p['name']} is being drawn right now; try again in a moment")
        conn.execute(
            "UPDATE jobs SET status = 'cancelled', error = 'picture removed', updated_at = ?"
            " WHERE career_id = ? AND unique_id = ? AND status = 'queued'",
            (_now(), career_id, uid),
        )
        _set_pin(conn, career_id, uid, "never")
        active = active_career_id(conn) == career_id
        removed = unpublish(conn, career_id, career_key, uid, fm_user_dir, active=active)
        classify_and_plan(conn, career_id, settings)
    return removed


def retry(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    *,
    now: float | None = None,
) -> ActionResult:
    """Queue the player's job from the review list again; it draws with new seeds."""
    with transaction(conn):
        p = _player(conn, career_id, uid)
        if why := _refusal(p):
            raise ManualError(f"{p['name']}: {why}")
        failed = conn.execute(
            "SELECT kind, group_name FROM jobs WHERE career_id = ? AND unique_id = ?"
            " AND status = 'failed' ORDER BY id DESC LIMIT 1",
            (career_id, uid),
        ).fetchone()
        if failed is None:
            raise ManualError(f"{p['name']} has nothing to retry")
        _queue(
            conn,
            career_id,
            p,
            failed["kind"],
            failed["group_name"],
            request_rank(now),
            "requested: retry",
            settings,
        )
        classify_and_plan(conn, career_id, settings)
    return ActionResult(queued=[uid])


def cancel(conn: sqlite3.Connection, career_id: int, uid: int, settings: Settings) -> bool:
    """Cancel the player's queued manual request, then let the rules decide again.

    Cancelling a Generate request also clears the pin it set, so the hand-pick is undone;
    any other pin stays.
    """
    with transaction(conn):
        job = conn.execute(
            "SELECT id, group_name, reason FROM jobs WHERE career_id = ? AND unique_id = ?"
            " AND status = 'queued' AND manual = 1",
            (career_id, uid),
        ).fetchone()
        if job is None:
            return False
        conn.execute(
            "UPDATE jobs SET status = 'cancelled', error = 'cancelled in the UI', updated_at = ?"
            " WHERE id = ?",
            (_now(), job["id"]),
        )
        if job["reason"] == _generate_reason(job["group_name"]):
            conn.execute(
                "UPDATE players SET pin = NULL WHERE career_id = ? AND unique_id = ? AND pin = ?",
                (career_id, uid, job["group_name"]),
            )
        classify_and_plan(conn, career_id, settings)
    return True


@dataclass(frozen=True, slots=True)
class Restored:
    version: int
    pinned_never: bool
    """The player stays pinned to never: the picture is shown but never updated."""
    publish: PublishReport


def restore_picture(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    uid: int,
    image_id: int,
    settings: Settings,
) -> Restored:
    """Make an earlier picture from the player's history current again, and publish it.

    Only a picture that passed the quality check and whose file is still in data/ can come
    back (spec section 3: a picture that failed the check is never published). It stands for
    the player's group, club and age band now, so the rules do not re-draw it until one of
    them changes; queued automatic jobs it makes unnecessary are cancelled. Pins and manual
    requests are left alone.
    """
    data_dir: Path = settings.paths.data_dir
    with transaction(conn):
        p = _player(conn, career_id, uid)
        if p["job_status"] == "running":
            raise ManualError(f"{p['name']} is being drawn right now; try again in a moment")
        img = conn.execute(
            "SELECT * FROM images WHERE id = ? AND career_id = ? AND unique_id = ?",
            (image_id, career_id, uid),
        ).fetchone()
        if img is None:
            raise ManualError(f"{p['name']} has no picture {image_id}")
        if img["is_current"]:
            raise ManualError(f"v{img['version']} is already the current picture")
        if not restorable(img["master_path"], json.loads(img["check_result"] or "{}")):
            raise ManualError(f"v{img['version']} failed the quality check; it is never published")
        if not (data_dir / img["master_path"]).is_file():
            raise ManualError(f"v{img['version']}'s picture file is missing from data/")
        group = p["group_name"] or img["group_name"]
        band = age_wording(p["age"], group, bool(p["at_my_club"]), settings.age)
        conn.execute(
            "UPDATE images SET is_current = 0, published_path = NULL"
            " WHERE career_id = ? AND unique_id = ? AND is_current = 1",
            (career_id, uid),
        )
        conn.execute(
            "UPDATE images SET is_current = 1, published_path = NULL, restored_at = ?,"
            " restored_group = ?, restored_club_id = ?, restored_band = ? WHERE id = ?",
            (_now(), group, p["club_id"], band, image_id),
        )
        classify_and_plan(conn, career_id, settings)
    report = publish_pack(conn, settings, career_id, career_key)
    return Restored(img["version"], p["pin"] == "never", report)


def restorable(master_path: str | None, check: dict) -> bool:
    """Whether a picture passed its check and so may be shown in FM27 again."""
    if not master_path or master_path.endswith("-rejected.png"):
        return False
    return bool(check.get("passed"))
