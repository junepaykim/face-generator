"""Manual requests from the UI (docs/spec.md section 7). No GPU, no web.

Each action runs in one transaction: it writes pins and manual jobs, then re-plans its own
players from the stored snapshot, as a settings change does for everyone. It uses the settings
saved by the time it holds the write lock (`read_settings`), as a sync does, so a settings
change saved meanwhile is not undone. The worker picks the jobs up.

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
  their club, age band or group changes again.
- **Pick** shows another candidate of the current picture in FM27 instead of the one the
  check chose. Only a candidate that passed the check can be picked; picking the check's
  own choice again puts its master back.

Manual jobs run before everything else (spec section 2). Among them the newest request runs
first, so one "new face" click is drawn next even behind a large selection; within one
request, players run in the order given.

Players outside men's football are refused: the rules ignore them even when pinned.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from facegen.core.age import age_wording, audience
from facegen.core.classify import GROUP_RANK, Group
from facegen.core.settings_values import Settings
from facegen.persistence.db import active_career_id, transaction
from facegen.persistence.image_store import mark_pictures_changed, save_png
from facegen.persistence.jobs import queue_manual as _queue
from facegen.persistence.sheet_store import select_sheet
from facegen.pipeline.publish import PublishReport, publish_lock, publish_pack, unpublish
from facegen.pipeline.reclassification import classify_and_plan

PINS = ("a", "b", "c", "never")
DEFAULT_GROUP = "b"
"""A hand-picked player's group unless the user picks another."""


class ManualError(ValueError):
    """A request that cannot be carried out. The message is for the user in English (logs, the
    command line); `code` and `values` let a page say it in its own language (Phase 12e)."""

    def __init__(self, message: str, code: str = "", **values: Any) -> None:
        super().__init__(message)
        self.code = code
        self.values = values


class CareerChanged(ManualError):
    def __init__(self) -> None:
        super().__init__(
            "This form no longer matches the active career. Reload the page and try again.",
            "career_changed",
        )


SettingsReader = Callable[[], Settings]
"""Reads the settings saved now, from the action's own connection."""


@contextmanager
def _action_transaction(
    conn: sqlite3.Connection,
    career_id: int,
    active_only: bool,
    settings: Settings,
    read_settings: SettingsReader | None,
) -> Iterator[Settings]:
    """A form's target must still be active when the writer acquires the database. Yields the
    settings to act with: those saved by then (`read_settings`), else `settings`."""
    with transaction(conn):
        if active_only and active_career_id(conn) != career_id:
            raise CareerChanged()
        yield settings if read_settings is None else read_settings()


REFUSALS: dict[str, str] = {
    "absent": "not in the latest save",
    "not_mens": "not in men's football",
    "drawing": "being drawn right now",
    "never": "pinned to never; unpin first",
    "no_picture": "no picture to draw again; use Generate",
    "no_group": "in no group; use Generate and choose one",
    "no_failed": "no failed picture to retry",
}
"""Why no job can be queued for a player, by code: the reasons ActionResult.skipped and
ManualError carry."""
REFUSAL_CODES = {text: code for code, text in REFUSALS.items()}


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
        raise ManualError(f"no player {uid} in this career", "no_player", uid=uid)
    return row


def _refusal(p: sqlite3.Row) -> str | None:
    """Why no job can be queued for this player now (REFUSALS), or None."""
    if not p["present"]:
        return REFUSALS["absent"]
    if not p["in_scope"]:
        return REFUSALS["not_mens"]
    if p["job_status"] == "running":
        return REFUSALS["drawing"]
    return None


def _refused(why: str, **values: Any) -> ManualError:
    """A refusal as an error; with a name, the message names the player."""
    name = values.get("name")
    return ManualError(f"{name}: {why}" if name else why, REFUSAL_CODES.get(why, ""), **values)


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
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """Pin the players to `group` and queue a picture for those who need one."""
    if group not in ("a", "b", "c"):
        raise ManualError(f"unknown group {group!r}")
    result = ActionResult()
    if now is None:
        now = time.time()
    selected = list(dict.fromkeys(uids))
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        for uid in selected:
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
            _queue(conn, career_id, p, kind, group, rank, _generate_reason(group), current)
            result.queued.append(uid)
        classify_and_plan(conn, career_id, current, only=selected)
    return result


def _redraw_group(conn: sqlite3.Connection, career_id: int, p: sqlite3.Row, kind: str) -> str:
    """The group a re-draw is drawn at. Raises ManualError(reason) if there can be none; the
    reason leaves out the player's name, so many players' reasons can be summed up."""
    if why := _refusal(p):
        raise _refused(why)
    if p["pin"] == "never":
        raise _refused(REFUSALS["never"])
    if kind == "redraw" and p["image_group"] is None:
        raise _refused(REFUSALS["no_picture"])
    last_failed = conn.execute(
        "SELECT group_name FROM jobs WHERE career_id = ? AND unique_id = ?"
        " AND status = 'failed' ORDER BY id DESC LIMIT 1",
        (career_id, p["unique_id"]),
    ).fetchone()
    group = p["group_name"] or p["image_group"] or (last_failed[0] if last_failed else None)
    if group is None:
        raise _refused(REFUSALS["no_group"])
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
    active_only: bool,
    read_settings: SettingsReader | None,
) -> ActionResult:
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        p = _player(conn, career_id, uid)
        try:
            group = _redraw_group(conn, career_id, p, kind)
        except ManualError as exc:
            raise _refused(str(exc), name=p["name"]) from None
        _queue(conn, career_id, p, kind, group, request_rank(now), _redraw_reason(kind), current)
        classify_and_plan(conn, career_id, current, only=(uid,))
    return ActionResult(queued=[uid])


def new_face(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    *,
    now: float | None = None,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """Re-roll the character sheet and draw a picture from the new one."""
    return _redraw(conn, career_id, uid, settings, "reroll", now, active_only, read_settings)


def new_picture(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    *,
    now: float | None = None,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """Keep the character sheet and draw a new picture."""
    return _redraw(conn, career_id, uid, settings, "redraw", now, active_only, read_settings)


def _redraw_many(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    settings: Settings,
    kind: str,
    now: float | None,
    active_only: bool,
    read_settings: SettingsReader | None,
) -> ActionResult:
    """A re-draw for many players at once; those who cannot have one are skipped with the
    reason. Players run in the order given."""
    result = ActionResult()
    if now is None:
        now = time.time()
    selected = list(dict.fromkeys(uids))
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        for uid in selected:
            p = _player(conn, career_id, uid)
            try:
                group = _redraw_group(conn, career_id, p, kind)
            except ManualError as exc:
                result.skipped.append((uid, str(exc)))
                continue
            rank = request_rank(now, len(result.queued))
            _queue(conn, career_id, p, kind, group, rank, _redraw_reason(kind), current)
            result.queued.append(uid)
        classify_and_plan(conn, career_id, current, only=selected)
    return result


def new_pictures(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    settings: Settings,
    *,
    now: float | None = None,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """New picture for many players at once (the players page)."""
    return _redraw_many(conn, career_id, uids, settings, "redraw", now, active_only, read_settings)


def new_faces(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    settings: Settings,
    *,
    now: float | None = None,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """New face for many players at once (the review page, Phase 12i)."""
    return _redraw_many(conn, career_id, uids, settings, "reroll", now, active_only, read_settings)


def set_pin(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    pin: str | None,
    settings: Settings,
    *,
    now: float | None = None,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """Pin players to a group (a hand-pick: Generate), to "never", or clear the pin (None)."""
    if pin in ("a", "b", "c"):
        return generate(
            conn,
            career_id,
            uids,
            settings,
            pin,
            now=now,
            active_only=active_only,
            read_settings=read_settings,
        )
    if pin is not None and pin not in PINS:
        raise ManualError(f"unknown pin {pin!r}")
    result = ActionResult()
    selected = list(dict.fromkeys(uids))
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        for uid in selected:
            _player(conn, career_id, uid)
            _set_pin(conn, career_id, uid, pin)
            if pin == "never":
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
        classify_and_plan(conn, career_id, current, only=selected)
    return result


def remove_picture(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    uid: int,
    settings: Settings,
    *,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> bool:
    """Take the player's picture out of FM27 and keep them from being drawn again.

    The master pictures and their history stay in data/. Returns whether a published
    picture or mapping was removed.
    """
    # The publish lock before the transaction, as publish_pack takes them.
    with (
        publish_lock(settings.paths.data_dir),
        _action_transaction(conn, career_id, active_only, settings, read_settings) as current,
    ):
        p = _player(conn, career_id, uid)
        if p["job_status"] == "running":
            raise ManualError(
                f"{p['name']} is being drawn right now; try again in a moment",
                "busy",
                name=p["name"],
            )
        conn.execute(
            "UPDATE jobs SET status = 'cancelled', error = 'picture removed', updated_at = ?"
            " WHERE career_id = ? AND unique_id = ? AND status = 'queued'",
            (_now(), career_id, uid),
        )
        _set_pin(conn, career_id, uid, "never")
        active = active_career_id(conn) == career_id
        fm_user_dir = current.paths.fm_user_dir
        removed = unpublish(conn, career_id, career_key, uid, fm_user_dir, active=active)
        classify_and_plan(conn, career_id, current, only=(uid,))
    return removed


def _queue_retry(
    conn: sqlite3.Connection,
    career_id: int,
    p: sqlite3.Row,
    settings: Settings,
    now: float | None,
    position: int = 0,
) -> str | None:
    """Queue the most recent failed job, or return the player's refusal reason."""
    if why := _refusal(p):
        return why
    failed = conn.execute(
        "SELECT kind, group_name FROM jobs WHERE career_id = ? AND unique_id = ?"
        " AND status = 'failed' ORDER BY id DESC LIMIT 1",
        (career_id, p["unique_id"]),
    ).fetchone()
    if failed is None:
        return REFUSALS["no_failed"]
    _queue(
        conn,
        career_id,
        p,
        failed["kind"],
        failed["group_name"],
        request_rank(now, position),
        "requested: retry",
        settings,
    )
    return None


def retry(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    *,
    now: float | None = None,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """Queue the player's job from the review list again; it draws with new seeds."""
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        p = _player(conn, career_id, uid)
        why = _queue_retry(conn, career_id, p, current, now)
        if why == REFUSALS["no_failed"]:
            raise ManualError(
                f"{p['name']} has nothing to retry", "nothing_to_retry", name=p["name"]
            )
        if why:
            raise _refused(why, name=p["name"])
        classify_and_plan(conn, career_id, current, only=(uid,))
    return ActionResult(queued=[uid])


def retry_many(
    conn: sqlite3.Connection,
    career_id: int,
    uids: Iterable[int],
    settings: Settings,
    *,
    now: float | None = None,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> ActionResult:
    """`retry` for many players at once (the review page, Phase 12i): each one's failed job
    queued again with new seeds; those who cannot be retried are skipped with the reason."""
    result = ActionResult()
    if now is None:
        now = time.time()
    selected = list(dict.fromkeys(uids))
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        for uid in selected:
            p = _player(conn, career_id, uid)
            if why := _queue_retry(conn, career_id, p, current, now, len(result.queued)):
                result.skipped.append((uid, why))
            else:
                result.queued.append(uid)
        classify_and_plan(conn, career_id, current, only=selected)
    return result


def cancel(
    conn: sqlite3.Connection,
    career_id: int,
    uid: int,
    settings: Settings,
    *,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> bool:
    """Cancel the player's queued manual request, then let the rules decide again.

    Cancelling a Generate request also clears the pin it set, so the hand-pick is undone;
    any other pin stays.
    """
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
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
        classify_and_plan(conn, career_id, current, only=(uid,))
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
    *,
    active_only: bool = False,
    read_settings: SettingsReader | None = None,
) -> Restored:
    """Make an earlier picture from the player's history current again, and publish it.

    Only a picture that passed the quality check and whose file is still in data/ can come
    back (spec section 3: a picture that failed the check is never published). It stands for
    the player's group, club and age band now, so the rules do not re-draw it until one of
    them changes; queued automatic jobs it makes unnecessary are cancelled. Pins and manual
    requests are left alone.
    """
    data_dir: Path = settings.paths.data_dir
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        p = _player(conn, career_id, uid)
        if p["job_status"] == "running":
            raise ManualError(
                f"{p['name']} is being drawn right now; try again in a moment",
                "busy",
                name=p["name"],
            )
        img = conn.execute(
            "SELECT * FROM images WHERE id = ? AND career_id = ? AND unique_id = ?",
            (image_id, career_id, uid),
        ).fetchone()
        if img is None:
            raise ManualError(
                f"{p['name']} has no picture {image_id}", "no_image", name=p["name"], image=image_id
            )
        version = img["version"]
        if img["is_current"]:
            raise ManualError(
                f"v{version} is already the current picture", "already_current", version=version
            )
        if not restorable(img["master_path"], json.loads(img["check_result"] or "{}")):
            raise ManualError(
                f"v{version} failed the quality check; it is never published",
                "failed_check",
                version=version,
            )
        if not (data_dir / img["master_path"]).is_file():
            raise ManualError(
                f"v{version}'s picture file is missing from data/", "file_missing", version=version
            )
        group = p["group_name"] or img["group_name"]
        band = age_wording(p["age"], group, bool(p["at_my_club"]), current.age, **audience(p))
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
        if img["traits_version"] is not None:
            select_sheet(conn, career_id, uid, img["traits_version"])
        # A queued keep-face redraw must follow the restored selection. A pending
        # new-face request retains its separately allocated sheet.
        conn.execute(
            "UPDATE jobs SET traits_version = NULL, revision = revision + 1"
            " WHERE career_id = ? AND unique_id = ? AND status = 'queued' AND kind != 'reroll'",
            (career_id, uid),
        )
        classify_and_plan(conn, career_id, current, only=(uid,))
    report = publish_pack(conn, current, career_id, career_key)
    return Restored(img["version"], p["pin"] == "never", report)


@dataclass(frozen=True, slots=True)
class Picked:
    version: int
    publish: PublishReport


def pick_candidate(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    uid: int,
    attempt: int,
    index: int,
    settings: Settings,
    *,
    active_only: bool = False,
    version: int | None = None,
    read_settings: SettingsReader | None = None,
) -> Picked:
    """Show another candidate of the current picture in FM27, and publish it.

    Only a candidate that passed the quality check can be picked (spec section 3). It is
    written as `vNNN-pick-aAcI.png` beside the master and becomes the picture's master; the
    check's own choice is kept under `auto` in the check result, so picking it again puts its
    master (with the detail pass, when there was one) back. Group, club and age band stay the
    same, so nothing is re-planned.

    `version` is the picture the page showed: when another has been drawn since, nothing is
    picked. The pick is stored before the pack is written; picking the same candidate again
    while it is not published yet (that write failed) publishes it.
    """
    data_dir: Path = settings.paths.data_dir
    with _action_transaction(conn, career_id, active_only, settings, read_settings) as current:
        p = _player(conn, career_id, uid)
        if p["job_status"] == "running":
            raise ManualError(
                f"{p['name']} is being drawn right now; try again in a moment",
                "busy",
                name=p["name"],
            )
        img = conn.execute(
            "SELECT * FROM images WHERE career_id = ? AND unique_id = ? AND is_current = 1",
            (career_id, uid),
        ).fetchone()
        if img is None:
            raise ManualError(f"{p['name']} has no picture", "no_current", name=p["name"])
        if version is not None and img["version"] != version:
            raise ManualError(
                f"v{img['version']} has been drawn since the page showed v{version}",
                "picture_changed",
                version=img["version"],
            )
        version = img["version"]
        check = json.loads(img["check_result"] or "{}")
        cand = next(
            (
                c
                for c in check.get("candidates", [])
                if c.get("attempt") == attempt and c.get("index") == index
            ),
            None,
        )
        if cand is None:
            raise ManualError(
                f"v{version} has no candidate {attempt}/{index}", "no_candidate", version=version
            )
        if not cand.get("passed") or not restorable(img["master_path"], check):
            raise ManualError(
                "that candidate failed the quality check; it is never published",
                "candidate_failed",
            )
        chosen = check.get("chosen") or {}
        if (chosen.get("attempt"), chosen.get("index")) != (attempt, index):
            _choose(conn, data_dir, img, check, cand, attempt, index)
        elif img["published_path"] is not None:
            raise ManualError(
                f"that candidate is already v{version}'s picture",
                "already_picked",
                version=version,
            )
    report = publish_pack(conn, current, career_id, career_key)
    return Picked(version, report)


def _choose(
    conn: sqlite3.Connection,
    data_dir: Path,
    img: sqlite3.Row,
    check: dict[str, Any],
    cand: dict[str, Any],
    attempt: int,
    index: int,
) -> None:
    """Make the candidate the picture's master, waiting to be published (in the caller's
    transaction)."""
    version = img["version"]
    chosen = check.get("chosen") or {}
    auto = check.get("auto")
    if auto is not None and (auto.get("attempt"), auto.get("index")) == (attempt, index):
        master, seed = auto["master"], auto["seed"]
        if not (data_dir / master).is_file():
            raise ManualError(
                f"v{version}'s picture file is missing from data/",
                "file_missing",
                version=version,
            )
        del check["auto"]
    else:
        folder = Path(img["master_path"]).parent
        master = (folder / f"v{version:03d}-pick-a{attempt}c{index}.png").as_posix()
        seed = int(cand["seed"])
        if not (data_dir / master).is_file():
            source = Path(str(cand.get("path") or ""))
            if (
                source.is_absolute()
                or ".." in source.parts
                or source.parts[:1] != ("masters",)
                or not (data_dir / source).is_file()
            ):
                raise ManualError(
                    f"v{version}'s candidate file is missing from data/",
                    "candidate_missing",
                    version=version,
                )
            _write_pick(data_dir / source, data_dir / master, img, seed, attempt, index)
        if auto is None:
            check["auto"] = {
                "attempt": chosen.get("attempt"),
                "index": chosen.get("index"),
                "seed": img["seed"],
                "master": img["master_path"],
            }
    check["chosen"] = {"attempt": attempt, "index": index, "seed": seed}
    conn.execute(
        "UPDATE images SET master_path = ?, seed = ?, published_path = NULL,"
        " check_result = ? WHERE id = ?",
        (master, seed, json.dumps(check), img["id"]),
    )
    mark_pictures_changed(conn, img["career_id"])


def _write_pick(
    source: Path, dest: Path, img: sqlite3.Row, seed: int, attempt: int, index: int
) -> None:
    """The picked candidate (WebP, already 3:4) as a PNG master, as the publisher reads."""
    from PIL import Image

    with Image.open(source) as im:
        picture = im.convert("RGB")
    meta = {
        "model": img["model"],
        "prompt": img["prompt"],
        "negative": img["negative_prompt"],
        "seed": seed,
        "picked": {"attempt": attempt, "index": index},
    }
    save_png(dest, picture, meta)


def restorable(master_path: str | None, check: dict) -> bool:
    """Whether a picture passed its check and so may be shown in FM27 again."""
    if not master_path or master_path.endswith("-rejected.png"):
        return False
    return bool(check.get("passed"))
