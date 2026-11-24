"""What the pipeline's results say on the page, in the request's language.

The pipeline keeps its own English summaries for logs and the command line; the pages word
the same fields through the message catalog.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from facegen.core.classify import Why
from facegen.core.cutoffs import Met, share_text
from facegen.core.problems import UNKNOWN, diagnose
from facegen.i18n import Translator
from facegen.pipeline.manual import REFUSAL_CODES, REFUSALS, ActionResult, ManualError
from facegen.pipeline.reclassification import GROUP_COUNT_KEYS, ChangeReport
from facegen.runtime.contracts import PollerStatus, WorkerStatus


def manual_summary(result: ActionResult, t: Translator) -> str:
    """ "3 queued, 1 pinned (already have a picture)" and the like."""
    parts = []
    if result.queued:
        parts.append(t("manual.queued", count=len(result.queued)))
    if result.pinned:
        parts.append(t("manual.pinned", count=len(result.pinned)))
    if result.skipped:
        why = "; ".join(sorted({refusal_text(reason, t) for _, reason in result.skipped}))
        parts.append(t("manual.skipped", count=len(result.skipped), why=why))
    return t("manual.join").join(parts) or t("manual.nothing")


def report_summary(report: ChangeReport, t: Translator) -> str:
    """The group counts that changed and what happened to the queue, on one line."""
    parts = []
    counts = t("manual.join").join(
        t(
            "report.count",
            name=t("report.tracked") if k == "tracked" else t(f"group.{k}"),
            before=report.before[k],
            after=report.after[k],
        )
        for k in GROUP_COUNT_KEYS
        if report.before[k] != report.after[k]
    )
    if counts:
        parts.append(counts)
    queue = [
        t(key, count=n)
        for n, key in (
            (sum(report.enqueue.values()), "report.queued"),
            (report.updated, "report.updated"),
            (report.cancelled, "report.cancelled"),
        )
        if n
    ]
    if queue:
        parts.append(t("report.queue", what=t("manual.join").join(queue)))
    return "; ".join(parts) or t("report.unchanged")


def drawing_key(kind: str, t: Translator, fallback: str) -> str:
    """The sentence for a picture being drawn, one per job kind ("worker.drawing.<kind>"), so
    an upgrade names its group once: "Upgrading {name} to {group}" (Phase 17e). `fallback`
    (with {kind}) for a kind the catalog does not word."""
    key = f"worker.drawing.{kind}"
    return key if kind and t.has(key) else fallback


def worker_sentence(status: WorkerStatus, t: Translator) -> str:
    """What the worker is doing or waiting for, in plain words: its reason code worded by the
    catalog ("worker.reason.<code>"), or its English detail line when it gives none."""
    key = f"worker.reason.{status.reason}"
    if not status.reason or not t.has(key):
        return status.detail
    values = dict(status.values)
    if status.reason == "drawing":
        key = drawing_key(str(values.get("kind") or ""), t, key)
    if "kind" in values:
        values["kind"] = t(f"kind.{values['kind']}")
    if "group" in values:
        values["group"] = t(f"group.{str(values['group']).lower()}")
    if "groups" in values:
        values["groups"] = ", ".join(
            t(f"group.{g.strip().lower()}") for g in str(values["groups"]).split(",")
        )
    if "cause" in values:
        values["cause"] = t(f"worker.cause.{values['cause']}")
    if "review" in values:
        review = values.pop("review")
        name = values.get("name", "")
        values["then"] = (
            t("worker.then.review" if review else "worker.then.queue", name=name) if name else ""
        )
    try:
        return t(key, **values)
    except (KeyError, IndexError, ValueError):  # a value missing: the English line instead
        return status.detail


WATCHER_TONES = {
    "synced": "ok",
    "settling": "accent",
    "due": "accent",
    "syncing": "accent",
    "not set": "warn",
    "error": "bad",
    "stopped": "bad",
}
"""The save watcher's badge colour by state (Phase 17i); grey for the rest."""


def short_time(stamp: str, today: str | None = None) -> str:
    """ "YYYY-MM-DD HH:MM" as "HH:MM" when it is today, else as it is."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    day, _, time = stamp.partition(" ")
    return time if day == today and time else stamp


@dataclass(frozen=True, slots=True)
class WatcherLine:
    """The save watcher on the Log page (Phase 17i): a badge, then what it is about."""

    badge: str
    tone: str
    text: str
    checked: str | None
    error: str | None
    """An error's own text, folded away under Technical details."""


def watcher_line(status: PollerStatus, t: Translator, today: str | None = None) -> WatcherLine:
    """The watcher's state as a badge and its reason in the page's language
    ("log.watcher.<reason>"); its English line when the reason is unknown."""
    state_key = "log.watcher_state." + status.state.replace(" ", "_")
    badge = t(state_key) if t.has(state_key) else status.state
    values = dict(status.values)
    error = values.pop("error", None)
    if "saved" in values:
        values["saved"] = short_time(str(values["saved"]), today)
    key = f"log.watcher.{status.reason}"
    text = status.detail
    if status.reason and t.has(key):
        try:
            text = t(key, **values)
        except (KeyError, IndexError, ValueError):  # a value missing: the English line instead
            text = status.detail
    return WatcherLine(
        badge=badge,
        tone=WATCHER_TONES.get(status.state, ""),
        text=text,
        checked=status.checked,
        error=str(error) if error else None,
    )


def side_line(side: Mapping[str, Any], t: Translator) -> str:
    """The sidebar's state in a few words (Phase 17d): "Generating · 821 left". The worker's
    reason in short ("worker.short.<code>"), or its state's name when it gives none."""
    state, reason, queued = side.get("state"), side.get("reason"), side.get("queued") or 0
    if not state:
        return ""
    key = f"worker.short.{reason}"
    words = t(key) if reason and t.has(key) else t(f"worker.state.{state}")
    return t("side.left", words=words, count=queued) if queued else words


def seconds_since(iso: str | None, now: datetime | None = None) -> float | None:
    """Seconds from an ISO 8601 time (WorkerStatus.since) to now; None without one."""
    if not iso:
        return None
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return max(0.0, ((now or datetime.now(UTC)) - then).total_seconds())


def duration(seconds: float, t: Translator) -> str:
    """ "45 s", "12 min", "2 h 5 min": rounded up to the unit shown."""
    if seconds < 60:
        return t("dur.seconds", count=max(1, math.ceil(seconds)))
    minutes = math.ceil(seconds / 60)
    if minutes < 60:
        return t("dur.minutes", count=minutes)
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return t("dur.hours", count=hours)
    return t("dur.hours_minutes", hours=hours, minutes=rest)


def clock(seconds: float, t: Translator) -> str:
    """Time gone by, as a ticking clock shows it (Phase 17e): "19 s", "1 min 5 s", "1 h 2 min";
    rounded down, so it never runs ahead (app.js `clockText` words it the same way)."""
    whole = max(0, math.floor(seconds))
    if whole < 60:
        return t("dur.seconds", count=whole)
    minutes = whole // 60
    if minutes < 60:
        return t("dur.minutes_seconds", minutes=minutes, seconds=whole % 60)
    return t("dur.hours_minutes", hours=minutes // 60, minutes=minutes % 60)


def band_name(wording: str | None, t: Translator) -> str:
    """An age wording for people: "앳된 얼굴" for W02; a code the catalog lacks as itself."""
    if not wording:
        return "-"
    key = f"band.{wording}"
    return t(key) if t.has(key) else wording


# --- Phase 12e: errors as what happened and what to do ----------------------------------------


@dataclass(frozen=True, slots=True)
class Problem:
    """An error as a page shows it: what happened, what to do, where to go, and the error's
    own text folded away as the technical details."""

    id: str
    """Ties the folded details to the page part, so a live update keeps them open."""
    title: str
    cause: str
    todo: str
    detail: str
    more: str
    """The label of the folded details."""
    links: tuple[tuple[str, str], ...] = field(default=())


LINKS = {
    "gpu_memory": ("/hardware", "problem.link.autoconfig"),
    "models_missing": ("/#setup", "problem.link.models"),
    "folder": ("/settings#folders", "problem.link.settings"),
}


def problem(
    id: str, title: str, raw: str | None, t: Translator, *, log: bool = True
) -> Problem | None:
    """The error's kind (core.problems) worded for the page; None without an error."""
    if not raw:
        return None
    kind = diagnose(raw)
    links = []
    if kind in LINKS:
        href, label = LINKS[kind]
        links.append((href, t(label)))
    if log:
        links.append(("/log", t("problem.link.log")))
    return Problem(
        id=id,
        title=title,
        cause=t(f"problem.cause.{kind}"),
        todo=t(f"problem.do.{kind}"),
        detail=raw,
        more=t("problem.details"),
        links=tuple(links),
    )


def problem_line(raw: str, t: Translator) -> str:
    """An error in one plain line, for a message at the top of a page that cannot fold the
    details away: what happened and what to do; the error's text only when its kind is
    unknown."""
    kind = diagnose(raw)
    line = f"{t(f'problem.cause.{kind}')} {t(f'problem.do.{kind}')}"
    return f"{line} ({raw})" if kind == UNKNOWN else line


def review_problem(id: str, raw: str | None, t: Translator) -> Problem | None:
    """Why a job went to review: failed checks, a graphics fault, or another error."""
    if not raw:
        return None
    if raw.startswith("failed the quality check"):
        title = t("problem.review.qc")
    elif raw.startswith("the graphics driver failed"):
        title = t("problem.review.gpu")
    else:
        return problem(id, t("problem.review.failed"), raw, t, log=False)
    return Problem(id=id, title=title, cause="", todo="", detail=raw, more=t("problem.details"))


CHECK_FAILURES = (
    (re.compile(r"rated (?P<rating>\w+) \(general [\d.]+\)"), "rating"),
    (re.compile(r"content: (?P<tags>.+)"), "content"),
    (re.compile(r"hair colour not seen"), "hair"),
    (re.compile(r"eye colour not seen"), "eyes"),
    (re.compile(r"not one girl: (?P<what>.+)"), "people"),
    (re.compile(r"no figure found against the background"), "figure"),
    (re.compile(r"head starts (?P<share>\d+%) down the picture"), "head"),
)


def check_failure_text(raw: str, t: Translator) -> str:
    """Why a candidate failed its checks (render.quality_checks), in the page's language; the
    check's own text when it is not one of these. Only the wording: the checks are unchanged."""
    for pattern, key in CHECK_FAILURES:
        if m := pattern.fullmatch(raw):
            return t(f"review.failure.{key}", **m.groupdict())
    return raw


def refusal_text(reason: str, t: Translator) -> str:
    """Why a player was skipped (manual.REFUSALS), in the page's language."""
    code = REFUSAL_CODES.get(reason)
    return t(f"manual.refusal.{code}") if code else reason


def manual_error_text(exc: ManualError, t: Translator) -> str:
    """A refused request (ManualError) in the page's language; its English text when it has
    no code a catalog words."""
    code = getattr(exc, "code", "")
    values = dict(getattr(exc, "values", {}))
    if code in REFUSALS:
        why = t(f"manual.refusal.{code}")
        return t("manual.error.named", name=values["name"], why=why) if "name" in values else why
    if code and t.has(f"manual.error.{code}"):
        return t(f"manual.error.{code}", **values)
    return refusal_text(str(exc), t)


_SETTINGS_ERRORS = (
    (
        re.compile(r"must be between (?P<low>-?[\d.]+) and (?P<high>-?[\d.]+)"),
        "settings.error.range",
    ),
    (re.compile(r"must be one of (?P<choices>.+?), got "), "settings.error.choice"),
    (re.compile(r"first ages must rise strictly"), "settings.error.ages_rise"),
    (re.compile(r"needs at least one wording"), "settings.error.one_wording"),
    (re.compile(r"must be above 0 and at most 1"), "settings.error.scale"),
    (re.compile(r"needs an? (?:int|float|bool|str|list)"), "settings.error.type"),
)


def settings_error_text(message: str, t: Translator) -> str:
    """A settings check's message (persistence.settings.SettingsError) in the page's language,
    for the common checks; the message itself otherwise."""
    for pattern, key in _SETTINGS_ERRORS:
        if match := pattern.search(message):
            return t(key, **match.groupdict())
    return message


def _met_chip(met: Met | None, t: Translator) -> str:
    return t("why.met", attr=met.attr, at=met.at) if met else ""


def _met_sentence(met: Met | None, t: Translator) -> str:
    if met is None:
        return ""
    if met.share is not None:
        share = share_text(met.share)
        return t("why.met_share", attr=met.attr, value=met.value, at=met.at, share=share)
    return t("why.met_full", attr=met.attr, value=met.value, at=met.at)


def why_chip(why: Why, t: Translator) -> str:
    """A reason as a short chip: "My league · CA 98+", "Academy" (Phase 17h)."""
    match why.code:
        case "cut":
            met = _met_chip(why.met, t)
            return f"{t(f'why.where.{why.where}')} · {met}" if why.where else met
        case "part":
            return t(f"why.part.{why.part}")
        case "part_cut":
            return f"{t(f'why.part.{why.part}')} · {_met_chip(why.met, t)}"
    return t(f"why.{why.code}")


def why_sentence(why: Why, t: Translator) -> str:
    """A reason in full: a chip's tooltip, and a line on the player page."""
    match why.code:
        case "cut":
            who, met = t(f"why.who.{why.who}"), _met_sentence(why.met, t)
            if why.where is None:
                return t("why.full.cut_everywhere", who=who, met=met)
            return t("why.full.cut", who=who, where=t(f"why.full.where.{why.where}"), met=met)
        case "part":
            return t(f"why.full.part.{why.part}")
        case "part_cut":
            part = t(f"why.full.part.{why.part}")
            return t("why.full.part_cut", part=part, met=_met_sentence(why.met, t))
        case "pin":
            return t("why.full.pin", group=t(f"group.{why.group}"))
    return t(f"why.full.{why.code}")


def why_chips(row: Mapping[str, Any], t: Translator) -> list[tuple[str, str]] | None:
    """The Players list's Why: (chip, its sentence) for each reason for the group. Why someone
    is tracked goes with the auto-update tag (`why_tracking`). None for a row not classified
    since the reasons were stored as records: the page shows its English `reasons` then."""
    whys = row.get("why")
    if whys is None:
        return None
    return [(why_chip(w, t), why_sentence(w, t)) for w in whys if not w.tracking]


def why_lines(row: Mapping[str, Any], t: Translator) -> list[str]:
    """Every reason in full (the player page); the stored English ones for an older row."""
    whys = row.get("why")
    if whys is None:
        return list(row.get("reasons") or ())
    return [why_sentence(w, t) for w in whys]


def why_tracking(row: Mapping[str, Any], t: Translator) -> str:
    """Why someone is tracked, for the auto-update tag; the term's meaning when no reason
    says (group A, or a row not classified since the records were stored)."""
    for w in row.get("why") or ():
        if w.tracking:
            return why_sentence(w, t)
    return t("term.tracked")
