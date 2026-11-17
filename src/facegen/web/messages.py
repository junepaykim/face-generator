"""What the pipeline's results say on the page, in the request's language.

The pipeline keeps its own English summaries for logs and the command line; the pages word
the same fields through the message catalog.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from facegen.i18n import Translator
from facegen.pipeline.manual import ActionResult
from facegen.pipeline.sync import GROUP_COUNT_KEYS, ChangeReport
from facegen.runtime.contracts import WorkerStatus


def manual_summary(result: ActionResult, t: Translator) -> str:
    """ "3 queued, 1 pinned (already have a picture)" and the like."""
    parts = []
    if result.queued:
        parts.append(t("manual.queued", count=len(result.queued)))
    if result.pinned:
        parts.append(t("manual.pinned", count=len(result.pinned)))
    if result.skipped:
        why = "; ".join(sorted({reason for _, reason in result.skipped}))
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


def worker_sentence(status: WorkerStatus, t: Translator) -> str:
    """What the worker is doing or waiting for, in plain words: its reason code worded by the
    catalog ("worker.reason.<code>"), or its English detail line when it gives none."""
    key = f"worker.reason.{status.reason}"
    if not status.reason or not t.has(key):
        return status.detail
    values = dict(status.values)
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


def band_name(wording: str | None, t: Translator) -> str:
    """An age wording for people: "앳된 얼굴" for W02; a code the catalog lacks as itself."""
    if not wording:
        return "-"
    key = f"band.{wording}"
    return t(key) if t.has(key) else wording
