"""The Log page's report (Phase 17i, decision D11): plain text for the user to paste where they
ask for help.

It holds the app's version, the system, the graphics card, the image model and how its weights
are held, the picture settings that matter (shape, sizes, candidates, detail pass, VAE tiling),
the worker's state, the save watcher, the last sync and the last LOG_LINES worker log lines,
oldest first, with their levels. No pictures. Player and career names stay; the user's home
folder in every path becomes %USERPROFILE% (Windows) or ~ (Linux). The words are English, like
the log file: the report is read by whoever helps.

`build` is pure: the caller gathers the facts (web/routes/log.py).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kitshot.core.settings_values import (
    IMAGE_MODELS,
    RENDER_SIZES,
    GroupSpec,
    Settings,
    detail_size,
)
from kitshot.runtime.contracts import LogLine, PollerStatus, SyncStatus, WorkerStatus
from kitshot.setup.hardware import Hardware

GROUP_NAMES = {"a": "Detailed", "b": "Standard", "c": "Light"}
WINDOWS_HOME = "%USERPROFILE%"

# C:\Users\<name>\ (or with forward or doubled slashes) and WSL's /mnt/c/Users/<name>/: the
# name runs to the next separator, so one with a space goes whole. A path that ends at the name
# (no separator after it) takes the name up to a space or quote.
_HOMES = (
    re.compile(r"(?i)\b[a-z]:[\\/]+users[\\/]+[^\\/\r\n]+(?=[\\/])"),
    re.compile(r"(?i)/mnt/[a-z]/users/[^/\r\n]+(?=/)"),
    re.compile(r"(?i)\b[a-z]:[\\/]+users[\\/]+[^\\/\s\"'<>|]+"),
    re.compile(r"(?i)/mnt/[a-z]/users/[^/\s\"'<>|]+"),
)


_SEP = r"[\\/]+"
_PATH_END = r"""(?![^\\/\s"'<>|])"""
"""A home folder ends at a separator, a space, a quote or the end, not inside a longer name."""


@dataclass(frozen=True, slots=True)
class ReportFacts:
    """What the report says, gathered by the caller."""

    made_at: str
    """Local time, "YYYY-MM-DD HH:MM:SS"."""
    version: str
    install: str
    """"installed", "git checkout" or "" when unknown."""
    system: str
    python: str
    card: Hardware | None
    """None: not known (not asked here, or still being asked)."""
    card_asked: bool
    """This process can ask for the card."""
    settings: Settings
    models_ready: bool | None
    """The chosen model's files are on this PC; None: not checked in this process."""
    career: str | None
    worker: WorkerStatus | None
    poller: PollerStatus | None
    watching: bool
    last_sync: Mapping[str, Any] | None
    """The latest sync_runs row."""
    sync: SyncStatus | None
    """This run's last sync, with its warnings."""
    homes: Sequence[str] = ()
    """Home folders to hide, besides any C:\\Users\\<name>: this process's, as found."""


def redact(text: str, homes: Iterable[str] = ()) -> str:
    """The text with every home folder replaced: C:\\Users\\<name> and /mnt/c/Users/<name> by
    %USERPROFILE%, and each of `homes` (Path.home() and the like) by %USERPROFILE% when it is
    a Windows one, else by ~."""
    for home in sorted({h.rstrip("\\/") for h in homes if h}, key=len, reverse=True):
        parts = [re.escape(part) for part in re.split(r"[\\/]+", home) if part]
        if len(parts) < 2:  # "/" or "C:": nothing to hide, and everything would match
            continue
        lead = _SEP if home[0] in "\\/" else ""
        pattern = re.compile("(?i)" + lead + _SEP.join(parts) + _PATH_END)
        windows = re.match(r"(?i)[a-z]:[\\/]|/mnt/[a-z]/", home) is not None
        text = pattern.sub(lambda _, mark=WINDOWS_HOME if windows else "~": mark, text)
    for found in _HOMES:
        text = found.sub(lambda _: WINDOWS_HOME, text)
    return text


def _yes(value: bool) -> str:
    return "on" if value else "off"


def _card(facts: ReportFacts) -> str:
    card = facts.card
    if card is None:
        return "not known yet" if facts.card_asked else "not asked in this process"
    extra = ", ".join(
        part
        for part in (
            f"torch {card.torch}" if card.torch else "",
            f"CUDA {card.cuda}" if card.cuda else "",
        )
        if part
    )
    return card.describe() + (f" ({extra})" if extra else "")


def _group(name: str, spec: GroupSpec, aspect: str) -> str:
    detail = detail_size(spec.detail, aspect)
    pass_text = (
        f"detail pass {spec.detail} ({detail[0]}x{detail[1]})" if detail else "no detail pass"
    )
    return (
        f"  {GROUP_NAMES[name]} ({name.upper()}): published {spec.published_long_edge} px,"
        f" {spec.candidates} candidate(s), {pass_text}"
    )


def _settings(facts: ReportFacts) -> list[str]:
    s = facts.settings
    model = IMAGE_MODELS.get(s.style.model)
    if model is None:
        model_text = f"{s.style.model} (unknown)"
    else:
        model_text = f"{model.name} ({model.weights})" if model.weights else model.name
    files = {True: "on this PC", False: "missing", None: "not checked in this process"}
    width, height = RENDER_SIZES.get(s.picture.aspect, (0, 0))
    lines = [
        f"Image model: {model_text}; files {files[facts.models_ready]}",
        f"Drawn as: {s.style.gender}",
        f"Picture shape: {s.picture.aspect}, rendered at {width}x{height}",
        "Groups:",
    ]
    for name in ("a", "b", "c"):
        lines.append(_group(name, getattr(s.groups, name), s.picture.aspect))
    auto = ", ".join(g.upper() for g in sorted(s.schedule.auto_groups)) or "none"
    lines += [
        f"Drawn automatically: {auto}",
        f"Draw while FM27 runs: {_yes(s.schedule.generate_while_fm_running)}",
        f"VAE tiling: {_yes(s.render.vae_tiling)}",
    ]
    return lines


def _worker(worker: WorkerStatus | None) -> list[str]:
    if worker is None:
        return ["Worker: not in this process"]
    state = worker.state + (f" ({worker.reason})" if worker.reason else "")
    if worker.paused and worker.state != "paused":
        state += ", paused"
    lines = [f"Worker: {state}" + (f" since {worker.since}" if worker.since else "")]
    if worker.detail:
        lines.append(f"  {worker.detail}")
    job = worker.current
    if job is not None:
        lines.append(f"  Drawing: {job.name} ({job.unique_id}), {job.kind} {job.group.upper()}")
    lines.append(
        f"  This run: {worker.done} made, {worker.failed} sent to review;"
        f" models {'loaded' if worker.models_loaded else 'not loaded'}"
        + ("; Generate now on" if worker.generate_now else "")
    )
    if worker.last_error:
        lines.append(f"  Last error: {worker.last_error}")
    return lines


def _watcher(facts: ReportFacts) -> str:
    poller = facts.poller
    if not facts.watching or poller is None:
        return "Save watcher: off"
    checked = f" (checked {poller.checked})" if poller.checked else ""
    return f"Save watcher: {poller.state}{checked}" + (
        f": {poller.detail}" if poller.detail else ""
    )


def _sync(facts: ReportFacts) -> list[str]:
    row = facts.last_sync
    if row is None:
        lines = ["Last sync: none yet"]
    else:
        when = row.get("finished_at") or row.get("started_at") or "?"
        lines = [
            f"Last sync: {row.get('save_file')} at {when}",
            (
                f"  Build: {row.get('build') or 'unknown'};"
                f" in-game date {row.get('game_date') or '?'};"
                f" started by {row.get('trigger') or '?'}"
            ),
        ]
        if row.get("error"):
            unsupported = " (the build is not supported)" if row.get("unsupported") else ""
            lines.append(f"  Error{unsupported}: {row['error']}")
    sync = facts.sync
    if sync is not None:
        if sync.running:
            lines.append(f"  Running now, started {sync.started} by {sync.trigger}")
        if sync.message:
            lines.append(f"  This run: {sync.message}")
        if sync.error and not sync.running:
            lines.append(f"  This run's error: {sync.error}")
        lines.extend(f"  Warning: {w}" for w in sync.warnings)
    return lines


def log_line(line: LogLine) -> str:
    """One worker log line for the report: time, level, text."""
    return f"{line.time} {line.level.upper():<7} {line.text}"


def build(facts: ReportFacts) -> str:
    """The report's text, the home folder hidden in every path."""
    s = facts.settings
    lines = [
        f"Kitshot report, {facts.made_at}",
        "",
        f"Version: {facts.version}" + (f" ({facts.install})" if facts.install else ""),
        f"System: {facts.system}; Python {facts.python}",
        f"Graphics card: {_card(facts)}",
        *_settings(facts),
        "",
        f"Career: {facts.career or 'none yet'}",
        *_worker(facts.worker),
        _watcher(facts),
        *_sync(facts),
        "",
        f"Data folder: {s.paths.data_dir}",
        f"FM27 folder: {s.paths.fm_user_dir or 'not set'}",
        "",
    ]
    log = facts.worker.log if facts.worker is not None else ()
    lines.append(f"Worker log, oldest first ({len(log)} line(s)):")
    lines.extend(log_line(line) for line in log)
    return redact("\n".join(lines) + "\n", facts.homes)
