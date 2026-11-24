"""The English and Korean message catalogs hold the same texts, and every key the templates,
routes and settings use is in them (Phase 12a).

Invariant: a page in either language never shows a raw key or misses a value to fill in.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest
from support.paths import ROOT

from kitshot.core import presets
from kitshot.core.classify import MY_CLUB_PARTS, WHERE_TEXT, WHO_TEXT, WHY_CODES, Why
from kitshot.core.cutoffs import SCOPES as CUTOFF_SCOPES
from kitshot.core.domain import JOB_KINDS
from kitshot.core.problems import KINDS as PROBLEM_KINDS
from kitshot.core.settings_values import (
    ASPECT_CHOICES,
    CUTOFF_MODES,
    DETAIL_SIZES,
    GENDER_CHOICES,
    LANGUAGE_CHOICES,
    MODEL_CHOICES,
    THEME_CHOICES,
)
from kitshot.i18n import LANGUAGES, Translator, catalog
from kitshot.persistence.settings import load_defaults
from kitshot.pipeline.manual import REFUSALS
from kitshot.runtime.contracts import (
    SYNC_NUMBERS,
    WATCHER_REASONS,
    WATCHER_STATES,
    WORKER_REASONS,
)
from kitshot.setup import hardware
from kitshot.setup.models import REPOS
from kitshot.web import presentation
from kitshot.web.forms import TABS, editable_keys, settings_sections
from kitshot.web.messages import CHECK_FAILURES
from kitshot.web.queries import (
    GALLERY_SORTS,
    GROUP_FILTERS,
    LEAGUES,
    ORIGINS,
    PHOTO_FILTERS,
    PICTURE_FILTERS,
    ROLES,
    SCOPES,
    SHOWN,
    SORTS,
    WHERE_FILTERS,
)

WEB = ROOT / "src" / "kitshot" / "web"
PLURAL = re.compile(r"\.(one|other)$")
# t("key" ...), tr("key" ...), t('key' ...) and the catalog keys handed to the script.
LITERAL_KEY = re.compile(r"""\btr?\(\s*["']([a-z_][a-z0-9_.\- ]*)["']""")
CONDITIONAL_KEY = re.compile(r"""\btr?\(\s*["']([a-z_][a-z0-9_.]*)["']\s+if\b""")
TERNARY_ELSE = re.compile(
    r"""\belse\s+["']((?:msg|player|players|status|dash|setup)\.[a-z0-9_.]+)["']"""
)
WORKER_STATES = (
    "starting", "idle", "paused", "waiting", "setup", "running", "error", "gpu-fault", "stopped",
    "restarting",
)  # fmt: skip


def base_keys(lang: str) -> dict[str, list[str]]:
    """Key without its plural form -> its texts."""
    keys: dict[str, list[str]] = {}
    for key, text in catalog(lang).items():
        keys.setdefault(PLURAL.sub("", key), []).append(text)
    return keys


def fields(text: str) -> set[str]:
    return {name.split(":")[0] for _, name, _, _ in string.Formatter().parse(text) if name}


def test_both_catalogs_have_the_same_keys_and_values_to_fill_in() -> None:
    en, ko = base_keys("en"), base_keys("ko")
    assert sorted(set(en) - set(ko)) == []
    assert sorted(set(ko) - set(en)) == []
    for key, texts in en.items():
        wanted = set().union(*(fields(t) for t in texts))
        for text in ko[key]:
            # Korean may leave out {count} where English needs it only to pick the form.
            assert fields(text) <= wanted, key
            assert fields(text) >= wanted - {"count"}, key


def test_every_text_formats() -> None:
    for lang in LANGUAGES:
        for key, text in catalog(lang).items():
            names = fields(text)
            text.format(**{n: 1 for n in names})  # no stray braces
            assert "{{" not in text, key


# Values that count something: shown with a thousands separator, "1,552" (Phase 17a). The
# durations ("{count} min") and the download sizes are worded by their own helpers.
COUNT_FIELDS = {"count", "current", "published", "picked", "mapped", "done", "failed", "before",
                "after", "page", "pages", "index", "files"}  # fmt: skip
NOT_COUNTS = ("dur.", "setup.models.progress")


def test_every_count_has_a_thousands_separator() -> None:
    for lang in LANGUAGES:
        for key, text in catalog(lang).items():
            if key.startswith(NOT_COUNTS):
                continue
            for _, name, spec, _ in string.Formatter().parse(text):
                if name in COUNT_FIELDS:
                    assert spec == ",", f"{lang} {key}: {{{name}}}"


@pytest.mark.parametrize(
    ("lang", "expected"),
    [("en", "1,552</a> (1,203 in FM27)"), ("ko", "1,552장</a> (FM27 적용 1,203장)")],
)
def test_big_numbers_read_the_same_way_everywhere(lang: str, expected: str) -> None:
    t = presentation.HtmlText(Translator(lang))
    assert expected in t("dash.counts.pictures_value", current=1552, published=1203)
    num = presentation.create_templates().env.filters["num"]
    assert (num(1552), num(12), num("?")) == ("1,552", "12", "?")


def used_keys() -> set[str]:
    sources = [*(WEB / "templates").glob("*.html"), *WEB.rglob("*.py")]
    keys: set[str] = set()
    for path in sources:
        source = path.read_text(encoding="utf-8")
        keys.update(LITERAL_KEY.findall(source))
        keys.update(CONDITIONAL_KEY.findall(source))
        keys.update(TERNARY_ELSE.findall(source))
    return {k for k in keys if not k.endswith(".")}


def test_every_literal_key_is_in_the_catalog() -> None:
    known = set(base_keys("en"))
    used = used_keys()
    assert len(used) > 300  # the scan found the templates
    assert sorted(used - known) == []


def vocabulary_keys() -> list[str]:
    keys = [f"kind.{k}" for k in JOB_KINDS] + [f"tier.{n}" for n in range(5)]
    keys += [f"worker.state.{s}" for s in WORKER_STATES]
    keys += [f"job_status.{s}" for s in ("queued", "running", "done", "failed", "cancelled")]
    keys += [f"players.league.{k}" for k in LEAGUES] + [f"players.origin.{k}" for k in ORIGINS]
    keys += [f"players.role.{k}" for k in ROLES] + [f"players.group.{k}" for k in GROUP_FILTERS]
    keys += [f"players.picture.{k}" for k in PICTURE_FILTERS]
    keys += [f"players.photo.{k}" for k in PHOTO_FILTERS] + [f"players.scope.{k}" for k in SCOPES]
    keys += [f"players.sort.{k}" for k in SORTS] + [f"players.shown.{k}" for k in SHOWN]
    # A player's reasons (Phase 17h): web/messages.py words them by code.
    built = {"part", "part_cut", "cut"}
    keys += [f"why.{c}" for c in WHY_CODES if c not in built and not Why(c).tracking]
    keys += [f"why.full.{c}" for c in WHY_CODES if c not in built]
    keys += ["why.full.cut", "why.full.cut_everywhere", "why.full.part_cut"]
    keys += [f"why.{kind}.{part}" for part in MY_CLUB_PARTS for kind in ("part", "full.part")]
    keys += [f"why.{kind}.{where}" for where in WHERE_TEXT for kind in ("where", "full.where")]
    keys += [f"why.who.{who}" for who in WHO_TEXT]
    keys += ["why.met", "why.met_full", "why.met_share"]
    keys += [f"settings.choice.detail_size.{k}" for k in DETAIL_SIZES]
    keys += [f"settings.choice.mode.{k}" for k in CUTOFF_MODES]
    keys += [f"settings.choice.aspect.{k}" for k in ASPECT_CHOICES]
    keys += [f"settings.tab.{k}" for k in TABS]
    keys += [f"cutoffs.found.{k}" for k in CUTOFF_SCOPES]
    keys += [f"settings.choice.language.{k}" for k in LANGUAGE_CHOICES]
    keys += [f"settings.choice.theme.{k}" for k in THEME_CHOICES]
    keys += [f"settings.choice.model.{k}" for k in MODEL_CHOICES]
    keys += [f"settings.choice.gender.{k}" for k in GENDER_CHOICES]
    for preset in (hardware.HIGH, hardware.MEDIUM, hardware.LOW, hardware.NONE):
        keys += [f"hardware.preset.{preset.name}.title", f"hardware.preset.{preset.name}.note"]
    keys += [f"tag.media.{role}" for role in ("journalist", "press officer")]
    keys += [f"gallery.role.{role}" for role in ("journalist", "press officer")]
    keys += [f"dash.sync.trigger.{k}" for k in ("poller", "sync now", "command")]
    keys += [f"dash.sync.chip.{k}" for k in SYNC_NUMBERS]
    keys += [f"dash.worker.generate_now_why.{k}" for k in ("fm", "held", "both")]
    keys += [f"worker.drawing.{k}" for k in JOB_KINDS]
    keys += list(presentation.JS_TEXTS)
    keys += [f"worker.reason.{code}" for code in WORKER_REASONS]
    keys += [f"worker.short.{code}" for code in WORKER_REASONS]
    keys += [f"worker.cause.{c}" for c in ("gpu", "crash", "spawn")]
    keys += [f"worker.then.{c}" for c in ("review", "queue")]
    keys += [f"log.watcher.{code}" for code in WATCHER_REASONS]
    keys += [f"log.watcher_state.{state.replace(' ', '_')}" for state in WATCHER_STATES]
    keys += [f"msg.folder_open.{k}" for k in ("unavailable", "missing", "failed")]
    keys += [
        f"dur.{k}" for k in ("seconds", "minutes", "hours", "hours_minutes", "minutes_seconds")
    ]
    keys += ["settings.field_tip.banned_traits", "settings.field_tip.fallback_min_unique_id"]
    keys += ["settings.field_tip.continental_competition_ids"]
    # Phase 17k: a quiet line under a setting, a box's example and FM27's folders by name.
    keys += ["settings.hint.generate_while_fm_running"]
    keys += ["settings.placeholder.continental_competition_ids"]
    keys += [f"settings.folders.{k}" for k in ("user", "install")]
    keys += [f"term.{k}" for k in ("a", "b", "c", "tracked", "pinned")]
    keys += [f"problem.{part}.{kind}" for kind in PROBLEM_KINDS for part in ("cause", "do")]
    keys += [f"manual.refusal.{code}" for code in REFUSALS]
    keys += [
        f"manual.error.{code}"
        for code in (
            "no_player", "busy", "nothing_to_retry", "no_image", "already_current",
            "failed_check", "file_missing", "career_changed", "no_current", "no_candidate",
            "candidate_failed", "already_picked", "candidate_missing", "picture_changed",
        )
    ]  # fmt: skip
    keys += [f"problem.link.{k}" for k in ("log", "autoconfig", "models", "settings")]
    keys += [f"models.repo.{repo}" for repo in REPOS]
    keys += [f"settings.basic.{kind}" for kind in presets.KINDS]
    keys += [f"settings.basic.reach_option.{name}" for name in presets.REACH_STEPS]
    keys += [f"settings.basic.quality_option.{name}" for name in presets.QUALITY]
    keys += [f"hardware.preset.{name}.title" for name in presets.QUALITY]
    keys += [f"gallery.where_option.{k}" for k in WHERE_FILTERS]
    keys += [f"gallery.sort.{k}" for k in GALLERY_SORTS]
    keys += [f"review.failure.{key}" for _, key in CHECK_FAILURES]
    return keys


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_every_key_built_from_a_value_is_in_the_catalog(lang: str) -> None:
    t = Translator(lang)
    assert [k for k in vocabulary_keys() if not t.has(k)] == []


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_every_setting_has_a_title_and_a_label(lang: str, tmp_path: Path) -> None:
    t = Translator(lang)
    defaults = load_defaults()
    sections = settings_sections(dict(defaults), defaults, t)
    shown = {f.key for _, _, _, fs in sections for f in fs}
    assert shown == set(editable_keys(defaults))
    for section, title, _, fs in sections:
        assert title != section, f"no title for {section}"
        for f in fs:
            assert "_" not in f.label or f.key.startswith("age."), f"no label for {f.key}"
