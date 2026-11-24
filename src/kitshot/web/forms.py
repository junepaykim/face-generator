"""Settings form vocabulary, presentation and validation, independent of routes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request
from starlette.datastructures import FormData

from kitshot.core.age import WORDINGS
from kitshot.core.cutoffs import OFF
from kitshot.core.settings_values import (
    AGE_MODES,
    ASPECT_CHOICES,
    CANDIDATE_RANGE,
    CUTOFF_MODES,
    DETAIL_SIZES,
    GENDER_CHOICES,
    LANGUAGE_CHOICES,
    MODEL_CHOICES,
    PICTURE_HEIGHTS,
    SOFTEN_SKIN_STEPS,
    THEME_CHOICES,
    TIER_CHOICES,
)
from kitshot.i18n import ENGLISH, Translator
from kitshot.persistence.settings import SKIP_WORDING, is_age_wording, is_cut_cell, is_fixed
from kitshot.web.messages import band_name

# Who is picked first: it is what the page is mostly for (spec section 7). Each section's title
# is the catalog's "settings.title.<section>", its note "settings.note.<top section>" (most
# sections have none: Phase 12 keeps the page short), its ⓘ tip "settings.tip.<top>". Quality
# reads from what is drawn to how (Phase 17j): the style (with the picture's shape), the groups'
# quality, rendering and the reference. Appearance (Phase 17l) holds how people look: age looks,
# then players at my club, height and build, personality. Cutoffs opens with who counts; System
# ends with the FM27 database IDs (Phase 17k).
SECTION_ORDER = (
    "eligibility", "cutoffs", "style", "picture", "groups", "render", "reference", "age",
    "my_club", "body", "personality", "staff", "media", "no_photo", "schedule", "display",
    "updates", "database_ids",
)  # fmt: skip
TABS: dict[str, tuple[str, ...]] = {
    "cutoffs": ("eligibility", "cutoffs"),
    "quality": ("groups", "picture", "render", "style", "reference"),
    "targets": ("staff", "media", "no_photo"),
    "appearance": ("age", "my_club", "body", "personality"),
    "system": ("schedule", "display", "updates", "database_ids"),
}  # fmt: skip
"""The Advanced page's tabs (Phase 16h, user 2026-11-20) and the top sections each holds:
who is picked; how pictures are drawn; who else gets one; how people look (Appearance, D10,
Phase 17l); how Kitshot runs. Who counts ("eligibility") opens Cutoffs (Phase 17k)."""
KEY_SECTIONS: dict[str, str] = {
    "eligibility.loaned_out_at_my_club": "cutoffs.my_club",
    "notifications.new_pictures": "schedule",
    "eligibility.continental_competition_ids": "database_ids",
    "newgen.fallback_min_unique_id": "database_ids",
}
"""Settings shown in another section than their key's, and so maybe in another tab (Phase 17k):
loaned-out players beside "Players I loaned out", the notification with Running, and the two
FM27 database IDs in one folded card at the bottom of System. Only where a setting is shown:
its key, its history line and its label stay."""
TAB_ANCHORS = {
    "ages": "appearance",
    "about": "system",
    "folders": "system",
    "graphics-card": "quality",
    "notifications": "system",
    "newgen": "system",
}
"""Anchors from before the tabs, or of sections whose settings moved (Phase 17k), and the tab each
opens (sections: their top section's tab). "ages" is the Age looks tab before Appearance
(Phase 17l): `#tab-ages` opens Appearance."""
LABELLED_IN_FULL = frozenset(
    {
        "cutoffs.mode",
        "cutoffs.light_above",
        "cutoffs.light_below",
        "staff.enabled",
        "no_photo.enabled",
        "media.journalists",
        "media.press_officers",
        "reference.enabled",
        "display.language",
        "display.theme",
        "notifications.new_pictures",
        "updates.check",
    }
)
"""Keys labelled by the whole key ("settings.label_in.<key>"); the others by their last part
("settings.label.<name>"), as the same name means the same in every section."""

CHOICES: dict[str, tuple[str, ...]] = {
    "detail_size": tuple(DETAIL_SIZES),
    "language": LANGUAGE_CHOICES,
    "theme": THEME_CHOICES,
    "model": MODEL_CHOICES,
    "gender": GENDER_CHOICES,
    "mode": CUTOFF_MODES,
    "aspect": ASPECT_CHOICES,
    "first_team": TIER_CHOICES,
    "youth": TIER_CHOICES,
    "loans": TIER_CHOICES,
    "from": tuple(WORDINGS),
    "to": tuple(WORDINGS),
    "age_mode": AGE_MODES,
}
"""Text settings with a fixed set of values, by `choice_name` (mostly the key's last part), shown
as a drop-down list; each value's label is the catalog's "settings.choice.<name>.<value>", a
group's name for a tier and a look's name for a look."""
TIER_KEYS = frozenset({"first_team", "youth", "loans"})
LOOK_KEYS = frozenset({"from", "to"})

INT_CHOICES: dict[str, tuple[int, ...]] = {
    "published_long_edge": PICTURE_HEIGHTS,
    "soften_skin": SOFTEN_SKIN_STEPS,
}
"""Whole numbers offered as a short list, by the key's last part (Phase 17j). A value's label is
the catalog's "settings.choice.<name>.<value>" if it has one, else the number. Any value the
settings accept still is: a stored one outside the list stays selectable, marked as custom."""
SEGMENTED = frozenset({"soften_skin", "age_mode"})
"""Short lists shown as a row of buttons (a radio group) instead of a drop-down list, by
`choice_name`."""
STEPPERS: dict[str, tuple[int, int]] = {"candidates": CANDIDATE_RANGE}
"""Small whole numbers shown with − and + buttons, with their limits, by the key's last part."""

GROUP_LIST_KEYS = frozenset({"schedule.auto_groups"})
"""Lists of picture groups, shown as one tick box per group."""

BAND_TAGS = {
    "W01": "child",
    "W02": "young girl",
    "W03": "youthful face",
    "W04": "young teenage girl, round face",
    "W05": "young teenage girl",
    "W06": "teenage girl",
    "W07": "young adult woman",
    "W08": "young woman, adult",
    "W09": "adult woman, mature face",
    "W10": "mature female",
}
"""One or two words of each age wording's prompt, shown with its name (core.age.WORDINGS).
Every table's looks draw these words, staff's and media's too (Phase 17l: they showed none)."""


def look_words() -> dict[str, tuple[str, str]]:
    """Each look's short prompt words and its whole wording, for the age tables' rows: shown
    as they are drawn, never changed here (ground rule 2)."""
    return {look: (BAND_TAGS[look], wording.text) for look, wording in WORDINGS.items()}


def is_age_mode(key: str) -> bool:
    """An age table's mode ("age.b.mode"): its looks set by hand, or spread (Phase 17l)."""
    return key.startswith("age.") and key.endswith(".mode")


def choice_name(key: str) -> str:
    """The name a setting's choices and labels go by: an age table's mode is "age_mode", apart
    from the cutoffs' "mode" (Phase 17l); any other setting, its key's last part."""
    return "age_mode" if is_age_mode(key) else key.rpartition(".")[2]


def choice_label(name: str, value: str, t: Translator = ENGLISH) -> str:
    if name in TIER_KEYS:
        return t(f"group.{value}")
    if name in LOOK_KEYS:
        return band_name(value, t)
    return t(f"settings.choice.{name}.{value}")


def choices_of(key: str, t: Translator = ENGLISH) -> tuple[tuple[str, str], ...] | None:
    name = choice_name(key)
    values = CHOICES.get(name)
    if values is None:
        return None
    return tuple((value, choice_label(name, value, t)) for value in values)


def number_label(name: str, value: int, t: Translator = ENGLISH) -> str:
    """A listed whole number for people to read: "0 (Off)", or the number itself."""
    key = f"settings.choice.{name}.{value}"
    return t(key) if t.has(key) else str(value)


def number_choices_of(key: str, t: Translator = ENGLISH) -> tuple[tuple[str, str], ...] | None:
    name = key.rpartition(".")[2]
    values = INT_CHOICES.get(name)
    if values is None:
        return None
    return tuple((str(value), number_label(name, value, t)) for value in values)


def control_of(key: str) -> str:
    """How a setting is edited when not by its kind's usual field: "segmented", "stepper" or
    "select" (a list of whole numbers); "" for the usual field."""
    name = choice_name(key)
    if name in SEGMENTED:
        return "segmented"
    if name in STEPPERS:
        return "stepper"
    return "select" if name in INT_CHOICES else ""


@dataclass(frozen=True, slots=True)
class Field:
    key: str
    label: str
    kind: str  # bool, int, float, list, groups, band, choice, text
    value: Any
    default: Any
    changed: bool
    """Differs from the default."""
    choices: tuple[tuple[str, str], ...] = ()
    """For a choice: (value, label) each."""
    default_text: str = ""
    """The default for people to read."""
    label_key: str = ""
    """The label's catalog key, for a page to word it with group chips; `label` is plain."""
    label_values: dict[str, Any] = field(default_factory=dict)
    control: str = ""
    """`control_of`: a segmented row, a stepper or a list of numbers; "" for the usual field."""
    limits: tuple[int, int] | None = None
    """A stepper's lowest and highest value."""

    @property
    def text(self) -> str:
        return input_text(self.value, self.key)

    @property
    def default_input(self) -> str:
        """The default as the field holds it: what "use default" puts back."""
        if isinstance(self.default, bool):
            return "on" if self.default else ""
        return input_text(self.default, self.key)


def input_text(value: Any, key: str = "") -> str:
    """A setting's value as its text field holds it; a whole share reads "6", not "6.0". A look
    an age table leaves out is an empty cell, shown as "Not used" (Phase 17l)."""
    if is_age_wording(key) and value == SKIP_WORDING:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def value_text(value: Any, t: Translator = ENGLISH, key: str = "") -> str:
    """A setting's value for people to read; a choice by its label, groups by their names."""
    if key in GROUP_LIST_KEYS and isinstance(value, list):
        return ", ".join(t(f"group.{g}") for g in value) or t("settings.value.none")
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or t("settings.value.none")
    if isinstance(value, bool):
        return t("settings.value.on") if value else t("settings.value.off")
    if is_age_wording(key) and value == SKIP_WORDING:
        return t("settings.value.skip")
    if is_cut_cell(key):
        if value == OFF:
            return t("settings.value.off")
        if key.startswith("cutoffs.share."):
            return t("settings.value.share", share=f"{float(value):g}")
        return t("settings.value.at_least", value=value)
    name = choice_name(key)
    if name in CHOICES and value in CHOICES[name]:
        return choice_label(name, value, t)
    if name in INT_CHOICES and value in INT_CHOICES[name]:
        return number_label(name, value, t)
    return str(value)


def _kind(default: Any, key: str = "") -> str:
    if is_age_wording(key):
        return "band"
    if is_cut_cell(key):
        return "cell"
    if key in GROUP_LIST_KEYS:
        return "groups"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "choice" if choices_of(key) else "text"
    return "list"


def label_parts(key: str, t: Translator = ENGLISH) -> tuple[str, dict[str, Any]]:
    """A setting's label as a catalog key and its values; "" for a key the catalog lacks."""
    if key in LABELLED_IN_FULL:
        return f"settings.label_in.{key}", {}
    if is_age_mode(key):
        return "settings.label.age_mode", {}
    name = key.rpartition(".")[2]
    if is_age_wording(key):
        return "settings.age_label", {"band": band_name(name, t), "tags": BAND_TAGS[name]}
    label = f"settings.label.{name}"
    return (label, {}) if t.has(label) else ("", {})


def _label(key: str, t: Translator = ENGLISH) -> str:
    label_key, values = label_parts(key, t)
    return t(label_key, **values) if label_key else key.rpartition(".")[2].replace("_", " ")


def section_title(section: str, t: Translator = ENGLISH) -> str:
    key = f"settings.title.{section}"
    return t(key) if t.has(key) else section


def describe_key(key: str, t: Translator = ENGLISH) -> str:
    """A setting's section and label, short enough for one line: "Detailed group · CA above"."""
    section, _, name = key.rpartition(".")
    if is_age_wording(key):
        return f"{section_title(section, t)} · {band_name(name, t)}"
    return f"{section_title(section, t)} · {_label(key, t)}"


@dataclass(frozen=True, slots=True)
class ChangeLine:
    key: str
    label: str
    before: str
    after: str


def change_lines(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    defaults: Mapping[str, Any],
    t: Translator = ENGLISH,
) -> list[ChangeLine]:
    """Each key a history entry touched, with its value before and after.

    `before` and `after` hold stored overrides, None for none: then the default applies.
    """
    lines = []
    for key in sorted(after):
        default = defaults.get(key)
        old = before.get(key)
        new = after[key]
        lines.append(
            ChangeLine(
                key,
                describe_key(key, t),
                value_text(default if old is None else old, t, key),
                value_text(default if new is None else new, t, key),
            )
        )
    return lines


def editable_keys(defaults: dict[str, Any]) -> list[str]:
    return [k for k in defaults if not is_fixed(k)]


def section_of(key: str) -> str:
    """The section a setting is shown in: its key's, unless KEY_SECTIONS places it elsewhere."""
    return KEY_SECTIONS.get(key) or key.rpartition(".")[0]


def settings_sections(
    values: dict[str, Any], defaults: dict[str, Any], t: Translator = ENGLISH
) -> list[tuple[str, str, str, list[Field]]]:
    """(section key, title, note, fields) in the order of the defaults file; each setting in
    the section it is shown in (`section_of`)."""
    sections: dict[str, list[Field]] = {}
    for key in editable_keys(defaults):
        section, name = section_of(key), key.rpartition(".")[2]
        label_key, label_values = label_parts(key, t)
        kind = _kind(defaults[key], key)
        sections.setdefault(section, []).append(
            Field(
                key=key,
                label=_label(key, t),
                kind=kind,
                value=values[key],
                default=defaults[key],
                changed=values[key] != defaults[key],
                choices=choices_of(key, t) or number_choices_of(key, t) or (),
                default_text=value_text(defaults[key], t, key),
                label_key=label_key,
                label_values=label_values,
                control=control_of(key) if kind in ("int", "choice") else "",
                limits=STEPPERS.get(name),
            )
        )

    def order(section: str) -> int:
        top = section.split(".")[0]
        return SECTION_ORDER.index(top) if top in SECTION_ORDER else len(SECTION_ORDER)

    def note(section: str) -> str:
        key = f"settings.note.{section.split('.')[0]}"
        return t(key) if t.has(key) else ""

    return [
        (section, section_title(section, t), note(section), fields)
        for section, fields in sorted(sections.items(), key=lambda kv: order(kv[0]))
    ]


def tab_of(section: str) -> str:
    """The Advanced page's tab holding a section ("age.staff" -> "ages")."""
    top = section.split(".")[0]
    return next((tab for tab, tops in TABS.items() if top in tops), "system")


def tab_anchors() -> dict[str, str]:
    """Anchor name (an old card's id, or a top section) to the tab that holds it, for the page's
    script: `#about` or `#section-age-b` opens its tab."""
    return TAB_ANCHORS | {top: tab for tab, tops in TABS.items() for top in tops}


def settings_tabs(
    sections: list[tuple[str, str, str, list[Field]]],
) -> dict[str, list[tuple[str, str, str, list[Field]]]]:
    """The sections of each tab, in page order."""
    tabs: dict[str, list[tuple[str, str, str, list[Field]]]] = {tab: [] for tab in TABS}
    for section in sections:
        tabs[tab_of(section[0])].append(section)
    return tabs


def _typed(default: Any, text: str, key: str = "") -> Any:
    """Text from a field, typed as the default. ValueError if it does not fit."""
    kind = _kind(default, key)
    if kind == "band":  # a whole number, or "-" (or nothing) for a wording left out
        return SKIP_WORDING if text in ("", SKIP_WORDING) else int(text)
    if kind == "cell":  # a share or a number, or "-" (or nothing) to switch the cell off
        if text in ("", OFF):
            return OFF
        text = text.removesuffix("%").strip().replace(",", ".")
        return float(text) if key.startswith("cutoffs.share.") else int(text)
    if kind == "groups":
        return [x.lower() for x in re.split(r"[,\s]+", text) if x]
    if kind in ("choice", "text"):  # a choice's value is checked with the settings
        return text
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    items = [x for x in re.split(r"[,\s]+", text) if x]
    if default and isinstance(default[0], str):
        return [x.lower() for x in items]
    return [int(x) for x in items]


def _error(default: Any, t: Translator = ENGLISH, key: str = "") -> str:
    kind = _kind(default, key)
    if kind == "band":
        return t("settings.error.band")
    if kind == "cell":
        return t("settings.error.cell")
    return t("settings.error.whole_numbers" if kind == "list" else "settings.error.number")


def parse_settings_form(
    form: FormData, defaults: dict[str, Any], t: Translator = ENGLISH
) -> tuple[dict[str, Any], dict[str, str]]:
    """Every editable key's posted value, typed as its default; and errors per key.

    The whole form, as a browser posts it: a checkbox left unticked is not sent at all.
    """
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key in editable_keys(defaults):
        default = defaults[key]
        raw = form.get(key)
        if isinstance(default, bool):
            # A tick box sends "on"; the page's hidden "off" before it is sent either way.
            values[key] = raw is not None and str(raw).strip().lower() not in _FALSE
            continue
        text = raw.strip() if isinstance(raw, str) else ""
        try:
            values[key] = _typed(default, text, key)
        except ValueError:
            errors[key] = _error(default, t, key)
            values[key] = text
    return values, errors


_TRUE = frozenset({"on", "true", "1"})
_FALSE = frozenset({"", "off", "false", "0"})


def parse_settings_changes(
    form: FormData, defaults: dict[str, Any], t: Translator = ENGLISH
) -> tuple[dict[str, Any], dict[str, str]]:
    """Only the editable keys posted (the page's autosave), typed; and errors per key.

    A checkbox is sent either way, "on" when ticked and empty when not. Other posted names,
    fixed or unknown keys among them, are ignored.
    """
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key in editable_keys(defaults):
        raw = form.get(key)
        if not isinstance(raw, str):
            continue
        default = defaults[key]
        text = raw.strip()
        if isinstance(default, bool):
            if text.lower() in _TRUE | _FALSE:
                values[key] = text.lower() in _TRUE
            else:
                errors[key] = t("settings.error.on_off")
            continue
        try:
            values[key] = _typed(default, text, key)
        except ValueError:
            errors[key] = _error(default, t, key)
    return values, errors


def blamed_keys(message: str, keys: list[str]) -> list[str]:
    """The posted keys a settings error names, by key or by its age table or section."""
    return [k for k in keys if k in message or f"{k.rpartition('.')[0]}:" in message]


async def form_of(request: Request) -> FormData:
    return await request.form()
