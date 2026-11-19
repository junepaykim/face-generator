"""Settings form vocabulary, presentation and validation, independent of routes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request
from starlette.datastructures import FormData

from facegen.core.settings_values import DETAIL_SIZES, LANGUAGE_CHOICES, THEME_CHOICES
from facegen.i18n import ENGLISH, Translator
from facegen.persistence.settings import SKIP_WORDING, is_age_wording, is_fixed
from facegen.web.messages import band_name

# The thresholds first: they are what the page is mostly for (spec section 7). Each section's
# title is the catalog's "settings.title.<section>", its note "settings.note.<top section>"
# (most sections have none: Phase 12 keeps the page short), its ⓘ tip "settings.tip.<top>".
SECTION_ORDER = (
    "groups", "staff", "youth", "media", "no_photo", "age", "schedule", "display",
    "notifications", "eligibility", "my_club", "body", "personality", "reference", "render", "updates",
    "newgen",
)  # fmt: skip
LABELLED_IN_FULL = frozenset(
    {
        "groups.b.league_ca_above",
        "groups.b.league_pa_above",
        "groups.b.any_ca_above",
        "groups.b.any_pa_above",
        "groups.c.league_ca_above",
        "groups.c.league_pa_above",
        "staff.enabled",
        "no_photo.enabled",
        "youth.enabled",
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
}
"""Text settings with a fixed set of values, by the key's last part, shown as a drop-down list;
each value's label is the catalog's "settings.choice.<name>.<value>"."""

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
"""One or two words of each age wording's prompt, shown after its name (core.age.WORDINGS)."""
ADULT_WORDED_TABLES = frozenset({"age.staff", "age.media"})
"""Staff and media prompts use adult professional wording in every band (decisions.md,
2026-11-14), so their tables show the band names without the players' prompt words."""


def choices_of(key: str, t: Translator = ENGLISH) -> tuple[tuple[str, str], ...] | None:
    name = key.rpartition(".")[2]
    values = CHOICES.get(name)
    if values is None:
        return None
    return tuple((value, t(f"settings.choice.{name}.{value}")) for value in values)


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

    @property
    def text(self) -> str:
        return input_text(self.value)

    @property
    def default_input(self) -> str:
        """The default as the field holds it: what "use default" puts back."""
        if isinstance(self.default, bool):
            return "on" if self.default else ""
        if isinstance(self.default, list):
            return ", ".join(str(v) for v in self.default)
        return str(self.default)


def input_text(value: Any) -> str:
    """A setting's value as its text field holds it."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
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
    name = key.rpartition(".")[2]
    if name in CHOICES and value in CHOICES[name]:
        return t(f"settings.choice.{name}.{value}")
    return str(value)


def _kind(default: Any, key: str = "") -> str:
    if is_age_wording(key):
        return "band"
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
    section, _, name = key.rpartition(".")
    if section in ADULT_WORDED_TABLES:
        return "settings.age_label_plain", {"band": band_name(name, t)}
    if section.startswith("age."):
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
    if section.startswith("age."):
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


def settings_sections(
    values: dict[str, Any], defaults: dict[str, Any], t: Translator = ENGLISH
) -> list[tuple[str, str, str, list[Field]]]:
    """(section key, title, note, fields) in the order of the defaults file."""
    sections: dict[str, list[Field]] = {}
    for key in editable_keys(defaults):
        section = key.rpartition(".")[0]
        label_key, label_values = label_parts(key, t)
        sections.setdefault(section, []).append(
            Field(
                key=key,
                label=_label(key, t),
                kind=_kind(defaults[key], key),
                value=values[key],
                default=defaults[key],
                changed=values[key] != defaults[key],
                choices=choices_of(key, t) or (),
                default_text=value_text(defaults[key], t, key),
                label_key=label_key,
                label_values=label_values,
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


def _typed(default: Any, text: str, key: str = "") -> Any:
    """Text from a field, typed as the default. ValueError if it does not fit."""
    kind = _kind(default, key)
    if kind == "band":  # a whole number, or "-" (or nothing) for a wording left out
        return SKIP_WORDING if text in ("", SKIP_WORDING) else int(text)
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
            values[key] = raw is not None
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
