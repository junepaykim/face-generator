"""Settings form vocabulary, presentation and validation, independent of routes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from starlette.datastructures import FormData

from facegen.age import WORDINGS
from facegen.settings import is_fixed

SECTION_NOTES = {
    "groups": "Thresholds re-classify the stored save at once. Sizes, candidates and the "
    "detail pass apply to pictures drawn from now on.",
    "eligibility": "Applies at the next sync: these decide, when a save is read, who is in "
    "men's football, at my club and in a continental competition.",
    "newgen": "Used only when a new career is first synced.",
    "staff": "Re-classifies at once. Staff are drawn at group A quality in club wear for their "
    "job (suits for the board, training wear for coaches, lab coats for doctors) in the "
    "club's colours; turning this off cancels their queued pictures and keeps the drawn ones.",
    "youth": "Re-classifies at once. My club's teams other than the first team are the "
    "reserves, under-23s and under-18s. A player moved to the first team rises to group A and "
    "is re-drawn at A quality.",
    "media": "Re-classifies at once. Journalists and press officers are read from the save at "
    "each sync and drawn once, at group B quality, as the gender the save gives them. Turning "
    "this off cancels their queued pictures and keeps the drawn ones.",
    "no_photo": "Re-classifies at once. Without a photo means FM27 ships none for them and no "
    "facepack in the graphics folder maps them; facegen's own pictures do not count. Both are "
    "read again at each sync. The group thresholds apply as for newgens, the academy rules do "
    "not. Turning this off cancels their queued pictures and keeps the drawn ones.",
    "age": "Re-classifies at once. Tracked players whose age wording changes are queued for a "
    "re-draw.",
    "reference": "Applies to re-draws from now on.",
    "my_club": "Applies to pictures drawn from now on.",
    "body": "Applies to character sheets sampled from now on; stored sheets keep their class.",
    "personality": "Applies to character sheets sampled from now on (a first picture or New "
    "face); stored sheets keep their expression.",
    "schedule": "The worker and the save watcher use it at once.",
}
# The thresholds first: they are what the page is mostly for (spec section 7).
SECTION_ORDER = (
    "groups", "staff", "youth", "media", "no_photo", "age", "schedule", "eligibility",
    "my_club", "body", "personality", "reference", "newgen",
)  # fmt: skip
SECTION_TITLES = {
    "groups.a": "Group A",
    "groups.b": "Group B",
    "groups.c": "Group C",
    "staff": "Staff at my club",
    "youth": "My club's youth teams",
    "media": "Journalists and press officers",
    "no_photo": "Pre-existing players without a photo",
    "eligibility": "Who counts",
    "newgen": "Newgens",
    "age.a_my_club": "Age wordings: group A at my club",
    "age.a_other_club": "Age wordings: group A elsewhere",
    "age.b": "Age wordings: group B",
    "age.c": "Age wordings: group C",
    "age.staff": "Age wordings: staff at my club",
    "age.media": "Age wordings: journalists and press officers drawn as women",
    "age.male": "Age wordings: staff and media drawn as men",
    "reference": "Reference picture",
    "my_club": "Players at my club",
    "body": "Height and build",
    "personality": "Personality",
    "schedule": "Worker",
}
LABELS = {
    "ca_above": "CA above",
    "pa_above": "PA above",
    "published_long_edge": "Published height (px; pictures are 3:4)",
    "candidates": "Candidates per picture",
    "detail_pass": "Detail pass",
    "my_academy": "Academy newgens stay in B and tracked after they leave",
    "league_ca_above": "In my league or a continental competition: CA above",
    "league_pa_above": "In my league or a continental competition: PA above",
    "any_ca_above": "Anywhere: CA above",
    "any_pa_above": "Anywhere: PA above",
    "mens_football_only": "Men's football only",
    "continental_competition_ids": "Continental competition IDs (editor database IDs)",
    "loaned_out_at_my_club": "Players I loaned out count as at my club",
    "fallback_min_unique_id": "Fallback newgen boundary (unique ID)",
    "enabled": "Use the existing picture as a reference on re-draws",
    "scale": "Reference strength (0 to 1)",
    "other_clubs": "Also for players outside my club",
    "banned_traits": "Banned traits (comma-separated)",
    "soften_skin": "Draw tan and dark skin lighter by (steps, 0 to 2)",
    "round_eyes": "Share of sharp eyes drawn round (0 to 1)",
    "short_below_cm": "Petite below (cm)",
    "tall_from_cm": "Tall from (cm)",
    "slim_strength_at_most": "Skinny at Strength at most",
    "muscular_strength_from": "Muscular from Strength",
    "weight": "Expression weight of each notable personality attribute (0 is off)",
    "poll_seconds": "Check the saves folder every (seconds)",
    "generate_while_fm_running": "Generate while FM27 is running",
    "auto_groups": "Groups drawn without a request (comma-separated)",
}
LABELS_IN = {
    "groups.c.league_ca_above": "In my league: CA above",
    "groups.c.league_pa_above": "In my league: PA above",
    "staff.enabled": "Draw my club's staff (coaches, physios, scouts, the board and you)",
    "no_photo.enabled": "Pick them by the newgen thresholds for groups A, B and C",
    "youth.enabled": "Draw my club's reserve, under-23 and under-18 players at group B "
    "(group A once they reach A's CA or PA, or join the first team)",
    "media.journalists": "Draw journalists from my club's country",
    "media.press_officers": "Draw my club's press officers",
}


@dataclass(frozen=True, slots=True)
class Field:
    key: str
    label: str
    kind: str  # bool, int, float, list
    value: Any
    default: Any
    changed: bool
    """Differs from the default."""

    @property
    def text(self) -> str:
        return input_text(self.value)

    @property
    def default_text(self) -> str:
        return value_text(self.default)

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


def value_text(value: Any) -> str:
    """A setting's value for people to read."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _kind(default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    return "list"


def _label(key: str) -> str:
    if key in LABELS_IN:
        return LABELS_IN[key]
    section, _, name = key.rpartition(".")
    if section.startswith("age."):
        return f"{name} from age ({WORDINGS[name].text})"
    return LABELS.get(name, name.replace("_", " "))


def describe_key(key: str) -> str:
    """A setting's section and label, short enough for one line: "Group A · CA above"."""
    section, _, name = key.rpartition(".")
    if section.startswith("age."):
        return f"{SECTION_TITLES.get(section, section)} · {name} from age"
    return f"{SECTION_TITLES.get(section, section)} · {_label(key)}"


@dataclass(frozen=True, slots=True)
class ChangeLine:
    key: str
    label: str
    before: str
    after: str


def change_lines(
    before: Mapping[str, Any], after: Mapping[str, Any], defaults: Mapping[str, Any]
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
                describe_key(key),
                value_text(default if old is None else old),
                value_text(default if new is None else new),
            )
        )
    return lines


def editable_keys(defaults: dict[str, Any]) -> list[str]:
    return [k for k in defaults if not is_fixed(k)]


def settings_sections(
    values: dict[str, Any], defaults: dict[str, Any]
) -> list[tuple[str, str, str, list[Field]]]:
    """(section key, title, note, fields) in the order of the defaults file."""
    sections: dict[str, list[Field]] = {}
    for key in editable_keys(defaults):
        section = key.rpartition(".")[0]
        sections.setdefault(section, []).append(
            Field(
                key=key,
                label=_label(key),
                kind=_kind(defaults[key]),
                value=values[key],
                default=defaults[key],
                changed=values[key] != defaults[key],
            )
        )

    def order(section: str) -> int:
        top = section.split(".")[0]
        return SECTION_ORDER.index(top) if top in SECTION_ORDER else len(SECTION_ORDER)

    return [
        (
            section,
            SECTION_TITLES.get(section, section),
            SECTION_NOTES.get(section.split(".")[0], ""),
            fields,
        )
        for section, fields in sorted(sections.items(), key=lambda kv: order(kv[0]))
    ]


def _typed(default: Any, text: str) -> Any:
    """Text from a field, typed as the default. ValueError if it does not fit."""
    kind = _kind(default)
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    items = [x for x in re.split(r"[,\s]+", text) if x]
    if default and isinstance(default[0], str):
        return [x.lower() for x in items]
    return [int(x) for x in items]


def _error(default: Any) -> str:
    return "needs whole numbers" if _kind(default) == "list" else "needs a number"


def parse_settings_form(
    form: FormData, defaults: dict[str, Any]
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
            values[key] = _typed(default, text)
        except ValueError:
            errors[key] = _error(default)
            values[key] = text
    return values, errors


_TRUE = frozenset({"on", "true", "1"})
_FALSE = frozenset({"", "off", "false", "0"})


def parse_settings_changes(
    form: FormData, defaults: dict[str, Any]
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
                errors[key] = "needs on or off"
            continue
        try:
            values[key] = _typed(default, text)
        except ValueError:
            errors[key] = _error(default)
    return values, errors


def blamed_keys(message: str, keys: list[str]) -> list[str]:
    """The posted keys a settings error names, by key or by its age table or section."""
    return [k for k in keys if k in message or f"{k.rpartition('.')[0]}:" in message]
