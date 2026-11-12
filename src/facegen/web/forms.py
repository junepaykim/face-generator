"""Settings form vocabulary, presentation and validation, independent of routes."""

from __future__ import annotations

import re
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
    "groups", "age", "schedule", "eligibility", "my_club", "body", "personality", "reference",
    "newgen",
)  # fmt: skip
SECTION_TITLES = {
    "groups.a": "Group A",
    "groups.b": "Group B",
    "groups.c": "Group C",
    "eligibility": "Who counts",
    "newgen": "Newgens",
    "age.a_my_club": "Age wordings: group A at my club",
    "age.a_other_club": "Age wordings: group A elsewhere",
    "age.b": "Age wordings: group B",
    "age.c": "Age wordings: group C",
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
        if isinstance(self.value, list):
            return ", ".join(str(v) for v in self.value)
        return str(self.value)

    @property
    def default_text(self) -> str:
        if isinstance(self.default, list):
            return ", ".join(str(v) for v in self.default) or "none"
        if isinstance(self.default, bool):
            return "on" if self.default else "off"
        return str(self.default)


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


def parse_settings_form(
    form: FormData, defaults: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Every editable key's posted value, typed as its default; and errors per key."""
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key in editable_keys(defaults):
        default = defaults[key]
        raw = form.get(key)
        text = raw.strip() if isinstance(raw, str) else ""
        kind = _kind(default)
        try:
            if kind == "bool":
                values[key] = raw is not None
            elif kind == "int":
                values[key] = int(text)
            elif kind == "float":
                values[key] = float(text)
            else:
                items = [x for x in re.split(r"[,\s]+", text) if x]
                if default and isinstance(default[0], str):
                    values[key] = [x.lower() for x in items]
                else:
                    values[key] = [int(x) for x in items]
        except ValueError:
            errors[key] = "needs a number" if kind != "list" else "needs whole numbers"
            values[key] = text
    return values, errors
