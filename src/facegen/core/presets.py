"""Named sets of setting values for the Basic settings page, each applied as one change that
one Undo reverts (Phase 12g). Pure: the defaults and current values come in as mappings.

Reach moves every group's CA and PA thresholds together, so more or fewer players at other
clubs qualify; players at my club always get a picture. Quality is the graphics-card presets
(`setup.hardware`, measured on the RTX 4080, docs/research.md section 12).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REACH_KEYS = (
    "groups.a.ca_above", "groups.a.pa_above",
    "groups.b.league_ca_above", "groups.b.league_pa_above",
    "groups.b.any_ca_above", "groups.b.any_pa_above",
    "groups.c.league_ca_above", "groups.c.league_pa_above",
)  # fmt: skip
REACH_STEPS = {"fewer": 10, "default": 0, "more": -10}
"""How far each threshold moves from its default ("above" is strict, so higher is fewer)."""
THRESHOLD_RANGE = (0, 200)

QUALITY: dict[str, dict[str, Any]] = {
    "high": {
        "groups.a.candidates": 3, "groups.a.detail_size": "1344x1728", "render.vae_tiling": False,
    },
    "medium": {
        "groups.a.candidates": 3, "groups.a.detail_size": "1152x1472", "render.vae_tiling": False,
    },
    "low": {
        "groups.a.candidates": 2, "groups.a.detail_size": "off", "render.vae_tiling": True,
    },
}  # fmt: skip
"""Named like the card presets: high for 12 GB or more, medium for 10 to 12, low below."""

KINDS = ("reach", "quality")


def options(kind: str, defaults: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each preset of a kind by name, as the setting values it sets. KeyError: no such kind."""
    if kind == "reach":
        low, high = THRESHOLD_RANGE
        return {
            name: {k: min(high, max(low, int(defaults[k]) + step)) for k in REACH_KEYS}
            for name, step in REACH_STEPS.items()
        }
    if kind == "quality":
        return {name: dict(values) for name, values in QUALITY.items()}
    raise KeyError(kind)


def values_of(kind: str, name: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """The setting values one preset sets. KeyError: no such kind or preset."""
    return options(kind, defaults)[name]


def current(kind: str, values: Mapping[str, Any], defaults: Mapping[str, Any]) -> str | None:
    """The preset the current values match, or None when they were set by hand."""
    for name, wanted in options(kind, defaults).items():
        if all(values.get(k) == v for k, v in wanted.items()):
            return name
    return None


def label(kind: str, name: str) -> str:
    """How the settings history names the change: "preset.reach.more"."""
    return f"preset.{kind}.{name}"


def parse_label(text: str) -> tuple[str, str] | None:
    """(kind, name) from a history label, or None for any other label."""
    parts = text.split(".")
    if len(parts) == 3 and parts[0] == "preset" and parts[1] in KINDS:
        return parts[1], parts[2]
    return None
