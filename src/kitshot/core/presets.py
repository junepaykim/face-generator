"""Named sets of setting values for the Basic settings page, each applied as one change that
one Undo reverts (Phase 12g). Pure: the defaults and current values come in as mappings.

Reach moves every cutoff together, so more or fewer players at other clubs qualify; players at
my club always get a picture (Phase 16): "fewer" halves every share and raises every number by
10, "more" takes one and a half times every share (at most 100%) and lowers every number by 10.
A cell switched off ("-") stays off. Quality is the graphics-card presets (`setup.hardware`,
measured on the RTX 4080, docs/research.md section 12, and again for every image model in
section 20).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kitshot.core.cutoffs import LEAGUE_CELLS, OFF, OTHER_CELLS

SHARE_KEYS = tuple(f"cutoffs.share.{c}" for c in (*LEAGUE_CELLS, *OTHER_CELLS))
NUMBER_KEYS = tuple(
    f"cutoffs.{kind}.{c}" for kind in ("manual", "everywhere") for c in LEAGUE_CELLS
)
REACH_KEYS = SHARE_KEYS + NUMBER_KEYS
REACH_STEPS: dict[str, tuple[float, int]] = {
    "fewer": (0.5, 10),
    "default": (1.0, 0),
    "more": (1.5, -10),
}
"""Each reach's factor for the shares and step for the numbers (higher numbers pick fewer)."""
THRESHOLD_RANGE = (1, 200)
SHARE_RANGE = (0.1, 100.0)


def _share(value: Any, factor: float) -> Any:
    if value == OFF:
        return OFF
    low, high = SHARE_RANGE
    return min(high, max(low, round(float(value) * factor, 1)))


def _number(value: Any, step: int) -> Any:
    if value == OFF:
        return OFF
    low, high = THRESHOLD_RANGE
    return min(high, max(low, int(value) + step))


QUALITY: dict[str, dict[str, Any]] = {
    "high": {
        "groups.a.candidates": 3, "groups.a.detail_size": "x1.5", "render.vae_tiling": False,
    },
    "medium": {
        "groups.a.candidates": 3, "groups.a.detail_size": "smaller", "render.vae_tiling": False,
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
        return {
            name: {k: _share(defaults[k], factor) for k in SHARE_KEYS}
            | {k: _number(defaults[k], step) for k in NUMBER_KEYS}
            for name, (factor, step) in REACH_STEPS.items()
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
