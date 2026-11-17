"""CUDA detection and pure recommendations for picture settings.

`detect` loads torch lazily and returns failures as reasons. The web process uses `probe`
(`facegen hardware --json` in a child), keeping torch out of the server.
Presets change candidates, detail size and VAE tiling; eligibility thresholds stay fixed.

The RTX 4080 measurements (docs/research.md section 12) put reference redraws at 7.70,
6.66 and 5.00 GiB for high, medium and low, with about 3 GB held by Windows. The defaults
therefore target 12 GB or more; medium targets 10–11 GB; lower cards use low. No CUDA
card means no changes. Round reported memory to whole GiB first: 11.99 GiB takes high,
and a 9 GB card takes low.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from facegen.platform import procs

PROBE_SECONDS = 180.0
"""How long `probe` waits for the child: importing torch takes several seconds, longer on a
cold disk."""

PRESET_KEYS = ("groups.a.candidates", "groups.a.detail_size", "render.vae_tiling")


@dataclass(frozen=True, slots=True)
class Hardware:
    name: str | None = None
    """The CUDA card's name; None when there is none."""
    total_bytes: int | None = None
    reason: str | None = None
    """Why there is no CUDA card to draw on."""
    torch: str | None = None
    cuda: str | None = None

    @property
    def has_gpu(self) -> bool:
        return self.name is not None and self.total_bytes is not None

    @property
    def vram_gib(self) -> float | None:
        return None if self.total_bytes is None else self.total_bytes / 2**30

    def describe(self) -> str:
        if self.has_gpu:
            assert self.vram_gib is not None
            return f"{self.name}, {self.vram_gib:.1f} GiB"
        return f"no CUDA graphics card ({self.reason or 'unknown reason'})"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Hardware:
        total = data.get("total_bytes")
        return cls(
            name=str(data["name"]) if data.get("name") else None,
            total_bytes=int(total) if total is not None else None,
            reason=str(data["reason"]) if data.get("reason") else None,
            torch=str(data["torch"]) if data.get("torch") else None,
            cuda=str(data["cuda"]) if data.get("cuda") else None,
        )


def detect() -> Hardware:
    """This PC's CUDA card, through torch. Never raises: a failure is the reason."""
    try:
        from facegen.render.gpu import prepare_environment

        prepare_environment()
        import torch
    except Exception as exc:  # noqa: BLE001 - no torch, or a broken one
        return Hardware(reason=f"torch could not be loaded: {type(exc).__name__}: {exc}")
    version = getattr(torch, "__version__", None)
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        if cuda is None:
            return Hardware(
                reason="this torch was built without CUDA (on Windows, PyPI's torch has none)",
                torch=version,
            )
        if not torch.cuda.is_available():
            return Hardware(
                reason="torch finds no CUDA device: no NVIDIA card, or its driver is missing",
                torch=version,
                cuda=cuda,
            )
        props = torch.cuda.get_device_properties(0)
        return Hardware(
            name=str(props.name), total_bytes=int(props.total_memory), torch=version, cuda=cuda
        )
    except Exception as exc:  # noqa: BLE001 - a driver error is an answer too
        return Hardware(
            reason=f"the card could not be read: {type(exc).__name__}: {exc}",
            torch=version,
            cuda=cuda,
        )


def command(program: Sequence[str] | None = None, config: Path | None = None) -> list[str]:
    """The command line of `facegen hardware --json`: this Python, unless `program` says."""
    base = list(program) if program else [sys.executable, "-m", "facegen.cli"]
    return [*base, *(["--config", str(config)] if config else []), "hardware", "--json"]


def probe(
    program: Sequence[str] | None = None,
    config: Path | None = None,
    *,
    timeout: float = PROBE_SECONDS,
) -> Hardware:
    """`detect` in a child process, so the caller never loads torch. Never raises."""
    try:
        done = procs.run_quiet(
            command(program, config),
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return Hardware(reason=f"the check did not answer within {timeout:.0f} s")
    except OSError as exc:
        return Hardware(reason=f"the check could not be started: {exc}")
    for line in reversed(done.stdout.splitlines()):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            try:
                return Hardware.from_json(data)
            except (KeyError, TypeError, ValueError):
                break
    last = (done.stderr.strip().splitlines() or [f"exit code {done.returncode}"])[-1]
    return Hardware(reason=f"the check failed: {last[:300]}")


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    """high, medium, low or none."""
    title: str
    changes: dict[str, Any]
    """Setting key to value; empty for "none"."""
    note: str


HIGH = Preset(
    "high",
    "12 GB or more",
    {"groups.a.candidates": 3, "groups.a.detail_size": "1344x1728", "render.vae_tiling": False},
    "The defaults: group A is the best of 3 candidates, with the detail pass at 1344x1728.",
)
MEDIUM = Preset(
    "medium",
    "10 to under 12 GB",
    {"groups.a.candidates": 3, "groups.a.detail_size": "1152x1472", "render.vae_tiling": False},
    "Group A is the best of 3, with a smaller detail pass (1152x1472), which needs less memory.",
)
LOW = Preset(
    "low",
    "under 10 GB",
    {"groups.a.candidates": 2, "groups.a.detail_size": "off", "render.vae_tiling": True},
    "Group A is the best of 2 with no detail pass, and the picture is decoded in tiles to save"
    " memory. Pictures take longer; who gets one does not change.",
)
NONE = Preset(
    "none",
    "no CUDA graphics card",
    {},
    "Nothing is changed: facegen draws on an NVIDIA card with CUDA, and none was found, so"
    " pictures cannot be drawn on this PC.",
)


def recommend(vram_gib: float | None) -> Preset:
    """The preset for a card with this much memory (GiB, as torch reports it); None: no card.
    Pure."""
    if vram_gib is None:
        return NONE
    whole = round(vram_gib)
    if whole >= 12:
        return HIGH
    if whole >= 10:
        return MEDIUM
    return LOW


@dataclass(frozen=True, slots=True)
class Change:
    key: str
    before: Any
    after: Any


def changes(preset: Preset, current: Mapping[str, Any]) -> list[Change]:
    """What applying the preset would change, from the current values. Pure."""
    return [
        Change(key, current.get(key), value)
        for key, value in preset.changes.items()
        if current.get(key) != value
    ]
