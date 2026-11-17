"""The graphics card, and the picture settings that suit it (Phase 10d).

- `detect()` asks torch for the CUDA card: its name and total memory, or why there is none
  (torch without CUDA, no NVIDIA driver, no card). torch is imported inside the function and
  any failure is an answer, not an error. The web process never imports torch: it runs
  `facegen hardware --json` as a short child process (`probe`), as the worker runs in one
  (Phase 7b), and keeps the answer for the process's life.
- `recommend(vram_gib)` is pure: the preset for a card, as setting keys. Only picture
  settings: group A's candidates, the detail pass size and VAE tiling. The group thresholds
  are never touched: they decide who gets a picture, not whether the card copes, and changing
  them would re-classify the save (user, 2026-11-13). A small card is slower, not different in
  who is drawn.

| Card memory            | Preset                                                     |
|------------------------|------------------------------------------------------------|
| 12 GB or more          | the defaults: A best of 3, detail pass 1344x1728           |
| 10 to under 12 GB      | A best of 3, detail pass 1152x1472                         |
| under 10 GB            | A best of 2, no detail pass, `render.vae_tiling` on        |
| no CUDA card           | nothing changed: pictures cannot be drawn on this PC       |

The boundaries come from the measurement on the RTX 4080 (docs/research.md section 12; the
user, 2026-11-16): a group A re-draw with a reference picture peaks at 7.70 GiB with the
defaults, 6.66 with the middle preset and 5.00 with the low one, and Windows keeps about 3 GB
of the card. So the defaults fit a 12 GB card, the middle preset a 10 or 11 GB one.

Cards are sold by whole gigabytes and torch reports a little less (a 16 GB card is 15.99 GiB),
so the memory is rounded to whole GiB first: a 12 GB card that reports 11.99 GiB takes the
defaults, a 10 GB card the middle preset, a 9 GB one the low preset.
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
