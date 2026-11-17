"""Environment for the GPU code. Call `prepare_environment` before torch is imported.

- Models load from the local Hugging Face cache only; nothing reaches the network while
  rendering or tagging (AGENTS.md hard rule 6).
- `is_fatal` tells an error that ends the process's use of the GPU (a dead CUDA context)
  from one it can get past, such as running out of memory.
- `expandable_segments` keeps reserved-but-unused GPU memory low. Windows holds about 3 GB
  of the card's 16; without it the driver spills into system RAM and rendering slows down
  (docs/research.md section 5).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def prepare_environment() -> None:
    for key, value in _ENV.items():
        os.environ.setdefault(key, value)
    if "torch" in sys.modules and "expandable_segments" not in os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    ):
        print("warning: torch was imported before the GPU environment was set", file=sys.stderr)


_FATAL_TYPES = frozenset({"AcceleratorError"})
_FATAL_TEXT = (
    "cuda error",
    "cudaerror",
    "device-side assert",
    "unspecified launch failure",
    "illegal memory access",
)


def is_fatal(exc: BaseException) -> bool:
    """Whether the error left the process's CUDA context dead.

    After such an error every later CUDA call fails too, freeing memory included: only a
    new process draws again. Running out of memory is not one: the context lives and the
    job can be tried again. Told from the type's name and the message, so no torch is
    needed; an error raised from another (`raise ... from`) counts if its cause does.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).lower()
        if "out of memory" not in text and (
            type(current).__name__ in _FATAL_TYPES or any(mark in text for mark in _FATAL_TEXT)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def describe() -> str:
    """torch's version and the card, for the log; never raises and never imports torch."""
    torch = sys.modules.get("torch")
    if torch is None:
        return "torch not loaded"
    try:
        return (
            f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}"
        )
    except Exception as exc:  # noqa: BLE001 - a dead context fails here too
        return (
            f"torch {getattr(torch, '__version__', '?')} (card not readable: {type(exc).__name__})"
        )


@contextmanager
def on_gpu(*modules: Any, device: str = "cuda", enabled: bool = True) -> Iterator[None]:
    """Move models that wait in system memory onto the GPU for the block, then back.

    For the models a job uses only briefly (the text encoders, the tagger, the IP-Adapter
    image encoder), so they hold no VRAM while the UNet runs: with the UNet's INT8 weights
    this keeps a group A job's peak reserved memory under 8 GiB, for about 1 s of transfers
    per job (user, 2026-11-11; docs/research.md, run `inference`). With `enabled` off the
    models are left where they are.

    Afterwards PyTorch's cache is handed back to the driver, so between a job's steps and
    between jobs the process holds about what it uses, not its peak. It hardly lowers the
    peak itself (the detail pass, 7.76 to 7.70 GiB reserved).
    """
    if not enabled:
        yield
        return
    for m in modules:
        m.to(device)
    try:
        yield
    finally:
        for m in modules:
            m.to("cpu")
        import torch  # loaded already: the models are torch modules

        torch.cuda.empty_cache()
