"""Environment for the GPU code. Call `prepare_environment` before torch is imported.

- Models load from the local Hugging Face cache only; nothing reaches the network while
  rendering or tagging (AGENTS.md hard rule 6).
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
