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
