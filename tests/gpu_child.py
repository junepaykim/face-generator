"""`facegen worker` with a fake GPU, for the Phase 7 exit test: `facegen serve` starts this
in place of the real worker process (cli.Gpu.program).

    python gpu_child.py MARKER [--config X] worker [--generate-now]

The renderer raises a fatal CUDA error on the first picture of its third job, and freeing
it fails too, as on 2026-11-13. That happens once: MARKER records it, so the next process
draws normally.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeRenderer, FakeTagger

from facegen.cli import Gpu, main
from facegen.prompt import Prompt

FAULT = "CUDA error: an illegal memory access was encountered"


class FaultOnThirdJob(FakeRenderer):
    def __init__(self, marker: Path) -> None:
        super().__init__()
        self.marker = marker
        self.jobs: list[str] = []
        self.faulted = False

    def render(self, prompt: Prompt, seed: int, reference: Any = None):  # type: ignore[no-untyped-def]
        if prompt.positive not in self.jobs:
            self.jobs.append(prompt.positive)
        if len(self.jobs) == 3 and not self.marker.exists():
            self.marker.write_text("faulted", encoding="utf-8")
            self.faulted = True
            raise RuntimeError(FAULT)
        return super().render(prompt, seed, reference)

    def close(self) -> None:
        if self.faulted:
            raise RuntimeError(FAULT)


if __name__ == "__main__":
    marker = Path(sys.argv[1])
    gpu = Gpu(renderer=lambda: FaultOnThirdJob(marker), tagger=FakeTagger, fm_check=lambda: False)
    raise SystemExit(main(sys.argv[2:], gpu=gpu))
