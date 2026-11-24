"""`kitshot worker` with a fake GPU, for the Phase 7 exit test: `kitshot serve` starts this
in place of the real worker process (cli.Gpu.program).

    python gpu_child.py MARKER [--slow] [--config X] worker [--generate-now]

The renderer raises a fatal CUDA error on the first picture of its third job, and freeing
it fails too, as on 2026-11-13. That happens once: MARKER records it, so the next process
draws normally.

With `--slow` the renderer instead writes "drawing" to MARKER and takes a minute over each
picture, for Quit while a picture is being drawn.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeRenderer, FakeTagger

from kitshot.cli import Gpu, main
from kitshot.core.prompt import Prompt

FAULT = "CUDA error: an illegal memory access was encountered"


class FaultOnThirdJob(FakeRenderer):
    def __init__(self, marker: Path) -> None:
        super().__init__()
        self.marker = marker
        self.jobs: list[str] = []
        self.faulted = False

    def render(self, prompt: Prompt, seed: int, reference: Any = None, **kw: Any):  # type: ignore[no-untyped-def]
        if prompt.positive not in self.jobs:
            self.jobs.append(prompt.positive)
        if len(self.jobs) == 3 and not self.marker.exists():
            self.marker.write_text("faulted", encoding="utf-8")
            self.faulted = True
            raise RuntimeError(FAULT)
        return super().render(prompt, seed, reference, **kw)

    def close(self) -> None:
        if self.faulted:
            raise RuntimeError(FAULT)


class Slow(FakeRenderer):
    def __init__(self, marker: Path) -> None:
        super().__init__()
        self.marker = marker

    def render(self, prompt: Prompt, seed: int, reference: Any = None, **kw: Any):  # type: ignore[no-untyped-def]
        self.marker.write_text("drawing", encoding="utf-8")
        time.sleep(60)
        return super().render(prompt, seed, reference, **kw)


if __name__ == "__main__":
    marker = Path(sys.argv[1])
    args = sys.argv[2:]
    slow = args[:1] == ["--slow"]
    if slow:
        args = args[1:]
    gpu = Gpu(
        renderer=lambda _model: Slow(marker) if slow else FaultOnThirdJob(marker),
        tagger=FakeTagger,
        fm_check=lambda: False,
    )
    raise SystemExit(main(args, gpu=gpu))
