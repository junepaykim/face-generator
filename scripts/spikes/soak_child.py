"""`kitshot worker` with the real renderer, measured: the child that soak_test.py starts.

    python soak_child.py SANDBOX FAULT_ON_JOB [--config X] worker [--generate-now]

Each render and detail pass appends a line to SANDBOX/calls.jsonl: process, kind, whether a
reference picture was given, seconds, allocated and peak reserved GPU memory. With
FAULT_ON_JOB > 0, the first picture of that job (counted by distinct prompts in this
process) first causes a real, sticky CUDA error, once: an out-of-range index on the GPU,
which ends in "CUDA error: device-side assert triggered" like any dead context.
SANDBOX/faulted records that it happened, so the next worker process draws normally.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kitshot.cli import Gpu, _default_renderer, _default_tagger, main


def real_fault() -> None:
    import torch

    x = torch.zeros(4, device="cuda")
    torch.index_select(x, 0, torch.tensor([1 << 20], device="cuda"))
    torch.cuda.synchronize()


class Measured:
    """The real renderer, with each call timed and its memory written down."""

    def __init__(self, inner: Any, sandbox: Path, fault_on_job: int) -> None:
        self._inner = inner
        self._log = sandbox / "calls.jsonl"
        self._marker = sandbox / "faulted"
        self._fault_on_job = fault_on_job
        self._prompts: list[str] = []
        self._calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def render(self, prompt: Any, seed: int, reference: Any = None, **kw: Any) -> Any:
        if prompt.positive not in self._prompts:
            self._prompts.append(prompt.positive)
            n = len(self._prompts)
            if n == self._fault_on_job and not self._marker.exists():
                self._marker.write_text(f"job {n}, process {os.getpid()}\n", encoding="utf-8")
                self._write({"kind": "fault", "job": n})
                real_fault()
        return self._timed(
            "render", reference, lambda: self._inner.render(prompt, seed, reference, **kw)
        )

    def detail(
        self, image: Any, prompt: Any, seed: int, reference: Any = None, *, size: Any
    ) -> Any:
        return self._timed(
            "detail",
            reference,
            lambda: self._inner.detail(image, prompt, seed, reference, size=size),
        )

    def _timed(self, kind: str, reference: Any, call: Callable[[], Any]) -> Any:
        import torch

        self._calls += 1
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ok = False
        try:
            out = call()
            ok = True
            return out
        finally:
            record: dict[str, Any] = {
                "kind": kind,
                "call": self._calls,
                "job": len(self._prompts),
                "reference": reference is not None,
                "seconds": round(time.perf_counter() - t0, 3),
                "ok": ok,
            }
            try:
                record["allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
                record["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
            except Exception as exc:  # noqa: BLE001 - a dead context fails here too
                record["memory_error"] = type(exc).__name__
            self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        record = {"t": round(time.time(), 3), "pid": os.getpid(), **record}
        with self._log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    sandbox, fault_on_job = Path(sys.argv[1]), int(sys.argv[2])
    gpu = Gpu(
        renderer=lambda: Measured(_default_renderer(), sandbox, fault_on_job),
        tagger=_default_tagger,
    )
    raise SystemExit(main(sys.argv[3:], gpu=gpu))
