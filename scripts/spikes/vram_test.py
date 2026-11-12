"""Side test: peak VRAM of group A jobs on the production renderer and tagger.

Asked for by the user on 2026-11-11: does keeping the text encoders and the tagger in system
memory while the UNet runs bring the peak under 8.0 GiB? Production now does that by default
(`offload_text_encoders` on the renderer, `offload` on the tagger); "resident" turns both off.
Two made-up players (IDs 920001 and
920002, not anyone in a save) are drawn as the worker draws group A: the prompt encoded, three
candidates, the tagger on the three, the detail pass, the tagger on it. The second player is a
re-draw with the first one's picture as the reference (IP-Adapter at 0.1), the heaviest path.

    uv run python scripts/spikes/vram_test.py resident   # everything stays on the card
    uv run python scripts/spikes/vram_test.py offload    # text encoders and tagger wait in RAM
    uv run python scripts/spikes/vram_test.py offload --fp16-weights   # INT8 weights off

Peaks are PyTorch's allocated and reserved memory, per job and phase, after a warm-up job.
Reserved is what the process holds from the driver, so what other programs lose; the CUDA
context (about 0.4 GiB) comes on top. Results go to data/bakeoff/vram/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from facegen import qc
from facegen.gpu import prepare_environment
from facegen.kits import Kit
from facegen.picture import face_box
from facegen.prompt import build_prompt
from facegen.traits import pick_pose, picture_seed, sample_sheet

prepare_environment()

import torch  # after prepare_environment

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "bakeoff" / "vram"
GIB = 2**30


def main(args: list[str]) -> None:
    from facegen.worker import AnimagineRenderer, Reference

    mode = args[0] if args else ""
    if mode not in ("resident", "offload") or not set(args[1:]) <= {"--fp16-weights"}:
        sys.exit(__doc__)
    offload = mode == "offload"
    int8 = "--fp16-weights" not in args
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    renderer = AnimagineRenderer(int8_weights=int8, offload_text_encoders=offload)
    tagger = qc.Eva02Tagger(offload=offload)
    load_s = time.perf_counter() - t0

    phases: dict[str, dict[str, float]] = {}

    def phase(name: str, fn: Any, *args: Any) -> Any:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*args)
        torch.cuda.synchronize()
        p = phases.setdefault(name, {"allocated": 0.0, "reserved": 0.0})
        p["allocated"] = max(p["allocated"], torch.cuda.max_memory_allocated() / GIB)
        p["reserved"] = max(p["reserved"], torch.cuda.max_memory_reserved() / GIB)
        return out

    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")

    def job(label: str, uid: int, reference: Any) -> tuple[Any, float]:
        t = time.perf_counter()
        sheet = sample_sheet("vram-test", uid, 1)
        prompt = build_prompt(sheet, "W07", kit, pick_pose(sheet.base_seed, 1))
        phase(f"{label}: encode prompt", renderer.prompt_chunks, prompt)
        if reference is not None:
            # What the first render of a re-draw does first: load the adapter (once) and
            # embed the reference with the image encoder, which then goes back to RAM.
            phase(f"{label}: embed reference", renderer._conditioning, reference)
        images = [
            phase(f"{label}: render", renderer.render, prompt, picture_seed(sheet.base_seed, 1, 0, k), reference)
            for k in range(3)
        ]  # fmt: skip
        phase(f"{label}: tag candidates", tagger, images)
        detailed = phase(
            f"{label}: detail pass", renderer.detail, images[0], prompt, sheet.base_seed, reference
        )
        phase(f"{label}: tag detail", tagger, [detailed])
        return detailed, time.perf_counter() - t

    job("warm-up", 920000, None)
    phases.clear()
    resident_gib = torch.cuda.memory_allocated() / GIB  # held between jobs
    first, s1 = job("plain", 920001, None)
    crop = first.crop(face_box(first))
    ref = Reference(key="vram-test", image=crop, scale=0.1)
    _, s2 = job("reference", 920002, ref)  # as a re-draw at my club
    after = {
        "allocated": round(torch.cuda.memory_allocated() / GIB, 2),
        "reserved": round(torch.cuda.memory_reserved() / GIB, 2),
    }

    def peak(prefix: str, kind: str) -> float:
        return round(max(p[kind] for k, p in phases.items() if k.startswith(prefix)), 2)

    result = {
        "mode": mode,
        "renderer": renderer.settings(),
        "load_seconds": round(load_s, 1),
        "resident_between_jobs_gib": round(resident_gib, 2),
        "held_after_jobs_gib": after,  # adapter loaded; reserved is what the driver sees
        "phases": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in phases.items()},
        "peak_allocated_gib": {"plain": peak("plain", "allocated"), "reference": peak("reference", "allocated")},
        "peak_reserved_gib": {"plain": peak("plain", "reserved"), "reference": peak("reference", "reserved")},
        "job_seconds": {"plain": round(s1, 1), "reference": round(s2, 1)},
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }  # fmt: skip
    name = f"{'int8' if int8 else 'fp16'}-{mode}"
    (RUN / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(name, json.dumps({k: result[k] for k in list(result)[2:] if k != "phases"}))
    for k, v in result["phases"].items():
        print(f"  {k:28s} allocated {v['allocated']:5.2f}  reserved {v['reserved']:5.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
