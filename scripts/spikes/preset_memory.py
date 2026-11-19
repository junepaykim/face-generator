"""Phase 10d spike: does each GPU preset fit the card it is meant for?

"Measure before promising" (implementation plan, Phase 10d). On the RTX 4080, PyTorch's
allocator is capped at 8 and then 12 GiB (`torch.cuda.set_per_process_memory_fraction`), and a
group A job is drawn with each preset (hardware.py): the candidates and their check, then the
detail pass at the preset's size and its check, as the worker draws them. For each it records
the peak reserved and allocated memory, the seconds, and whether it ran out of memory.

The cap is PyTorch's allocator only: the CUDA context (a few hundred MB) and other programs
are on top of it, so a real 8 GB card has less room than an 8 GiB cap. FM27 must be closed.

Each cap runs in a process of its own (a fresh allocator and context). Made-up character, no
player data; the pictures stay in memory. Results: data/phase10/preset-memory.json (not in git).

With `--reference` each job is a re-draw with a reference picture, as my club's tracked
players get (IP-Adapter plus-face at 0.1 on every candidate and the detail pass): the most
memory a job takes. Its results go to data/phase10/preset-memory-reference.json.

    uv run python scripts/spikes/preset_memory.py            # caps 8 and 12 GiB, all presets
    uv run python scripts/spikes/preset_memory.py --reference
    uv run python scripts/spikes/preset_memory.py --cap 8 --preset low
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "phase10" / "preset-memory.json"


def one_cap(
    cap_gib: float, presets: list[str], with_reference: bool = False
) -> list[dict[str, Any]]:
    from facegen.render.gpu import prepare_environment

    prepare_environment()
    import torch

    from facegen.core.prompt import build_prompt
    from facegen.core.traits import sample_sheet
    from facegen.datasource.kits import Kit
    from facegen.persistence.settings import DETAIL_SIZES
    from facegen.pipeline.worker import AnimagineRenderer, Reference
    from facegen.render import quality_checks as qc
    from facegen.render.picture import crop_to_aspect, face_box
    from facegen.render.tagger import Eva02Tagger
    from facegen.setup import hardware

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib * 2**30 / total), 0)
    t0 = time.perf_counter()
    renderer = AnimagineRenderer()
    tagger = Eva02Tagger()
    load = time.perf_counter() - t0
    sheet = sample_sheet("preset-memory", 1, 1)
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    expect = qc.expectation(sheet, kit)
    by_name = {p.name: p for p in (hardware.HIGH, hardware.MEDIUM, hardware.LOW)}
    results = []
    for k, name in enumerate(presets):
        preset = by_name[name]
        candidates = preset.changes["groups.a.candidates"]
        size = DETAIL_SIZES[preset.changes["groups.a.detail_size"]]
        renderer.set_vae_tiling(preset.changes["render.vae_tiling"])
        prompt = build_prompt(sheet, ("W03", "W05", "W07")[k % 3], kit, "crossed")
        reference = None
        if with_reference:  # the existing picture, drawn before the measured job
            existing = crop_to_aspect(renderer.render(prompt, sheet.base_seed + 100 + k))
            box = face_box(existing)
            reference = Reference(f"preset-memory-{k}", existing.crop(box), 0.1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row: dict[str, Any] = {
            "cap_gib": cap_gib,
            "preset": name,
            "candidates": candidates,
            "detail_size": list(size) if size else None,
            "vae_tiling": preset.changes["render.vae_tiling"],
            "reference": with_reference,
            "model_load_s": round(load, 1),
        }
        t1 = time.perf_counter()
        try:
            renderer.prepare([prompt])
            images = [
                renderer.render(prompt, sheet.base_seed + i, reference) for i in range(candidates)
            ]
            row["render_s"] = round(time.perf_counter() - t1, 2)
            results_ = tagger(images)
            passed = [
                qc.check(t, expect, qc.head_top(im)).passed
                for t, im in zip(results_, images, strict=True)
            ]
            row["passed"] = sum(passed)
            if size is not None:
                t2 = time.perf_counter()
                detailed = renderer.detail(images[0], prompt, sheet.base_seed, reference, size=size)
                tagger([detailed])
                row["detail_s"] = round(time.perf_counter() - t2, 2)
            row["ok"] = True
        except torch.OutOfMemoryError as exc:
            row["ok"] = False
            row["error"] = f"out of memory: {str(exc).splitlines()[0][:200]}"
        row["job_s"] = round(time.perf_counter() - t1, 2)
        row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        results.append(row)
        print(json.dumps(row), flush=True)
    renderer.close()
    tagger.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap", type=float, action="append", help="GiB (default: 8 and 12)")
    parser.add_argument(
        "--preset", action="append", choices=("high", "medium", "low"), help="default: all three"
    )
    parser.add_argument(
        "--reference", action="store_true", help="each job a re-draw with a reference picture"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    caps = args.cap or [8.0, 12.0]
    presets = args.preset or ["high", "medium", "low"]
    if args.child:
        one_cap(caps[0], presets, args.reference)
        return 0
    from facegen.pipeline.worker import fm_running
    from facegen.platform import procs

    if fm_running() is not False:
        print("FM27 is running (or it cannot be told): close it first.", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]] = []
    for cap in caps:
        command = [sys.executable, __file__, "--child", "--cap", str(cap)]
        if args.reference:
            command.append("--reference")
        for p in presets:
            command += ["--preset", p]
        done = procs.run_quiet(command, timeout=3600, capture_output=True, text=True)
        for line in done.stdout.splitlines():
            if line.startswith("{"):
                rows.append(json.loads(line))
        if done.returncode != 0:
            print(done.stderr[-2000:], file=sys.stderr)
    out = OUT.with_name("preset-memory-reference.json") if args.reference else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{'cap':>5} {'preset':<7} {'ok':<4} {'reserved':>9} {'allocated':>10} {'job s':>7}")
    for r in rows:
        print(
            f"{r['cap_gib']:>5.0f} {r['preset']:<7} {'yes' if r['ok'] else 'OOM':<4}"
            f" {r['peak_reserved_gib']:>9.2f} {r['peak_allocated_gib']:>10.2f} {r['job_s']:>7.1f}"
        )
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
