"""Phase 10d / 15i spike: does each GPU preset fit the card it is meant for, on every model?

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

Phase 15i: `--model` and `--gender` pick the image model (`style.model`) and who is drawn
(`style.gender`), each in a process of its own, with the prompt and check the worker uses
(core/style.py). A model that takes no reference (Z-Image-Turbo) is skipped with
`--reference`. Results then go to `--out` (default data/phase15/preset-memory.json), one row
per cap, model, gender, reference and preset. `--allow-fm` measures with FM27 open: PyTorch's
peaks are its own, but the seconds are then not representative.

    uv run python scripts/spikes/preset_memory.py            # caps 8 and 12 GiB, all presets
    uv run python scripts/spikes/preset_memory.py --reference
    uv run python scripts/spikes/preset_memory.py --cap 8 --preset low
    uv run python scripts/spikes/preset_memory.py --cap 8 --model realvis --gender man --reference
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
OUT_15 = ROOT / "data" / "phase15" / "preset-memory.json"
AGES = (19, 24, 29)
"""A man's age in years for each preset's job, beside the women's wordings W03, W05, W07."""


def one_cap(
    cap_gib: float,
    presets: list[str],
    with_reference: bool = False,
    model: str = "animagine",
    gender: str = "woman",
) -> list[dict[str, Any]]:
    from kitshot.render.gpu import prepare_environment

    prepare_environment()
    import torch

    from kitshot.core.man_traits import sample_man_sheet
    from kitshot.core.settings_values import detail_size
    from kitshot.core.shoot import DEFAULT_BACKDROP
    from kitshot.core.style import portrait_prompt
    from kitshot.core.traits import sample_sheet
    from kitshot.datasource.kits import Kit
    from kitshot.pipeline.worker import Reference
    from kitshot.render import quality_checks as qc
    from kitshot.render.picture import crop_to_aspect, face_box
    from kitshot.render.tagger import Eva02Tagger
    from kitshot.runtime.dependencies import default_renderer
    from kitshot.setup import hardware

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib * 2**30 / total), 0)
    t0 = time.perf_counter()
    renderer = default_renderer(model)
    tagger = Eva02Tagger()
    load = time.perf_counter() - t0
    sheet = sample_sheet("preset-memory", 1, 1)
    man = sample_man_sheet("preset-memory", 1, 1, nation=765) if gender == "man" else None
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    by_name = {p.name: p for p in (hardware.HIGH, hardware.MEDIUM, hardware.LOW)}
    results = []
    for k, name in enumerate(presets):
        preset = by_name[name]
        candidates = preset.changes["groups.a.candidates"]
        size = detail_size(preset.changes["groups.a.detail_size"], "3:4")
        renderer.set_vae_tiling(preset.changes["render.vae_tiling"])
        age = AGES[k % 3]
        prompt = portrait_prompt(
            model,
            gender,
            sheet=sheet,
            man=man,
            wording=("W03", "W05", "W07")[k % 3],
            age=age,
            wear=kit,
            pose="crossed",
            backdrop=DEFAULT_BACKDROP,
        )
        assert prompt is not None
        expect = qc.man_expectation(man, age, kit) if man else qc.expectation(sheet, kit)
        reference = None
        if with_reference:  # the existing picture, drawn before the measured job
            existing = crop_to_aspect(renderer.render(prompt, sheet.base_seed + 100 + k))
            box = face_box(existing)
            reference = Reference(f"preset-memory-{k}", existing.crop(box), 0.1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        chunks = getattr(renderer, "prompt_chunks", None)
        tokens = getattr(renderer, "prompt_tokens", None)
        row: dict[str, Any] = {
            "cap_gib": cap_gib,
            "model": model,
            "gender": gender,
            "prompt_length": (
                f"{chunks(prompt)} chunks"
                if chunks
                else f"{tokens(prompt)} tokens"
                if tokens
                else None
            ),
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
                (detail_tags,) = tagger([detailed])
                row["detail_passed"] = qc.check(detail_tags, expect, qc.head_top(detailed)).passed
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
    from kitshot.core.settings_values import GENDER_CHOICES, IMAGE_MODELS, MODEL_CHOICES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap", type=float, action="append", help="GiB (default: 8 and 12)")
    parser.add_argument(
        "--preset", action="append", choices=("high", "medium", "low"), help="default: all three"
    )
    parser.add_argument(
        "--reference", action="store_true", help="each job a re-draw with a reference picture"
    )
    parser.add_argument("--model", action="append", choices=MODEL_CHOICES)
    parser.add_argument("--gender", action="append", choices=GENDER_CHOICES)
    parser.add_argument("--out", type=Path, help="default: data/phase15/preset-memory.json")
    parser.add_argument("--allow-fm", action="store_true", help="measure with FM27 open")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    caps = args.cap or [8.0, 12.0]
    presets = args.preset or ["high", "medium", "low"]
    phase_15 = bool(args.model or args.gender)
    models = args.model or ["animagine"]
    genders = args.gender or ["woman"]
    if args.child:
        one_cap(caps[0], presets, args.reference, models[0], genders[0])
        return 0
    from kitshot.pipeline.worker import fm_running
    from kitshot.platform import procs

    running = fm_running()
    if running is not False and not args.allow_fm:
        print("FM27 is running (or it cannot be told): close it first.", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]] = []
    for cap in caps:
        for model in models:
            if args.reference and not IMAGE_MODELS[model].reference:
                print(f"{model} takes no reference picture: skipped", file=sys.stderr)
                continue
            for gender in genders:
                command = [sys.executable, __file__, "--child", "--cap", str(cap)]
                command += ["--model", model, "--gender", gender]
                if args.reference:
                    command.append("--reference")
                for p in presets:
                    command += ["--preset", p]
                done = procs.run_quiet(command, timeout=3600, capture_output=True, text=True)
                for line in done.stdout.splitlines():
                    if line.startswith("{"):
                        rows.append({**json.loads(line), "fm_running": running})
                if done.returncode != 0:
                    print(done.stderr[-2000:], file=sys.stderr)
    if phase_15:
        out = args.out or OUT_15
        if out.is_file():  # one file for every run: earlier rows are kept
            rows = json.loads(out.read_text(encoding="utf-8")) + rows
    else:
        out = args.out or (OUT.with_name("preset-memory-reference.json") if args.reference else OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(
        f"\n{'cap':>5} {'model':<13} {'gender':<6} {'ref':<4} {'preset':<7} {'ok':<4}"
        f" {'reserved':>9} {'allocated':>10} {'job s':>7} {'passed':>7}"
    )
    for r in rows:
        print(
            f"{r['cap_gib']:>5.0f} {r.get('model', 'animagine'):<13} {r.get('gender', 'woman'):<6}"
            f" {'yes' if r['reference'] else 'no':<4} {r['preset']:<7} {'yes' if r['ok'] else 'OOM':<4}"
            f" {r['peak_reserved_gib']:>9.2f} {r['peak_allocated_gib']:>10.2f} {r['job_s']:>7.1f}"
            f" {r.get('passed', '-')!s:>7}"
        )
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
