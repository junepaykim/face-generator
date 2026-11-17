"""Side test: the text encoders' trips to the GPU, per job and per batch (user, 2026-11-12).

With the text encoders offloaded (production since 2026-11-11), each job moved them (1.5 GiB)
onto the card to encode its prompt and back. The renderer now keeps prompt embeddings in system
memory (`facegen.prompt_cache`) and the worker encodes the next jobs' prompts in one trip. This
draws made-up group A players (IDs 930001-930006, not from a save) as the worker does: the
prompt encoded, three candidates, the tagger on them, the detail pass, the tagger on it. The
last is a re-draw with the first one's picture as the reference (IP-Adapter at 0.1).

    uv run python scripts/spikes/encode_cache_test.py per-job [--tag T]   # one trip per job
    uv run python scripts/spikes/encode_cache_test.py batch [--tag T]     # one trip for all jobs
    uv run python scripts/spikes/encode_cache_test.py resident [--tag T]  # encoders stay on the card

`--save` also keeps the pictures, in data/bakeoff/encode-cache/<tag>-<mode>/.

`per-job` and `resident` also run on the code from before the cache (tag it `before`). The tagger
is offloaded in every mode, as in production. Recorded: each phase's time and peak allocated and
reserved memory (after a warm-up job); what one trip of each offloaded model costs on its own
(moved onto the card and back, nothing run); and a SHA-256 of every picture's pixels, to compare
modes and code versions bit for bit. Results go to data/bakeoff/encode-cache/<tag>-<mode>.json.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from facegen import qc
from facegen.core.prompt import Prompt, build_prompt
from facegen.core.traits import pick_pose, picture_seed, sample_sheet
from facegen.datasource.kits import Kit
from facegen.render.gpu import on_gpu, prepare_environment
from facegen.render.picture import face_box

prepare_environment()

import torch  # after prepare_environment

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "bakeoff" / "encode-cache"
GIB = 2**30
KITS = [
    Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue"),
    Kit(shirt=("red",), pattern="plain", shorts="white"),
    Kit(shirt=("green", "white"), pattern="hoops", shorts="green"),
]
# (unique ID, age wording): the young wordings add to the negative prompt.
PLAYERS = [(930001, "W07"), (930002, "W03"), (930003, "W09"), (930004, "W05"), (930005, "W08")]
REDRAW = (930006, "W06")  # drawn with 930001's picture as the reference
MODES = ("per-job", "batch", "resident")


def main(args: list[str]) -> None:
    from facegen.pipeline.worker import AnimagineRenderer, Reference

    mode = args[0] if args else ""
    tag = args[args.index("--tag") + 1] if "--tag" in args else "after"
    save = RUN / f"{tag}-{mode}" if "--save" in args else None
    if mode not in MODES:
        sys.exit(__doc__)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    renderer = AnimagineRenderer(offload_text_encoders=mode != "resident")
    tagger = qc.Eva02Tagger()
    load_s = time.perf_counter() - t0
    prepare = getattr(renderer, "prepare", None)
    if mode == "batch" and prepare is None:
        sys.exit("batch needs the renderer's prepare (the code with the prompt cache)")

    phases: dict[str, dict[str, float]] = {}
    pixels: dict[str, str] = {}

    def keep(name: str, image: Any) -> None:
        pixels[name] = hashlib.sha256(image.tobytes()).hexdigest()
        if save is not None:
            save.mkdir(exist_ok=True)
            image.save(save / f"{name.replace(' ', '-')}.png")

    def phase(name: str, fn: Any, *a: Any) -> Any:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        out = fn(*a)
        torch.cuda.synchronize()
        p = phases.setdefault(name, {"seconds": 0.0, "allocated": 0.0, "reserved": 0.0})
        p["seconds"] += time.perf_counter() - t
        p["allocated"] = max(p["allocated"], torch.cuda.max_memory_allocated() / GIB)
        p["reserved"] = max(p["reserved"], torch.cuda.max_memory_reserved() / GIB)
        return out

    def encode(prompts: list[Prompt]) -> None:
        """One trip of the text encoders for these prompts."""
        if prepare is not None:
            prepare(prompts)
        else:  # the code before the cache: one prompt per trip, cached for its job only
            (prompt,) = prompts
            renderer._embeddings(prompt)

    def prompt_for(uid: int, wording: str) -> tuple[Any, Prompt]:
        sheet = sample_sheet("encode-cache-test", uid, 1)
        kit = KITS[uid % len(KITS)]
        return sheet, build_prompt(sheet, wording, kit, pick_pose(sheet.base_seed, 1))

    def job(label: str, uid: int, wording: str, reference: Any = None) -> tuple[Any, float]:
        t = time.perf_counter()
        sheet, prompt = prompt_for(uid, wording)
        if mode != "batch":
            phase(f"{label}: encode prompt", encode, [prompt])
        if reference is not None:
            # What the first render of a re-draw does first: load the adapter (once) and embed
            # the reference with the image encoder, which then goes back to system memory.
            phase(f"{label}: embed reference", renderer._conditioning, reference)
        images = []
        for k in range(3):
            seed = picture_seed(sheet.base_seed, 1, 0, k)
            image = phase(f"{label}: render", renderer.render, prompt, seed, reference)
            keep(f"{uid} render {k}", image)
            images.append(image)
        phase(f"{label}: tag candidates", tagger, images)
        detailed = phase(
            f"{label}: detail pass", renderer.detail, images[0], prompt, sheet.base_seed, reference
        )
        keep(f"{uid} detail", detailed)
        phase(f"{label}: tag detail", tagger, [detailed])
        return detailed, time.perf_counter() - t

    job("warm-up", 930000, "W07")
    phases.clear()
    pixels.clear()

    t_all = time.perf_counter()
    if mode == "batch":
        prompts = [prompt_for(uid, w)[1] for uid, w in [*PLAYERS, REDRAW]]
        phase("batch: encode all prompts", encode, prompts)
    seconds = {}
    first = None
    for uid, wording in PLAYERS:
        image, seconds[uid] = job("plain", uid, wording)
        first = first or image
    ref = Reference(key="encode-cache-test", image=first.crop(face_box(first)), scale=0.1)
    _, seconds[REDRAW[0]] = job("reference", *REDRAW, ref)
    total = time.perf_counter() - t_all

    # One trip of each offloaded model on its own: onto the card and back, nothing run.
    trips: dict[str, float] = {}
    pipe = renderer.pipe
    models = {
        "text encoders": (pipe.text_encoder, pipe.text_encoder_2),
        "tagger": (tagger.model,),
        "IP-Adapter image encoder": (pipe.image_encoder,),
    }
    for name, mods in models.items():
        if mode == "resident" and name == "text encoders":
            continue
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t = time.perf_counter()
            with on_gpu(*mods):
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        trips[name] = round(statistics.median(times), 3)

    def peak(prefix: str, kind: str) -> float:
        return round(max(p[kind] for k, p in phases.items() if k.startswith(prefix)), 2)

    n_jobs = len(PLAYERS) + 1
    encode_s = sum(p["seconds"] for k, p in phases.items() if "encode" in k)
    result = {
        "mode": mode,
        "tag": tag,
        "renderer": renderer.settings(),
        "load_seconds": round(load_s, 1),
        "jobs": n_jobs,
        "total_seconds": round(total, 1),
        "seconds_per_job": round(total / n_jobs, 2),
        "encode_seconds_total": round(encode_s, 3),
        "encode_seconds_per_job": round(encode_s / n_jobs, 3),
        "trip_seconds": trips,
        "peak_reserved_gib": {"plain": peak("plain", "reserved"), "reference": peak("reference", "reserved"),
                              "encode": peak("batch" if mode == "batch" else "plain: encode", "reserved")},
        "peak_allocated_gib": {"plain": peak("plain", "allocated"), "reference": peak("reference", "allocated")},
        "job_seconds": {str(k): round(v, 2) for k, v in seconds.items()},
        "phases": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in phases.items()},
        "pixels": pixels,
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }  # fmt: skip
    name = f"{tag}-{mode}"
    (RUN / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(
        name, json.dumps({k: result[k] for k in list(result)[3:] if k not in ("phases", "pixels")})
    )
    for k, v in result["phases"].items():
        print(
            f"  {k:28s} {v['seconds']:7.2f} s  allocated {v['allocated']:5.2f}"
            f"  reserved {v['reserved']:5.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
