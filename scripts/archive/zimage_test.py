"""Bake-off spike (run "zimage"): Z-Image (6B) in FP8 against Animagine XL 4.0.

Renders the round 1/2 cases (20 character sheets drawn twice, plus the two-sheet age
ladder) on Z-Image and Z-Image-Turbo with FP8 transformer weights, re-renders the same
cases on Animagine (the chosen model) so the page compares like for like, checks every
picture with the EVA02 tagger exactly as bakeoff.py does, and writes
data/bakeoff/<run>/index.html. A ten-case subset is also rendered in BF16 and with FP8
storage only, to see what FP8 costs, and with the tag prompt, to see whether Z-Image
needs plain English.

    uv run python scripts/archive/zimage_test.py prompts               # no GPU
    uv run python scripts/archive/zimage_test.py smoke                 # 2 pictures, zimage-smoke
    uv run python scripts/archive/zimage_test.py render --columns zimage~fp8 --cases S01a
    uv run python scripts/archive/zimage_test.py tag
    uv run python scripts/archive/zimage_test.py report
    uv run python scripts/archive/zimage_test.py all                   # everything, then report

Precision (column suffix):
  fp8   FP8 (e4m3) weights with a per-output-channel scale; activations quantised per
        row (token) on the fly; FP8 matmul through torch._scaled_mm's per-tensor kernel
        with the scales applied afterwards. On the RTX 4080 (2026-11-11) that kernel ran
        twice as fast as BF16, and its row-wise scaled mode slower than BF16. Per-tensor
        activation scaling was tried first and drew noise (see quantize_rows).
  fp8w  The same FP8 weights, upcast to BF16 for a BF16 matmul: half the weight memory,
        no speed-up.
  bf16  The published weights as they are; the quality reference.

The text encoder (Qwen3-4B, 8 GB) runs first for every prompt and is freed before the
transformer loads, so only one of them is on the card at a time.

Pictures stay on this PC (AGENTS.md rule 6). Rendering is resumable: a picture already on
disk is skipped. Close FM27 first.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bakeoff
from bakeoff import Case

SIZE = 1024
ZIMAGE = "Tongyi-MAI/Z-Image"
ZTURBO = "Tongyi-MAI/Z-Image-Turbo"  # transformer only; the rest is identical to ZIMAGE's

# Let bakeoff.tag() pick up the new folders ("zimage~fp8" etc.).
for _key, _label, _repo in (("zimage", "Z-Image", ZIMAGE), ("zturbo", "Z-Image-Turbo", ZTURBO)):
    bakeoff.MODELS[_key] = bakeoff.ModelSpec(_key, _label, _repo, None, "zimage")


@dataclass(frozen=True)
class Column:
    key: str  # output folder and report column
    label: str
    repo: str  # where the transformer and scheduler come from
    precision: str  # "fp8", "fp8w" or "bf16"
    style: str  # prompt style: "natural" or "tags"
    steps: int
    cfg: float  # 0 turns CFG off, and with it the negative prompt
    subset: bool  # only the ten precision-check cases


COLUMNS = {
    c.key: c
    for c in (
        # Model card: 28-50 steps, CFG 3-5. 30 steps and CFG 4 as the card's anime fine-tune.
        Column("zimage~fp8", "Z-Image · FP8", ZIMAGE, "fp8", "natural", 30, 4.0, False),
        # Model card: 9 steps (8 transformer passes), guidance 0. No negative prompt.
        Column("zturbo~fp8", "Z-Image-Turbo · FP8", ZTURBO, "fp8", "natural", 9, 0.0, False),
        Column("zimage~fp8w", "Z-Image · FP8 storage", ZIMAGE, "fp8w", "natural", 30, 4.0, True),
        Column(
            "zimage~fp8-tags", "Z-Image · FP8 · tag prompt", ZIMAGE, "fp8", "tags", 30, 4.0, True
        ),
        Column("zimage~bf16", "Z-Image · BF16", ZIMAGE, "bf16", "natural", 30, 4.0, True),
    )
}
ANIMAGINE = "animagine"
MAIN = [ANIMAGINE, "zimage~fp8", "zturbo~fp8"]
PRECISION = ["zimage~bf16", "zimage~fp8w", "zimage~fp8", "zimage~fp8-tags"]
# Render order. BF16 goes last: 12.3 GB of weights may not fit beside Windows' share.
DEFAULT_COLUMNS = [*MAIN, "zimage~fp8w", "zimage~fp8-tags", "zimage~bf16"]
LABELS = {ANIMAGINE: "Animagine XL 4.0 Opt · FP16 (chosen)"} | {
    k: c.label for k, c in COLUMNS.items()
}
# One case per sheet in bands 1-5, two per band; stripes, hoops and goalkeepers included.
SUBSET = ["S01a", "S03a", "S05a", "S08a", "S10a", "S12a", "S13a", "S16a", "S18a", "S20a"]
SMOKE = ["S01a", "S13a"]


def make_cases() -> list[Case]:
    """The round 1/2 cases. Teenage bands (1, 2) carry the young negative, as the
    generation recipe does for every under-18 wording (content rule)."""
    return [
        replace(
            c,
            wording=(bakeoff.BANDS[c.band], bakeoff.YOUNG_NEGATIVE if c.band <= 2 else ""),
        )
        for c in bakeoff.make_cases()
    ]


def cases_for(key: str, cases: list[Case]) -> list[Case]:
    if key in COLUMNS and COLUMNS[key].subset:
        return [c for c in cases if c.id in SUBSET]
    return cases


# ------------------------------------------------------------------- plain-English prompt
# Z-Image reads prompts through an LLM (Qwen3-4B) and was trained on sentence captions,
# so the sheet is written out as sentences. Same facts as the tag prompt, nothing added.

NL_AGE = {
    1: "a young teenage girl with a youthful face",
    2: "a teenage girl",
    3: "a young adult woman",
    4: "an adult woman with a mature face",
    5: "a mature adult woman with a mature face",
}
NL_HAIR = {
    "long": "long straight {c} hair",
    "ponytail": "{c} hair in a high ponytail",
    "twintails": "{c} hair in twin tails",
    "bob": "{c} hair in a bob cut",
    "short": "short {c} hair",
    "braid": "{c} hair in a long braid",
}
NL_FRINGE = {"blunt": "blunt bangs", "parted": "parted bangs", "swept": "side-swept bangs"}
NL_EYE_COLOUR = {
    "brown": "brown",
    "blue": "blue",
    "green": "green",
    "amber": "amber",
    "violet": "purple",
}
NL_EYE_SHAPE = {
    "gentle": "soft, gently drooping",
    "sharp": "sharp, upturned",
    "round": "large, round",
}
NL_SKIN = ["pale skin", "light skin", "lightly tanned skin", "tanned skin", "dark brown skin"]
NL_ACCESSORY = {
    "none": "",
    "hairband": "She wears a hairband.",
    "ribbon": "She has a ribbon in her hair.",
    "clip": "She wears a hair clip.",
    "glasses": "She wears glasses.",
}
NL_MARK = {
    "none": "",
    "freckles": "She has freckles on her cheeks.",
    "mole": "She has a small mole under one eye.",
}
NL_EXPRESSION = {
    "cheerful": "She smiles cheerfully with her mouth open.",
    "serious": "Her expression is serious, with her mouth closed.",
    "shy": "She looks shy, with a light blush and her mouth closed.",
    "confident": "She has a confident smirk.",
}
NL_POSE = {
    "crossed": "Her arms are crossed.",
    "hips": "Her hands are on her hips.",
    "ball": "She holds a soccer ball.",
    "behind": "Her arms are behind her back.",
    "thumb": "She gives a thumbs up.",
    "v": "She makes a peace sign with one hand.",
}
NL_KIT = {
    "blue_white_stripes": (
        "a soccer jersey with blue and white vertical stripes, short sleeves, and blue shorts"
    ),
    "red_white": "a red soccer jersey with a white collar, short sleeves, and white shorts",
    "yellow_green_hoops": (
        "a soccer jersey with yellow and green horizontal hoops, short sleeves, and green shorts"
    ),
    "black_white_stripes": (
        "a soccer jersey with black and white vertical stripes, short sleeves, and black shorts"
    ),
    "sky_blue": "a light blue soccer jersey with a white collar, short sleeves, and white shorts",
    "claret_blue": "a dark red soccer jersey with light blue sleeves, and white shorts",
    "orange_black": "an orange soccer jersey with black trim, short sleeves, and black shorts",
    "white_navy": (
        "a white soccer jersey with a navy blue collar, short sleeves, and navy blue shorts"
    ),
    "gk_green": "a green long-sleeved goalkeeper jersey and black goalkeeper gloves",
    "gk_orange": "an orange long-sleeved goalkeeper jersey and black goalkeeper gloves",
    "gk_purple": "a purple long-sleeved goalkeeper jersey and grey goalkeeper gloves",
}
NL_NEGATIVE = (
    "multiple people, photo, photorealistic, realistic, 3d render, lowres, blurry, "
    "worst quality, low quality, bad anatomy, bad hands, extra fingers, missing fingers"
)


def natural_prompt(case: Case) -> tuple[str, str]:
    s = case.sheet
    hair = NL_HAIR[s.style].format(c=s.hair)
    eyes = f"{NL_EYE_SHAPE[s.eye_shape]} {NL_EYE_COLOUR[s.eyes]} eyes"
    skin = NL_SKIN[bakeoff.SKIN_STEP[s.skin]]
    extras = " ".join(x for x in (NL_ACCESSORY[s.accessory], NL_MARK[s.mark]) if x)
    positive = " ".join(
        x
        for x in (
            "Anime illustration in a clean, modern Japanese anime style with cel shading.",
            (
                f"An upper-body portrait of {NL_AGE[case.band]}, a soccer player, standing "
                "and looking at the viewer in front of a plain grey studio background."
            ),
            f"She has {hair} with {NL_FRINGE[s.fringe]}, {eyes}, and {skin}.",
            extras,
            f"She wears {NL_KIT[case.kit]}.",
            NL_POSE[case.pose],
            NL_EXPRESSION[s.expression],
            "She is fully clothed in sportswear. She is the only person in the picture.",
        )
        if x
    )
    _, band_negative = case.wording or ("", "")
    negative = bakeoff._join(
        bakeoff.CONTENT_NEGATIVE, band_negative, bakeoff.COMPOSITION_NEGATIVE, NL_NEGATIVE
    )
    return positive, negative


def tag_prompt(case: Case) -> tuple[str, str]:
    """Animagine's prompt with its Danbooru score tags swapped for plain style words."""
    positive, negative = bakeoff.build_prompt(case, "animagine")
    quality = bakeoff.QUALITY["animagine"][0]
    positive = "anime illustration, " + positive.replace(", " + quality, "") + ", high quality"
    return positive, negative


def prompt_for(col: Column, case: Case) -> tuple[str, str]:
    return natural_prompt(case) if col.style == "natural" else tag_prompt(case)


# ----------------------------------------------------------------------------- FP8

FP8_MAX = 448.0  # largest float8_e4m3fn value


def _fp8_linear_class(compile_ops: bool):
    import torch
    import torch.nn.functional as F

    def quantize_rows(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # One scale per row (token). A single scale for the whole tensor wrecks the
        # picture: the feed-forward output projection in blocks 10-28 sees a few tokens up
        # to ~850,000x larger than the median token, and every ordinary token rounds to 0.
        # The scale is rounded to BF16 first so quantising and de-quantising agree.
        s = (x2.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
        s = s.to(torch.bfloat16)
        return (x2 / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s

    def rescale(y: torch.Tensor, s_row: torch.Tensor, s_col: torch.Tensor) -> torch.Tensor:
        return y * s_row * s_col

    if compile_ops:  # fuses each into one kernel; eager mode ate most of the FP8 gain
        quantize_rows = torch.compile(quantize_rows, dynamic=True)
        rescale = torch.compile(rescale, dynamic=True)

    class FP8Linear(torch.nn.Module):
        """nn.Linear with FP8 weights and one scale per output channel.

        fast=True: the input is scaled per row into FP8 and multiplied by the FP8 weights
        through torch._scaled_mm with unit scales, the fast per-tensor kernel on this card
        (its row-wise mode ran slower than BF16); the row and channel scales are applied
        to the BF16 product afterwards, which is exact. fast=False: the weights are upcast
        to BF16 for a BF16 matmul.
        """

        def __init__(self, linear: torch.nn.Linear, fast: bool) -> None:
            super().__init__()
            w = linear.weight.detach().to("cuda", torch.float32)
            scale = (w.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
            self.register_buffer("weight8", (w / scale[:, None]).to(torch.float8_e4m3fn))
            self.register_buffer("scale", scale.to(torch.bfloat16))
            self.register_buffer("one", torch.ones((), device="cuda"))
            bias = linear.bias
            self.bias = (
                None
                if bias is None
                else torch.nn.Parameter(bias.detach().to("cuda", torch.bfloat16), False)
            )
            self.fast = fast
            self.in_features, self.out_features = linear.in_features, linear.out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.fast:
                w = self.weight8.to(torch.bfloat16) * self.scale[:, None]
                return F.linear(x.to(torch.bfloat16), w, self.bias)
            shape = x.shape
            x8, s_row = quantize_rows(x.reshape(-1, shape[-1]).to(torch.bfloat16))
            y = torch._scaled_mm(
                x8, self.weight8.t(), scale_a=self.one, scale_b=self.one,
                out_dtype=torch.bfloat16,
            )  # fmt: skip
            y = rescale(y, s_row, self.scale)
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*shape[:-1], self.out_features)

    return FP8Linear


# Inside each transformer block: attention projections and the feed-forward. The
# embedders, the adaLN modulation (one row per picture) and the final layer stay BF16.
FP8_SCOPES = ("layers.", "noise_refiner.", "context_refiner.")


def to_fp8(transformer, fast: bool) -> int:
    """Swap the block Linear layers for FP8 ones on the GPU; returns how many."""
    import os

    import torch

    fp8_linear = _fp8_linear_class(compile_ops=os.environ.get("FP8_COMPILE", "1") == "1")
    targets = [
        (name, m)
        for name, m in transformer.named_modules()
        if isinstance(m, torch.nn.Linear)
        and name.startswith(FP8_SCOPES)
        and "adaLN" not in name
        and m.in_features % 16 == 0
        and m.out_features % 16 == 0
    ]
    for name, m in targets:
        parent_name, _, child = name.rpartition(".")
        parent = transformer.get_submodule(parent_name)
        setattr(parent, child, fp8_linear(m, fast))
        del m
    gc.collect()
    transformer.to("cuda")  # whatever stayed BF16
    return len(targets)


# ------------------------------------------------------------------------ rendering


def fm27_running() -> bool:
    try:
        out = subprocess.run(
            ["tasklist.exe", "/FI", "IMAGENAME eq fm.exe", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            timeout=30,
        ).stdout  # bytes: tasklist prints in the Windows code page, not UTF-8
    except (OSError, subprocess.SubprocessError):
        return False
    return b"fm.exe" in out.lower()


def gpu_guard() -> None:
    import torch

    if fm27_running():
        sys.exit("FM27 (fm.exe) is running; close it first")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
    if free < 11 * 2**30:
        sys.exit("less than 11 GiB free on the GPU; is another render running?")


def encode(texts: list[str]) -> tuple[dict[str, object], float]:
    """Every prompt through Qwen3-4B; returns embeddings on the CPU and seconds per text."""
    import torch
    from diffusers import ZImagePipeline

    t0 = time.perf_counter()
    # local_files_only: with sharded weights diffusers asks the Hub for the file list
    # unless told not to, and HF_HUB_OFFLINE alone makes that an error.
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE, transformer=None, vae=None, dtype=torch.bfloat16, local_files_only=True
    )
    pipe.text_encoder.to("cuda")
    load_s = time.perf_counter() - t0
    out: dict[str, object] = {}
    t1 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), 8):
            batch = texts[start : start + 8]
            embeds = pipe._encode_prompt(list(batch), device="cuda", max_sequence_length=512)
            for text, e in zip(batch, embeds, strict=True):
                out[text] = e.to("cpu")
    per_text = (time.perf_counter() - t1) / max(len(texts), 1)
    print(f"text encoder: loaded in {load_s:.0f} s, {per_text:.2f} s per prompt")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out, per_text


def load_pipeline(col: Column):
    import torch
    from diffusers import AutoencoderKL, ZImagePipeline, ZImageTransformer2DModel

    transformer = ZImageTransformer2DModel.from_pretrained(
        col.repo, subfolder="transformer", dtype=torch.bfloat16, local_files_only=True
    )
    if col.precision == "bf16":
        transformer.to("cuda")
    else:
        n = to_fp8(transformer, fast=col.precision == "fp8")
        print(f"{col.key}: {n} Linear layers in FP8")
    vae = AutoencoderKL.from_pretrained(
        ZIMAGE, subfolder="vae", dtype=torch.bfloat16, local_files_only=True
    )
    pipe = ZImagePipeline.from_pretrained(
        col.repo,
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def render_z(run_dir: Path, keys: list[str], cases: list[Case]) -> None:
    import torch

    todo = {}
    for key in keys:
        out_dir = run_dir / key
        missing = [c for c in cases_for(key, cases) if not (out_dir / f"{c.id}.png").exists()]
        if missing:
            todo[key] = missing
        else:
            print(f"{key}: nothing to render")
    if not todo:
        return
    texts = sorted({t for key, cs in todo.items() for c in cs for t in prompt_for(COLUMNS[key], c)})
    embeds, encode_s = encode(texts)

    for key, cs in todo.items():
        try:
            render_column(run_dir, COLUMNS[key], cs, embeds, encode_s)
        except torch.cuda.OutOfMemoryError as e:
            print(f"{key}: out of GPU memory, skipped ({str(e).splitlines()[0]})")
            models_path = run_dir / "models.json"
            stats = json.loads(models_path.read_text()) if models_path.exists() else {}
            stats[key] = {"label": COLUMNS[key].label, "error": "out of GPU memory"}
            models_path.write_text(json.dumps(stats, indent=2))
        gc.collect()
        torch.cuda.empty_cache()


def render_column(
    run_dir: Path, col: Column, cs: list[Case], embeds: dict[str, object], encode_s: float
) -> None:
    import torch
    from PIL.PngImagePlugin import PngInfo

    key = col.key
    log_path = run_dir / "renders.jsonl"
    models_path = run_dir / "models.json"
    out_dir = run_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{key}: loading {col.label}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = load_pipeline(col)
    load_s = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / 2**30
    print(f"{key}: loaded in {load_s:.0f} s, {weights_gb:.1f} GiB on the GPU")

    for n, case in enumerate(cs, 1):
        positive, negative = prompt_for(col, case)
        kwargs = {}
        if col.cfg > 0:
            kwargs["negative_prompt_embeds"] = [embeds[negative].to("cuda")]
        t1 = time.perf_counter()
        with torch.no_grad():
            image = pipe(
                prompt_embeds=[embeds[positive].to("cuda")],
                height=SIZE,
                width=SIZE,
                num_inference_steps=col.steps,
                guidance_scale=col.cfg,
                cfg_normalization=False,
                generator=torch.Generator("cuda").manual_seed(case.seed),
                **kwargs,
            ).images[0]
        seconds = time.perf_counter() - t1
        record = {
            "model": key.partition("~")[0],
            "column": key,
            "model_label": col.label,
            "repo": col.repo,
            "precision": col.precision,
            "prompt_style": col.style,
            "case": case.id,
            "seed": case.seed,
            "steps": col.steps,
            "cfg": col.cfg,
            "sampler": "flow-match euler (model default)",
            "width": SIZE,
            "height": SIZE,
            "prompt": positive,
            "negative": negative if col.cfg > 0 else "",
            "seconds": round(seconds, 2),
        }
        meta = PngInfo()
        meta.add_text("facegen", json.dumps(record))
        tmp = out_dir / f".{case.id}.png.tmp"
        image.save(tmp, format="PNG", pnginfo=meta)
        tmp.replace(out_dir / f"{case.id}.png")
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [{n}/{len(cs)}] {case.id} {seconds:.1f} s", flush=True)

    stats = json.loads(models_path.read_text()) if models_path.exists() else {}
    stats[key] = {
        "label": col.label,
        "load_seconds": round(load_s, 1),
        "weights_gib": round(weights_gb, 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "encode_seconds_per_prompt": round(encode_s, 2),
    }
    models_path.write_text(json.dumps(stats, indent=2))
    print(f"{key}: peak {stats[key]['peak_reserved_gib']:.1f} GiB reserved")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def render(run_dir: Path, keys: list[str], case_ids: set[str]) -> None:
    cases = [c for c in make_cases() if not case_ids or c.id in case_ids]
    gpu_guard()
    if ANIMAGINE in keys:
        bakeoff.render(run_dir, [ANIMAGINE], cases)
    z_keys = [k for k in keys if k in COLUMNS]
    if z_keys:
        render_z(run_dir, z_keys, cases)


# --------------------------------------------------------------------------- report


def _psnr(a: Path, b: Path) -> float:
    import numpy as np
    from PIL import Image

    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def report(run_dir: Path) -> Path:
    cases = {c.id: c for c in make_cases()}
    renders: dict[tuple[str, str], dict] = {}
    if (run_dir / "renders.jsonl").exists():
        for line in (run_dir / "renders.jsonl").read_text().splitlines():
            r = json.loads(line)
            renders[(r.get("column", r["model"]), r["case"])] = r
    stats = (
        json.loads((run_dir / "models.json").read_text())
        if (run_dir / "models.json").exists()
        else {}
    )
    tags_path = run_dir / "tags_eva02.json"
    tags = json.loads(tags_path.read_text())["images"] if tags_path.exists() else {}
    columns = [
        k
        for k in [*MAIN, *PRECISION]
        if (run_dir / k).is_dir() and any((run_dir / k).glob("*.png"))
    ]
    columns = list(dict.fromkeys(columns))

    rows: dict[tuple[str, str], dict] = {}
    for col in columns:
        for cid, case in cases.items():
            if not (run_dir / col / f"{cid}.png").exists():
                continue
            info = tags.get(f"{col}/{cid}")
            row = {"seconds": renders.get((col, cid), {}).get("seconds"), "band": case.band}
            if info:
                row["general"] = info["rating"] == "general"
                row["p_general"] = info["ratings"].get("general")
                row["tags"] = info["tags"]
                row["p_child"] = info.get("probes", {}).get("child")
                row.update(bakeoff.check(case.expect, info["tags"]))
                row["all"] = all(row[k] for k in ("general", "hair", "eyes", "kit", "solo"))
            reference = run_dir / "zimage~bf16" / f"{cid}.png"
            if col in ("zimage~fp8", "zimage~fp8w") and reference.exists():
                row["psnr"] = _psnr(run_dir / col / f"{cid}.png", reference)
            rows[(col, cid)] = row

    def pct(vals: list[bool]) -> str:
        return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "–"

    summary = []
    for col in columns:
        mine = [r for (c, _), r in rows.items() if c == col]
        secs = [r["seconds"] for r in mine if r.get("seconds")]
        st = stats.get(col, {})
        psnrs = [r["psnr"] for r in mine if "psnr" in r]
        child = [r["p_child"] for r in mine if r["band"] == 1 and r.get("p_child") is not None]
        summary.append(
            {
                "Column": LABELS.get(col, col),
                "Pictures": str(len(mine)),
                "s / picture": f"{statistics.median(secs):.1f}" if secs else "–",
                "Peak VRAM": f"{st['peak_reserved_gib']:.1f} GiB" if st else "–",
                "Weights on GPU": f"{st['weights_gib']:.1f} GiB" if st else "–",
                "Load": f"{st['load_seconds']:.0f} s" if st else "–",
                "General": pct([r.get("general", False) for r in mine]),
                "Hair": pct([r.get("hair", False) for r in mine]),
                "Eyes": pct([r.get("eyes", False) for r in mine]),
                "Kit": pct([r.get("kit", False) for r in mine]),
                "One girl": pct([r.get("solo", False) for r in mine]),
                "All checks": pct([r.get("all", False) for r in mine]),
                "Band 1 child": f"{statistics.mean(child):.2f}" if child else "–",
                "PSNR vs BF16": f"{statistics.mean(psnrs):.1f} dB" if psnrs else "–",
            }
        )

    def cell(col: str, cid: str) -> str:
        row = rows.get((col, cid))
        if row is None:
            return '<div class="pic missing">not rendered</div>'
        badges = []
        if "general" in row:
            cls = "ok" if row["general"] else "bad"
            word = "general" if row["general"] else "NOT general"
            badges.append(
                f'<span class="b {cls}" title="p(general)={row["p_general"]}">{word}</span>'
            )
            for name in ("hair", "eyes", "kit", "solo"):
                badges.append(f'<span class="b {"ok" if row[name] else "bad"}">{name}</span>')
        if "psnr" in row:
            badges.append(f'<span class="b">{row["psnr"]:.1f} dB</span>')
        r = renders.get((col, cid), {})
        tag_text = ", ".join(list(row.get("tags", {}))[:40])
        title = html.escape(
            f"{cid} · seed {r.get('seed')} · {row.get('seconds')} s\n\n{r.get('prompt', '')}"
            f"\n\nTagger: {tag_text}"
        )
        return (
            f'<figure class="pic"><a href="{col}/{cid}.png" target="_blank">'
            f'<img src="{col}/{cid}.png" loading="lazy" title="{title}" alt="{cid}"></a>'
            f"<figcaption>{''.join(badges)}</figcaption></figure>"
        )

    head = "".join(f"<th>{html.escape(k)}</th>" for k in (summary[0] if summary else {}))
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in s.values()) + "</tr>"
        for s in summary
    )
    sections = [
        f"<h2>Summary</h2><div class='scroll'><table><tr>{head}</tr>{body}</table></div>",
        (
            "<p class='note'>Checks use the EVA02 tagger at 0.35, as in rounds 1–5. "
            "<b>General</b> is the content rule: a picture passes only if its top rating is "
            "general. <b>All checks</b> = general, hair, eyes, kit and one girl. <b>s / "
            "picture</b> excludes the text encoder, which runs once per prompt before the "
            "transformer loads. <b>PSNR vs BF16</b> compares the same seed and prompt against "
            "the BF16 picture; higher means FP8 changed less (above ~30 dB is hard to see). "
            "Hover a picture for its prompt and tags.</p>"
        ),
    ]
    main = [c for c in MAIN if c in columns]
    heads = "".join(f"<th>{html.escape(LABELS[c])}</th>" for c in main)
    sections.append(
        "<h2>Character sheets</h2><p class='note'>Each cell: variant a and variant b, a "
        "different kit, pose and seed. Is it the same character? Does it look like the "
        "style we want?</p>"
    )
    for band, wording in bakeoff.BANDS.items():
        sections.append(f"<h3>Band {band}: {html.escape(wording)}</h3>")
        trs = []
        for s in (x for x in bakeoff.SHEETS if x.band == band):
            desc = html.escape(
                f"{s.id}: {s.hair} {s.style}, {s.fringe} fringe, {s.eyes} {s.eye_shape} eyes, "
                f"{s.skin} skin, {s.accessory}, {s.mark}, {s.expression}. "
                f"a: {s.kits[0]} / {s.poses[0]}; b: {s.kits[1]} / {s.poses[1]}"
            )
            tds = "".join(
                f"<td><div class='pair'>{cell(c, s.id + 'a')}{cell(c, s.id + 'b')}</div></td>"
                for c in main
            )
            trs.append(f"<tr><td class='desc'>{desc}</td>{tds}</tr>")
        sections.append(
            f"<div class='scroll'><table class='grid'><tr><th></th>{heads}</tr>"
            f"{''.join(trs)}</table></div>"
        )

    sections.append(
        "<h2>Age ladder</h2><p class='note'>Same sheet, kit, pose and seed; only the band "
        "changes.</p>"
    )
    for s in bakeoff.LADDER:
        for c in main:
            pics = "".join(
                f"<td><div class='band'>Band {b}</div>{cell(c, f'{s.id}-b{b}')}</td>"
                for b in bakeoff.BANDS
            )
            sections.append(
                f"<h3>{s.id} · {html.escape(LABELS[c])}</h3>"
                f"<div class='scroll'><table class='grid'><tr>{pics}</tr></table></div>"
            )

    prec = [c for c in PRECISION if c in columns]
    if len(prec) > 1:
        heads = "".join(f"<th>{html.escape(LABELS[c])}</th>" for c in prec)
        trs = "".join(
            f"<tr><td class='desc'>{cid}</td>"
            + "".join(f"<td>{cell(c, cid)}</td>" for c in prec)
            + "</tr>"
            for cid in SUBSET
        )
        sections.append(
            "<h2>What FP8 costs</h2><p class='note'>Same seed and prompt. BF16 is the "
            "reference; FP8 storage upcasts the FP8 weights for a BF16 matmul; FP8 also "
            "multiplies in FP8. The last column uses the tag prompt instead of sentences.</p>"
            f"<div class='scroll'><table class='grid'><tr><th></th>{heads}</tr>{trs}"
            "</table></div>"
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Z-Image FP8 test</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4; --card:#fff;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2c323c; --card:#1c2027;
          --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px 24px 48px; background:var(--bg); color:var(--fg);
        font:14px/1.45 system-ui, sans-serif; }}
h1 {{ margin:0 0 4px; font-size:22px; }} h2 {{ margin:32px 0 8px; font-size:18px; }}
h3 {{ margin:20px 0 6px; font-size:15px; color:var(--muted); font-weight:600; }}
.note {{ color:var(--muted); max-width:90ch; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }} th, td {{ padding:6px 10px; border-bottom:1px solid
  var(--line); text-align:left; vertical-align:top; white-space:nowrap; }}
.grid td {{ border:none; padding:4px; }}
.desc {{ white-space:normal; width:220px; min-width:220px; color:var(--muted); font-size:12px; }}
.pair {{ display:flex; gap:4px; }}
.pic {{ margin:0; width:208px; background:var(--card); border:1px solid var(--line);
        border-radius:6px; overflow:hidden; }}
.pic img {{ display:block; width:208px; height:208px; }}
.pic.missing {{ height:208px; display:grid; place-items:center; color:var(--muted); }}
figcaption {{ display:flex; flex-wrap:wrap; gap:3px; padding:4px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; color:var(--muted);
      background:var(--line); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
.band {{ font-size:12px; color:var(--muted); margin-bottom:2px; }}
</style></head><body>
<h1>Z-Image (6B) in FP8 against Animagine · {html.escape(run_dir.name)}</h1>
<p class="note">{SIZE}×{SIZE}. Animagine: the generation recipe (28 steps, CFG 5.5,
Euler a, tag prompt). Z-Image: 30 steps, CFG 4, plain-English prompt with the same content
negative. Z-Image-Turbo: 9 steps, no CFG, so no negative prompt. Teenage bands carry the
young negative in every column. Local file only; these pictures do not leave this PC.</p>
{"".join(sections)}
</body></html>"""
    path = run_dir / "index.html"
    path.write_text(page)
    return path


# ------------------------------------------------------------------------------ CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="zimage", help="run name under data/bakeoff/")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts", help="print the prompts; no GPU")
    p.add_argument("--style", choices=("natural", "tags"), default="natural")
    sub.add_parser("smoke", help=f"render {SMOKE} on Z-Image FP8 into <run>-smoke")
    for name in ("render", "all"):
        p = sub.add_parser(name)
        p.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
        p.add_argument("--cases", default="", help="comma-separated case IDs; default all")
    sub.add_parser("tag")
    sub.add_parser("report")
    args = parser.parse_args()

    run_dir = bakeoff.BAKEOFF_DIR / args.run
    if args.command == "prompts":
        for case in make_cases():
            pos, neg = natural_prompt(case) if args.style == "natural" else tag_prompt(case)
            print(f"{case.id} seed={case.seed}\n  + {pos}\n  - {neg}")
        return
    if args.command == "smoke":
        run_dir = bakeoff.BAKEOFF_DIR / f"{args.run}-smoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        render(run_dir, ["zimage~fp8"], set(SMOKE))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command in ("render", "all"):
        keys = [k for k in args.columns.split(",") if k]
        unknown = [k for k in keys if k != ANIMAGINE and k not in COLUMNS]
        if unknown:
            sys.exit(f"unknown column(s) {unknown}; choose from {[ANIMAGINE, *COLUMNS]}")
        render(run_dir, keys, {c for c in args.cases.split(",") if c})
    if args.command in ("tag", "all"):
        bakeoff.tag(run_dir, ["eva02"])
    if args.command in ("report", "all"):
        print("report:", report(run_dir))


if __name__ == "__main__":
    main()
