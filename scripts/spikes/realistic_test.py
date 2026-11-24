"""Side test (run "realistic"): photoreal models for male players at the anime model's VRAM.

Asked for by the user on 2026-11-19 (docs/decisions.md): a realistic model for male
players, quantised so it uses about as much graphics memory as Animagine, the anime model (a
group A job peaks at 6.9 GiB reserved, 7.7 GiB with a reference picture). Each column is one
model in one weight format:

    animagine         the production renderer on its own prompts: the VRAM and speed target
    realvis~int8      RealVisXL V5.0 (SDXL), the production INT8 path; ~fp16 is its reference
    juggernaut~int8   Juggernaut XL v9 (SDXL), likewise; ~fp16 is its reference
    zturbo~nf4/int8   Z-Image-Turbo (6B DiT, 9 steps); ~bf16 is its reference
    klein~nf4/int8    FLUX.2 klein 4B (4B DiT, 4 steps); ~bf16 is its reference

Weight formats, for the Linear layers inside the numbered transformer blocks (embedders,
modulation, norms and the final layer keep the model's dtype):

    int8  one scale per output channel, rebuilt per call (kitshot.render.quantise, production)
    nf4   4-bit NormalFloat codes, one absmax per 64 weights, rebuilt per call (pure PyTorch,
          no new package)
    bf16  the published weights (fp16 for SDXL): the quality reference for PSNR and SSIM

The Qwen3 text encoders are INT8 for the int8 and nf4 columns and BF16 for the bf16 ones.
They run before the transformer loads and leave the card afterwards, so their peak is
reported apart, as the "encode trip".

    uv run --no-sync python scripts/spikes/realistic_test.py prompts         # no GPU
    uv run --no-sync python scripts/spikes/realistic_test.py nf4-check       # one layer
    uv run --no-sync python scripts/spikes/realistic_test.py render [--columns a,b] [--cases S01a]
    uv run --no-sync python scripts/spikes/realistic_test.py tag
    uv run --no-sync python scripts/spikes/realistic_test.py report
    uv run --no-sync python scripts/spikes/realistic_test.py all

The sheets are made up; no save data is read. Pictures and the page stay in
data/bakeoff/realistic/ (AGENTS.md rule 6). Rendering is resumable: a picture already on
disk is skipped. Close FM27 first.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kitshot.render.gpu import prepare_environment

prepare_environment()

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "data" / "bakeoff" / "realistic"
GIB = 2**30
WIDTH, HEIGHT = 896, 1152  # the production render size (3:4 after the master's crop)
DETAIL = (1344, 1728)  # group A's detail-pass size, rendered once per column as a memory probe
TARGET_GIB = 7.7  # Animagine's group A job with a reference picture (research.md, run inference)

ANIMAGINE = "cagliostrolab/animagine-xl-4.0"  # also the source of the fp16-fix SDXL VAE
REALVIS = "SG161222/RealVisXL_V5.0"
JUGGERNAUT = "RunDiffusion/Juggernaut-XL-v9"
ZIMAGE = "Tongyi-MAI/Z-Image"  # text encoder, tokenizer and VAE (identical in Turbo's repo)
ZTURBO = "Tongyi-MAI/Z-Image-Turbo"
KLEIN = "black-forest-labs/FLUX.2-klein-4B"  # its Qwen3-4B text encoder is Z-Image's


# ---------------------------------------------------------------------------- cases


@dataclass(frozen=True)
class Sheet:
    id: str
    age: int
    look: str  # ancestry words, as a facepack prompt would carry them
    skin: str
    hair: str
    hair_tag: str  # the tagger's hair colour tag, "" for a shaved head
    beard: str
    eyes: str
    expression: str
    kits: tuple[str, str]
    backdrops: tuple[str, str]


KITS = {
    "blue_white": "a football shirt with blue and white vertical stripes",
    "red_white": "a red football shirt with a white collar",
    "yellow_green": "a football shirt with yellow and green horizontal hoops",
    "black_white": "a football shirt with black and white vertical stripes",
    "sky": "a sky blue football shirt with a white crew neck",
    "claret": "a claret football shirt with light blue sleeves",
    "orange": "an orange football shirt with black trim",
    "white_navy": "a white football shirt with a navy blue collar",
    "gk_green": "a green long-sleeved goalkeeper jersey",
    "gk_orange": "an orange long-sleeved goalkeeper jersey",
    "gk_purple": "a purple long-sleeved goalkeeper jersey",
}
BACKDROPS = {
    "grey": "a plain mid-grey",
    "blue": "a plain light blue",
    "navy": "a dark navy blue",
    "white": "a plain off-white",
    "red": "a plain deep red",
}
CLEAN = "clean-shaven"

SHEETS = [
    Sheet("S01", 18, "Northern European", "fair", "short blond hair with a side parting",
          "blonde hair", CLEAN, "blue", "a slight smile",
          ("blue_white", "red_white"), ("grey", "blue")),
    Sheet("S02", 22, "Mediterranean", "olive", "dark brown wavy hair swept back",
          "brown hair", "light stubble", "brown", "a neutral expression",
          ("claret", "white_navy"), ("navy", "grey")),
    Sheet("S03", 27, "West African", "dark brown", "short black hair in a high-top fade",
          "black hair", "a short trimmed beard", "dark brown", "a confident smile",
          ("orange", "sky"), ("white", "red")),
    Sheet("S04", 31, "Scottish", "pale freckled", "ginger hair in a short crew cut",
          "orange hair", "a full ginger beard", "green", "a serious expression",
          ("gk_green", "black_white"), ("grey", "navy")),
    Sheet("S05", 19, "Brazilian", "light brown", "short black curly hair",
          "black hair", CLEAN, "brown", "a wide smile",
          ("yellow_green", "blue_white"), ("blue", "white")),
    Sheet("S06", 34, "English", "fair", "short brown hair going grey at the temples",
          "brown hair", "light stubble", "hazel", "a neutral expression",
          ("white_navy", "gk_purple"), ("red", "grey")),
    Sheet("S07", 25, "Caribbean", "brown", "short black dreadlocks tied back",
          "black hair", "a moustache and goatee", "brown", "a serious expression",
          ("red_white", "claret"), ("grey", "white")),
    Sheet("S08", 21, "Japanese", "light", "straight black hair with a short fringe",
          "black hair", CLEAN, "dark brown", "a slight smile",
          ("sky", "orange"), ("navy", "blue")),
    Sheet("S09", 29, "Moroccan", "light brown", "a shaved head",
          "", "a full black beard", "brown", "a neutral expression",
          ("black_white", "gk_orange"), ("white", "grey")),
    Sheet("S10", 17, "Dutch", "fair", "messy short light brown hair",
          "brown hair", CLEAN, "blue", "a cheerful smile",
          ("blue_white", "yellow_green"), ("red", "blue")),
    Sheet("S11", 36, "Ghanaian", "dark brown", "a closely shaved head",
          "", "a short greying beard", "brown", "a calm, serious expression",
          ("gk_purple", "white_navy"), ("grey", "navy")),
    Sheet("S12", 24, "South Asian", "medium brown", "short black hair with a neat side parting",
          "black hair", "light stubble", "brown", "a slight smile",
          ("claret", "red_white"), ("blue", "white")),
]  # fmt: skip
LADDER_SHEET = "S02"
LADDER_AGES = (17, 21, 26, 31, 36)


@dataclass(frozen=True)
class Case:
    id: str
    sheet: Sheet
    age: int
    kit: str
    backdrop: str
    seed: int


def _seed(case_id: str) -> int:
    return zlib.crc32(f"realistic:{case_id}".encode())


def make_cases() -> list[Case]:
    """Each sheet twice (a, b: another kit, backdrop and seed), then the age ladder: one
    sheet, kit, backdrop and seed at five ages."""
    cases = [
        Case(f"{s.id}{v}", s, s.age, s.kits[k], s.backdrops[k], _seed(f"{s.id}{v}"))
        for s in SHEETS
        for k, v in enumerate("ab")
    ]
    ladder = next(s for s in SHEETS if s.id == LADDER_SHEET)
    cases += [
        Case(f"L{age}", ladder, age, ladder.kits[0], ladder.backdrops[0], _seed("ladder"))
        for age in LADDER_AGES
    ]
    return cases


# Every prompt carries the clothing terms; CFG models also get this negative. Content rule
# (AGENTS.md rule 3): fully clothed, non-sexual sports portraits only.
CONTENT_NEGATIVE = (
    "nsfw, nude, naked, shirtless, bare chest, topless, underwear, swimwear, "
    "revealing clothes, suggestive"
)
COMPOSITION_NEGATIVE = "multiple people, two people, text, watermark, signature, logo, frame"
STYLE_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, 3d render, cgi, plastic skin, "
    "deformed, bad anatomy, asymmetric eyes, blurry, lowres"
)
NEGATIVE = f"{CONTENT_NEGATIVE}, {COMPOSITION_NEGATIVE}, {STYLE_NEGATIVE}"


def prompt_for(case: Case) -> str:
    s = case.sheet
    beard = "no facial hair" if s.beard == CLEAN else s.beard
    article = "an" if str(case.age).startswith(("8", "11", "18")) else "a"
    parts = (
        (
            f"Professional studio headshot photograph of {article} {case.age}-year-old {s.look} "
            "male football player, head and shoulders, centred, facing the camera and looking "
            "into the lens."
        ),
        f"He has {s.skin} skin, {s.hair}, {beard}, {s.eyes} eyes and {s.expression}.",
        f"He wears {KITS[case.kit]}, fully clothed.",
        (
            f"{BACKDROPS[case.backdrop].capitalize()} studio background, soft even lighting, "
            "sharp focus, natural skin texture, realistic photo."
        ),
        "He is the only person in the picture.",
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------- columns


@dataclass(frozen=True)
class Column:
    key: str  # output folder
    label: str
    family: str  # "sdxl", "zimage", "klein"
    repo: str  # transformer / UNet source
    weights: str  # "int8", "nf4", "bf16" ("bf16" means fp16 for SDXL)
    steps: int
    cfg: float  # 0 (or klein's 1 when distilled): no negative prompt
    sampler: str
    reference: str = ""  # the column this one is compared with (PSNR, SSIM)


COLUMNS = {
    c.key: c
    for c in (
        # RealVis card: DPM++ SDE Karras 30+ steps (needs torchsde, not installed) or
        # DPM++ 2M Karras 50+; DPM++ 2M SDE Karras at 30 is the nearest built-in substitute.
        Column("realvis~int8", "RealVisXL V5.0 · INT8", "sdxl", REALVIS, "int8", 30, 5.0,
               "dpm++ 2m sde karras", "realvis~fp16"),
        Column("realvis~fp16", "RealVisXL V5.0 · FP16", "sdxl", REALVIS, "bf16", 30, 5.0,
               "dpm++ 2m sde karras"),
        # Juggernaut card: DPM++ 2M Karras, 30-40 steps, CFG 3-7 (lower is more realistic).
        Column("juggernaut~int8", "Juggernaut XL v9 · INT8", "sdxl", JUGGERNAUT, "int8", 30,
               4.5, "dpm++ 2m karras", "juggernaut~fp16"),
        Column("juggernaut~fp16", "Juggernaut XL v9 · FP16", "sdxl", JUGGERNAUT, "bf16", 30,
               4.5, "dpm++ 2m karras"),
        # Z-Image-Turbo card: 9 steps (8 transformer passes), guidance 0.
        Column("zturbo~nf4", "Z-Image-Turbo · NF4", "zimage", ZTURBO, "nf4", 9, 0.0,
               "flow-match euler", "zturbo~bf16"),
        Column("zturbo~int8", "Z-Image-Turbo · INT8", "zimage", ZTURBO, "int8", 9, 0.0,
               "flow-match euler", "zturbo~bf16"),
        Column("zturbo~bf16", "Z-Image-Turbo · BF16", "zimage", ZTURBO, "bf16", 9, 0.0,
               "flow-match euler"),
        # klein card: 4 steps, guidance 1.0 (distilled; no negative prompt).
        Column("klein~nf4", "FLUX.2 klein 4B · NF4", "klein", KLEIN, "nf4", 4, 1.0,
               "flow-match euler", "klein~bf16"),
        Column("klein~int8", "FLUX.2 klein 4B · INT8", "klein", KLEIN, "int8", 4, 1.0,
               "flow-match euler", "klein~bf16"),
        Column("klein~bf16", "FLUX.2 klein 4B · BF16", "klein", KLEIN, "bf16", 4, 1.0,
               "flow-match euler"),
    )
}  # fmt: skip
REFERENCES = {c.reference for c in COLUMNS.values() if c.reference}
BASELINE = "animagine"
LABELS = {BASELINE: "Animagine XL 4.0 · INT8 (anime, production)"} | {
    k: c.label for k, c in COLUMNS.items()
}
DEFAULT_COLUMNS = [BASELINE, *COLUMNS]


# --------------------------------------------------------------------- quantisation

NF4_CODE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)  # fmt: skip
"""The 16 NormalFloat-4 values of QLoRA (Dettmers et al. 2023), as bitsandbytes uses them."""
NF4_GROUP = 64

# Inside numbered block lists ("layers.3.", "single_transformer_blocks.12."), minus
# modulation and norms, which see one row per picture and are cheap to keep exact.
_BLOCK = re.compile(r"(^|\.)\d+\.")
_KEEP = ("adaLN", "norm", "modulation", "embed", "time_", "proj_out")


def block_linears(model: Any, group: int = 1) -> list[str]:
    import torch

    return [
        name
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and _BLOCK.search(name)
        and not any(k in name for k in _KEEP)
        and m.in_features % group == 0
    ]


def _nf4_class() -> type:
    import torch
    import torch.nn.functional as F

    class NF4Linear(torch.nn.Module):
        """nn.Linear with 4-bit NF4 codes, one absmax per NF4_GROUP inputs, two codes per
        byte. The weight is rebuilt in the model's dtype for each call (eager, no compile)."""

        def __init__(self, linear: torch.nn.Linear, device: str) -> None:
            super().__init__()
            dtype = linear.weight.dtype
            w = linear.weight.detach().to(device, torch.float32)
            out_f, in_f = w.shape
            g = w.reshape(out_f, in_f // NF4_GROUP, NF4_GROUP)
            absmax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8).to(dtype)
            code = torch.tensor(NF4_CODE, device=device)
            # Nearest code: the midpoints between neighbours bound each code's bucket.
            idx = torch.bucketize(g / absmax.float(), (code[1:] + code[:-1]) / 2)
            idx = idx.to(torch.uint8)
            self.register_buffer("packed", (idx[..., 0::2] | (idx[..., 1::2] << 4)).contiguous())
            self.register_buffer("absmax", absmax)
            # Byte -> its two values (low nibble first): one lookup per byte through
            # F.embedding is 3.4x faster than indexing per nibble (0.65 against 2.2 ms for a
            # 10240x3840 layer on the RTX 4080, where its matmul at 4,500 tokens takes 5.5 ms).
            b = torch.arange(256, device=device)
            self.register_buffer("pairs", torch.stack((code[b & 15], code[b >> 4]), -1).to(dtype))
            bias = linear.bias
            self.bias = (
                None
                if bias is None
                else torch.nn.Parameter(bias.detach().to(device), requires_grad=False)
            )
            self.in_features, self.out_features = in_f, out_f

        def weight(self) -> torch.Tensor:
            p = self.packed
            w = F.embedding(p.int(), self.pairs).reshape(*p.shape[:-1], NF4_GROUP)
            return (w * self.absmax).reshape(self.out_features, self.in_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.weight(), self.bias)

    return NF4Linear


def quantise(model: Any, weights: str, device: str = "cuda") -> int:
    """Swap the block Linear layers for INT8 or NF4 ones, built on `device` layer by layer;
    returns how many. The rest of the model stays where it is."""
    if weights == "int8":
        from kitshot.render.quantise import Int8Linear as cls

        names = block_linears(model)
    elif weights == "nf4":
        cls = _nf4_class()
        names = block_linears(model, NF4_GROUP)
    else:
        raise ValueError(weights)
    for name in names:
        parent, _, child = name.rpartition(".")
        holder = model.get_submodule(parent)
        setattr(holder, child, cls(getattr(holder, child), device))
    gc.collect()
    return len(names)


# ------------------------------------------------------------------------ GPU helpers


def fm27_running() -> bool:
    try:
        out = subprocess.run(
            ["tasklist.exe", "/FI", "IMAGENAME eq fm.exe", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return b"fm.exe" in out.lower()


def gpu_guard() -> None:
    import torch

    if fm27_running():
        sys.exit("FM27 (fm.exe) is running; close it first")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {free / GIB:.1f} of {total / GIB:.1f} GiB free")
    if free < 11 * GIB:
        sys.exit("less than 11 GiB free on the GPU; is another render running?")


def free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


class Peaks:
    """Peak allocated / reserved memory per named phase."""

    def __init__(self) -> None:
        self.phases: dict[str, dict[str, float]] = {}

    def run(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        p = self.phases.setdefault(name, {"allocated": 0.0, "reserved": 0.0})
        p["allocated"] = max(p["allocated"], round(torch.cuda.max_memory_allocated() / GIB, 2))
        p["reserved"] = max(p["reserved"], round(torch.cuda.max_memory_reserved() / GIB, 2))
        return out


def update_json(path: Path, key: str, value: dict[str, Any]) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=1))


def save_picture(run_dir: Path, col: str, case: Case, image: Any, record: dict) -> None:
    from PIL.PngImagePlugin import PngInfo

    out_dir = run_dir / col
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = PngInfo()
    meta.add_text("facegen", json.dumps(record))
    tmp = out_dir / f".{case.id}.png.tmp"
    image.save(tmp, format="PNG", pnginfo=meta)
    tmp.replace(out_dir / f"{case.id}.png")
    with (run_dir / "renders.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------- text encoders


def _qwen3_te(dtype_weights: str) -> Any:
    """Z-Image's Qwen3-4B (also klein's), BF16 or with INT8 block layers, on the GPU."""
    import torch
    from transformers import AutoModelForCausalLM

    from kitshot.render.quantise import quantise_linears

    te = AutoModelForCausalLM.from_pretrained(
        ZIMAGE, subfolder="text_encoder", dtype=torch.bfloat16, local_files_only=True
    )
    if dtype_weights != "bf16":
        quantise_linears(te, "model.layers.", "cuda")
    return te.to("cuda").eval()


def encode_qwen3(texts_by_family: dict[str, list[str]], te_weights: str, peaks: Peaks) -> dict:
    """Embeddings for Z-Image and klein prompts, through one shared Qwen3-4B; on the CPU."""
    import torch
    from diffusers import Flux2KleinPipeline, ZImagePipeline
    from transformers import Qwen2TokenizerFast

    out: dict[tuple[str, str], Any] = {}
    phase = f"encode trip · Qwen3-4B {te_weights.upper()}"
    te = peaks.run(phase, _qwen3_te, te_weights)
    t0 = time.perf_counter()
    n = 0
    with torch.no_grad():
        if texts_by_family.get("zimage"):
            pipe = ZImagePipeline.from_pretrained(
                ZIMAGE, transformer=None, vae=None, text_encoder=te, local_files_only=True
            )
            for text in texts_by_family["zimage"]:
                (e,) = peaks.run(phase, pipe._encode_prompt, [text], device="cuda")
                out[("zimage", text)] = e.to("cpu")
                n += 1
            del pipe
        if texts_by_family.get("klein"):
            # The class model_index.json names; AutoTokenizer wants a config.json at the root.
            tok = Qwen2TokenizerFast.from_pretrained(
                KLEIN, subfolder="tokenizer", local_files_only=True
            )
            for text in texts_by_family["klein"]:
                e = peaks.run(
                    phase,
                    Flux2KleinPipeline._get_qwen3_prompt_embeds,
                    text_encoder=te,
                    tokenizer=tok,
                    prompt=[text],
                    device="cuda",
                )
                out[("klein", text)] = e.to("cpu")
                n += 1
    per_text = (time.perf_counter() - t0) / max(n, 1)
    print(f"{phase}: {n} prompts, {per_text:.2f} s each")
    del te
    free_gpu()
    return {"embeds": out, "seconds_per_prompt": round(per_text, 3), "phase": phase}


# ------------------------------------------------------------------------- loading


def load_sdxl(col: Column) -> Any:
    """The production SDXL setup on another checkpoint: fp16 with Animagine's fp16-fix VAE,
    INT8 transformer blocks (or fp16), text encoders waiting in system memory."""
    import torch
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    from kitshot.render.quantise import quantise_linears
    from kitshot.render.renderer import INT8_SCOPE

    vae = AutoencoderKL.from_pretrained(ANIMAGINE, subfolder="vae", dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        col.repo, vae=vae, variant="fp16", dtype=torch.float16, use_safetensors=True
    )
    if col.weights == "int8":
        quantise_linears(pipe.unet, INT8_SCOPE, "cuda")
    sde = "sde" in col.sampler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++" if sde else "dpmsolver++",
    )
    pipe.unet.to("cuda")
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_dit(col: Column) -> Any:
    import diffusers
    import torch

    t_cls, p_cls = {
        "zimage": ("ZImageTransformer2DModel", "ZImagePipeline"),
        "klein": ("Flux2Transformer2DModel", "Flux2KleinPipeline"),
    }[col.family]
    transformer = getattr(diffusers, t_cls).from_pretrained(
        col.repo, subfolder="transformer", dtype=torch.bfloat16, local_files_only=True
    )
    if col.weights != "bf16":
        n = quantise(transformer, col.weights)
        print(f"{col.key}: {n} Linear layers in {col.weights.upper()}")
    transformer.to("cuda")
    extra: dict[str, Any] = {}
    if col.family == "zimage":  # Turbo's repository was fetched without its (identical) VAE
        extra["vae"] = diffusers.AutoencoderKL.from_pretrained(
            ZIMAGE, subfolder="vae", dtype=torch.bfloat16, local_files_only=True
        )
    pipe = getattr(diffusers, p_cls).from_pretrained(
        col.repo,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        **extra,
    )
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ----------------------------------------------------------------------- rendering


def _sdxl_embeddings(pipe: Any, cache: dict, text: str, peaks: Peaks) -> Any:
    import torch

    from kitshot.render.gpu import on_gpu
    from kitshot.render.renderer import _encode_text

    if text not in cache:
        with torch.no_grad(), on_gpu(pipe.text_encoder, pipe.text_encoder_2):
            cache[text] = peaks.run("encode trip · CLIP L+G FP16", _encode_text, pipe, text)
    return cache[text]


def render_sdxl(pipe: Any, col: Column, case: Case, size: tuple[int, int], peaks: Peaks,
                cache: dict, phase: str) -> Any:  # fmt: skip
    import torch

    from kitshot.render.renderer import pair_embeddings

    positive = _sdxl_embeddings(pipe, cache, prompt_for(case), peaks)
    negative = _sdxl_embeddings(pipe, cache, NEGATIVE, peaks)
    empty = _sdxl_embeddings(pipe, cache, "", peaks)
    pe, pp, ne, npool, _ = (
        t.to("cuda") if hasattr(t, "to") else t for t in pair_embeddings(positive, negative, empty)
    )
    matmul = torch.backends.cuda.matmul
    before = matmul.allow_fp16_accumulation
    matmul.allow_fp16_accumulation = True  # production: fp16 accumulation in the pipeline only
    try:
        with torch.no_grad():
            return peaks.run(
                phase,
                pipe,
                prompt_embeds=pe,
                pooled_prompt_embeds=pp,
                negative_prompt_embeds=ne,
                negative_pooled_prompt_embeds=npool,
                width=size[0],
                height=size[1],
                num_inference_steps=col.steps,
                guidance_scale=col.cfg,
                generator=torch.Generator("cuda").manual_seed(case.seed),
            ).images[0]
    finally:
        matmul.allow_fp16_accumulation = before


def render_dit(pipe: Any, col: Column, case: Case, size: tuple[int, int], peaks: Peaks,
               embeds: dict, phase: str) -> Any:  # fmt: skip
    import torch

    e = embeds[(col.family, prompt_for(case))]
    kwargs: dict[str, Any] = {
        "width": size[0],
        "height": size[1],
        "num_inference_steps": col.steps,
        "guidance_scale": col.cfg,
        "generator": torch.Generator("cuda").manual_seed(case.seed),
    }
    if col.family == "zimage":
        kwargs["prompt_embeds"] = [e.to("cuda")]
        kwargs["cfg_normalization"] = False
    else:
        kwargs["prompt_embeds"] = e.to("cuda")
    with torch.no_grad():
        return peaks.run(phase, pipe, **kwargs).images[0]


def detail_probe(draw: Any, vae: Any, peaks: Peaks) -> dict[str, Any]:
    """One render at the detail size, then (if it went over the target) one with VAE tiling."""
    import torch

    probe: dict[str, Any] = {}
    for tiled in (False, True):
        phase = "detail-size render, VAE tiled" if tiled else "detail-size render"
        if tiled:
            over = peaks.phases.get("detail-size render", {}).get("reserved", 99) > TARGET_GIB
            if not (over or "error" in probe) or not hasattr(vae, "enable_tiling"):
                break
            vae.enable_tiling()
        prefix = "tiled_" if tiled else ""
        try:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            image = draw(phase)
            torch.cuda.synchronize()
            probe[prefix + "seconds"] = round(time.perf_counter() - t1, 2)
            if not tiled:
                probe["image"] = image
        except torch.cuda.OutOfMemoryError:
            probe[prefix + "error"] = "out of GPU memory"
        finally:
            if tiled:
                vae.disable_tiling()
        free_gpu()
    return probe


def render_column(run_dir: Path, col: Column, cases: list[Case], encoded: dict | None) -> None:
    import torch

    todo = [c for c in cases if not (run_dir / col.key / f"{c.id}.png").exists()]
    if not todo:
        print(f"{col.key}: nothing to render")
        return
    free_gpu()
    peaks = Peaks()
    print(f"{col.key}: loading {col.label}")
    t0 = time.perf_counter()
    loader = load_sdxl if col.family == "sdxl" else load_dit
    pipe = peaks.run("load", loader, col)
    load_s = time.perf_counter() - t0
    weights_gib = round(torch.cuda.memory_allocated() / GIB, 2)
    print(f"{col.key}: loaded in {load_s:.0f} s, {weights_gib:.2f} GiB on the GPU")
    sdxl_cache: dict[str, Any] = {}

    def draw(case: Case, size: tuple[int, int], phase: str) -> Any:
        if col.family == "sdxl":
            return render_sdxl(pipe, col, case, size, peaks, sdxl_cache, phase)
        return render_dit(pipe, col, case, size, peaks, encoded["embeds"], phase)

    warm = todo[0]
    draw(warm, (WIDTH, HEIGHT), "warm-up")  # first call pays for kernel selection
    seconds = []
    for n, case in enumerate(todo, 1):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        image = draw(case, (WIDTH, HEIGHT), "render")
        torch.cuda.synchronize()
        s = time.perf_counter() - t1
        seconds.append(s)
        record = {
            "column": col.key,
            "label": col.label,
            "repo": col.repo,
            "weights": col.weights,
            "case": case.id,
            "seed": case.seed,
            "steps": col.steps,
            "cfg": col.cfg,
            "sampler": col.sampler,
            "width": WIDTH,
            "height": HEIGHT,
            "prompt": prompt_for(case),
            "negative": NEGATIVE if col.cfg > 1 else "",
            "seconds": round(s, 2),
        }
        save_picture(run_dir, col.key, case, image, record)
        print(f"  [{n}/{len(todo)}] {case.id} {s:.1f} s", flush=True)

    # Group A's detail pass runs at 1344x1728; a full render at that size is its memory probe,
    # measured as Animagine's is in render_baseline. Over the target, again with VAE tiling.
    probe: dict[str, Any] = {}
    if col.key not in REFERENCES:  # a reference's memory is not the question
        probe = detail_probe(lambda phase: draw(warm, DETAIL, phase), pipe.vae, peaks)
    if "image" in probe:
        image = probe.pop("image")
        image.resize((WIDTH, HEIGHT)).save(run_dir / col.key / "_detail_probe.webp", quality=88)

    stats = {
        "label": col.label,
        "family": col.family,
        "weights": col.weights,
        "steps": col.steps,
        "cfg": col.cfg,
        "sampler": col.sampler,
        "load_seconds": round(load_s, 1),
        "weights_gib": weights_gib,
        "seconds_median": round(statistics.median(seconds), 2),
        "phases": peaks.phases,
        "detail_probe": probe,
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }
    if encoded:
        stats["encode"] = {
            "phase": encoded["phase"],
            "seconds_per_prompt": encoded["seconds_per_prompt"],
        }
    path = run_dir / "models.json"
    old = json.loads(path.read_text()).get(col.key, {}) if path.exists() else {}
    if old and len(todo) < len(cases):  # resumed: keep the larger peaks of both runs
        for name, p in old.get("phases", {}).items():
            q = stats["phases"].setdefault(name, p)
            for k in ("allocated", "reserved"):
                q[k] = max(q[k], p[k])
    update_json(path, col.key, stats)
    render_peak = peaks.phases.get("render", {}).get("reserved")
    print(f"{col.key}: {stats['seconds_median']:.1f} s per picture, peak {render_peak} GiB")
    pipe = None  # the draw closure holds the name; drop the pipeline before the next column
    free_gpu()


def render_baseline(run_dir: Path) -> None:
    """The production Animagine renderer on its own prompts: render and detail-pass peaks."""
    import torch

    from kitshot.core.prompt import build_prompt
    from kitshot.core.traits import pick_pose, picture_seed, sample_sheet
    from kitshot.datasource.kits import Kit
    from kitshot.render.renderer import AnimagineRenderer

    path = run_dir / "models.json"
    if path.exists() and BASELINE in json.loads(path.read_text()):
        print(f"{BASELINE}: already measured")
        return
    peaks = Peaks()
    t0 = time.perf_counter()
    renderer = peaks.run("load", AnimagineRenderer)
    load_s = time.perf_counter() - t0
    weights_gib = round(torch.cuda.memory_allocated() / GIB, 2)
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    seconds = []
    for uid in (930000, 930001, 930002, 930003):  # made-up IDs, not anyone in a save
        sheet = sample_sheet("realistic-test", uid, 1)
        prompt = build_prompt(sheet, "W07", kit, pick_pose(sheet.base_seed, 1))
        peaks.run("encode trip · CLIP L+G FP16", renderer.prepare, [prompt])
        seed = picture_seed(sheet.base_seed, 1, 0, 0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        image = peaks.run("render" if uid > 930000 else "warm-up", renderer.render, prompt, seed)
        torch.cuda.synchronize()
        if uid > 930000:
            seconds.append(time.perf_counter() - t1)
    t1 = time.perf_counter()
    peaks.run("detail pass", renderer.detail, image, prompt, seed, size=DETAIL)
    detail_s = time.perf_counter() - t1

    def big(phase: str) -> Any:  # like the other columns' probe: a full render at that size
        pe, pp, ne, npool, _ = renderer._embeddings(prompt)
        with torch.no_grad(), renderer._fp16_accumulation():
            return peaks.run(
                phase,
                renderer.pipe,
                prompt_embeds=pe,
                pooled_prompt_embeds=pp,
                negative_prompt_embeds=ne,
                negative_pooled_prompt_embeds=npool,
                width=DETAIL[0],
                height=DETAIL[1],
                num_inference_steps=28,
                guidance_scale=5.5,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]

    probe = detail_probe(big, renderer.pipe.vae, peaks)
    probe.pop("image", None)
    update_json(
        path,
        BASELINE,
        {
            "label": LABELS[BASELINE],
            "family": "sdxl",
            "weights": "int8",
            "steps": 28,
            "cfg": 5.5,
            "sampler": "euler a",
            "load_seconds": round(load_s, 1),
            "weights_gib": weights_gib,
            "seconds_median": round(statistics.median(seconds), 2),
            "phases": peaks.phases,
            "detail_probe": probe,
            "detail_pass_seconds": round(detail_s, 2),
            "when": time.strftime("%Y-%m-%d %H:%M"),
        },
    )
    renderer.close()
    free_gpu()
    print(f"{BASELINE}: {statistics.median(seconds):.1f} s, phases {peaks.phases}")


def available(col: Column) -> bool:
    """Whether the column's weights are in the local cache."""
    from huggingface_hub import try_to_load_from_cache

    probe = {
        "sdxl": "unet/diffusion_pytorch_model.fp16.safetensors",
        "zimage": "transformer/diffusion_pytorch_model.safetensors.index.json",
        "klein": "transformer/diffusion_pytorch_model.safetensors",
    }[col.family]
    return isinstance(try_to_load_from_cache(col.repo, probe), str)


def render(run_dir: Path, keys: list[str], case_ids: set[str]) -> None:
    import torch

    cases = [c for c in make_cases() if not case_ids or c.id in case_ids]
    gpu_guard()
    if BASELINE in keys:
        try:
            render_baseline(run_dir)
        except Exception as exc:  # noqa: BLE001 - the model columns can still run
            print(f"{BASELINE}: failed ({type(exc).__name__}: {exc}); continuing")
            free_gpu()
    cols = [COLUMNS[k] for k in keys if k in COLUMNS]
    missing = [c.key for c in cols if not available(c)]
    if missing:
        print(f"not downloaded, skipped: {', '.join(missing)}")
    cols = [c for c in cols if c.key not in missing]
    todo = {
        c.key: [x for x in cases if not (run_dir / c.key / f"{x.id}.png").exists()] for c in cols
    }
    cols = [c for c in cols if todo[c.key]]
    texts = sorted({prompt_for(x) for c in cols for x in cases})

    # One encode trip per text-encoder format, before any transformer is on the card.
    encode_peaks = Peaks()
    encoded: dict[str, dict] = {}
    for te_weights in ("int8", "bf16"):
        fams = {
            c.family
            for c in cols
            if c.family in ("zimage", "klein") and (c.weights == "bf16") == (te_weights == "bf16")
        }
        if fams:
            encoded[f"qwen3-{te_weights}"] = encode_qwen3(
                {f: texts for f in fams}, te_weights, encode_peaks
            )
    if encode_peaks.phases:
        update_json(run_dir / "models.json", "_encode", {"phases": encode_peaks.phases})

    for col in cols:
        enc = None
        if col.family in ("zimage", "klein"):
            enc = encoded[f"qwen3-{'bf16' if col.weights == 'bf16' else 'int8'}"]
        try:
            render_column(run_dir, col, cases, enc)
        except torch.cuda.OutOfMemoryError as e:
            print(f"{col.key}: out of GPU memory, skipped ({str(e).splitlines()[0]})")
            update_json(
                run_dir / "models.json", col.key, {"label": col.label, "error": "out of memory"}
            )
        free_gpu()


# --------------------------------------------------------------------------- checks


def nf4_check() -> None:
    """NF4 and INT8 error on Z-Image-Turbo's largest block layers, against BF16."""
    import torch
    from diffusers import ZImageTransformer2DModel

    from kitshot.render.quantise import Int8Linear

    nf4 = _nf4_class()
    t = ZImageTransformer2DModel.from_pretrained(
        ZTURBO, subfolder="transformer", dtype=torch.bfloat16, local_files_only=True
    )
    names = block_linears(t, NF4_GROUP)
    print(f"{len(names)} block Linear layers")
    torch.manual_seed(0)
    for name in names[:6] + names[-3:]:
        lin = t.get_submodule(name).to("cuda")
        x = torch.randn(1024, lin.in_features, device="cuda", dtype=torch.bfloat16)
        ref = lin(x).float()
        for label, q in (("int8", Int8Linear(lin, "cuda")), ("nf4", nf4(lin, "cuda"))):
            y = q(x).float()
            rel = ((y - ref).norm() / ref.norm()).item()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                q(x)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1000
            print(f"  {name:44s} {label}: output error {rel:6.2%}, {ms:5.2f} ms")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            lin(x)
        torch.cuda.synchronize()
        print(f"  {'':44s} bf16: {(time.perf_counter() - t0) / 20 * 1000:5.2f} ms")
        lin.to("cpu")


# The production check expects one anime girl (quality_checks.REQUIRED_TAGS), so it fails
# every picture here; these spike-only measures read the same tagger. Its content terms are
# reused as they are, plus the male ones it lacks. The tagger was trained on anime, so on
# photos its numbers are a rough guide, not a gate.
PHOTO_CONTENT_TAGS = ("topless male", "shirtless", "male underwear")
BEARD_TAGS = ("facial hair", "beard", "stubble", "mustache", "goatee")
REALISM_TAGS = ("realistic", "photorealistic", "photo (medium)")
ONE_MAN_FORBIDDEN = ("1girl", "multiple boys", "2boys", "multiple views", "no humans")
PROBES = ("1boy", "solo", "male focus", "realistic", "photorealistic", "photo (medium)",
          "anime coloring", "facial hair", "very short hair", "bald", "upper body", "portrait",
          "soccer uniform", "sportswear", "child", "old man")  # fmt: skip


def tag(run_dir: Path) -> None:
    from PIL import Image

    from kitshot.render.quality_checks import CONTENT_TAGS, TAG_THRESHOLD, head_top
    from kitshot.render.tagger import Eva02Tagger

    path = run_dir / "tags_eva02.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    cases = {c.id: c for c in make_cases()}
    pics = sorted(p for k in COLUMNS if (run_dir / k).is_dir() for p in (run_dir / k).glob("*.png"))
    todo = [p for p in pics if f"{p.parent.name}/{p.stem}" not in data and p.stem in cases]
    if not todo:
        print("tag: nothing new")
        return
    tagger = Eva02Tagger()
    for start in range(0, len(todo), 8):
        batch = todo[start : start + 8]
        images = [Image.open(p).convert("RGB") for p in batch]
        for p, im, res in zip(batch, images, tagger(images), strict=True):
            sheet = cases[p.stem].sheet
            flagged = [t for t in (*CONTENT_TAGS, *PHOTO_CONTENT_TAGS) if res.present(t)]
            bearded = any(res.present(t) for t in BEARD_TAGS)
            top = head_top(im)
            data[f"{p.parent.name}/{p.stem}"] = {
                "rating": res.rating,
                "ratings": {k: round(v, 4) for k, v in res.ratings.items()},
                "content_flags": flagged,
                "content_ok": res.rating == "general" and not flagged,
                "one_man": res.present("1boy")
                and res.present("solo")
                and not any(res.present(t) for t in ONE_MAN_FORBIDDEN),
                "head_top": None if top is None else round(top, 3),
                "framing_ok": top is not None and top <= 0.20,
                "beard_ok": bearded == (sheet.beard != CLEAN),
                "hair_ok": (res.present(sheet.hair_tag) if sheet.hair_tag else True),
                "realism": round(max(res.p(t) for t in REALISM_TAGS), 3),
                "probes": {t: round(res.p(t), 3) for t in PROBES},
                "tags": res.above_threshold(40),
            }
        print(f"tag: {min(start + 8, len(todo))}/{len(todo)}", flush=True)
    path.write_text(json.dumps(data, indent=1))
    tagger.close()
    print(f"tag: threshold {TAG_THRESHOLD}, {len(data)} pictures in {path.name}")


# --------------------------------------------------------------------------- report

TEMPLATE = Path(__file__).with_name("realistic_report.html")
LICENCES = {
    "animagine": "OpenRAIL++",
    "realvis": "OpenRAIL++",
    "juggernaut": "CreativeML OpenRAIL-M",
    "zturbo": "Apache-2.0",
    "klein": "Apache-2.0",
}
THUMB = (448, 576)


def _luma(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _fidelity(a: Path, b: Path) -> tuple[float, float]:
    """PSNR (dB, RGB) and SSIM (luma, 11x11 Gaussian, sigma 1.5) of a against b."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    g = (g / g.sum()).float()
    kx, ky = g.view(1, 1, 1, 11), g.view(1, 1, 11, 1)

    def blur(t: Any) -> Any:
        return F.conv2d(F.conv2d(t, kx), ky)

    u = torch.from_numpy(_luma(a))[None, None]
    v = torch.from_numpy(_luma(b))[None, None]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu, mv = blur(u), blur(v)
    su, sv, suv = blur(u * u) - mu**2, blur(v * v) - mv**2, blur(u * v) - mu * mv
    ssim = ((2 * mu * mv + c1) * (2 * suv + c2)) / ((mu**2 + mv**2 + c1) * (su + sv + c2))
    return round(float(psnr), 2), round(float(ssim.mean()), 4)


def _thumb(src: Path, dst: Path) -> None:
    from PIL import Image

    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").resize(THUMB, Image.Resampling.LANCZOS).save(
        dst, "WEBP", quality=86, method=5
    )


def _pct(values: list[bool]) -> float | None:
    return round(100 * sum(values) / len(values), 1) if values else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def describe(case: Case) -> str:
    s = case.sheet
    return (
        f"{case.age} · {s.look} · {s.skin} skin · {s.hair} · {s.beard} · {s.eyes} eyes · "
        f"{case.kit.replace('_', ' ')} kit · {case.backdrop} backdrop"
    )


def report(run_dir: Path) -> Path:
    cases = make_cases()
    models_path = run_dir / "models.json"
    stats = json.loads(models_path.read_text()) if models_path.exists() else {}
    tags_path = run_dir / "tags_eva02.json"
    tags = json.loads(tags_path.read_text()) if tags_path.exists() else {}
    renders: dict[str, dict] = {}
    if (run_dir / "renders.jsonl").exists():
        for line in (run_dir / "renders.jsonl").read_text().splitlines():
            r = json.loads(line)
            renders[f"{r['column']}/{r['case']}"] = r
    fid_path = run_dir / "fidelity.json"
    fid = json.loads(fid_path.read_text()) if fid_path.exists() else {}
    encode_phases = stats.get("_encode", {}).get("phases", {})

    pics: dict[str, dict] = {}
    columns = []
    for key, col in COLUMNS.items():
        present = [c for c in cases if (run_dir / key / f"{c.id}.png").exists()]
        if not present and key not in stats:
            continue
        rows = []
        for case in present:
            ck = f"{key}/{case.id}"
            src = run_dir / key / f"{case.id}.png"
            _thumb(src, run_dir / "_thumbs" / key / f"{case.id}.webp")
            pic: dict[str, Any] = {
                "t": f"_thumbs/{key}/{case.id}.webp",
                "f": f"{key}/{case.id}.png",
                "s": renders.get(ck, {}).get("seconds"),
            }
            info = tags.get(ck)
            if info:
                pic.update(
                    ok=info["content_ok"], man=info["one_man"], fr=info["framing_ok"],
                    bd=info["beard_ok"], hr=info["hair_ok"], re=info["realism"],
                    rating=info["rating"], flags=info["content_flags"],
                    top=info["head_top"], tags=list(info["tags"])[:18],
                )  # fmt: skip
            ref = run_dir / col.reference / f"{case.id}.png" if col.reference else None
            if ref is not None and ref.exists():
                stamp = f"{src.stat().st_mtime_ns}:{ref.stat().st_mtime_ns}"
                if fid.get(ck, {}).get("stamp") != stamp:
                    psnr, ssim = _fidelity(src, ref)
                    fid[ck] = {"stamp": stamp, "psnr": psnr, "ssim": ssim}
                pic["psnr"], pic["ssim"] = fid[ck]["psnr"], fid[ck]["ssim"]
            pics[ck] = pic
            rows.append(pic)
        st = stats.get(key, {})
        phases = st.get("phases", {})
        enc_phase = st.get("encode", {}).get("phase") or "encode trip · CLIP L+G FP16"
        enc = (encode_phases.get(enc_phase) or phases.get(enc_phase) or {}).get("reserved")
        probe = run_dir / key / "_detail_probe.webp"
        tagged = [r for r in rows if "ok" in r]
        columns.append(
            {
                "key": key,
                "label": col.label,
                "model": key.partition("~")[0],
                "family": col.family,
                "weights": col.weights,
                "steps": col.steps,
                "cfg": col.cfg,
                "sampler": col.sampler,
                "repo": col.repo,
                "licence": LICENCES.get(key.partition("~")[0], ""),
                "reference": col.reference,
                "is_reference": key in REFERENCES,
                "error": st.get("error"),
                "pictures": len(rows),
                "weights_gib": st.get("weights_gib"),
                "load_s": st.get("load_seconds"),
                "s_median": st.get("seconds_median"),
                "render_peak": phases.get("render", {}).get("reserved"),
                "render_alloc": phases.get("render", {}).get("allocated"),
                "detail_peak": phases.get("detail-size render", {}).get("reserved"),
                "detail_s": st.get("detail_probe", {}).get("seconds"),
                "detail_error": st.get("detail_probe", {}).get("error"),
                "tiled_peak": phases.get("detail-size render, VAE tiled", {}).get("reserved"),
                "tiled_s": st.get("detail_probe", {}).get("tiled_seconds"),
                "probe": f"{key}/_detail_probe.webp" if probe.exists() else None,
                "encode_phase": enc_phase if enc is not None else None,
                "encode_peak": enc,
                "encode_s": st.get("encode", {}).get("seconds_per_prompt"),
                "metrics": {
                    "tagged": len(tagged),
                    "content_ok": _pct([r["ok"] for r in tagged]),
                    "one_man": _pct([r["man"] for r in tagged]),
                    "framing": _pct([r["fr"] for r in tagged]),
                    "beard": _pct([r["bd"] for r in tagged]),
                    "hair": _pct([r["hr"] for r in tagged]),
                    "realism": _mean([r["re"] for r in tagged]),
                    "psnr": _mean([r["psnr"] for r in rows if "psnr" in r]),
                    "ssim": _mean([r["ssim"] for r in rows if "ssim" in r]),
                },
            }
        )
    fid_path.write_text(json.dumps(fid, indent=1))

    base = stats.get(BASELINE)
    baseline = None
    if base:
        ph = base.get("phases", {})
        baseline = {
            "key": BASELINE,
            "label": LABELS[BASELINE],
            "licence": LICENCES[BASELINE],
            "weights_gib": base.get("weights_gib"),
            "load_s": base.get("load_seconds"),
            "s_median": base.get("seconds_median"),
            "render_peak": ph.get("render", {}).get("reserved"),
            "detail_peak": ph.get("detail-size render", {}).get("reserved"),
            "detail_s": base.get("detail_probe", {}).get("seconds"),
            "tiled_peak": ph.get("detail-size render, VAE tiled", {}).get("reserved"),
            "tiled_s": base.get("detail_probe", {}).get("tiled_seconds"),
            "detail_pass_peak": ph.get("detail pass", {}).get("reserved"),
            "detail_pass_s": base.get("detail_pass_seconds"),
            "encode_peak": ph.get("encode trip · CLIP L+G FP16", {}).get("reserved"),
        }
    data = {
        "run": run_dir.name,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "size": [WIDTH, HEIGHT],
        "detail": list(DETAIL),
        "target": TARGET_GIB,
        "baseline": baseline,
        "columns": columns,
        "cases": [
            {
                "id": c.id,
                "kind": "ladder" if c.id.startswith("L") else "sheet",
                "sheet": c.sheet.id,
                "age": c.age,
                "desc": describe(c),
                "seed": c.seed,
                "prompt": prompt_for(c),
            }
            for c in cases
        ],
        "negative": NEGATIVE,
        "pics": pics,
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", blob)
    path = run_dir / "index.html"
    path.write_text(page)
    return path


# ------------------------------------------------------------------------------ CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--run", default="realistic", help="run name under data/bakeoff/")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prompts", help="print the cases and prompts; no GPU")
    sub.add_parser("nf4-check", help="NF4 and INT8 error and speed on single layers")
    for name in ("render", "all"):
        p = sub.add_parser(name)
        p.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
        p.add_argument("--cases", default="", help="comma-separated case IDs; default all")
    sub.add_parser("tag")
    sub.add_parser("report")
    args = parser.parse_args()

    run_dir = RUN_DIR.parent / args.run
    if args.command == "prompts":
        print(f"negative (CFG models): {NEGATIVE}\n")
        for case in make_cases():
            print(f"{case.id} seed={case.seed}\n  {prompt_for(case)}")
        return
    if args.command == "nf4-check":
        nf4_check()
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command in ("render", "all"):
        keys = [k for k in args.columns.split(",") if k]
        unknown = [k for k in keys if k != BASELINE and k not in COLUMNS]
        if unknown:
            sys.exit(f"unknown column(s) {unknown}; choose from {DEFAULT_COLUMNS}")
        render(run_dir, keys, {c for c in args.cases.split(",") if c})
    if args.command in ("tag", "all"):
        tag(run_dir)
    if args.command in ("report", "all"):
        print("report:", report(run_dir))


if __name__ == "__main__":
    main()
