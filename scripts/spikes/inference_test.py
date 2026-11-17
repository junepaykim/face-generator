"""Side test: faster inference for Animagine on this card (FP8, INT8, fp16 accumulation, compile).

Asked for by the user on 2026-11-11. The model, prompt, sampler and sizes stay as in
production; only how the UNet computes changes. Variants (joined with "+"):

  fp16acc  cuBLAS accumulates fp16 matmuls in fp16 (torch flag allow_fp16_accumulation).
           GeForce cards run fp16 tensor cores at half rate with fp32 accumulation.
  compile  torch.compile on the UNet (inductor, static shapes).
  fp8      the 700 Linear layers inside the UNet's transformer blocks in FP8 (e4m3): one
           scale per output channel for weights, one per token for activations, through
           torch._scaled_mm's per-tensor kernel with unit scales (as the Z-Image test).
  int8     the same layers in INT8 through torch._int_mm, same scaling.
  bf16     the UNet in bf16 instead of fp16: a quality yardstick (a precision change nobody
           would call harmful), not a speed-up.
  sage     SageAttention 1.0.6 (INT8 Q and K, fp16 P and V; Triton) in place of SDPA flash
           for every attention call with head dim 64/96/128 (all 140 in the UNet; the VAE's
           single 512-wide head stays on SDPA). Not a project dependency; run with
           `uv run --with sageattention==1.0.6` (approved by the user for this test).
  w8       weight-only INT8 as production now has it (facegen.quantise): the same 700
           layers stored as INT8 per output channel and rebuilt in fp16 per call; no compile.

The baseline is AnimagineRenderer with fp16_accumulation and int8_weights off, which draws
exactly as production did before 2026-11-11.

Timing. Other programs on Windows share the GPU and its speed drifts by tens of percent within
minutes, so each variant runs in its own process with the plain fp16 UNet loaded alongside,
and every picture is drawn by both, alternating which goes first. Speed-ups are the median of
the per-picture ratios.

Ten made-up players (unique IDs 910001-910010, not anyone in a save) through the real code:
traits, kits, prompt and AnimagineRenderer, at 896x1152 plus group A's detail pass. Pictures,
results and the page go to data/bakeoff/inference/ and stay on this PC (AGENTS.md rule 6).

    uv run python scripts/spikes/inference_test.py render fp16       # the baseline pictures
    uv run python scripts/spikes/inference_test.py render fp8+compile
    uv run --with sageattention==1.0.6 python scripts/spikes/inference_test.py render fp16acc+sage
    uv run python scripts/spikes/inference_test.py render w8+fp16acc
    uv run python scripts/spikes/inference_test.py report [VARIANT ...]   # page: these only
"""

from __future__ import annotations

import gc
import html
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from facegen import qc
from facegen.core.prompt import Prompt, build_prompt
from facegen.core.traits import pick_pose, picture_seed, sample_sheet
from facegen.datasource.kits import Kit, load_kits
from facegen.persistence.settings import load_settings
from facegen.render.gpu import prepare_environment
from facegen.render.picture import face_box

prepare_environment()

import torch  # after prepare_environment

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "bakeoff" / "inference"
BASE = "fp16"
FLAGS = {"fp16", "fp16acc", "compile", "fp8", "int8", "bf16", "sage", "w8"}
# (made-up unique ID, age wording, club ID, goalkeeper, at my club)
CASES = [
    (910001, "W01", 636, False, True),  # Colchester stripes, youngest wording
    (910002, "W03", 636, True, True),  # Colchester goalkeeper
    (910003, "W05", 658, False, False),  # Grimsby black and white stripes
    (910004, "W07", 620, False, False),  # Bristol Rovers quarters
    (910005, "W09", 605, False, False),  # Barnet amber, black collar
    (910006, "W10", None, False, False),  # fallback kit
    (910007, "W02", 109210, False, False),  # Accrington red
    (910008, "W04", 631, False, False),  # Cheltenham
    (910009, "W06", 641, True, False),  # Crewe goalkeeper
    (910010, "W08", 651, False, False),  # Exeter
]
FP8_MAX = 448.0


@dataclass(frozen=True)
class Case:
    uid: int
    wording: str
    kit: Kit
    pose: str
    prompt: Prompt
    expect: qc.Expectation
    seed: int


def make_cases() -> list[Case]:
    settings = load_settings()
    kits = load_kits()
    out = []
    for uid, wording, club, gk, mine in CASES:
        sheet = sample_sheet("inference-test", uid, 1, height_cm=180, strength=10)
        kit, _ = kits.kit_for(club, gk)
        pose = pick_pose(sheet.base_seed, 1)
        my_club = settings.my_club if mine else None
        prompt = build_prompt(sheet, wording, kit, pose, my_club)
        expect = qc.expectation(sheet, kit, my_club)
        out.append(
            Case(uid, wording, kit, pose, prompt, expect, picture_seed(sheet.base_seed, 1, 0, 0))
        )
    return out


# ------------------------------------------------------------------- quantised Linear layers


class QLinear(torch.nn.Module):
    """nn.Linear in FP8 or INT8: weights scaled per output channel, activations per token.

    The quantised operands go through the fast per-tensor kernel with unit scales
    (torch._scaled_mm for FP8, torch._int_mm for INT8); the row and channel scales are
    applied to the product afterwards, which is exact. Run inside torch.compile so the
    quantise and rescale steps fuse with their neighbours; eagerly they cost most of the gain
    (UNet-level test: 158 ms per call against 184 for fp16 and 145 for fp16acc).
    """

    def __init__(self, linear: torch.nn.Linear, kind: str) -> None:
        super().__init__()
        w = linear.weight.detach().to("cuda", torch.float32)
        self.qmax = FP8_MAX if kind == "fp8" else 127.0
        scale = (w.abs().amax(dim=1) / self.qmax).clamp(min=1e-12)
        wq = w / scale[:, None]
        wq = (
            wq.to(torch.float8_e4m3fn)
            if kind == "fp8"
            else wq.round().clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("wq", wq.contiguous())
        self.register_buffer("wscale", scale)
        self.register_buffer("one", torch.ones((), device="cuda"))
        bias = linear.bias
        self.bias = None if bias is None else torch.nn.Parameter(bias.detach().to("cuda"), False)
        self.kind = kind
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        s_row = (x2.abs().amax(dim=1, keepdim=True).float() / self.qmax).clamp(min=1e-12)
        if self.kind == "fp8":
            xq = (x2.float() / s_row).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            # bf16 out: the unit-scale product can exceed fp16's range.
            y = torch._scaled_mm(
                xq, self.wq.t(), scale_a=self.one, scale_b=self.one, out_dtype=torch.bfloat16
            )
        else:
            xq = (x2.float() / s_row).round().clamp(-127, 127).to(torch.int8)
            y = torch._int_mm(xq, self.wq.t())
        y = y.float() * s_row * self.wscale
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*shape[:-1], self.out_features)


def quantised_unet(kind: str) -> tuple[torch.nn.Module, int]:
    """A second UNet with its transformer-block Linear layers in FP8 or INT8, on the GPU."""
    from diffusers import UNet2DConditionModel

    from facegen.pipeline.worker import MODEL_REPO

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_REPO, subfolder="unet", dtype=torch.float16, use_safetensors=True
    )  # on the CPU; each layer is quantised straight onto the GPU
    targets = [
        name
        for name, m in unet.named_modules()
        if isinstance(m, torch.nn.Linear) and "transformer_blocks" in name
    ]
    for name in targets:
        parent, _, child = name.rpartition(".")
        holder = unet.get_submodule(parent)
        setattr(holder, child, QLinear(getattr(holder, child), kind))
    unet.to("cuda")
    unet.eval()
    return unet, len(targets)


def weight_only_unet() -> tuple[torch.nn.Module, int]:
    """A second UNet quantised exactly as AnimagineRenderer does by default."""
    from diffusers import UNet2DConditionModel

    from facegen.pipeline.worker import INT8_SCOPE, MODEL_REPO
    from facegen.render.quantise import quantise_linears

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_REPO, subfolder="unet", dtype=torch.float16, use_safetensors=True
    )
    n = quantise_linears(unet, INT8_SCOPE, "cuda")
    return unet.to("cuda").eval(), n


def bf16_unet() -> torch.nn.Module:
    """A second UNet in bf16, casting the pipeline's fp16 inputs and its output."""
    from diffusers import UNet2DConditionModel

    from facegen.pipeline.worker import MODEL_REPO

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_REPO, subfolder="unet", dtype=torch.bfloat16, use_safetensors=True
    ).to("cuda")

    def cast(v: Any) -> Any:
        if torch.is_tensor(v) and v.dtype == torch.float16:
            return v.to(torch.bfloat16)
        if isinstance(v, dict):
            return {k: cast(x) for k, x in v.items()}
        return v

    def pre(_m: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        return tuple(cast(a) for a in args), {k: cast(v) for k, v in kwargs.items()}

    def post(_m: Any, _args: tuple, _kwargs: dict, out: Any) -> Any:
        return tuple(o.to(torch.float16) if torch.is_tensor(o) else o for o in out)

    unet.register_forward_pre_hook(pre, with_kwargs=True)
    unet.register_forward_hook(post, with_kwargs=True)
    return unet


def module_bytes(m: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in [*m.parameters(), *m.buffers()])


SAGE = {"on": False, "sage_calls": 0, "sdpa_calls": 0}
"""While "on", attention goes to SageAttention where it can; counts what went where."""


def install_sage() -> None:
    """Route F.scaled_dot_product_attention through SageAttention while SAGE["on"] is set.

    diffusers' attention processors look the function up on torch.nn.functional at each
    call, so patching the module attribute reaches every UNet attention layer.
    """
    import torch.nn.functional as F
    from sageattention import sageattn

    sdpa = F.scaled_dot_product_attention

    def attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if not SAGE["on"]:
            return sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, **kw)
        if attn_mask is None and dropout_p == 0.0 and q.shape[-1] in (64, 96, 128):
            SAGE["sage_calls"] += 1
            return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
        SAGE["sdpa_calls"] += 1
        return sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = attention


# ------------------------------------------------------------------------------ rendering


def gpu_state() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()  # fmt: skip
    except OSError:
        return "?"


def render(name: str) -> None:
    flags = set(name.split("+"))
    if not flags <= FLAGS:
        sys.exit(f"unknown flags {flags - FLAGS}; known: {sorted(FLAGS)}")
    if name != BASE and not (RUN / BASE / "timing.json").exists():
        sys.exit(f"render the baseline first: {sys.argv[0]} render {BASE}")
    from PIL import Image

    from facegen.pipeline.worker import AnimagineRenderer

    out_dir = RUN / name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = RUN / BASE
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU before load: {gpu_state()}")

    # The baseline: production as it drew before fp16 accumulation and INT8 weights. The
    # variant's settings are applied by this script, per call.
    renderer = AnimagineRenderer(fp16_accumulation=False, int8_weights=False)
    cases = make_cases()
    # Every prompt encoded up front; the text encoders then wait in system memory so the
    # second UNet fits beside the first. Production keeps them on the card (1.6 GiB).
    embeds = {(c.prompt.positive, c.prompt.negative): renderer._embeddings(c.prompt) for c in cases}
    renderer._embeddings = lambda p: embeds[(p.positive, p.negative)]  # type: ignore[method-assign]
    te_bytes = module_bytes(renderer.pipe.text_encoder) + module_bytes(renderer.pipe.text_encoder_2)
    renderer.pipe.text_encoder.to("cpu")
    renderer.pipe.text_encoder_2.to("cpu")
    chunks = sorted({e[4] for e in embeds.values()})

    base_unet = renderer.pipe.unet
    var_unet: torch.nn.Module | None = None
    separate = 0  # bytes of the second UNet, when the variant has one
    n_quantised = 0
    if name != BASE:
        if "fp8" in flags or "int8" in flags:
            var_unet, n_quantised = quantised_unet("fp8" if "fp8" in flags else "int8")
            separate = module_bytes(var_unet)
        elif "bf16" in flags:
            var_unet = bf16_unet()
            separate = module_bytes(var_unet)
        elif "w8" in flags:
            var_unet, n_quantised = weight_only_unet()
            separate = module_bytes(var_unet)
        else:
            var_unet = base_unet
        if "compile" in flags:
            var_unet = torch.compile(var_unet, dynamic=False)
        if "sage" in flags:
            install_sage()
    gc.collect()
    torch.cuda.empty_cache()
    var_weights = separate or module_bytes(base_unet)
    print(f"variant {name}: UNet weights {var_weights / 2**30:.2f} GiB, {n_quantised} quantised")

    peaks = {"base": 0, "var": 0}

    def run(which: str, fn: Any, *args: Any) -> tuple[Any, float]:
        unet = base_unet if which == "base" else var_unet
        for pipe in (renderer.pipe, renderer.img2img):
            object.__setattr__(pipe, "unet", unet)
        torch.backends.cuda.matmul.allow_fp16_accumulation = which == "var" and "fp16acc" in flags
        SAGE["on"] = which == "var" and "sage" in flags
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        peaks[which] = max(peaks[which], torch.cuda.max_memory_allocated())
        return out, secs

    sides = ["base"] if var_unet is None else ["base", "var"]

    # Warm-up: one render and one detail pass per prompt length in chunks, so compiling for
    # every shape (two picture sizes, each chunk count) happens before the timed pictures.
    warm: dict[str, list[float]] = {"render": [], "detail": []}
    for n in chunks:
        c = next(c for c in cases if embeds[(c.prompt.positive, c.prompt.negative)][4] == n)
        for which in sides:
            img, s1 = run(which, renderer.render, c.prompt, c.seed + 7777)
            _, s2 = run(which, renderer.detail, img, c.prompt, c.seed + 7777)
            if which == sides[-1]:
                warm["render"].append(round(s1, 1))
                warm["detail"].append(round(s2, 1))
    print(f"warm-up (chunks {chunks}): {warm}")

    times: dict[str, dict[str, list[float]]] = {s: {"render": [], "detail": []} for s in sides}
    repeat: list[float] = []  # the baseline re-drawn in this process against the stored one
    for i, c in enumerate(cases):
        order = sides if i % 2 == 0 else sides[::-1]
        pics: dict[str, Any] = {}
        for which in order:
            pics[which], secs = run(which, renderer.render, c.prompt, c.seed)
            times[which]["render"].append(round(secs, 3))
        details: dict[str, Any] = {}
        for which in order:
            details[which], secs = run(which, renderer.detail, pics[which], c.prompt, c.seed)
            times[which]["detail"].append(round(secs, 3))
        for which in sides:
            folder = base_dir if which == "base" else out_dir
            for kind, image in (("render", pics[which]), ("detail", details[which])):
                path = folder / f"{c.uid}-{kind}.png"
                if which == "base" and path.exists():
                    if kind == "render":
                        repeat.append(psnr(Image.open(path), image))
                    continue
                image.save(path)
        line = "  ".join(
            f"{w} {times[w]['render'][-1]:.2f}+{times[w]['detail'][-1]:.2f} s" for w in sides
        )
        print(f"{c.uid}: {line}")

    # Peak allocated during each side's calls, as production would hold it: the other side's
    # separate UNet left out and the text encoders put back.
    peak_gib = {"base": round((peaks["base"] - separate + te_bytes) / 2**30, 2)}
    if var_unet is not None:
        other = module_bytes(base_unet) if separate else 0
        peak_gib["var"] = round((peaks["var"] - other + te_bytes) / 2**30, 2)
    result = {
        "variant": name,
        "flags": sorted(flags),
        "quantised_layers": n_quantised,
        "unet_weights_gib": round(var_weights / 2**30, 2),
        "warmup": warm,
        "times": times,
        "peak_alloc_gib": peak_gib,
        "baseline_repeat_psnr": [round(p, 1) for p in repeat],
        "attention_calls": {k: SAGE[k] for k in ("sage_calls", "sdpa_calls")},
        "gpu_after": gpu_state(),
        "torch": torch.__version__,
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }
    folder = base_dir if name == BASE else out_dir
    (folder / "timing.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("unet_weights_gib", "peak_alloc_gib", "warmup")}))


# --------------------------------------------------------------------------------- report


def psnr(a: Any, b: Any) -> float:
    import numpy as np

    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(a: Any, b: Any) -> float:
    """Mean SSIM on luminance, 11x11 Gaussian window (sigma 1.5), as in Wang et al. 2004."""
    import numpy as np
    import torch.nn.functional as F

    def lum(im: Any) -> torch.Tensor:
        return torch.from_numpy(np.asarray(im.convert("L"), dtype=np.float32))[None, None]

    x, y = lum(a), lum(b)
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None]

    def f(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, k)

    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return float(m.mean())


def report(shown: list[str]) -> None:
    """Score the variants and write the page; with names, the page and table show only them."""
    from PIL import Image

    cases = make_cases()
    variants = sorted(
        (p.name for p in RUN.iterdir() if p.is_dir() and p.name != BASE and (p / "timing.json").exists()),
        key=lambda n: (n != "bf16", n),
    )  # fmt: skip
    if shown:
        missing = set(shown) - set(variants)
        if missing:
            sys.exit(f"not rendered: {sorted(missing)}")
        variants = shown
    tagger = qc.Eva02Tagger()
    base_timing = json.loads((RUN / BASE / "timing.json").read_text())

    def checks(name: str) -> dict[str, Any]:
        """Tagger check and fidelity against the baseline, per picture; cached."""
        path = RUN / name / "qc.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        for c in cases:
            for kind in ("render", "detail"):
                key = f"{c.uid}-{kind}"
                if key in data:
                    continue
                im = Image.open(RUN / name / f"{key}.png").convert("RGB")
                tags = tagger([im])[0]
                res = qc.check(tags, c.expect, qc.head_top(im))
                entry = {"passed": res.passed, "failures": list(res.failures),
                         "p_general": round(res.p_general, 4)}  # fmt: skip
                if name != BASE:
                    ref = Image.open(RUN / BASE / f"{key}.png").convert("RGB")
                    entry["psnr"] = round(psnr(ref, im), 2)
                    entry["ssim"] = round(ssim(ref, im), 4)
                data[key] = entry
        path.write_text(json.dumps(data, indent=1))
        return data

    base_qc = checks(BASE)
    # Face crops of the detail pass, all from the baseline's box so they line up.
    boxes = {}
    for c in cases:
        ref = Image.open(RUN / BASE / f"{c.uid}-detail.png").convert("RGB")
        boxes[c.uid] = face_box(ref)
    for name in [BASE, *variants]:
        for c in cases:
            crop = RUN / name / f"{c.uid}-face.png"
            if not crop.exists():
                im = Image.open(RUN / name / f"{c.uid}-detail.png").convert("RGB")
                im.crop(boxes[c.uid]).resize((384, 384), Image.Resampling.LANCZOS).save(crop)
            thumb = RUN / name / f"{c.uid}-thumb.jpg"
            if not thumb.exists():
                im = Image.open(RUN / name / f"{c.uid}-render.png").convert("RGB")
                im.resize((336, 432), Image.Resampling.LANCZOS).save(thumb, quality=90)

    rows = []
    per_variant: dict[str, dict[str, Any]] = {}
    for name in variants:
        t = json.loads((RUN / name / "timing.json").read_text())
        q = checks(name)
        tb, tv = t["times"]["base"], t["times"]["var"]
        r_ratio = statistics.median(b / v for b, v in zip(tb["render"], tv["render"]))
        d_ratio = statistics.median(b / v for b, v in zip(tb["detail"], tv["detail"]))
        # Group A player: three candidates and one detail pass (the tagger is ~0.03 s).
        a_base = [3 * r + d for r, d in zip(tb["render"], tb["detail"])]
        a_var = [3 * r + d for r, d in zip(tv["render"], tv["detail"])]
        a_ratio = statistics.median(b / v for b, v in zip(a_base, a_var))
        rq = [q[f"{c.uid}-render"] for c in cases]
        dq = [q[f"{c.uid}-detail"] for c in cases]
        s = {
            "render_s": statistics.median(tv["render"]),
            "render_base_s": statistics.median(tb["render"]),
            "render_x": r_ratio,
            "detail_s": statistics.median(tv["detail"]),
            "detail_base_s": statistics.median(tb["detail"]),
            "detail_x": d_ratio,
            "group_a_x": a_ratio,
            "group_a_s": statistics.median(a_var),
            "group_a_base_s": statistics.median(a_base),
            "psnr_render": statistics.mean(e["psnr"] for e in rq),
            "ssim_render": statistics.mean(e["ssim"] for e in rq),
            "psnr_detail": statistics.mean(e["psnr"] for e in dq),
            "ssim_detail": statistics.mean(e["ssim"] for e in dq),
            "passed": sum(e["passed"] for e in rq + dq),
            "checked": len(rq + dq),
            "weights": t["unet_weights_gib"],
            "peak": t["peak_alloc_gib"]["var"],
            "peak_base": t["peak_alloc_gib"]["base"],
            "warmup_s": sum(t["warmup"]["render"]) + sum(t["warmup"]["detail"]),
            "repeat_psnr": t["baseline_repeat_psnr"],
        }
        per_variant[name] = s
    base_passed = sum(
        base_qc[f"{c.uid}-{k}"]["passed"] for c in cases for k in ("render", "detail")
    )
    summary = {"variants": per_variant, "baseline_passed": base_passed,
               "baseline_timing": base_timing["times"]}  # fmt: skip
    (RUN / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"baseline fp16: check {base_passed}/{2 * len(cases)}")
    print(
        "| variant | render x | detail x | group A x | PSNR r/d dB | SSIM r/d | check | UNet GiB | peak GiB | warm-up s |"
    )
    for name, s in per_variant.items():
        print(
            f"| {name} | {s['render_x']:.2f} | {s['detail_x']:.2f} | {s['group_a_x']:.2f} | "
            f"{s['psnr_render']:.1f}/{s['psnr_detail']:.1f} | {s['ssim_render']:.3f}/{s['ssim_detail']:.3f} | "
            f"{s['passed']}/{s['checked']} | {s['weights']} | {s['peak']} (base {s['peak_base']}) | "
            f"{s['warmup_s']:.0f} |"
        )  # fmt: skip

    for name, s in per_variant.items():
        # bf16 is a quality yardstick; its run held two full UNets and spilled out of VRAM.
        speed = (
            '<td colspan="5">— (quality yardstick only; not timed fairly)</td>'
            if name == "bf16"
            else f"<td>{s['render_x']:.2f}×</td>"
            f"<td>{s['render_s']:.2f} / {s['render_base_s']:.2f}</td>"
            f"<td>{s['detail_x']:.2f}×</td><td>{s['group_a_x']:.2f}×</td>"
            f"<td>{s['group_a_s']:.1f} / {s['group_a_base_s']:.1f}</td>"
        )
        rows.append(
            f"<tr><td><b>{html.escape(name)}</b></td>{speed}"
            f"<td>{s['psnr_render']:.1f} / {s['psnr_detail']:.1f}</td>"
            f"<td>{s['ssim_render']:.3f} / {s['ssim_detail']:.3f}</td>"
            f"<td>{s['passed']}/{s['checked']}</td><td>{s['weights']:.2f}</td>"
            f"<td>{s['peak']:.1f}</td>"
            f"<td>{'—' if name == 'bf16' else round(s['warmup_s'])}</td></tr>"
        )
    cols = [BASE, *variants]
    grid = []
    for c in cases:
        cells = []
        for name in cols:
            q = base_qc if name == BASE else json.loads((RUN / name / "qc.json").read_text())
            r, d = q[f"{c.uid}-render"], q[f"{c.uid}-detail"]
            ok = r["passed"] and d["passed"]
            badge = f'<span class="b {"ok" if ok else "bad"}">{"passed" if ok else "failed"}</span>'
            fid = (
                ""
                if name == BASE
                else f'<span class="b">{r["psnr"]:.1f} dB · {r["ssim"]:.3f}</span>'
            )
            cells.append(
                f'<td><a href="{name}/{c.uid}-render.png"><img class="t" src="{name}/{c.uid}-thumb.jpg" loading="lazy"></a>'
                f'<a href="{name}/{c.uid}-detail.png"><img class="f" src="{name}/{c.uid}-face.png" loading="lazy"></a>'
                f"<div>{badge} {fid}</div></td>"
            )  # fmt: skip
        desc = html.escape(f"{c.uid}: {c.wording}, {c.kit.describe()}, {c.pose}")
        grid.append(f"<tr><td class='d'>{desc}</td>{''.join(cells)}</tr>")
    when = base_timing["when"]
    yardstick = "bf16 is the yardstick for a harmless precision change"
    bf16_qc = RUN / "bf16" / "qc.json"
    if "bf16" not in variants and bf16_qc.exists():
        q = json.loads(bf16_qc.read_text())
        r = statistics.mean(q[f"{c.uid}-render"]["psnr"] for c in cases)
        d = statistics.mean(q[f"{c.uid}-detail"]["psnr"] for c in cases)
        yardstick = (
            f"for scale, the UNet in bf16 instead of fp16, a change nobody would call harmful,"
            f" gives {r:.1f} / {d:.1f} dB"
        )
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Inference speed-ups</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2;
  --line:#2c323c; --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px 24px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, sans-serif; }}
table {{ border-collapse:collapse; margin-bottom:24px; }}
td, th {{ padding:4px 6px; vertical-align:top; text-align:left; border-bottom:1px solid var(--line); }}
.s td, .s th {{ text-align:right; font-variant-numeric:tabular-nums; }}
.s td:first-child, .s th:first-child {{ text-align:left; }}
img.t {{ width:168px; height:216px; display:block; }}
img.f {{ width:168px; height:168px; display:block; margin-top:2px; }}
.d {{ width:150px; color:var(--muted); font-size:12px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; background:var(--line); color:var(--muted); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
.note {{ color:var(--muted); max-width:95ch; }}
</style></head><body>
<h1>Inference speed-ups for Animagine XL 4.0</h1>
<p class="note">RTX 4080, PyTorch {base_timing["torch"]}, {when}. 896×1152, 28 steps Euler a, CFG 5.5, then
group A's detail pass (×1.5, img2img 0.35). Ten made-up players. Each variant was timed in its own
process against the plain fp16 UNet loaded beside it, alternating picture by picture, because other
programs on Windows were using the GPU; "×" is the median per-picture speed-up, seconds are
variant / baseline in that run. PSNR and SSIM compare each picture with the fp16 baseline drawn from
the same seed ({yardstick}). Peak is the pipeline's
allocated memory as production holds it (text encoders on the card, tagger not counted). Warm-up is
the first render and detail pass per shape, including compiling. Baseline check: {base_passed}/{2 * len(cases)}.
Local file only.</p>
<table class="s"><tr><th>Variant</th><th>Render</th><th>s</th><th>Detail</th><th>Group A player</th><th>s</th>
<th>PSNR dB render / detail</th><th>SSIM render / detail</th><th>Check</th><th>UNet GiB</th><th>Peak GiB</th><th>Warm-up s</th></tr>
{"".join(rows)}</table>
<p class="note">Each cell: the 896×1152 render (click for full size), below it the face from the
detail pass at 1344×1728 (click for full size), cropped from the same box in every column.</p>
<table><tr><th></th>{"".join(f"<th>{html.escape(c)}</th>" for c in cols)}</tr>{"".join(grid)}</table>
</body></html>"""
    path = RUN / "index.html"
    path.write_text(text)
    print("page:", path)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["render", name]:
            render(name)
        case ["report", *shown]:
            report(shown)
        case _:
            sys.exit(__doc__)
