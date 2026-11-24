"""Bake-off spike (run "wai-anima"): WAI-illustrious-SDXL v17 and Anima-2.9B against Animagine.

The two models a web answer ranked first and second for this job (2026-11-11). Renders the
round 1/2 cases (20 character sheets drawn twice, plus the two-sheet age ladder) on each,
re-renders the same cases on Animagine (the chosen model), checks every picture with the
EVA02 tagger as the other runs do, and writes data/bakeoff/<run>/index.html.

Each model is run the way its own card says, not with Animagine's settings:

  wai          WAI-illustrious-SDXL v17 (Civitai's file; the Hugging Face copy used here has
               the same SHA256). Card: Euler a, 15-30 steps, CFG 5-7, quality tags
               "masterpiece, best quality, amazing quality", rating tag "general", a short
               negative. 28 steps and CFG 5.5 sit inside the card and match Animagine.
  wai~tall     The same at 1024x1344, the card's example size ("use size larger than
               1024x1024"); ten cases.
  wai~dmd2     WAI with the DMD2 4-step LoRA fused in, as its README: LCM scheduler,
               timesteps 999/749/499/249, no CFG. Without CFG the negative prompt has no
               effect, so the content terms in it do nothing here (measured, not adopted).
  wai~dmd2-8   The same LoRA at 8 LCM steps, as the web answer suggested; ten cases.
  anima29      Anima-2.9B preview v1 (a community layer expansion of Anima-Base) in ComfyUI,
               its card's supported path. Card: Euler + sgm_uniform, 28-50 steps, CFG
               3.5-5. Anima's README: prefix "masterpiece, best quality, score_7, safe",
               tags and sentences mixed, the more detail the better. 40 steps, CFG 4.

The content terms (spec section 3) and the young negative for teenage bands are in every
negative prompt. ComfyUI runs from its own environment in data/comfyui/ with --offline,
listening on 127.0.0.1 only. Pictures stay on this PC (AGENTS.md rule 6). Rendering is
resumable: a picture already on disk is skipped. Close FM27 first.

DMD2 needs peft, which the project does not install; run with `uv run --with peft`:

    uv run python scripts/archive/wai_anima_test.py prompts --style anima   # no GPU
    uv run --with peft python scripts/archive/wai_anima_test.py smoke       # 2 per column
    uv run --with peft python scripts/archive/wai_anima_test.py all
    uv run python scripts/archive/wai_anima_test.py render --columns anima29 --cases S01a
    uv run python scripts/archive/wai_anima_test.py tag
    uv run python scripts/archive/wai_anima_test.py report
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Self

import bakeoff
import zimage_test
from bakeoff import Case

from kitshot.qc import CONTENT_TAGS  # the quality check's content tags

WAI = ("LyliaEngine/waiIllustriousSDXL_v170", "waiIllustriousSDXL_v170.safetensors")
DMD2 = ("tianweiy/DMD2", "dmd2_sdxl_4step_lora_fp16.safetensors")
DMD2_TIMESTEPS = [999, 749, 499, 249]  # the 4-step LoRA's own schedule (DMD2 README)

COMFY_DIR = bakeoff.ROOT / "data" / "comfyui"
COMFY_PY = COMFY_DIR / ".venv" / "bin" / "python"
COMFY_PORT = 8189
ANIMA_UNET = "anima-2.9b-preview-v1.safetensors"  # links into the Hugging Face cache
ANIMA_TE = "qwen_3_06b_base.safetensors"
ANIMA_VAE = "qwen_image_vae.safetensors"

# Let bakeoff.tag() pick up the new folders.
bakeoff.MODELS["wai"] = bakeoff.ModelSpec("wai", "WAI-illustrious-SDXL v17", *WAI, "wai")
bakeoff.MODELS["anima29"] = bakeoff.ModelSpec(
    "anima29", "Anima-2.9B preview v1", "Gazingstars123/Anima-2.9B", None, "anima"
)


@dataclass(frozen=True)
class Column:
    key: str  # output folder and report column
    label: str
    engine: str  # "diffusers" or "comfyui"
    steps: int
    cfg: float  # 1 or less turns CFG off, and with it the negative prompt
    width: int
    height: int
    subset: bool  # only the ten subset cases
    dmd2: bool = False


COLUMNS = {
    c.key: c
    for c in (
        Column("wai", "WAI-illustrious v17", "diffusers", 28, 5.5, 1024, 1024, False),
        Column("wai~tall", "WAI v17 · 1024×1344", "diffusers", 28, 5.5, 1024, 1344, True),
        Column("wai~dmd2", "WAI v17 + DMD2 · 4 steps", "diffusers", 4, 0.0, 1024, 1024, False, True),
        Column("wai~dmd2-8", "WAI v17 + DMD2 · 8 steps", "diffusers", 8, 0.0, 1024, 1024, True, True),
        Column("anima29", "Anima-2.9B preview v1 · ComfyUI", "comfyui", 40, 4.0, 1024, 1024, False),
    )
}  # fmt: skip
ANIMAGINE = "animagine"
MAIN = [ANIMAGINE, "wai", "wai~dmd2", "anima29"]
EXTRA = ["wai~tall", "wai~dmd2-8"]
DEFAULT_COLUMNS = [ANIMAGINE, "wai", "wai~tall", "wai~dmd2", "wai~dmd2-8", "anima29"]
LABELS = {ANIMAGINE: "Animagine XL 4.0 Opt (chosen)"} | {k: c.label for k, c in COLUMNS.items()}
SUBSET = zimage_test.SUBSET
SMOKE = ["S01a", "S13a"]
make_cases = zimage_test.make_cases  # the generation recipe's band wording and young negative


def cases_for(key: str, cases: list[Case]) -> list[Case]:
    if key in COLUMNS and COLUMNS[key].subset:
        return [c for c in cases if c.id in SUBSET]
    return cases


# ---------------------------------------------------------------------------- prompts

WAI_QUALITY = "masterpiece, best quality, amazing quality"  # WAI card
WAI_RATING = "general"  # WAI card: general, sensitive, nsfw, explicit
WAI_NEGATIVE = "bad quality, worst quality, worst detail, sketch, censor"  # WAI card
# Anima README: recommended prefix, plus a period tag (the 2.9B card asks for one).
ANIMA_PREFIX = "masterpiece, best quality, score_7, newest, safe"
ANIMA_NEGATIVE = (  # Anima README's recommended negative
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, "
    "jpeg artifacts, chromatic aberration"
)


def body_tags(case: Case) -> str:
    """The sheet's tags without subject, rating or quality tags (same as every column)."""
    positive, _ = bakeoff.build_prompt(case, "animagine")
    head = bakeoff._join("1girl, solo", bakeoff.RATING) + ", "
    tail = ", " + bakeoff.QUALITY["animagine"][0]
    assert positive.startswith(head) and positive.endswith(tail), positive
    return positive[len(head) : -len(tail)]


def _negative(case: Case, quality: str) -> str:
    _, band_negative = case.wording or ("", "")
    return bakeoff._join(
        bakeoff.CONTENT_NEGATIVE, band_negative, bakeoff.COMPOSITION_NEGATIVE, quality
    )


def wai_prompt(case: Case) -> tuple[str, str]:
    positive = bakeoff._join(WAI_QUALITY, WAI_RATING, "1girl, solo", body_tags(case))
    return positive, _negative(case, WAI_NEGATIVE)


def anima_prompt(case: Case) -> tuple[str, str]:
    """Anima's tag order (quality, period, safety, subject, general tags), then the same
    sheet as sentences, since its cards ask for detailed prompts."""
    sentences, _ = zimage_test.natural_prompt(case)
    positive = f"{ANIMA_PREFIX}, 1girl, solo, {body_tags(case)}. {sentences}"
    return positive, _negative(case, ANIMA_NEGATIVE)


def prompt_for(key: str, case: Case) -> tuple[str, str]:
    return anima_prompt(case) if COLUMNS[key].engine == "comfyui" else wai_prompt(case)


# ------------------------------------------------------------------- shared rendering


def _save(out_dir: Path, case: Case, image, record: dict, log_path: Path) -> None:
    from PIL.PngImagePlugin import PngInfo

    meta = PngInfo()
    meta.add_text("facegen", json.dumps(record))
    tmp = out_dir / f".{case.id}.png.tmp"
    image.save(tmp, format="PNG", pnginfo=meta)
    tmp.replace(out_dir / f"{case.id}.png")
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _write_stats(run_dir: Path, key: str, entry: dict) -> None:
    path = run_dir / "models.json"
    stats = json.loads(path.read_text()) if path.exists() else {}
    stats[key] = entry
    path.write_text(json.dumps(stats, indent=2))


def _record(col: Column, case: Case, positive: str, negative: str, seconds: float) -> dict:
    return {
        "model": col.key.partition("~")[0],
        "column": col.key,
        "model_label": col.label,
        "engine": col.engine,
        "case": case.id,
        "seed": case.seed,
        "steps": col.steps,
        "cfg": col.cfg,
        "width": col.width,
        "height": col.height,
        "prompt": positive,
        "negative": negative if col.cfg > 1 else "",
        "seconds": round(seconds, 2),
    }


# ------------------------------------------------------------- WAI (diffusers, SDXL)


def load_wai(col: Column):
    pipe = bakeoff.load_pipeline(bakeoff.MODELS["wai"])  # fp16-fix VAE, Euler a, epsilon
    if col.dmd2:
        from diffusers import LCMScheduler
        from huggingface_hub import hf_hub_download

        pipe.load_lora_weights(hf_hub_download(*DMD2))
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()  # the fused weights stay
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe


def generate_wai(pipe, col: Column, positive: str, negative: str, seed: int):
    import torch

    use_cfg = col.cfg > 1
    texts = (positive, negative) if use_cfg else (positive,)
    with torch.no_grad():
        n_chunks = max(len(bakeoff._token_chunks(pipe.tokenizer, t)) for t in texts)
        pe, pp = bakeoff.encode(pipe, positive, n_chunks)
        kwargs: dict[str, object] = {"prompt_embeds": pe, "pooled_prompt_embeds": pp}
        if use_cfg:
            ne, npool = bakeoff.encode(pipe, negative, n_chunks)
            kwargs |= {"negative_prompt_embeds": ne, "negative_pooled_prompt_embeds": npool}
        if col.dmd2 and col.steps == len(DMD2_TIMESTEPS):
            kwargs["timesteps"] = DMD2_TIMESTEPS
        else:
            kwargs["num_inference_steps"] = col.steps
        return pipe(
            width=col.width,
            height=col.height,
            guidance_scale=col.cfg,
            generator=torch.Generator("cuda").manual_seed(seed),
            **kwargs,
        ).images[0]


def render_wai(run_dir: Path, col: Column, cases: list[Case]) -> None:
    import torch

    out_dir = run_dir / col.key
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases_for(col.key, cases) if not (out_dir / f"{c.id}.png").exists()]
    if not todo:
        print(f"{col.key}: nothing to render")
        return
    print(f"{col.key}: loading {col.label}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = load_wai(col)
    load_s = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / 2**30
    print(f"{col.key}: loaded in {load_s:.0f} s, {weights_gb:.1f} GiB on the GPU")
    for n, case in enumerate(todo, 1):
        positive, negative = wai_prompt(case)
        t1 = time.perf_counter()
        image = generate_wai(pipe, col, positive, negative, case.seed)
        seconds = time.perf_counter() - t1
        record = _record(col, case, positive, negative, seconds)
        record["sampler"] = "lcm" if col.dmd2 else "euler_a"
        record["vae"] = "/".join(bakeoff.FP16_VAE)
        if col.dmd2:
            record["lora"] = "/".join(DMD2)
        _save(out_dir, case, image, record, run_dir / "renders.jsonl")
        print(f"  [{n}/{len(todo)}] {case.id} {seconds:.1f} s", flush=True)
    _write_stats(
        run_dir,
        col.key,
        {
            "label": col.label,
            "load_seconds": round(load_s, 1),
            "weights_gib": round(weights_gb, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "vram_source": "torch",
        },
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------- Anima-2.9B (ComfyUI)


def _smi_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(out.split()[0])


class GpuPeak(threading.Thread):
    """Highest card-wide memory use while ComfyUI runs (it is another process)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak = max(self.peak, _smi_used_mib())
            self._stop_event.wait(0.5)

    def stop(self) -> int:
        self._stop_event.set()
        self.join()
        return self.peak


# Never use a proxy for the local server.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ComfyServer:
    def __init__(self, run_dir: Path) -> None:
        self.output = run_dir / ".comfy-output"
        self.log = run_dir / "comfyui.log"
        self.client_id = uuid.uuid4().hex

    def _http(self, path: str, data: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{COMFY_PORT}{path}",
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with _OPENER.open(req, timeout=60) as r:
            return json.loads(r.read())

    def __enter__(self) -> Self:
        if not COMFY_PY.exists():
            sys.exit(f"no ComfyUI environment at {COMFY_DIR}; see the module docstring")
        self.output.mkdir(parents=True, exist_ok=True)
        env = os.environ | {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "COMFYUI_DISABLE_PRODUCT_TIPS": "1",
        }
        self._logf = self.log.open("a")
        self.proc = subprocess.Popen(
            [
                str(COMFY_PY),
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                str(COMFY_PORT),
                "--offline",
                "--disable-all-custom-nodes",
                "--disable-auto-launch",
                "--output-directory",
                str(self.output),
            ],
            cwd=COMFY_DIR / "ComfyUI",
            env=env,
            stdout=self._logf,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                sys.exit(f"ComfyUI exited early; see {self.log}")
            try:
                self._http("/system_stats")
                return self
            except OSError:
                time.sleep(1)
        self.__exit__()
        sys.exit(f"ComfyUI did not start in 5 minutes; see {self.log}")

    def __exit__(self, *_: object) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._logf.close()

    def run(self, graph: dict, output_node: str) -> Path:
        try:
            pid = self._http("/prompt", {"prompt": graph, "client_id": self.client_id})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI refused the workflow: {e.read().decode()[:2000]}") from e
        prompt_id = pid["prompt_id"]
        while True:
            entry = self._http(f"/history/{prompt_id}").get(prompt_id)
            if entry and entry.get("status", {}).get("completed") is not None:
                status = entry["status"]
                if status.get("status_str") != "success":
                    raise RuntimeError(f"ComfyUI failed: {json.dumps(status)[:2000]}")
                img = entry["outputs"][output_node]["images"][0]
                return self.output / img.get("subfolder", "") / img["filename"]
            time.sleep(0.1)


def anima_graph(col: Column, positive: str, negative: str, seed: int, prefix: str) -> dict:
    """Anima's official example workflow (README's example.png), with the 2.9B card's
    sampler and scheduler."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": ANIMA_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": ANIMA_TE, "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": ANIMA_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": col.width, "height": col.height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": col.steps, "cfg": col.cfg, "sampler_name": "euler",
            "scheduler": "sgm_uniform", "denoise": 1.0, "model": ["1", 0],
            "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }  # fmt: skip


def render_comfy(run_dir: Path, col: Column, cases: list[Case]) -> None:
    from PIL import Image

    out_dir = run_dir / col.key
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases_for(col.key, cases) if not (out_dir / f"{c.id}.png").exists()]
    if not todo:
        print(f"{col.key}: nothing to render")
        return
    idle = _smi_used_mib()
    peak = GpuPeak()
    peak.start()
    t0 = time.perf_counter()
    first = None
    with ComfyServer(run_dir) as server:
        start_s = time.perf_counter() - t0
        print(f"{col.key}: ComfyUI up in {start_s:.0f} s")
        for n, case in enumerate(todo, 1):
            positive, negative = anima_prompt(case)
            t1 = time.perf_counter()
            src = server.run(anima_graph(col, positive, negative, case.seed, col.key), "9")
            seconds = time.perf_counter() - t1
            first = seconds if first is None else first
            with Image.open(src) as im:
                image = im.convert("RGB")
            src.unlink()
            record = _record(col, case, positive, negative, seconds)
            record |= {"sampler": "euler", "scheduler": "sgm_uniform", "unet": ANIMA_UNET}
            _save(out_dir, case, image, record, run_dir / "renders.jsonl")
            print(f"  [{n}/{len(todo)}] {case.id} {seconds:.1f} s", flush=True)
    top = peak.stop()
    _write_stats(
        run_dir,
        col.key,
        {
            "label": col.label,
            "load_seconds": round(start_s, 1),
            "first_picture_seconds": round(first or 0, 1),  # includes loading the models
            "peak_reserved_gib": round((top - idle) / 1024, 2),
            "vram_source": "nvidia-smi, above the idle card",
        },
    )


def render(run_dir: Path, keys: list[str], case_ids: set[str]) -> None:
    cases = [c for c in make_cases() if not case_ids or c.id in case_ids]
    zimage_test.gpu_guard()
    if ANIMAGINE in keys:
        bakeoff.render(run_dir, [ANIMAGINE], cases)
    for key in keys:
        if key in COLUMNS and COLUMNS[key].engine == "diffusers":
            if COLUMNS[key].dmd2:
                try:
                    import peft  # noqa: F401
                except ImportError:
                    print(f"{key}: skipped; DMD2 needs peft (uv run --with peft ...)")
                    continue
            render_wai(run_dir, COLUMNS[key], cases)
    for key in keys:
        if key in COLUMNS and COLUMNS[key].engine == "comfyui":
            render_comfy(run_dir, COLUMNS[key], cases)


# --------------------------------------------------------------------------- report


def report(run_dir: Path) -> Path:
    cases = {c.id: c for c in make_cases()}
    renders: dict[tuple[str, str], dict] = {}
    if (run_dir / "renders.jsonl").exists():
        for line in (run_dir / "renders.jsonl").read_text().splitlines():
            r = json.loads(line)
            renders[(r.get("column", r["model"]), r["case"])] = r
    stats_path = run_dir / "models.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    tags_path = run_dir / "tags_eva02.json"
    tags = json.loads(tags_path.read_text())["images"] if tags_path.exists() else {}
    columns = [
        k for k in [*MAIN, *EXTRA] if (run_dir / k).is_dir() and any((run_dir / k).glob("*.png"))
    ]

    rows: dict[tuple[str, str], dict] = {}
    for col in columns:
        for cid, case in cases.items():
            if not (run_dir / col / f"{cid}.png").exists():
                continue
            info = tags.get(f"{col}/{cid}")
            row: dict = {"seconds": renders.get((col, cid), {}).get("seconds"), "band": case.band}
            if info:
                t = info["tags"]
                row["general"] = info["rating"] == "general"
                row["p_general"] = info["ratings"].get("general")
                row["bad_tags"] = [x for x in CONTENT_TAGS if x in t]
                row["content"] = row["general"] and not row["bad_tags"]
                row["views"] = "multiple views" in t
                row["tags"] = t
                row["p_child"] = info.get("probes", {}).get("child")
                row["p_mature"] = info.get("probes", {}).get("mature female")
                row.update(bakeoff.check(case.expect, t))
                row["all"] = all(row[k] for k in ("content", "hair", "eyes", "kit", "solo"))
            rows[(col, cid)] = row

    def pct(vals: list[bool]) -> str:
        return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "–"

    def mean(vals: list[float]) -> str:
        return f"{statistics.mean(vals):.2f}" if vals else "–"

    summary = []
    for col in columns:
        mine = [r for (c, _), r in rows.items() if c == col]
        secs = [r["seconds"] for r in mine if r.get("seconds")]
        st = stats.get(col, {})
        summary.append(
            {
                "Column": LABELS.get(col, col),
                "Pictures": str(len(mine)),
                "s / picture": f"{statistics.median(secs):.1f}" if secs else "–",
                "Peak VRAM": f"{st['peak_reserved_gib']:.1f} GiB" if "peak_reserved_gib" in st else "–",
                "Content check": pct([r.get("content", False) for r in mine]),
                "Hair": pct([r.get("hair", False) for r in mine]),
                "Eyes": pct([r.get("eyes", False) for r in mine]),
                "Kit": pct([r.get("kit", False) for r in mine]),
                "One girl": pct([r.get("solo", False) for r in mine]),
                "All checks": pct([r.get("all", False) for r in mine]),
                "Multiple views": str(sum(r.get("views", False) for r in mine)),
                "Band 1 p(child)": mean([r["p_child"] for r in mine if r["band"] == 1 and r.get("p_child") is not None]),
                "Bands 4–5 p(mature female)": mean([r["p_mature"] for r in mine if r["band"] >= 4 and r.get("p_mature") is not None]),
            }
        )  # fmt: skip

    def cell(col: str, cid: str) -> str:
        row = rows.get((col, cid))
        if row is None:
            return '<div class="pic missing">not rendered</div>'
        badges = []
        if "content" in row:
            word = "content ok" if row["content"] else "CONTENT FAIL"
            extra = f" ({', '.join(row['bad_tags'])})" if row["bad_tags"] else ""
            badges.append(
                f'<span class="b {"ok" if row["content"] else "bad"}" '
                f'title="p(general)={row["p_general"]}">{word}{html.escape(extra)}</span>'
            )
            for name in ("hair", "eyes", "kit", "solo"):
                badges.append(f'<span class="b {"ok" if row[name] else "bad"}">{name}</span>')
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
            "<p class='note'>EVA02 tagger at 0.35, as in the other runs. <b>Content check</b> is "
            "the quality check's rule: top rating general and none of its content tags. <b>All "
            "checks</b> = content, hair, eyes, kit and one girl. <b>s / picture</b> includes "
            "prompt encoding. Peak VRAM: torch's peak for diffusers columns; for ComfyUI, the "
            "card's peak use above idle from nvidia-smi. Hover a picture for its prompt and tags."
            "</p>"
        ),
    ]
    main = [c for c in MAIN if c in columns]
    heads = "".join(f"<th>{html.escape(LABELS[c])}</th>" for c in main)
    sections.append(
        "<h2>Character sheets</h2><p class='note'>Each cell: variant a and b, a different "
        "kit, pose and seed. Same character? The style we want?</p>"
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
    extra = [c for c in ["wai", "wai~tall", "wai~dmd2", "wai~dmd2-8"] if c in columns]
    if any(c in columns for c in EXTRA):
        heads = "".join(f"<th>{html.escape(LABELS[c])}</th>" for c in extra)
        trs = "".join(
            f"<tr><td class='desc'>{cid}</td>"
            + "".join(f"<td>{cell(c, cid)}</td>" for c in extra)
            + "</tr>"
            for cid in SUBSET
        )
        sections.append(
            "<h2>WAI variants</h2><p class='note'>Same seed and prompt. The card's taller "
            "size, and DMD2 at 4 and 8 steps (no CFG, so no negative prompt).</p>"
            f"<div class='scroll'><table class='grid'><tr><th></th>{heads}</tr>{trs}"
            "</table></div>"
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WAI and Anima test</title>
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
.pic img {{ display:block; width:208px; height:auto; }}
.pic.missing {{ height:208px; display:grid; place-items:center; color:var(--muted); }}
figcaption {{ display:flex; flex-wrap:wrap; gap:3px; padding:4px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; color:var(--muted);
      background:var(--line); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
.band {{ font-size:12px; color:var(--muted); margin-bottom:2px; }}
</style></head><body>
<h1>WAI-illustrious v17 and Anima-2.9B against Animagine · {html.escape(run_dir.name)}</h1>
<p class="note">Animagine: the generation recipe (1024², 28 steps, CFG 5.5, Euler a). WAI:
its card's tags, 1024², 28 steps, CFG 5.5, Euler a, fp16-fix VAE. WAI + DMD2: LCM, no CFG,
so no negative prompt. Anima-2.9B: ComfyUI, 1024², 40 steps, CFG 4, Euler + sgm_uniform,
tags then sentences. Content terms and the young negative are in every negative prompt.
Local file only; these pictures do not leave this PC.</p>
{"".join(sections)}
</body></html>"""
    path = run_dir / "index.html"
    path.write_text(page)
    return path


# ------------------------------------------------------------------------------ CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="wai-anima", help="run name under data/bakeoff/")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts", help="print the prompts; no GPU")
    p.add_argument("--style", choices=("wai", "anima"), default="wai")
    sub.add_parser("smoke", help=f"render {SMOKE} in every column into <run>-smoke")
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
            pos, neg = anima_prompt(case) if args.style == "anima" else wai_prompt(case)
            print(f"{case.id} seed={case.seed}\n  + {pos}\n  - {neg}")
        return
    if args.command == "smoke":
        run_dir = bakeoff.BAKEOFF_DIR / f"{args.run}-smoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        keys = [k for k in DEFAULT_COLUMNS if k == ANIMAGINE or not COLUMNS[k].subset]
        render(run_dir, keys, set(SMOKE))
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
