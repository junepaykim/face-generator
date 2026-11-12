"""Phase 0.3 spike: does a reference picture keep the same character across a re-draw?

The tool re-draws a tracked player after a transfer (new kit) or an age-wording change
(spec section 5), from the same character sheet and base seed. This tests whether also
giving the model the player's existing picture as a reference, through IP-Adapter
(h94/IP-Adapter, SDXL ViT-H), keeps the face more recognisable, and what it costs.

Scenarios, on Animagine XL 4.0 Opt with the bake-off prompts:

- transfer: each of the 20 character sheets. The existing picture is round 2's "a"
  picture; the new one has kit and pose "b" and seed + 1. Without a reference that is
  round 2's "b" picture.
- age: round 3's W02 picture of L1-L4 is the existing one; the new one is drawn at W04,
  W07 and W09 with the same kit, pose and seed. Does the age still change?

Columns: none (round 2/3 picture, no reference), face-0.4/0.6/0.8 (plus-face adapter on
a face crop of the existing picture, at that scale), plus-0.4 (general adapter on the
whole existing picture; expected to carry the old kit over).

Pictures stay local, in data/bakeoff/r4/ (AGENTS.md hard rule 6).

    uv run python scripts/spikes/reference_test.py
"""

from __future__ import annotations

import gc
import html
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bakeoff as b  # sets offline mode and the CUDA allocator before torch loads

RUN = b.BAKEOFF_DIR / "r4"
IPA_REPO = "h94/IP-Adapter"
IPA_ENCODER = "models/image_encoder"  # ViT-H/14, shared by both adapters
ADAPTERS = {
    "face": ("ip-adapter-plus-face_sdxl_vit-h.safetensors", (0.1, 0.2, 0.4, 0.6, 0.8)),
    "plus": ("ip-adapter-plus_sdxl_vit-h.safetensors", (0.4,)),
}
AGE_TARGETS = ("W04", "W07", "W09")


@dataclass(frozen=True)
class Task:
    id: str  # "S01-kit" or "L1-W07"
    scenario: str  # "transfer" or "age"
    case: b.Case  # the new picture's case: prompt, seed and expected kit
    reference: Path  # the existing picture
    baseline: Path  # the same new picture drawn without a reference
    old_kit: str | None  # the existing picture's kit, when it differs


def tasks() -> list[Task]:
    pairs = {c.id: c for c in b.make_cases()}
    ages = {c.id: c for c in b.make_age_cases()}
    r2, r3 = b.BAKEOFF_DIR / "r2" / "animagine", b.BAKEOFF_DIR / "r3" / "animagine"
    out = [
        Task(f"{s.id}-kit", "transfer", pairs[f"{s.id}b"], r2 / f"{s.id}a.png",
             r2 / f"{s.id}b.png", s.kits[0] if s.kits[0] != s.kits[1] else None)
        for s in b.SHEETS
    ]  # fmt: skip
    out += [
        Task(f"{s.id}-{w}", "age", ages[f"{s.id}-{w}"], r3 / f"{s.id}-W02.png",
             r3 / f"{s.id}-{w}.png", None)
        for s in b.AGE_SHEETS
        for w in AGE_TARGETS
    ]  # fmt: skip
    return out


def face_box(image) -> tuple[int, int, int, int]:
    """A square around the head, found against the plain grey background.

    The top of the head is the first row with enough non-background pixels; the box is
    centred on the foreground in the band below it. Crude, but the pictures all share
    one framing. The crops are shown on the result page.
    """
    import numpy as np

    a = np.asarray(image.convert("RGB")).astype(np.int16)
    h, w, _ = a.shape
    corners = np.concatenate([a[:16, :16], a[:16, -16:], a[-16:, :16], a[-16:, -16:]])
    bg = np.median(corners.reshape(-1, 3), axis=0)
    fg = np.abs(a - bg).sum(axis=2) > 60
    rows = np.flatnonzero(fg.sum(axis=1) > 0.02 * w)
    side = int(0.36 * h)
    top = max(0, int(rows[0]) - int(0.03 * h)) if rows.size else int(0.05 * h)
    band = fg[top : top + side]
    weights = band.sum(axis=0)
    cx = int((weights * np.arange(w)).sum() / weights.sum()) if weights.sum() else w // 2
    left = min(max(0, cx - side // 2), w - side)
    top = min(top, h - side)
    return left, top, left + side, top + side


def prepare(all_tasks: list[Task]) -> None:
    """Copy the existing and baseline pictures in, and cut the reference face crops."""
    from PIL import Image

    for name in ("ref", "crops", "animagine~none"):
        (RUN / name).mkdir(parents=True, exist_ok=True)
    for t in all_tasks:
        ref_id = t.reference.stem if t.scenario == "transfer" else f"{t.case.sheet.id}-W02"
        if not (RUN / "ref" / f"{ref_id}.png").exists():
            shutil.copy2(t.reference, RUN / "ref" / f"{ref_id}.png")
        crop = RUN / "crops" / f"{ref_id}.png"
        if not crop.exists():
            img = Image.open(t.reference)
            img.crop(face_box(img)).save(crop)
        if not (RUN / "animagine~none" / f"{t.id}.png").exists():
            shutil.copy2(t.baseline, RUN / "animagine~none" / f"{t.id}.png")


def ref_id(t: Task) -> str:
    return t.reference.stem if t.scenario == "transfer" else f"{t.case.sheet.id}-W02"


def render(all_tasks: list[Task]) -> None:
    import torch
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    spec = b.MODELS["animagine"]
    pipe = None
    for adapter, (weight_name, scales) in ADAPTERS.items():
        columns = {s: RUN / f"animagine~{adapter}-{s}" for s in scales}
        todo = [(s, t) for s in scales for t in all_tasks
                if not (columns[s] / f"{t.id}.png").exists()]  # fmt: skip
        if not todo:
            print(f"{adapter}: nothing to render")
            continue
        if pipe is None:
            pipe = b.load_pipeline(spec)
        torch.cuda.reset_peak_memory_stats()
        pipe.load_ip_adapter(
            IPA_REPO, subfolder="sdxl_models", weight_name=weight_name,
            image_encoder_folder=IPA_ENCODER,
        )  # fmt: skip
        print(f"{adapter}: adapter loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        for n, (scale, t) in enumerate(todo, 1):
            pipe.set_ip_adapter_scale(scale)
            source = RUN / ("crops" if adapter == "face" else "ref") / f"{ref_id(t)}.png"
            reference = Image.open(source).convert("RGB")
            positive, negative = b.build_prompt(t.case, spec.family)
            t0 = time.perf_counter()
            with torch.no_grad():
                chunks = max(len(b._token_chunks(pipe.tokenizer, x)) for x in (positive, negative))
                pe, pp = b.encode(pipe, positive, chunks)
                ne, npool = b.encode(pipe, negative, chunks)
                image = pipe(
                    prompt_embeds=pe, pooled_prompt_embeds=pp, negative_prompt_embeds=ne,
                    negative_pooled_prompt_embeds=npool, ip_adapter_image=reference,
                    width=b.SIZE, height=b.SIZE, num_inference_steps=b.STEPS,
                    guidance_scale=b.CFG,
                    generator=torch.Generator("cuda").manual_seed(t.case.seed),
                ).images[0]  # fmt: skip
            seconds = time.perf_counter() - t0
            record = {
                "task": t.id, "scenario": t.scenario, "adapter": adapter,
                "weight": weight_name, "scale": scale, "reference": source.name,
                "seed": t.case.seed, "steps": b.STEPS, "cfg": b.CFG, "prompt": positive,
                "negative": negative, "seconds": round(seconds, 2),
                "column": columns[scale].name,
            }  # fmt: skip
            meta = PngInfo()
            meta.add_text("facegen", json.dumps(record))
            columns[scale].mkdir(parents=True, exist_ok=True)
            tmp = columns[scale] / f".{t.id}.png.tmp"
            image.save(tmp, format="PNG", pnginfo=meta)
            tmp.replace(columns[scale] / f"{t.id}.png")
            with (RUN / "renders.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"  [{n}/{len(todo)}] {adapter}-{scale} {t.id} {seconds:.1f} s")
        peak = torch.cuda.max_memory_reserved() / 2**30
        print(f"{adapter}: peak {peak:.1f} GiB reserved")
        pipe.unload_ip_adapter()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def similarity(all_tasks: list[Task]) -> dict[str, dict[str, float]]:
    """Cosine similarity of CLIP ViT-H embeddings of the reference and new face crops.

    A rough proxy for "same character". It is biased towards the adapter columns, which
    condition on this same encoder, so read it next to the pictures.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    enc = CLIPVisionModelWithProjection.from_pretrained(
        IPA_REPO, subfolder=IPA_ENCODER, dtype=torch.float16
    ).to("cuda").eval()  # fmt: skip
    proc = CLIPImageProcessor(size=224, crop_size=224)

    def embed(img):
        x = proc(images=img, return_tensors="pt").pixel_values.to("cuda", torch.float16)
        with torch.no_grad():
            return F.normalize(enc(pixel_values=x).image_embeds.float(), dim=-1)

    refs = {}
    out: dict[str, dict[str, float]] = {}
    for col in sorted(p for p in RUN.glob("animagine~*") if p.is_dir()):
        for t in all_tasks:
            path = col / f"{t.id}.png"
            if not path.exists():
                continue
            if ref_id(t) not in refs:
                refs[ref_id(t)] = embed(Image.open(RUN / "crops" / f"{ref_id(t)}.png"))
            img = Image.open(path)
            sim = float((refs[ref_id(t)] * embed(img.crop(face_box(img)))).sum())
            out.setdefault(col.name, {})[t.id] = round(sim, 3)
    (RUN / "similarity.json").write_text(json.dumps(out, indent=1))
    return out


# The user's follow-up (2026-11-11): low face scales against no reference and face 0.4.
LOW_COLUMNS = ["animagine~none", "animagine~face-0.1", "animagine~face-0.2", "animagine~face-0.4"]


def page(
    all_tasks: list[Task],
    sims: dict[str, dict[str, float]],
    columns: list[str] | None = None,
    name: str = "reference.html",
) -> Path:
    tags = json.loads((RUN / "tags_eva02.json").read_text())["images"]
    columns = columns or ["animagine~none"] + [
        f"animagine~{a}-{s}" for a, (_, scales) in ADAPTERS.items() for s in scales
    ]
    label = {c: c.partition("~")[2].replace("none", "no reference") for c in columns}
    renders = {}
    if (RUN / "renders.jsonl").exists():
        for line in (RUN / "renders.jsonl").read_text().splitlines():
            r = json.loads(line)
            renders[(r["column"], r["task"])] = r

    def result(col: str, t: Task) -> dict:
        info = tags.get(f"{col}/{t.id}", {})
        res = b.check(t.case.expect, info.get("tags", {}))
        res["general"] = info.get("rating") == "general"
        res["sim"] = sims.get(col, {}).get(t.id)
        if t.old_kit:
            old = b.expectations(t.case.sheet, t.old_kit)["kit"]
            res["old_kit"] = b.check({"kit": old}, info.get("tags", {}))["kit"]
        return res

    def pct(xs: list[bool]) -> str:
        return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"

    summary = []
    for col in columns:
        row = {"Column": label[col]}
        for scen in ("transfer", "age"):
            rs = [result(col, t) for t in all_tasks if t.scenario == scen]
            sims_ = [r["sim"] for r in rs if r["sim"] is not None]
            row[f"{scen}: similarity"] = f"{statistics.mean(sims_):.3f}" if sims_ else "–"
            row[f"{scen}: general"] = pct([r["general"] for r in rs])
            row[f"{scen}: hair+eyes"] = pct([r["hair"] and r["eyes"] for r in rs])
            row[f"{scen}: one girl"] = pct([r["solo"] for r in rs])
            if scen == "transfer":
                row["transfer: new kit"] = pct([r["kit"] for r in rs])
                row["transfer: old kit seen"] = pct([r["old_kit"] for r in rs if "old_kit" in r])
        secs = [r["seconds"] for (c, _), r in renders.items() if c == col]
        row["s / picture"] = f"{statistics.median(secs):.1f}" if secs else "–"
        summary.append(row)

    def fig(src: str, badges: list[str], small: str = "") -> str:
        extra = f'<img class="crop" src="{small}" alt="">' if small else ""
        return (f'<figure class="pic"><a href="{src}" target="_blank"><img src="{src}" '
                f'loading="lazy" alt=""></a>{extra}<figcaption>{"".join(badges)}'
                "</figcaption></figure>")  # fmt: skip

    def badge(ok: bool, text: str) -> str:
        return f'<span class="b {"ok" if ok else "bad"}">{html.escape(text)}</span>'

    def rows(scen: str) -> str:
        out = []
        for t in (x for x in all_tasks if x.scenario == scen):
            cells = [fig(f"ref/{ref_id(t)}.png", ['<span class="b">existing</span>'],
                         f"crops/{ref_id(t)}.png")]  # fmt: skip
            for col in columns:
                r = result(col, t)
                bs = [
                    badge(r["general"], "general" if r["general"] else "NOT general"),
                    badge(r["hair"] and r["eyes"], "hair+eyes"),
                    badge(r["solo"], "one girl"),
                ]
                if scen == "transfer":
                    bs.append(badge(r["kit"], "new kit"))
                    if r.get("old_kit"):
                        bs.append(badge(False, "old kit"))
                if r["sim"] is not None:
                    bs.append(f'<span class="b">sim {r["sim"]:.2f}</span>')
                cells.append(fig(f"{col}/{t.id}.png", bs))
            words = b.AGE_WORDINGS[t.id.split("-")[1]][0] if scen == "age" else ""
            desc = html.escape(f"{t.id} {words}".strip())
            out.append(f"<tr><td class='desc'>{desc}</td>"
                       + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")  # fmt: skip
        heads = "<th></th><th>existing picture (face crop below)</th>" + "".join(
            f"<th>{html.escape(label[c])}</th>" for c in columns
        )
        return (
            f"<div class='scroll'><table class='grid'><tr>{heads}</tr>{''.join(out)}</table></div>"
        )

    head = "".join(f"<th>{html.escape(k)}</th>" for k in summary[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in s.values()) + "</tr>"
                   for s in summary)  # fmt: skip
    html_page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reference picture test</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4; --card:#fff;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2c323c; --card:#1c2027;
          --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px 24px 48px; background:var(--bg); color:var(--fg);
        font:14px/1.45 system-ui, sans-serif; }}
h1 {{ margin:0 0 4px; font-size:22px; }} h2 {{ margin:28px 0 8px; font-size:18px; }}
.note {{ color:var(--muted); max-width:90ch; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }}
th {{ text-align:left; font-weight:600; padding:4px 6px; color:var(--muted); font-size:12px;
      vertical-align:bottom; }}
td {{ padding:4px 6px; vertical-align:top; border-bottom:1px solid var(--line); }}
.grid td {{ border:none; }}
.desc {{ width:90px; min-width:90px; font-size:12px; color:var(--muted); }}
.pic {{ margin:0; width:200px; background:var(--card); border:1px solid var(--line);
        border-radius:6px; overflow:hidden; }}
.pic img {{ display:block; width:200px; height:200px; }}
.pic img.crop {{ width:80px; height:80px; margin:4px; border-radius:4px; }}
figcaption {{ display:flex; flex-wrap:wrap; gap:3px; padding:4px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; color:var(--muted);
      background:var(--line); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
</style></head><body>
<h1>Reference picture test · r4</h1>
<p class="note">Does giving the model the player's existing picture keep the same character
after a re-draw? Animagine XL 4.0 Opt, {b.SIZE}&times;{b.SIZE}, {b.STEPS} steps, CFG {b.CFG}.
"face" columns use the face IP-Adapter on the face crop shown under the existing picture;
"plus" uses the general adapter on the whole picture. The number is the adapter scale.
"sim" is the CLIP similarity of the two face crops: a rough proxy, biased towards the
adapter columns. "old kit" means the tagger still sees the previous club's kit. Local file
only.</p>
<h2>Summary</h2><div class="scroll"><table><tr>{head}</tr>{body}</table></div>
<h2>Transfer: new kit and pose</h2>{rows("transfer")}
<h2>Age change: W02 picture as reference, drawn older</h2>
<p class="note">The age should still change. If a column keeps the W02 face too strongly,
it would freeze players at the age of their first picture.</p>{rows("age")}
</body></html>"""
    path = RUN / name
    path.write_text(html_page)
    return path


def main() -> None:
    all_tasks = tasks()
    RUN.mkdir(parents=True, exist_ok=True)
    prepare(all_tasks)
    render(all_tasks)
    b.tag(RUN, ["eva02"])
    sims = similarity(all_tasks)
    gc.collect()
    print("page:", page(all_tasks, sims))
    print("page:", page(all_tasks, sims, LOW_COLUMNS, "reference-low.html"))


if __name__ == "__main__":
    main()
