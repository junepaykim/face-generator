"""Phase 2 spike: what group A's detail pass costs and what it changes.

Six synthetic players (made-up IDs, not anyone in a save) drawn through the real Phase 2
code: traits, kits, prompt, AnimagineRenderer and the EVA02 check. Each base picture
(1024) is re-drawn by the detail pass (upscale x1.5 to 1536, img2img) at three strengths,
then checked again. Records seconds and peak VRAM with the tagger loaded alongside, as the
worker runs. Writes data/phase2/detail/detail.html: a local page, the pictures stay on this
PC (AGENTS.md hard rule 6). Resumable: pictures already on disk are skipped.

    uv run python scripts/spikes/detail_test.py
"""

from __future__ import annotations

import html
import json
import statistics
import time
from pathlib import Path

from facegen import qc
from facegen.gpu import prepare_environment
from facegen.kits import load_kits
from facegen.prompt import build_prompt
from facegen.settings import load_settings
from facegen.traits import pick_pose, picture_seed, sample_sheet

prepare_environment()

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "phase2" / "detail"
STRENGTHS = (0.25, 0.35, 0.45)
SCALE = 1.5
# (made-up unique ID, age wording, club ID, goalkeeper, at my club)
CASES = [
    (900001, "W01", 636, False, True),  # Colchester stripes, youngest wording
    (900002, "W03", 636, True, True),  # Colchester goalkeeper
    (900003, "W05", 658, False, False),  # Grimsby black and white stripes
    (900004, "W07", 689, False, False),  # Northampton claret
    (900005, "W09", 620, False, False),  # Bristol Rovers quarters
    (900006, "W10", None, False, False),  # fallback kit
]


def main() -> None:
    import torch
    from PIL import Image

    from facegen.worker import AnimagineRenderer

    RUN.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    kits = load_kits()
    results_path = RUN / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    t0 = time.perf_counter()
    renderer = AnimagineRenderer()
    load_s = time.perf_counter() - t0
    tagger = qc.Eva02Tagger()
    print(f"loaded in {load_s:.0f} s; weights {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    for uid, wording, club, gk, mine in CASES:
        sheet = sample_sheet("detail-test", uid, 1, height_cm=180, strength=10)
        kit, _ = kits.kit_for(club, gk)
        pose = pick_pose(sheet.base_seed, 1)
        my_club = settings.my_club if mine else None
        prompt = build_prompt(sheet, wording, kit, pose, my_club)
        expect = qc.expectation(sheet, kit, my_club)
        seed = picture_seed(sheet.base_seed, 1, 0, 0)
        entry = results.setdefault(
            str(uid),
            {"wording": wording, "kit": kit.describe(), "pose": pose, "prompt": prompt.positive},
        )

        base_path = RUN / f"{uid}-base.png"
        if base_path.exists():
            base = Image.open(base_path).convert("RGB")
        else:
            torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            base = renderer.render(prompt, seed)
            secs = time.perf_counter() - t1
            base.save(base_path)
            tags = tagger([base])[0]
            entry["base"] = {
                "seconds": round(secs, 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                **qc.check(tags, expect, qc.head_top(base)).to_json(tags),
            }
            print(f"{uid} base {secs:.1f} s")

        for strength in STRENGTHS:
            key = f"detail-{strength}"
            path = RUN / f"{uid}-{key}.png"
            if path.exists():
                continue
            renderer.detail_strength = strength
            torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            size = tuple(round(side * SCALE / 64) * 64 for side in base.size)
            big = renderer.detail(base, prompt, seed, size=size)
            secs = time.perf_counter() - t1
            big.save(path)
            # What FM27 would get: the 1536 master downscaled to group A's 1024.
            big.resize((1024, 1024), Image.Resampling.LANCZOS).save(RUN / f"{uid}-{key}-1024.png")
            tags = tagger([big])[0]
            entry[key] = {
                "seconds": round(secs, 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "size": big.width,
                **qc.check(tags, expect, qc.head_top(big)).to_json(tags),
            }
            print(f"{uid} {key} {secs:.1f} s, peak {entry[key]['peak_reserved_gib']} GiB")
        results_path.write_text(json.dumps(results, indent=1))

    # Tagger cost per picture, alone.
    imgs = [Image.open(RUN / f"{uid}-base.png").convert("RGB") for uid, *_ in CASES]
    t1 = time.perf_counter()
    for im in imgs:
        tagger([im])
    tag_s = (time.perf_counter() - t1) / len(imgs)
    summary = {"load_seconds": round(load_s, 1), "tag_seconds": round(tag_s, 3)}
    (RUN / "summary.json").write_text(json.dumps(summary, indent=1))
    print("page:", page(results, summary))


def page(results: dict, summary: dict) -> Path:
    cols = ["base", *(f"detail-{s}" for s in STRENGTHS)]
    stats = []
    for col in cols:
        secs = [r[col]["seconds"] for r in results.values() if col in r]
        peak = [r[col]["peak_reserved_gib"] for r in results.values() if col in r]
        ok = [r[col]["passed"] for r in results.values() if col in r]
        if secs:
            stats.append(
                f"<tr><td>{col}</td><td>{statistics.median(secs):.1f} s</td>"
                f"<td>{max(peak):.1f} GiB</td><td>{sum(ok)}/{len(ok)}</td></tr>"
            )
    rows = []
    for uid, r in results.items():
        cells = []
        for col in cols:
            if col not in r:
                cells.append("<td></td>")
                continue
            src = f"{uid}-{col}.png" if col == "base" else f"{uid}-{col}-1024.png"
            full = f"{uid}-{col}.png"
            c = r[col]
            cls = "ok" if c["passed"] else "bad"
            cells.append(
                f'<td><a href="{full}" target="_blank"><img src="{src}" loading="lazy"></a>'
                f'<div><span class="b {cls}">{"passed" if c["passed"] else "failed"}</span> '
                f'<span class="b">general {c["p_general"]:.2f}</span> '
                f'<span class="b">{c["seconds"]} s</span></div></td>'
            )
        desc = html.escape(f"{uid}: {r['wording']}, {r['kit']}, {r['pose']}")
        rows.append(f"<tr><td class='d'>{desc}</td>{''.join(cells)}</tr>")
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Detail pass</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2;
  --line:#2c323c; --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px 24px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, sans-serif; }}
table {{ border-collapse:collapse; }} td, th {{ padding:4px; vertical-align:top; text-align:left;
  border-bottom:1px solid var(--line); }}
img {{ width:400px; height:400px; display:block; }}
.d {{ width:180px; color:var(--muted); font-size:12px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; background:var(--line);
      color:var(--muted); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }} .b.bad {{ color:var(--bad);
  background:var(--badbg); }}
.note {{ color:var(--muted); max-width:90ch; }}
</style></head><body>
<h1>Group A detail pass</h1>
<p class="note">Left: the 1024 base picture. Right: the detail pass (upscaled x{SCALE} to 1536,
re-drawn by img2img at the given strength), shown downscaled to 1024 as FM27 would get it; click
for the 1536 master. Six made-up players, not from a save. Model load {summary["load_seconds"]} s;
tagger {summary["tag_seconds"]} s per picture. Local file only.</p>
<table><tr><th>Column</th><th>Median time</th><th>Peak VRAM (reserved)</th><th>Check</th></tr>
{"".join(stats)}</table>
<table><tr><th></th>{"".join(f"<th>{c}</th>" for c in cols)}</tr>{"".join(rows)}</table>
</body></html>"""
    path = RUN / "detail.html"
    path.write_text(text)
    return path


if __name__ == "__main__":
    main()
