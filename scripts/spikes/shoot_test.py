"""Spike: players' shoots on the GPU (shoot.py, 2026-11-12).

About 30 made-up players (IDs 920001 on, not anyone in a save) drawn through the real code:
traits, kits, shoot, prompt, AnimagineRenderer and the EVA02 check, one candidate each at
group B quality (no detail pass). Cases are picked so that every kit look shows up, plain,
gradient and two-tone backdrops all do, and "from below" a few times. For each picture it
records the check, and the head position by the new scan (figure.py, followed row by row)
and by the old one (the two top corners only), to show where the old one broke.

Writes data/shoot-test/shoot.html: a local page; the pictures stay on this PC (AGENTS.md
hard rule 6). Resumable: pictures already on disk are skipped. Refuses to run while FM27
does, like the worker, unless given --now.

    uv run python scripts/spikes/shoot_test.py [--now]
"""

from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter
from pathlib import Path

from facegen.core.prompt import build_prompt
from facegen.core.shoot import LOOKS, Shoot, shoot_for
from facegen.core.traits import picture_seed, sample_sheet
from facegen.datasource.kits import load_kits
from facegen.persistence.settings import load_settings
from facegen.pipeline.worker import fm_running
from facegen.render import quality_checks as qc
from facegen.render.gpu import prepare_environment
from facegen.render.picture import face_box
from facegen.render.tagger import Eva02Tagger

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "shoot-test"
CLUBS = (636, 658, 602, 1569, 679, 620, None)  # stripes, stripes, sleeves, hoops, catalog, ...
WORDINGS = ("W03", "W05", "W07", "W08", "W09")
QUOTA = {"plain": 8, "gradient": 10, "two-tone": 8, "below": 3, "look": 2}
LIMIT = 32


def cases(kits, settings) -> list[tuple[int, str, object, Shoot, object]]:
    """(unique ID, wording, sheet, shoot, kit source) until every quota is met."""
    from datetime import date

    have: Counter[str] = Counter()
    chosen = []
    for n in range(2000):
        uid = 920001 + n
        club = CLUBS[n % len(CLUBS)]
        goalkeeper = n % 6 == 5
        sheet = sample_sheet("shoot-test", uid, 1, height_cm=180, strength=10)
        kit, source = kits.kit_for(club, goalkeeper)
        home, _ = kits.kit_for(club, False)
        day = date(2026, 12, 1) if n % 2 else date(2027, 7, 1)
        shoot = shoot_for(
            kit, sheet.hair_colour, sheet.base_seed, 1, home=home, age=29, game_date=day
        )
        wanted = [
            have[shoot.backdrop.kind] < QUOTA[shoot.backdrop.kind],
            shoot.pose.angle == "below" and have["below"] < QUOTA["below"],
            have[f"look {shoot.wear.look}"] < QUOTA["look"],
        ]
        if not any(wanted):
            continue
        have[shoot.backdrop.kind] += 1
        have["below"] += shoot.pose.angle == "below"
        have[f"look {shoot.wear.look}"] += 1
        chosen.append((uid, WORDINGS[n % len(WORDINGS)], sheet, shoot, source))
        done = all(have[k] >= QUOTA[k] for k in ("plain", "gradient", "two-tone", "below"))
        if (done and all(have[f"look {x}"] >= QUOTA["look"] for x in LOOKS)) or len(
            chosen
        ) >= LIMIT:
            break
    return chosen


def old_head_top(image) -> float | None:
    """The scan before 2026-11-12: one backdrop colour, the median of both top corners."""
    small = image.convert("RGB").resize((128, 128))
    px = small.load()
    corner = [px[x, y] for y in range(8) for x in (*range(8), *range(120, 128))]
    bg = tuple(sorted(p[i] for p in corner)[len(corner) // 2] for i in range(3))
    for y in range(128):
        hits = sum(
            sum(abs(a - b) for a, b in zip(px[x, y], bg, strict=True)) > 60 for x in range(128)
        )
        if hits > 2:
            return y / 128
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", action="store_true", help="run even while FM27 is running")
    args = parser.parse_args()
    if not args.now and fm_running() is not False:
        raise SystemExit("FM27 is running (or cannot be checked); close it or pass --now")
    prepare_environment()
    from PIL import Image, ImageDraw

    from facegen.pipeline.worker import AnimagineRenderer

    RUN.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    kits = load_kits(data_dir=settings.paths.data_dir)
    results_path = RUN / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    renderer = tagger = None
    for uid, wording, sheet, shoot, source in cases(kits, settings):
        path = RUN / f"{uid}.png"
        prompt = build_prompt(sheet, wording, shoot.wear, shoot.pose, backdrop=shoot.backdrop)
        if not path.exists():
            if renderer is None:
                renderer, tagger = AnimagineRenderer(), Eva02Tagger()
            t0 = time.perf_counter()
            image = renderer.render(prompt, picture_seed(sheet.base_seed, 1, 0, 0))
            image.save(path)
            tags = tagger([image])[0]
            check = qc.check(tags, qc.expectation(sheet, shoot.wear), qc.head_top(image))
            results[str(uid)] = {
                "wording": wording,
                "kit_source": source,
                "shoot": shoot.context(),
                "wear": shoot.wear.describe(),
                "prompt": prompt.positive,
                "seconds": round(time.perf_counter() - t0, 1),
                "old_head_top": old_head_top(image),
                "face_box": list(face_box(image)),
                **check.to_json(tags),
            }
            results_path.write_text(json.dumps(results, indent=1))
            print(
                f"{uid} {shoot.backdrop.describe()}: {'passed' if check.passed else check.failures}"
            )
        image = Image.open(path).convert("RGB")
        r = results[str(uid)]
        draw = ImageDraw.Draw(image)
        h = image.height
        draw.rectangle(r["face_box"], outline=(0, 220, 0), width=5)
        if r["head_top"] is not None:
            draw.line(
                (0, r["head_top"] * h, image.width, r["head_top"] * h), fill=(0, 220, 0), width=4
            )
        if r["old_head_top"] is not None:
            y = r["old_head_top"] * h
            draw.line((0, y, image.width, y), fill=(230, 0, 0), width=4)
        image.thumbnail((360, 480))
        image.save(RUN / f"{uid}-marked.jpg", quality=88)
    print("page:", page(results))


def page(results: dict) -> Path:
    passed = sum(r["passed"] for r in results.values())
    by_kind: Counter[str] = Counter()
    ok_kind: Counter[str] = Counter()
    old_broken = 0
    for r in results.values():
        kind = r["shoot"]["backdrop"].split(" ")[0]
        kind = kind if kind in ("gradient", "two-tone") else "plain"
        by_kind[kind] += 1
        ok_kind[kind] += r["passed"]
        old, new = r["old_head_top"], r["head_top"]
        old_broken += new is not None and (old is None or abs(old - new) > 0.03)
    rows = []
    for uid, r in sorted(results.items()):
        s = r["shoot"]
        badge = "ok" if r["passed"] else "bad"
        fails = "" if r["passed"] else html.escape("; ".join(r["failures"]))
        old = "none" if r["old_head_top"] is None else f"{r['old_head_top']:.0%}"
        new = "none" if r["head_top"] is None else f"{r['head_top']:.0%}"
        rows.append(
            f'<figure><a href="{uid}.png" target="_blank"><img src="{uid}-marked.jpg" '
            f'loading="lazy"></a><figcaption><span class="b {badge}">'
            f"{'passed' if r['passed'] else 'failed'}</span> {fails}<br>"
            f"{html.escape(r['wear'])}<br>{s['arms']}, {s['angle']} · "
            f"<b>{html.escape(s['backdrop'])}</b><br>head: new {new}, old {old} · "
            f"{r['kit_source']} · {s['season']}</figcaption></figure>"
        )
    kinds = ", ".join(f"{k} {ok_kind[k]}/{by_kind[k]}" for k in sorted(by_kind))
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Player shoots</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2;
  --line:#2c323c; --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, sans-serif; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:12px; }}
figure {{ margin:0; }} img {{ width:100%; display:block; border-radius:4px; }}
figcaption {{ font-size:12px; color:var(--muted); }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
</style></head><body>
<h1>Player shoots</h1>
<p>{passed}/{len(results)} passed the check (by backdrop: {kinds}). Green: the head top and face
box by the new scan; red: the head top by the old corner-only scan, which was off by more than 3%
of the height on {old_broken} pictures. Made-up players, one candidate each. Local file only.</p>
<div class="grid">{"".join(rows)}</div></body></html>"""
    path = RUN / "shoot.html"
    path.write_text(text)
    return path


if __name__ == "__main__":
    main()
