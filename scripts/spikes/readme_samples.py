"""README sample pictures: made-up players drawn by the real worker, for the user to pick from.

Each model and gender runs in its own in-memory database and temporary folders: a made-up
career whose newgens (IDs above the created block, CA 170) play at clubs from the kit table,
so the everywhere cut puts them in Detailed (group A) and the worker draws them as it would in
a career: the sheet, the shoot, the prompt, three candidates, the detail pass and the check.
Only pictures that passed the check become current, and only current pictures are copied out.
Photo men carry a nation, so their look comes from its region.

Writes data/readme-samples/: each master as <label>.png, a 480x640 preview as <label>.jpg, the
existing hero's four tiles as H1-H4 for comparison, samples.json (prompt, seed, check) and
index.html to look at them. Pictures stay on this PC (AGENTS.md hard rule 6). Runs offline
with the files already in the Hugging Face cache; nothing is written outside data/.

    uv run --no-sync python scripts/spikes/readme_samples.py [--only realvis,zimage_turbo,...]
    uv run --no-sync python scripts/spikes/readme_samples.py --hero R1,R3,A2,H4

--hero draws nothing: it lays the four picked pictures (masters, or the old hero's H tiles)
side by side as docs/images/hero.jpg, 240x320 each, 8 px of white between them, as before.

Draws while FM27 runs (the user asked for it); the game may slow down meanwhile.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "readme-samples"
HERO = ROOT / "docs" / "images" / "hero.jpg"
HERO_TILE = (240, 320)
HERO_GAP = 8
PREVIEW = (480, 640)


@dataclass(frozen=True)
class Case:
    label: str
    nation: int | None
    nation_name: str
    age: int
    club: int
    kit: str


CLUBS = {  # config/kits.toml: club ID -> the home kit in words (for the page only)
    636: "blue and white stripes",
    605: "amber",
    109210: "red",
    634: "blue",
    620: "blue and white quarters",
    640: "red",
}

RUNS: dict[tuple[str, str], list[Case]] = {
    ("realvis", "man"): [
        Case("R1", 765, "England", 24, 636, CLUBS[636]),
        Case("R2", 1651, "Brazil", 21, 605, CLUBS[605]),
        Case("R3", 38, "Nigeria", 27, 109210, CLUBS[109210]),
        Case("R4", 116, "Japan", 23, 634, CLUBS[634]),
        Case("R5", 796, "Spain", 30, 620, CLUBS[620]),
        Case("R6", 41, "Senegal", 19, 640, CLUBS[640]),
    ],
    ("zimage_turbo", "man"): [
        Case("Z1", 776, "Italy", 26, 634, CLUBS[634]),
        Case("Z2", 1649, "Argentina", 22, 636, CLUBS[636]),
        Case("Z3", 135, "South Korea", 25, 109210, CLUBS[109210]),
        Case("Z4", 765, "England", 31, 605, CLUBS[605]),
    ],
    ("animagine", "woman"): [
        Case("A1", None, "-", 20, 636, CLUBS[636]),
        Case("A2", None, "-", 24, 605, CLUBS[605]),
        Case("A3", None, "-", 27, 109210, CLUBS[109210]),
        Case("A4", None, "-", 19, 634, CLUBS[634]),
        Case("A5", None, "-", 22, 620, CLUBS[620]),
        Case("A6", None, "-", 29, 640, CLUBS[640]),
    ],
}

MODEL_NAMES = {
    "animagine": "Animagine XL 4.0",
    "realvis": "RealVisXL V5.0",
    "zimage_turbo": "Z-Image-Turbo",
}


def draw(model: str, gender: str, cases: list[Case], tagger: Any) -> list[dict[str, Any]]:
    """Draws one model's cases through the worker; returns a row per current picture."""
    from factories import NEWGEN, club, player, snapshot

    from facegen.core.settings_values import PictureSettings, StyleSettings
    from facegen.datasource.kits import load_kits
    from facegen.persistence import db
    from facegen.persistence.settings import Paths, load_settings
    from facegen.pipeline.sync import run_sync
    from facegen.pipeline.worker import JobFilter, Worker, load_career
    from facegen.platform.memory import release_free_memory
    from facegen.runtime.dependencies import default_renderer

    tmp = Path(tempfile.mkdtemp(prefix=f"facegen-readme-{model}-"))
    s = replace(
        load_settings(),
        paths=Paths(fm_user_dir=tmp / "fm", data_dir=tmp / "data", fm_install_dir=tmp / "install"),
        style=StyleSettings(model=model, gender=gender),
        picture=PictureSettings(aspect="3:4"),
    )
    uids = {NEWGEN + i: c for i, c in enumerate(cases)}
    people = [
        replace(
            player(
                uid,
                name=f"Sample {c.label}",
                ca=170,
                pa=185,
                club_id=c.club,
                age=c.age,
                birth_date=date(2026 - c.age, 1, 1),
            ),
            nation=c.nation,
        )
        for uid, c in uids.items()
    ]
    from factories import DEFAULT_CLUBS

    clubs = dict(DEFAULT_CLUBS) | {c.club: club(c.club) for c in cases}
    conn = db.connect(":memory:")
    run_sync(
        conn, snapshot(people, career=f"readme-{model}", clubs=clubs), s, save_file="career.fm"
    )
    career = load_career(conn)
    assert career is not None

    t = time.perf_counter()
    renderer = default_renderer(model)
    print(f"{model}: loaded in {time.perf_counter() - t:.1f} s", flush=True)
    worker = Worker(
        conn,
        s,
        load_kits(),
        career,
        renderer_factory=lambda _key: renderer,
        tagger_factory=lambda: tagger,
        fm_check=lambda: False,
        log=lambda m: print(f"  {m}", flush=True),
    )
    t = time.perf_counter()
    summary = worker.run(JobFilter(groups=frozenset({"a"})))
    print(
        f"{model}: {len(summary.done)} done, {len(summary.failed)} failed "
        f"in {time.perf_counter() - t:.0f} s",
        flush=True,
    )
    for o in summary.failed:
        print(f"  failed {o.job.unique_id}: {o.message}", flush=True)
    renderer.close()
    release_free_memory()

    rows = []
    for row in conn.execute(
        "SELECT * FROM images WHERE is_current = 1 AND unique_id >= ? ORDER BY unique_id",
        (NEWGEN,),
    ):
        c = uids[row["unique_id"]]
        check = json.loads(row["check_result"])
        if not check.get("passed"):
            continue  # never use a picture that failed its check (hard rule 3)
        render = json.loads(row["render"])
        shutil.copyfile(s.paths.data_dir / row["master_path"], OUT / f"{c.label}.png")
        rows.append(
            {
                "label": c.label,
                "model": model,
                "gender": gender,
                "nation": c.nation_name,
                "age": c.age,
                "kit": c.kit,
                "seed": row["seed"],
                "prompt": row["prompt"],
                "negative": row["negative_prompt"],
                "score": round(chosen_score(check), 4),
                "rating": check["result"]["rating"],
                "detail_pass": render["detail"]["used"],
            }
        )
    shutil.rmtree(tmp, ignore_errors=True)
    return rows


def chosen_score(check: dict[str, Any]) -> float:
    """The check's score of the candidate the worker chose."""
    seed = check["chosen"]["seed"]
    return next(c["score"] for c in check["candidates"] if c["seed"] == seed)


def previews(labels: list[str]) -> None:
    from PIL import Image

    for label in labels:
        with Image.open(OUT / f"{label}.png") as im:
            im.convert("RGB").resize(PREVIEW, Image.Resampling.LANCZOS).save(
                OUT / f"{label}.jpg", quality=90
            )


def hero_tiles() -> list[str]:
    """The existing hero's four 240x320 tiles, cut by position (8 px apart)."""
    from PIL import Image

    labels = []
    w, h = HERO_TILE
    with Image.open(HERO) as im:
        for i in range(4):
            x = i * (w + HERO_GAP)
            label = f"H{i + 1}"
            im.crop((x, 0, x + w, h)).save(OUT / f"{label}.jpg", quality=95)
            labels.append(label)
    return labels


def compose_hero(labels: list[str]) -> None:
    """The README's hero from four picked pictures, left to right."""
    from PIL import Image

    assert len(labels) == 4, labels
    w, h = HERO_TILE
    hero = Image.new("RGB", (4 * w + 3 * HERO_GAP, h), "white")
    for i, label in enumerate(labels):
        source = OUT / f"{label}.png"
        if not source.is_file():
            source = OUT / f"{label}.jpg"  # an old hero tile
        with Image.open(source) as im:
            tile = im.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        hero.paste(tile, (i * (w + HERO_GAP), 0))
    hero.save(HERO, quality=92, optimize=True)
    print(f"wrote {HERO} from {', '.join(labels)}")


def page(rows: list[dict[str, Any]], hero: list[str]) -> None:
    def card(label: str, title: str, lines: list[str], full: str | None) -> str:
        img = f'<img src="{label}.jpg" alt="{label}">'
        if full:
            img = f'<a href="{full}">{img}</a>'
        text = "<br>".join(html.escape(x) for x in lines)
        return f"<figure>{img}<figcaption><b>{label}</b> {html.escape(title)}<br>{text}</figcaption></figure>"

    sections = []
    for model, gender in RUNS:
        mine = [r for r in rows if r["model"] == model]
        cards = [
            card(
                r["label"],
                f"{MODEL_NAMES[model]}, {gender}",
                [
                    f"{r['nation']}, age {r['age']}, {r['kit']}",
                    f"check {r['score']} ({r['rating']}), detail pass {'yes' if r['detail_pass'] else 'no'}",
                ],
                f"{r['label']}.png",
            )
            for r in mine
        ]
        sections.append(
            f"<h2>{MODEL_NAMES[model]} ({gender})</h2><div class=grid>{''.join(cards)}</div>"
        )
    sections.append(
        "<h2>Existing hero (Animagine, women)</h2><div class=grid>"
        + "".join(card(h, "docs/images/hero.jpg", [f"tile {h[1]} of 4"], None) for h in hero)
        + "</div>"
    )
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>README samples</title><style>"
        "body{font:14px system-ui;margin:16px;background:#f4f4f4}"
        ".grid{display:flex;flex-wrap:wrap;gap:12px}"
        "figure{margin:0;background:#fff;padding:8px;border-radius:6px;width:240px}"
        "img{width:240px;height:320px;object-fit:cover;display:block}"
        "figcaption{margin-top:6px;line-height:1.35}</style>"
        "<h1>README samples: pick 2 photo men and 2 anime women</h1>"
        "<p>Click a picture for the full-size master. Every picture passed the check.</p>"
        + "".join(sections),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--only", help="comma-separated model keys to draw")
    parser.add_argument("--hero", help="four comma-separated labels: write the README hero")
    args = parser.parse_args()
    if args.hero:
        compose_hero(args.hero.split(","))
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT / "tests"))
    from facegen.render.tagger import Eva02Tagger

    OUT.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    found = OUT / "samples.json"
    rows: list[dict[str, Any]] = json.loads(found.read_text()) if found.is_file() else []
    tagger = Eva02Tagger()
    for (model, gender), cases in RUNS.items():
        if only is not None and model not in only:
            continue
        rows = [r for r in rows if r["model"] != model]
        new = draw(model, gender, cases, tagger)
        previews([r["label"] for r in new])
        rows += new
        found.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tagger.close()
    page(rows, hero_tiles())
    print(f"{len(rows)} pictures; open {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
