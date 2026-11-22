"""Phase 15's exit test: Animagine draws women bit for bit as before Phase 15.

Runs the real worker with Animagine and the tagger on a small made-up world from the tests'
factories (my club's player and goalkeeper, a rival's star, my club's coach and a journalist),
then "new picture" for my club's player (a re-draw with a reference), in a temporary folder
with publishing on. Writes each picture's prompt, negative, seed, candidates and a hash of the
master's pixels and of the published file's pixels to a JSON file. Run it against the code
before Phase 15 and against today's, then compare:

    git worktree add --detach <scratch>/before-15 4c0bdf4
    uv run --no-sync python scripts/spikes/animagine_women_unchanged.py \
        --src <scratch>/before-15 --out <scratch>/before.json
    uv run --no-sync python scripts/spikes/animagine_women_unchanged.py --out <scratch>/now.json
    uv run --no-sync python scripts/spikes/animagine_women_unchanged.py \
        --compare <scratch>/before.json <scratch>/now.json

Needs the GPU, Animagine, h94/IP-Adapter and the tagger in the Hugging Face cache, and FM27
closed. Runs offline; nothing is written outside the temporary folder and --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _pixels(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


def run(src: Path, out: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path[:0] = [str(src / "src"), str(src / "tests")]
    import facegen

    assert Path(facegen.__file__).resolve().is_relative_to(src.resolve() / "src"), facegen.__file__
    from factories import RIVAL, media_person, player, snapshot, staff_member

    from facegen.datasource.kits import load_kits
    from facegen.persistence import db
    from facegen.persistence.settings import Paths, load_settings
    from facegen.pipeline import manual
    from facegen.pipeline.sync import run_sync
    from facegen.pipeline.worker import JobFilter, Worker, load_career
    from facegen.render.renderer import AnimagineRenderer
    from facegen.render.tagger import Eva02Tagger

    tmp = Path(tempfile.mkdtemp(prefix="facegen-women-"))
    settings = load_settings()
    if hasattr(settings, "style"):
        assert (settings.style.model, settings.style.gender) == ("animagine", "woman")
    s = replace(
        settings,
        paths=Paths(fm_user_dir=tmp / "fm", data_dir=tmp / "data", fm_install_dir=tmp / "install"),
    )
    world = [
        player(1001, name="Old Hand", age=29),
        player(1002, name="Keeper", age=17, goalkeeper=True),
        player(1003, name="Star", ca=160, pa=180, club_id=RIVAL, age=19),
    ]
    conn = db.connect(":memory:")
    run_sync(
        conn,
        snapshot(world, staff=[staff_member(5001, age=52)], media=[media_person(6001, age=38)]),
        s,
        save_file="career.fm",
    )
    renderer, tagger = AnimagineRenderer(), Eva02Tagger()
    career = load_career(conn)
    assert career is not None

    def worker() -> Any:
        return Worker(
            conn,
            s,
            load_kits(),
            career,
            renderer_factory=lambda *_: renderer,  # no argument before Phase 15d
            tagger_factory=lambda: tagger,
            fm_check=lambda: False,
        )

    summary = worker().run(JobFilter())
    print("first round:", len(summary.done), "done,", len(summary.failed), "failed")
    manual.new_picture(conn, career.id, 1001, s)
    summary = worker().run(JobFilter())
    print("new picture:", len(summary.done), "done,", len(summary.failed), "failed")

    pictures = []
    for row in conn.execute("SELECT * FROM images ORDER BY unique_id, version"):
        check = json.loads(row["check_result"] or "null")
        render = json.loads(row["render"] or "null")
        published = (
            s.paths.fm_user_dir / "graphics" / "facegen" / career.key / f"{row['unique_id']}.png"
        )
        pictures.append(
            {
                "uid": row["unique_id"],
                "version": row["version"],
                "current": row["is_current"],
                "model": row["model"],
                "prompt": row["prompt"],
                "negative": row["negative_prompt"],
                "seed": row["seed"],
                "candidates": [
                    (c["seed"], c["passed"], round(c["score"], 6))
                    for c in (check or {}).get("candidates", [])
                ],
                "master": _pixels(s.paths.data_dir / row["master_path"])
                if row["master_path"]
                else None,
                "published": _pixels(published)
                if row["is_current"] and published.is_file()
                else None,
                "reference": (render or {}).get("reference"),
                "render": render,
            }
        )
    out.write_text(json.dumps({"src": str(src), "pictures": pictures}, indent=1))
    print(f"{len(pictures)} pictures written to {out}")
    renderer.close()
    tagger.close()


SAME = ("uid", "version", "current", "model", "prompt", "negative", "seed", "candidates")
PIXELS = ("master", "published")


def _untimed(render: dict[str, Any] | None) -> dict[str, Any]:
    """A picture's record without its timings (the job's and the detail pass's seconds)."""
    out = {k: v for k, v in (render or {}).items() if k != "seconds"}
    if isinstance(out.get("detail"), dict):
        out["detail"] = {k: v for k, v in out["detail"].items() if k != "seconds"}
    return out


def compare(before: Path, now: Path) -> int:
    a = json.loads(before.read_text())["pictures"]
    b = json.loads(now.read_text())["pictures"]
    if len(a) != len(b):
        print(f"{len(a)} pictures before, {len(b)} now")
        return 1
    bad = 0
    for x, y in zip(a, b, strict=True):
        name = f"{x['uid']} v{x['version']}"
        diff = [k for k in (*SAME, *PIXELS) if x[k] != y[k]]
        ref = x["reference"] and {k: v for k, v in x["reference"].items() if k != "image_id"}
        ref_now = y["reference"] and {k: v for k, v in y["reference"].items() if k != "image_id"}
        if ref != ref_now:
            diff.append("reference")
        before_render, now_render = _untimed(x["render"]), _untimed(y["render"])
        extra = sorted(set(now_render) - set(before_render))
        changed = sorted(
            k
            for k in set(before_render) & set(now_render)
            if before_render[k] != now_render[k] and k != "reference"
        )
        status = "DIFFERENT " + ", ".join(diff) if diff else "same bytes"
        print(f"{name}: {status}; record keys added {extra}, changed {changed}")
        bad += bool(diff)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=ROOT, help="the source tree to run")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "NOW"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.out is None:
        ap.error("--out is needed to run")
    run(args.src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
