"""Make the bundled kit catalog, src/facegen/data/kit_catalog.json, from a kit colour probe
report (scripts/spikes/kit_colour_probe.py --all-home).

Keeps only what `kits.estimated_kits` reads, as the worker would read the report: each club's
one or two home-shirt colour names, plus the FM27 catalog hash, the probe version and when
the probe ran. No measured colours, texture names, GUIDs, bundle names or images: only our
own estimates leave this PC (decisions.md, 2026-11-13).

    uv run python scripts/build_kit_catalog.py [--report PATH] [--output PATH]

Refresh after an FM27 update: run the probe with --all-home, then this, then the tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from facegen.datasource.kits import (
    BUNDLE_FORMAT,
    CATALOG_REPORT,
    KitError,
    catalog_kit,
    parse_bundle,
    report_made,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / CATALOG_REPORT
OUTPUT = ROOT / "src" / "facegen" / "data" / "kit_catalog.json"
ABOUT = (
    "Home-shirt colours per FM27 club ID, estimated by facegen's kit colour probe from the"
    " game's 2D kit images. Colour names from facegen's vocabulary only. Made by"
    " scripts/build_kit_catalog.py; read by facegen.kits."
)


def reduce(report: Mapping[str, Any], modified: float | None = None) -> dict[str, Any]:
    """The bundle's content for a probe report. Raises KitError for a report that does not
    cover every home kit in the game's catalog (a `--club` run, say)."""
    summary = report.get("summary")
    results = report.get("results")
    if not isinstance(summary, Mapping) or not isinstance(results, list):
        raise KitError("not a probe report: needs 'summary' and 'results'")
    catalog_sha256 = summary.get("catalog_sha256")
    if not isinstance(catalog_sha256, str) or len(catalog_sha256) != 64:
        raise KitError("the report's summary has no catalog_sha256")
    homes = [r for r in results if isinstance(r, Mapping) and r.get("kind") == "home"]
    in_game = (summary.get("kinds") or {}).get("home")
    if not isinstance(in_game, int) or len(homes) < in_game:
        raise KitError(
            f"the report measures {len(homes)} of {in_game} home kits;"
            " run the probe with --all-home"
        )
    made = report_made(report, modified)
    if made is None:
        raise KitError("the report says not when it was made")
    clubs: dict[int, tuple[str, ...]] = {}
    for row in homes:
        kit = catalog_kit(row)
        if kit is not None and isinstance(row.get("club_id"), int):
            clubs[row["club_id"]] = kit.shirt
    return {
        "about": ABOUT,
        "format": BUNDLE_FORMAT,
        "source": {
            "catalog_sha256": catalog_sha256,
            # Reports from before the probe numbered itself came from its first version.
            "probe_version": int(summary.get("probe_version", 1)),
            "made": made.isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
        "clubs": dict(sorted(clubs.items())),
    }


def dumps(bundle: Mapping[str, Any]) -> str:
    """JSON with one club per line, so a refresh shows as a readable diff."""
    head = {k: v for k, v in bundle.items() if k != "clubs"}
    lines = [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    clubs = [f'    "{club}": {json.dumps(list(shirt))}' for club, shirt in bundle["clubs"].items()]
    return "{\n" + "\n".join(lines) + '\n  "clubs": {\n' + ",\n".join(clubs) + "\n  }\n}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=REPORT, help="the probe's report.json")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="the bundle to write")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        bundle = reduce(report, args.report.stat().st_mtime)
        text = dumps(bundle)
        parse_bundle(json.loads(text))  # what facegen will read back
    except (OSError, ValueError, KitError) as exc:
        print(f"{args.report}: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    two = sum(1 for shirt in bundle["clubs"].values() if len(shirt) == 2)
    print(
        f"{args.output}: {len(bundle['clubs'])} clubs ({two} with two colours),"
        f" {len(text.encode()):,} bytes; probe {bundle['source']['probe_version']},"
        f" made {bundle['source']['made']}, catalog {bundle['source']['catalog_sha256'][:12]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
