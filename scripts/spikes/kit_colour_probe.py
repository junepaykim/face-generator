"""Read-only feasibility probe: FM27 team ID -> bundled 2D kit -> estimated colours.

Requires UnityPy 1.25.4, numpy and Pillow in the probe interpreter, or --deps DIR.
Does not import fmsave, touch a save, or change the production kit table. Outputs
belong under data/, never in the game install. Colours are image estimates, not
the authoritative RGB fields from FM's database. No network access is performed.

Example:
  python scripts/spikes/kit_colour_probe.py --install-dir GAME --club 602 636

The catalog reader deliberately validates the observed length/pointer layout and
fails on changes, rather than associating a team with a neighbouring asset by guess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

KEY = re.compile(rb"kit2d_([0-9]+)_(home|away|third)(?![A-Za-z0-9_])")
HERE = Path(__file__).resolve().parents[2]


def catalog_entries(data: bytes) -> dict[str, dict[str, Any]]:
    entries = {}
    for match in KEY.finditer(data):
        start, end = match.span()
        key = match.group().decode("ascii")
        if start < 4 or end + 44 > len(data):
            raise ValueError(f"Truncated catalog entry: {key}")
        if struct.unpack_from("<I", data, start - 4)[0] != end - start:
            raise ValueError(f"Unrecognised catalog key length: {key}")
        if struct.unpack_from("<I", data, end)[0] != 32:
            raise ValueError(f"Unrecognised GUID layout: {key}")
        guid = data[end + 4 : end + 36]
        pointers = struct.unpack_from("<II", data, end + 36)
        if not re.fullmatch(rb"[0-9a-f]{32}", guid) or pointers != (start, end + 4):
            raise ValueError(f"Unverified key/GUID association: {key}")
        if key in entries:
            raise ValueError(f"Duplicate catalog key: {key}")
        entries[key] = {
            "club_id": int(match.group(1)),
            "kind": match.group(2).decode("ascii"),
            "key": key,
            "guid": guid.decode("ascii"),
        }
    if not entries:
        raise ValueError("No recognised 2D team-kit entries in the catalog")
    return entries


def estimate_colours(image: Any) -> dict[str, Any]:
    """A deliberately conservative first pass, excluding common crest/sponsor regions.

    Groups shades of the same hue so that folds do not become a second colour.
    A small logo or trim does not justify inventing a secondary shirt colour.
    Equal stripes/halves are flagged: pixel area cannot establish semantic priority.
    """
    import numpy as np

    rgba = np.asarray(image.convert("RGBA").resize((256, 256)))
    hsv = np.asarray(image.convert("RGB").resize((256, 256)).convert("HSV")) / 255.0
    y, x = np.mgrid[:256, :256] / 255.0
    opaque = rgba[:, :, 3] > 220
    body = (x > 0.28) & (x < 0.72) & (y > 0.54) & (y < 0.88)
    shoulders = (x > 0.28) & (x < 0.72) & (y > 0.10) & (y < 0.16)
    sleeves = ((x < 0.24) | (x > 0.76)) & (y > 0.17) & (y < 0.34)
    clean = opaque & (body | shoulders | sleeves)
    if clean.sum() < 200:
        return {"status": "insufficient_shirt_pixels", "primary": None, "secondary": None}

    h, s, v = hsv[:, :, 0] * 360, hsv[:, :, 1], hsv[:, :, 2]
    # Categories are broad on purpose; output also keeps measured RGB/hex values.
    labels = np.full(h.shape, "red", dtype="<U8")
    for low, high, label in [
        (18, 42, "orange"),
        (42, 72, "yellow"),
        (72, 172, "green"),
        (172, 245, "blue"),
        (245, 300, "purple"),
        (300, 345, "pink"),
    ]:
        labels[(h >= low) & (h < high)] = label
    # Low-saturation pale blue still belongs to blue. Neutral dark/light shades
    # are combined, rather than reporting grey folds as another shirt colour.
    labels[(s < 0.10) & (v >= 0.50)] = "white"
    labels[(s < 0.10) & (v < 0.50)] = "black"
    labels[v < 0.13] = "black"
    counts = Counter(labels[clean].tolist())
    total = int(clean.sum())
    palette = []
    for name, count in counts.most_common():
        mask = clean & (labels == name)
        brightness = v[mask]
        # The brighter middle of each colour is less affected by fabric shadows.
        lo, hi = np.quantile(brightness, [0.55, 0.85])
        selected = mask & (v >= lo) & (v <= hi)
        if not selected.any():
            selected = mask  # Very small groups can fall between quantile endpoints.
        rgb = np.median(rgba[:, :, :3][selected], axis=0).astype(int).tolist()
        rv, gv, bv = (channel / 255 for channel in rgb)
        if name == "blue" and max(rv, gv, bv) < 0.40:
            name = "navy"
        elif name == "blue" and max(rv, gv, bv) > 0.60 and min(rv, gv, bv) > 0.35:
            name = "sky"
        elif name == "red" and max(rv, gv, bv) < 0.50:
            name = "claret"
        palette.append({
            "name": name,
            "hex": "#" + "".join(f"{c:02X}" for c in rgb),
            "rgb": rgb,
            "sample_fraction": round(count / total, 4),
        })
    # A colour near a bucket boundary (cream vs white, for example) can land in
    # two labels. Merge visually near-identical representatives before ranking.
    merged: list[dict[str, Any]] = []
    for colour in palette:
        close = next(
            (
                p for p in merged
                if sum((a - b) ** 2 for a, b in zip(p["rgb"], colour["rgb"])) < 32**2
            ),
            None,
        )
        if close is None:
            merged.append(colour)
            continue
        a, b = close["sample_fraction"], colour["sample_fraction"]
        close["rgb"] = [
            round((x * a + y * b) / (a + b))
            for x, y in zip(close["rgb"], colour["rgb"])
        ]
        close["hex"] = "#" + "".join(f"{c:02X}" for c in close["rgb"])
        close["sample_fraction"] = round(a + b, 4)
    palette = sorted(merged, key=lambda p: p["sample_fraction"], reverse=True)
    primary = palette[0]
    secondary = palette[1] if len(palette) > 1 and palette[1]["sample_fraction"] >= 0.08 else None
    warnings = ["Estimated from shirt pixels; not FM database colour values."]
    if secondary and primary["sample_fraction"] - secondary["sample_fraction"] < 0.15:
        warnings.append("Similar colour areas; primary/secondary order is ambiguous.")
    if any(p["sample_fraction"] >= 0.08 for p in palette[2:]):
        warnings.append("More than two substantial colours; inspect the full palette.")
    return {
        "status": "estimated",
        "primary": primary,
        "secondary": secondary,
        "palette": palette,
        "sample_pixels": total,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--deps", type=Path, help="Optional existing probe-only dependency folder")
    parser.add_argument("--club", nargs="+", type=int, default=[])
    parser.add_argument("--kind", choices=("home", "away", "third", "all"), default="home")
    parser.add_argument("--all-home", action="store_true", help="Measure every mapped home kit")
    parser.add_argument("--index-cache", type=Path, help="Reuse a previously verified index.json")
    parser.add_argument("--output", type=Path, default=HERE / "data" / "kit-colour-probe")
    args = parser.parse_args()
    if not args.club and not args.all_home:
        parser.error("Specify --club ID ... or --all-home")
    if any(club < 0 for club in args.club):
        parser.error("Club IDs must be non-negative")
    install = args.install_dir.resolve()
    out = args.output.resolve()
    # Artifacts only in the project's ignored data tree, never another FM pack.
    if not out.is_relative_to((HERE / "data").resolve()):
        parser.error("--output must be inside this project's data/ folder")
    if out.is_relative_to(install):
        parser.error("Cannot write artifacts inside the game install")
    if args.deps:
        sys.path.insert(0, str(args.deps.resolve()))
    import UnityPy

    started = time.perf_counter()
    root = install / "fm_Data" / "StreamingAssets" / "AddrData"
    data = (root / "catalog_info.bin").read_bytes()
    entries = catalog_entries(data)
    out.mkdir(parents=True, exist_ok=True)
    by_guid: dict[str, list[dict[str, Any]]] = {}
    for entry in entries.values():
        by_guid.setdefault(entry["guid"], []).append(entry)
    kinds = ("home", "away", "third") if args.kind == "all" else (args.kind,)
    wanted = {f"kit2d_{club}_{kind}" for club in args.club for kind in kinds}
    if args.all_home:
        wanted.update(key for key, entry in entries.items() if entry["kind"] == "home")
    results: dict[str, dict[str, Any]] = {}
    for key in sorted(wanted - entries.keys()):
        results[key] = {"key": key, "status": "not_in_catalog", "primary": None, "secondary": None}
    mapped = set()
    seen_this_run = set()
    errors = []
    bundles = sorted(root.glob("kits2d-*.bundle"))
    fingerprint = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in bundles}
    catalog_hash = hashlib.sha256(data).hexdigest()
    scan_bundles = bundles
    if args.index_cache:
        cache = json.loads(args.index_cache.read_text(encoding="utf-8"))
        if cache["catalog_sha256"] != catalog_hash or cache["bundle_fingerprints"] != fingerprint:
            parser.error("The game assets changed; rebuild the index without --index-cache")
        old = cache["entries"]
        if old.keys() != entries.keys() or any(
            old[k]["guid"] != e["guid"] for k, e in entries.items()
        ):
            parser.error("Cached key/GUID map does not agree with the catalog")
        for key, entry in entries.items():
            if "bundle" in old[key]:
                if old[key]["bundle"] not in fingerprint:
                    parser.error(f"Unrecognised cached bundle for {key}")
                fields = ("bundle", "path_id", "texture_name")
                entry.update({field: old[key][field] for field in fields})
                mapped.add(key)
        needed = {entries[k]["bundle"] for k in wanted & mapped}
        scan_bundles = [p for p in bundles if p.name in needed]
    for number, path in enumerate(scan_bundles, 1):
        try:
            env = UnityPy.load(str(path))
            for guid, ref in env.container.items():
                if guid not in by_guid:
                    continue
                obj = ref.deref()
                if obj.type.name not in ("Texture2D", "Sprite"):
                    raise ValueError(f"Unexpected {obj.type.name} for {guid}")
                for entry in by_guid[guid]:
                    key = entry["key"]
                    if key in seen_this_run:
                        raise ValueError(f"Ambiguous asset mapping: {key}")
                    seen_this_run.add(key)
                    entry.update(
                        bundle=path.name, path_id=obj.path_id, texture_name=obj.peek_name()
                    )
                    mapped.add(key)
                    if key not in wanted:
                        continue
                    image = obj.read().image.convert("RGBA")
                    row = {**entry, "size": list(image.size), **estimate_colours(image)}
                    if entry["club_id"] in args.club:
                        image.save(out / f"{key}.png")
                        row["image"] = f"{key}.png"
                    results[key] = row
            del env
        except Exception as exc:  # noqa: BLE001 - recorded per bundle in the report
            errors.append({"bundle": path.name, "error": str(exc)})
        if number % 10 == 0 or number == len(scan_bundles):
            print(
                f"Scanned {number}/{len(scan_bundles)} bundles; mapped {len(mapped)} kit keys",
                flush=True,
            )
    for key in sorted(wanted & entries.keys() - results.keys()):
        results[key] = {
            **entries[key], "status": "extraction_failed", "primary": None, "secondary": None
        }
    summary = {
        "catalog_sha256": catalog_hash,
        "unitypy_version": UnityPy.__version__,
        "kit_keys": len(entries),
        "team_ids": len({e["club_id"] for e in entries.values()}),
        "kinds": dict(Counter(e["kind"] for e in entries.values())),
        "mapped_keys": len(mapped),
        "unmapped_keys": sorted(entries.keys() - mapped),
        "bundles": len(bundles),
        "scanned_bundles": len(scan_bundles),
        "mapping_source": "validated_cache" if args.index_cache else "full_scan",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "result_statuses": dict(Counter(r["status"] for r in results.values())),
        "errors": errors,
    }
    index = {"catalog_sha256": catalog_hash, "bundle_fingerprints": fingerprint, "entries": entries}
    (out / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {"summary": summary, "results": [results[k] for k in sorted(results)]}
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for row in results.values():
        if row.get("club_id") in args.club or row["status"] != "estimated":
            colours = [row.get(k) for k in ("primary", "secondary")]
            print(row["key"], row.get("texture_name", ""), row["status"],
                  [(c["name"], c["hex"], c["sample_fraction"]) for c in colours if c])


if __name__ == "__main__":
    main()
