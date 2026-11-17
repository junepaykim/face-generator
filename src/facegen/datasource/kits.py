"""Load kit colours per club from config/kits.toml and the local or bundled catalog.

A kit is described with a fixed vocabulary of colours and patterns, never free text, so the
table cannot put arbitrary words into a prompt: the wardrobe stays football kit (content
rule, spec section 3). Goalkeepers wear their club's goalkeeper kit. My club's staff do not
wear the kit: wardrobe.py dresses them, with the home kit's colours on one piece. A player's
picture wears the kit in one of shoot.py's looks.

Where a kit comes from, first match wins (`KitTable.kit_for`):

1. config/kits.toml, written by hand: colours, pattern, shorts and goalkeeper kit.
2. Kit colours estimated from FM27's own 2D home-shirt image of the club (user,
   2026-11-12), from one of two places, the newer first:
   - the kit catalog: `scripts/spikes/kit_colour_probe.py` reads the images from the game
     install on this PC and writes its estimates to data/kit-colour-probe/report.json;
   - the bundled catalog, src/facegen/data/kit_catalog.json, which ships with facegen so
     that a PC that never ran the probe has the colours too (Phase 9).
     `scripts/build_kit_catalog.py` makes it from a probe report: club IDs and colour names
     only, never the game's images or asset names.
   The report counts as newer when the probe ran after the bundle's report did (after an
   FM27 update, say); the other one still covers the clubs the newer one lacks, so a probe
   run for one club does not hide the bundle's other clubs.
   An estimate gives one or two shirt colours, never a pattern: two colours are drawn as a
   "mixed" shirt, with no layout asked for, because two colours alone do not say whether
   they are stripes, hoops or sleeves. Its shorts are unknown and left out, and its
   goalkeeper wears the first of KEEPER_COLOURS that is not like the home shirt (FM27's
   goalkeeper colours are not known). A club FM27 has no image for is in neither.
3. The neutral fallback kit.

Stripes and hoops are drawn as well as the model manages (decisions.md); the quality check
does not look at the kit. Its shirt colours only help rank group A's candidates.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from facegen.core.kit_values import (  # noqa: F401 - compatibility exports
    COLOURS,
    KEEPER_COLOURS,
    PATTERNS,
    ClubKits,
    Colour,
    Kit,
    KitError,
    KitSource,
    KitTable,
    _colour,
    estimated_kits,
    families,
    keeper_colour,
    parse_kit,
    parse_kits,
)
from facegen.locations import CONFIG_DIR

KITS_PATH = CONFIG_DIR / "kits.toml"
CATALOG_REPORT = Path("kit-colour-probe") / "report.json"
"""The kit catalog, relative to the data folder: the colour probe's output."""
BUNDLE_NAME = "kit_catalog.json"
"""The bundled catalog, in the package's data folder (src/facegen/data/)."""
BUNDLE_FORMAT = 1


def catalog_kit(row: Mapping[str, Any]) -> Kit | None:
    """The home kit for one row of the colour probe's report; None if it gives no colour."""
    if row.get("status") != "estimated":
        return None
    primary = (row.get("primary") or {}).get("name")
    secondary = (row.get("secondary") or {}).get("name")
    kits = estimated_kits(primary, secondary)
    return kits.home if kits is not None else None


def parse_catalog(report: Mapping[str, Any]) -> dict[int, ClubKits]:
    """Club ID to kits, from the colour probe's report: the home rows with an estimate."""
    results = report.get("results")
    if not isinstance(results, list):
        raise KitError("kit catalog: no 'results' list")
    clubs: dict[int, ClubKits] = {}
    for row in results:
        if not isinstance(row, Mapping) or row.get("kind") != "home":
            continue
        club_id = row.get("club_id")
        if not isinstance(club_id, int) or row.get("status") != "estimated":
            continue
        primary = (row.get("primary") or {}).get("name")
        secondary = (row.get("secondary") or {}).get("name")
        kits = estimated_kits(primary, secondary)
        if kits is not None:
            clubs[club_id] = kits
    return clubs


def report_made(report: Mapping[str, Any], modified: float | None = None) -> datetime | None:
    """When the probe made a report, to the second: its summary's `generated`, or else the
    file's time (`modified`, seconds since the epoch), for reports from before the probe wrote
    it. Whole seconds, as the bundle records it: the report a bundle was built from is not
    newer than the bundle."""
    summary = report.get("summary")
    generated = summary.get("generated") if isinstance(summary, Mapping) else None
    made: datetime | None = None
    if isinstance(generated, str):
        try:
            made = datetime.fromisoformat(generated)
        except ValueError:
            made = None
    if made is None and modified is not None:
        made = datetime.fromtimestamp(modified, UTC)
    if made is None:
        return None
    return (made if made.tzinfo else made.replace(tzinfo=UTC)).replace(microsecond=0)


@dataclass(frozen=True, slots=True)
class Bundle:
    """The bundled catalog: estimated home-shirt colours per club, and where they came from."""

    clubs: Mapping[int, ClubKits]
    made: datetime
    """When the probe made the report the bundle was built from."""
    catalog_sha256: str
    """The FM27 kit catalog the probe read (catalog_info.bin), so a refresh can tell."""
    probe_version: int


def parse_bundle(data: Mapping[str, Any]) -> Bundle:
    """The bundled catalog's JSON. Every row must be one or two known colour names: the file
    ships with facegen, so a row that is not is a mistake, not a club to skip."""
    if data.get("format") != BUNDLE_FORMAT:
        raise KitError(f"bundled kit catalog: format {data.get('format')!r}, not {BUNDLE_FORMAT}")
    source = data.get("source")
    clubs_data = data.get("clubs")
    if not isinstance(source, Mapping) or not isinstance(clubs_data, Mapping):
        raise KitError("bundled kit catalog: needs 'source' and 'clubs'")
    try:
        made = datetime.fromisoformat(str(source["made"]))
        catalog_sha256 = str(source["catalog_sha256"])
        probe_version = int(source["probe_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise KitError(f"bundled kit catalog: bad 'source': {exc}") from exc
    clubs: dict[int, ClubKits] = {}
    for key, shirt in clubs_data.items():
        where = f"bundled kit catalog: club {key}"
        if not str(key).isdigit():
            raise KitError(f"{where}: club keys are the game's unique club IDs")
        if not isinstance(shirt, list) or not 1 <= len(shirt) <= 2 or len(set(shirt)) < len(shirt):
            raise KitError(f"{where}: needs one or two different colours")
        kits = estimated_kits(_colour(where, shirt[0]), *(_colour(where, c) for c in shirt[1:]))
        assert kits is not None  # both names are known colours
        clubs[int(key)] = kits
    return Bundle(
        clubs=clubs,
        made=made if made.tzinfo else made.replace(tzinfo=UTC),
        catalog_sha256=catalog_sha256,
        probe_version=probe_version,
    )


def load_bundle() -> Bundle:
    """The bundled catalog, from inside the installed package."""
    try:
        text = resources.files("facegen").joinpath("data", BUNDLE_NAME).read_text("utf-8")
        data = json.loads(text)
    except (OSError, ValueError) as exc:
        raise KitError(f"bundled kit catalog: {exc}") from exc
    if not isinstance(data, Mapping):
        raise KitError("bundled kit catalog: not a JSON object")
    return parse_bundle(data)


def catalog_path(data_dir: Path) -> Path:
    return data_dir / CATALOG_REPORT


def with_estimates(
    table: KitTable,
    *,
    report: Mapping[int, ClubKits] | None = None,
    made: datetime | None = None,
    bundle: Bundle | None = None,
) -> KitTable:
    """The table with a probe report's and the bundle's estimates. The report, made at `made`,
    goes first when the probe made it after the bundle's report: a refresh after an FM27 update, say. A report
    of unknown age, or the very report the bundle was built from, comes after the bundle."""
    first = report is not None and bundle is not None and made is not None and made > bundle.made
    return replace(
        table,
        catalog=report or {},
        bundled=bundle.clubs if bundle is not None else {},
        report_first=first,
    )


def load_kits(
    path: Path | None = None, *, data_dir: Path | None = None, bundled: bool = True
) -> KitTable:
    """config/kits.toml (or `path`), the kit catalog if the colour probe has written one in
    `data_dir`, and the bundled catalog (unless `bundled` is false).

    No probe report means the bundle alone; a club in neither gets the fallback kit.
    """
    with (path or KITS_PATH).open("rb") as f:
        table = parse_kits(tomllib.load(f))
    bundle = load_bundle() if bundled else None
    catalog = catalog_path(data_dir) if data_dir is not None else None
    if catalog is None or not catalog.is_file():
        return with_estimates(table, bundle=bundle)
    try:
        report = json.loads(catalog.read_text(encoding="utf-8"))
        modified = catalog.stat().st_mtime
    except (OSError, ValueError) as exc:
        raise KitError(f"kit catalog {catalog}: {exc}") from exc
    if not isinstance(report, Mapping):
        raise KitError(f"kit catalog {catalog}: not a JSON object")
    return with_estimates(
        table,
        report=parse_catalog(report),
        made=report_made(report, modified),
        bundle=bundle,
    )
