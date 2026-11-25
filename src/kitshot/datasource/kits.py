"""Load fixed kit colours from configuration, FM27's database and local/bundled catalogs.

config/kits.toml wins; then FM27's own kits, read from the game's database on this PC
(game_kits.py); then the newer catalog estimate, the older catalog for missing clubs, and the
neutral fallback. Unknown report dates put the bundle first. Probe reports come from
scripts/spikes/kit_colour_probe.py; scripts/build_kit_catalog.py bundles only club IDs and
colour names, never game images or asset names. Nothing of FM27's database is bundled.

The vocabulary prevents free-text prompt injection and keeps player clothes football kit.
FM27's kits give the home shirt's colours and pattern and the goalkeeper's shirt colour, not
the shorts. Estimates contain shirt colours only: no known pattern or shorts, and goalkeeper
colours that contrast with the home shirt. Staff use home colours through the wardrobe.
Shirt-colour tags rank candidates only; they cannot fail QC.
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

from kitshot.core.kit_values import (  # noqa: F401 - compatibility exports
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
from kitshot.datasource.game_kits import load_game_kits
from kitshot.locations import CONFIG_DIR

KITS_PATH = CONFIG_DIR / "kits.toml"
CATALOG_REPORT = Path("kit-colour-probe") / "report.json"
"""The kit catalog, relative to the data folder: the colour probe's output."""
BUNDLE_NAME = "kit_catalog.json"
"""The bundled catalog, in the package's data folder (src/kitshot/data/)."""
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
    ships with Kitshot, so a row that is not is a mistake, not a club to skip."""
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
        assert kits is not None
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
        text = resources.files("kitshot").joinpath("data", BUNDLE_NAME).read_text("utf-8")
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
    path: Path | None = None,
    *,
    data_dir: Path | None = None,
    install_dir: Path | None = None,
    bundled: bool = True,
) -> KitTable:
    """config/kits.toml (or `path`), FM27's own kits when `install_dir` holds a database this
    version reads (cached in `data_dir`), the kit catalog if the colour probe has written one
    in `data_dir`, and the bundled catalog (unless `bundled` is false).

    No probe report means the bundle alone; a club in none of them gets the fallback kit. An
    unreadable database is left out with a warning in the log, never an error: the other
    sources still apply.
    """
    with (path or KITS_PATH).open("rb") as f:
        table = parse_kits(tomllib.load(f))
    game = load_game_kits(install_dir, data_dir) if install_dir is not None else None
    if game is not None:
        table = replace(table, game=game.kits())
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
