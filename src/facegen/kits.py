"""Kit colours per club (docs/spec.md section 3), from config/kits.toml. Pure, apart from the
loaders at the end.

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
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from facegen.locations import CONFIG_DIR

KITS_PATH = CONFIG_DIR / "kits.toml"
CATALOG_REPORT = Path("kit-colour-probe") / "report.json"
"""The kit catalog, relative to the data folder: the colour probe's output."""
BUNDLE_NAME = "kit_catalog.json"
"""The bundled catalog, in the package's data folder (src/facegen/data/)."""
BUNDLE_FORMAT = 1

type KitSource = Literal["table", "catalog", "bundled", "fallback"]
"""Where a kit came from: config/kits.toml, the kit catalog (a probe report on this PC), the
bundled catalog, or the fallback kit."""


@dataclass(frozen=True, slots=True)
class Colour:
    words: str
    """Prompt word."""
    shirt_tags: frozenset[str]
    """Tagger tags for a shirt in this colour."""
    families: frozenset[str]
    """Broad colour families, to tell which backdrops would blend with it (shoot.py)."""


def _c(words: str, tags: tuple[str, ...], families: tuple[str, ...]) -> Colour:
    return Colour(words, frozenset(tags), frozenset(families))


COLOURS: dict[str, Colour] = {
    "red": _c("red", ("red shirt",), ("red",)),
    "claret": _c("dark red", ("red shirt",), ("red",)),
    "white": _c("white", ("white shirt",), ("white",)),
    "black": _c("black", ("black shirt",), ("black",)),
    "blue": _c("blue", ("blue shirt",), ("blue",)),
    "sky": _c("light blue", ("blue shirt",), ("blue",)),
    "navy": _c("navy blue", ("blue shirt",), ("blue",)),
    "yellow": _c("yellow", ("yellow shirt",), ("yellow",)),
    "amber": _c("orange", ("orange shirt", "yellow shirt"), ("orange", "yellow")),
    "orange": _c("orange", ("orange shirt",), ("orange",)),
    "green": _c("green", ("green shirt",), ("green",)),
    "purple": _c("purple", ("purple shirt",), ("purple",)),
    "grey": _c("grey", ("grey shirt",), ("grey",)),
    "pink": _c("pink", ("pink shirt",), ("pink",)),
}

# pattern -> needs a second shirt colour
PATTERNS: dict[str, bool] = {
    "plain": False,
    "stripes": True,  # vertical
    "hoops": True,  # horizontal
    "sleeves": True,  # body in the first colour, sleeves in the second
    "two-tone": True,  # halves or quarters
    "mixed": True,  # two colours, layout not known (the kit catalog's estimates)
}

KEEPER_COLOURS = ("green", "yellow", "orange", "purple")
"""A catalog club's goalkeeper wears the first of these unlike every home shirt colour."""


def families(colours: tuple[str, ...]) -> frozenset[str]:
    """The colour families of some kit colours."""
    return frozenset(f for c in colours for f in COLOURS[c].families)


class KitError(ValueError):
    """config/kits.toml has an entry the vocabulary does not cover."""


@dataclass(frozen=True, slots=True)
class Kit:
    shirt: tuple[str, ...]
    pattern: str = "plain"
    shorts: str | None = None
    trim: str | None = None
    """Collar colour."""
    gloves: str | None = None
    goalkeeper: bool = False

    def words(self) -> str:
        """Prompt words for this kit as a plain match kit: short sleeves and shorts, or for a
        goalkeeper long sleeves and gloves (as tested in the bake-off: no shorts)."""
        if self.goalkeeper:
            return f"goalkeeper, {self.shirt_words()}, long sleeves, {self.gloves_words()}"
        return ", ".join(p for p in (self.shirt_words(), "short sleeves", self.shorts_words()) if p)

    def shirt_words(self) -> str:
        """The shirt alone: colours, pattern and collar."""
        c = [COLOURS[x].words for x in self.shirt]
        match self.pattern:
            case "plain":
                parts = [f"{c[0]} shirt"]
            case "stripes":
                parts = [f"{c[0]} and {c[1]} vertical-striped shirt"]
            case "hoops":
                parts = [f"{c[0]} and {c[1]} horizontal-striped shirt"]
            case "sleeves":
                parts = [f"{c[0]} shirt", f"{c[1]} sleeves"]
            case "two-tone":
                parts = [f"{c[0]} and {c[1]} two-tone shirt"]
            case "mixed":
                parts = [f"{c[0]} and {c[1]} shirt"]
            case _:
                raise KitError(f"unknown pattern {self.pattern!r}")
        if self.trim:
            parts.append(f"{COLOURS[self.trim].words} collar")
        return ", ".join(parts)

    def jacket_words(self) -> str:
        """The club track jacket in this kit's shirt colours (and collar)."""
        c = [COLOURS[x].words for x in self.shirt]
        if self.pattern == "sleeves":
            parts = [f"{c[0]} track jacket", f"{c[1]} sleeves"]
        else:  # stripes and hoops are not asked of a jacket: its two colours are enough
            parts = [f"{' and '.join(c)} track jacket"]
        if self.trim:
            parts.append(f"{COLOURS[self.trim].words} collar")
        return ", ".join(parts)

    def shorts_words(self) -> str:
        """The shorts, or nothing where they are not known."""
        return f"{COLOURS[self.shorts].words} shorts" if self.shorts else ""

    def gloves_words(self) -> str:
        return f"{COLOURS[self.gloves or 'black'].words} gloves"

    def colours(self) -> tuple[str, ...]:
        """Every colour of the shirt, collar and shorts, without repeats."""
        return tuple(dict.fromkeys(c for c in (*self.shirt, self.trim, self.shorts) if c))

    def shirt_tag_groups(self, garment: str = "shirt") -> tuple[frozenset[str], ...]:
        """One group per shirt colour; any tag in a group shows that colour was drawn.

        `garment` names what the colour is on: "jacket" gives "blue jacket" for "blue shirt".
        """
        groups = tuple(COLOURS[x].shirt_tags for x in dict.fromkeys(self.shirt))
        if garment == "shirt":
            return groups
        return tuple(frozenset(t.replace(" shirt", f" {garment}") for t in g) for g in groups)

    def rank_tags(self) -> dict[str, frozenset[str]]:
        """Ranking only: each shirt colour to the tags that show it was drawn."""
        return {f"shirt colour {i}": g for i, g in enumerate(self.shirt_tag_groups(), 1)}

    def describe(self) -> str:
        shirt = " and ".join(self.shirt)
        text = f"{shirt} {self.pattern}" if self.pattern not in ("plain", "mixed") else shirt
        if self.goalkeeper:
            return f"goalkeeper {text}"
        return f"{text}, {self.shorts} shorts" if self.shorts else text


@dataclass(frozen=True, slots=True)
class ClubKits:
    home: Kit
    goalkeeper: Kit


@dataclass(frozen=True, slots=True)
class KitTable:
    clubs: Mapping[int, ClubKits]
    fallback: ClubKits
    catalog: Mapping[int, ClubKits] = field(default_factory=dict)
    """Clubs whose colours a probe report on this PC estimated, by unique club ID."""
    bundled: Mapping[int, ClubKits] = field(default_factory=dict)
    """Clubs whose colours the bundled catalog has, by unique club ID."""
    report_first: bool = False
    """The probe report is newer than the bundle, so its estimates come before the bundle's."""

    def kit_for(self, club_id: int | None, goalkeeper: bool) -> tuple[Kit, KitSource]:
        """The kit to draw and where it came from: the table, the catalog, the bundle or the
        fallback.

        Staff, goalkeeping coaches included, take the home kit: its colours dress them.
        """
        kits: ClubKits = self.fallback
        source: KitSource = "fallback"
        estimates: list[tuple[Mapping[int, ClubKits], KitSource]] = [
            (self.catalog, "catalog"),
            (self.bundled, "bundled"),
        ]
        if not self.report_first:
            estimates.reverse()
        if club_id is not None:
            for found, where in ((self.clubs, "table"), *estimates):
                if club_id in found:
                    kits, source = found[club_id], where
                    break
        return (kits.goalkeeper if goalkeeper else kits.home), source


def _colour(where: str, value: Any) -> str:
    if value not in COLOURS:
        raise KitError(f"{where}: unknown colour {value!r}; use one of {', '.join(COLOURS)}")
    return value


def parse_kit(where: str, data: Mapping[str, Any], *, goalkeeper: bool) -> Kit:
    allowed = {"shirt", "pattern", "trim"} | ({"gloves"} if goalkeeper else {"shorts"})
    unknown = set(data) - allowed
    if unknown:
        raise KitError(f"{where}: unknown keys {sorted(unknown)}")
    shirt = data.get("shirt")
    if not isinstance(shirt, list) or not 1 <= len(shirt) <= 2:
        raise KitError(f"{where}: 'shirt' must list one or two colours")
    shirt_t = tuple(_colour(f"{where}.shirt", x) for x in shirt)
    pattern = data.get("pattern", "plain")
    if pattern not in PATTERNS:
        raise KitError(f"{where}: unknown pattern {pattern!r}; use one of {', '.join(PATTERNS)}")
    if PATTERNS[pattern] != (len(shirt_t) == 2):
        need = "two shirt colours" if PATTERNS[pattern] else "one shirt colour"
        raise KitError(f"{where}: pattern {pattern!r} needs {need}")
    shorts = data.get("shorts")
    if not goalkeeper and shorts is None:
        raise KitError(f"{where}: 'shorts' is required")
    return Kit(
        shirt=shirt_t,
        pattern=pattern,
        shorts=_colour(f"{where}.shorts", shorts) if shorts is not None else None,
        trim=_colour(f"{where}.trim", data["trim"]) if "trim" in data else None,
        gloves=_colour(f"{where}.gloves", data["gloves"]) if "gloves" in data else None,
        goalkeeper=goalkeeper,
    )


def _club_kits(where: str, data: Mapping[str, Any]) -> ClubKits:
    unknown = set(data) - {"home", "goalkeeper"}
    if unknown or "home" not in data or "goalkeeper" not in data:
        raise KitError(f"{where}: needs exactly 'home' and 'goalkeeper'")
    return ClubKits(
        home=parse_kit(f"{where}.home", data["home"], goalkeeper=False),
        goalkeeper=parse_kit(f"{where}.goalkeeper", data["goalkeeper"], goalkeeper=True),
    )


def parse_kits(data: Mapping[str, Any]) -> KitTable:
    if "fallback" not in data:
        raise KitError("kits: a [fallback] kit is required")
    clubs: dict[int, ClubKits] = {}
    for key, value in data.get("clubs", {}).items():
        if not str(key).isdigit():
            raise KitError(f"clubs.{key}: club keys are the game's unique club IDs")
        clubs[int(key)] = _club_kits(f"clubs.{key}", value)
    return KitTable(clubs=clubs, fallback=_club_kits("fallback", data["fallback"]))


def keeper_colour(home: Kit) -> str:
    """A goalkeeper colour unlike the home shirt's, for a club whose keeper kit is unknown."""
    taken = families((*home.shirt, *((home.trim,) if home.trim else ())))
    for colour in KEEPER_COLOURS:
        if not COLOURS[colour].families & taken:
            return colour
    return KEEPER_COLOURS[0]


def estimated_kits(primary: Any, secondary: Any = None) -> ClubKits | None:
    """Kits from estimated home-shirt colour names; None if the first is not a known colour.

    Only the broad colour names are used, never measured RGB values. A second colour makes a
    "mixed" shirt: an estimate does not say how the colours are laid out.
    """
    if primary not in COLOURS:
        return None
    if secondary in COLOURS and secondary != primary:
        home = Kit(shirt=(primary, secondary), pattern="mixed")
    else:
        home = Kit(shirt=(primary,))
    return ClubKits(home=home, goalkeeper=Kit(shirt=(keeper_colour(home),), goalkeeper=True))


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
