from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path

import pytest
from support.paths import ROOT

from kitshot.core.prompt import CONTENT_NEGATIVE
from kitshot.datasource.kits import (
    BUNDLE_NAME,
    CATALOG_REPORT,
    COLOURS,
    KEEPER_COLOURS,
    PATTERNS,
    Bundle,
    Kit,
    KitError,
    catalog_kit,
    families,
    keeper_colour,
    load_bundle,
    load_kits,
    parse_bundle,
    parse_catalog,
    parse_kits,
    report_made,
    with_estimates,
)

UNKNOWN_CLUB = 987654321
"""An ID in no kit source: not in the table or the bundle."""


def table(**clubs: dict) -> dict:
    return {
        "fallback": {"home": {"shirt": ["white"], "shorts": "black"}, "goalkeeper": {"shirt": ["green"]}},
        "clubs": clubs,
    }  # fmt: skip


def test_kit_words_per_pattern() -> None:
    def home(**kw: object) -> str:
        return Kit(**kw).words()  # type: ignore[arg-type]

    assert home(shirt=("red",), shorts="white") == "red shirt, short sleeves, white shorts"
    assert (
        home(shirt=("blue", "white"), pattern="stripes", shorts="blue")
        == "blue and white vertical-striped shirt, short sleeves, blue shorts"
    )
    assert (
        home(shirt=("yellow", "green"), pattern="hoops", shorts="green")
        == "yellow and green horizontal-striped shirt, short sleeves, green shorts"
    )
    assert (
        home(shirt=("claret", "sky"), pattern="sleeves", shorts="white")
        == "dark red shirt, light blue sleeves, short sleeves, white shorts"
    )
    assert (
        home(shirt=("white",), trim="navy", shorts="navy")
        == "white shirt, navy blue collar, short sleeves, navy blue shorts"
    )


def test_goalkeeper_kit_words_match_the_bake_off() -> None:
    gk = Kit(shirt=("green",), goalkeeper=True)
    assert gk.words() == "goalkeeper, green shirt, long sleeves, black gloves"
    gk = Kit(shirt=("purple",), gloves="grey", goalkeeper=True)
    assert gk.words() == "goalkeeper, purple shirt, long sleeves, grey gloves"


def test_unknown_club_gets_the_fallback() -> None:
    kits = parse_kits(
        table(**{"636": {"home": {"shirt": ["blue"], "shorts": "blue"},
                         "goalkeeper": {"shirt": ["yellow"]}}})
    )  # fmt: skip
    kit, source = kits.kit_for(636, goalkeeper=False)
    assert source == "table" and kit.shirt == ("blue",) and not kit.goalkeeper
    kit, source = kits.kit_for(636, goalkeeper=True)
    assert source == "table" and kit.shirt == ("yellow",) and kit.goalkeeper
    kit, source = kits.kit_for(999, goalkeeper=False)
    assert source == "fallback" and kit.shirt == ("white",)
    kit, source = kits.kit_for(None, goalkeeper=True)
    assert source == "fallback" and kit.goalkeeper


@pytest.mark.parametrize(
    "bad",
    [
        {"home": {"shirt": ["gold"], "shorts": "black"}, "goalkeeper": {"shirt": ["green"]}},
        {"home": {"shirt": ["red"]}, "goalkeeper": {"shirt": ["green"]}},  # no shorts
        {"home": {"shirt": ["red"], "pattern": "stripes", "shorts": "red"},
         "goalkeeper": {"shirt": ["green"]}},  # stripes need two colours
        {"home": {"shirt": ["red", "white"], "shorts": "red"},
         "goalkeeper": {"shirt": ["green"]}},  # plain takes one
        {"home": {"shirt": ["red"], "shorts": "red", "words": "anything"},
         "goalkeeper": {"shirt": ["green"]}},  # no free text
        {"home": {"shirt": ["red"], "shorts": "red"}},  # no goalkeeper kit
    ],
)  # fmt: skip
def test_bad_entries_are_refused(bad: dict) -> None:
    with pytest.raises(KitError):
        parse_kits(table(**{"636": bad}))


def test_club_keys_are_ids() -> None:
    entry = {"home": {"shirt": ["red"], "shorts": "red"}, "goalkeeper": {"shirt": ["green"]}}
    with pytest.raises(KitError):
        parse_kits(table(colchester=entry))
    with pytest.raises(KitError):
        parse_kits({"clubs": {}})  # no fallback


def test_colour_words_are_plain_colours() -> None:
    """Kits can only add colour and garment words to a prompt (content rule)."""
    for colour in COLOURS.values():
        assert all(part.isalpha() for part in colour.words.split()), colour
        assert colour.words not in CONTENT_NEGATIVE


def test_the_shipped_table_loads_and_covers_my_league() -> None:
    kits = load_kits(ROOT / "config" / "kits.toml")
    league_two = {636, 109210, 605, 620, 631, 634, 640, 641, 651, 5103834, 656, 658, 581, 689,
                  694, 698, 703, 704, 5103842, 710, 725, 729, 731, 744}  # fmt: skip
    assert league_two <= set(kits.clubs)
    colchester, source = kits.kit_for(636, goalkeeper=False)
    assert source == "table"
    assert colchester.pattern == "stripes" and set(colchester.shirt) == {"blue", "white"}
    for club in kits.clubs.values():
        assert club.goalkeeper.goalkeeper and not club.home.goalkeeper
        assert club.home.words() and club.goalkeeper.words()


def test_rank_tags_name_each_shirt_colour() -> None:
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    assert kit.rank_tags() == {
        "shirt colour 1": frozenset({"blue shirt"}),
        "shirt colour 2": frozenset({"white shirt"}),
    }


# --- the kit catalog (colours estimated from FM27's kit images) --------------------------------


def estimate(club_id: int, primary: str | None, secondary: str | None = None, **row: object):
    def colour(name: str | None) -> dict | None:
        return None if name is None else {"name": name, "hex": "#000000", "rgb": [0, 0, 0]}

    return {
        "club_id": club_id,
        "kind": "home",
        "status": "estimated",
        "primary": colour(primary),
        "secondary": colour(secondary),
        **row,
    }


REPORT = {
    "summary": {},
    "results": [
        estimate(602, "red", "white"),
        estimate(679, "sky"),
        estimate(1569, "white", "green"),
        estimate(688, "black", "white", kind="away"),  # only home kits are used
        {"key": "kit2d_1708_home", "club_id": 1708, "kind": "home", "status": "not_in_catalog",
         "primary": None, "secondary": None},
        estimate(7, "gold"),  # a name outside the vocabulary: unusable, never guessed
        estimate(8, "blue", "blue"),  # the same name twice is one colour
    ],
}  # fmt: skip


def test_catalog_rows_become_kits_without_a_pattern() -> None:
    clubs = parse_catalog(REPORT)
    assert set(clubs) == {602, 679, 1569, 8}
    arsenal = clubs[602].home
    # Two colours, no layout: the probe does not say whether they are stripes or sleeves.
    assert arsenal.shirt == ("red", "white") and arsenal.pattern == "mixed"
    assert arsenal.words() == "red and white shirt, short sleeves"  # shorts unknown
    assert arsenal.describe() == "red and white"
    assert clubs[679].home == Kit(shirt=("sky",))
    assert clubs[8].home == Kit(shirt=("blue",))
    for club in clubs.values():
        assert club.goalkeeper.goalkeeper and len(club.goalkeeper.shirt) == 1


def test_catalog_skips_what_it_cannot_use() -> None:
    assert catalog_kit({"status": "not_in_catalog", "primary": None}) is None
    assert catalog_kit({"status": "insufficient_shirt_pixels", "primary": None}) is None
    assert catalog_kit(estimate(1, None)) is None
    with pytest.raises(KitError):
        parse_catalog({"summary": {}})


def test_catalog_goalkeepers_contrast_with_the_home_shirt() -> None:
    assert keeper_colour(Kit(shirt=("red",))) == "green"
    assert keeper_colour(Kit(shirt=("white", "green"), pattern="mixed")) == "yellow"
    assert keeper_colour(Kit(shirt=("green", "yellow"), pattern="mixed")) == "orange"
    # Amber is both orange and yellow.
    assert keeper_colour(Kit(shirt=("green", "amber"), pattern="mixed")) == "purple"
    assert set(KEEPER_COLOURS) <= set(COLOURS)


def write_report(data_dir: Path, *rows: dict, generated: str | None = None) -> Path:
    """A probe report where the worker looks for it: data/kit-colour-probe/report.json."""
    path = data_dir / CATALOG_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"generated": generated} if generated else {}
    path.write_text(json.dumps({"summary": summary, "results": list(rows)}))
    return path


def test_the_table_comes_before_the_catalog(tmp_path: Path) -> None:
    write_report(tmp_path, estimate(636, "red"), estimate(602, "red"))
    kits = load_kits(ROOT / "config" / "kits.toml", data_dir=tmp_path, bundled=False)
    colchester, source = kits.kit_for(636, goalkeeper=False)
    assert source == "table" and colchester.pattern == "stripes"
    _, source = kits.kit_for(602, goalkeeper=False)
    assert source == "table"  # hand-written: verified against FM27's image
    other = tmp_path / "other"
    write_report(other, estimate(679, "sky"))
    kits = load_kits(data_dir=other, bundled=False)
    city, source = kits.kit_for(679, goalkeeper=False)
    assert source == "catalog" and city.shirt == ("sky",)
    keeper, source = kits.kit_for(679, goalkeeper=True)
    assert source == "catalog" and keeper.goalkeeper and keeper.shirt == ("green",)
    _, source = kits.kit_for(UNKNOWN_CLUB, goalkeeper=False)
    assert source == "fallback"


def test_no_catalog_file_means_the_bundle_alone(tmp_path: Path) -> None:
    kits = load_kits(data_dir=tmp_path)
    assert kits.catalog == {} and len(kits.bundled) > 800
    assert load_kits(data_dir=tmp_path, bundled=False).bundled == {}
    broken = tmp_path / CATALOG_REPORT
    broken.parent.mkdir()
    broken.write_text("{not json")
    with pytest.raises(KitError):
        load_kits(data_dir=tmp_path)


# --- the bundled catalog (src/kitshot/data/kit_catalog.json) -----------------------------------


def test_a_club_in_the_table_in_the_bundle_or_in_neither(tmp_path: Path) -> None:
    """On a PC that never ran the probe: the table first, then the bundle, then the fallback."""
    kits = load_kits(data_dir=tmp_path)
    colchester, source = kits.kit_for(636, goalkeeper=False)  # in both: the table wins
    assert source == "table" and colchester.pattern == "stripes"
    assert 636 in kits.bundled
    city, source = kits.kit_for(679, goalkeeper=False)  # Manchester City: the bundle only
    assert source == "bundled" and city.shirt == ("sky",) and city.shorts is None
    keeper, source = kits.kit_for(679, goalkeeper=True)
    assert source == "bundled" and keeper.goalkeeper and keeper.shirt == ("green",)
    _, source = kits.kit_for(UNKNOWN_CLUB, goalkeeper=False)  # in no source
    assert source == "fallback"
    # FM27 ships no kit image for Barcelona: without the game's database (test_game_kits.py),
    # it has no kit either.
    assert kits.kit_for(1708, goalkeeper=False)[1] == "fallback"
    assert kits.kit_for(None, goalkeeper=False) == (kits.fallback.home, "fallback")


def test_a_newer_local_report_comes_before_the_bundle(tmp_path: Path) -> None:
    bundle = load_bundle()
    after = (bundle.made + timedelta(days=30)).isoformat()
    write_report(tmp_path, estimate(679, "red", "white"), generated=after)
    kits = load_kits(data_dir=tmp_path)
    assert kits.report_first
    city, source = kits.kit_for(679, goalkeeper=False)
    assert source == "catalog" and city.shirt == ("red", "white")
    # A report for one club does not hide the bundle's other clubs.
    _, source = kits.kit_for(602, goalkeeper=False)
    assert source == "table"
    _, source = kits.kit_for(992, goalkeeper=False)  # Ajax: in the bundle only
    assert source == "bundled"


def test_an_older_local_report_comes_after_the_bundle(tmp_path: Path) -> None:
    bundle = load_bundle()
    before = (bundle.made - timedelta(days=30)).isoformat()
    write_report(tmp_path, estimate(679, "red"), estimate(9, "purple"), generated=before)
    kits = load_kits(data_dir=tmp_path)
    assert not kits.report_first
    assert kits.kit_for(679, goalkeeper=False)[1] == "bundled"
    # It still covers a club the bundle lacks.
    nine, source = kits.kit_for(9, goalkeeper=False)
    assert source == "catalog" and nine.shirt == ("purple",)


def test_the_report_a_bundle_was_built_from_is_not_newer(tmp_path: Path) -> None:
    """A report without `generated` dates from its file; to the second, as the bundle does."""
    bundle = load_bundle()
    path = write_report(tmp_path, estimate(679, "red"))
    same = bundle.made.timestamp() + 0.48
    os.utime(path, (same, same))
    assert report_made(json.loads(path.read_text()), path.stat().st_mtime) == bundle.made
    assert not load_kits(data_dir=tmp_path).report_first
    later = same + 1
    os.utime(path, (later, later))
    assert load_kits(data_dir=tmp_path).report_first
    assert report_made({"summary": {}}) is None
    assert report_made({"summary": {"generated": "2026-11-20T10:00:00"}}) == datetime(
        2026, 11, 20, 10, tzinfo=UTC
    )


def test_with_estimates_without_a_bundle_or_a_date() -> None:
    kits_table = parse_kits(table())
    report = parse_catalog({"results": [estimate(679, "sky")]})
    kits = with_estimates(kits_table, report=report)
    assert kits.kit_for(679, goalkeeper=False)[1] == "catalog" and not kits.report_first
    bundle = Bundle(clubs=report, made=datetime(2026, 11, 13, tzinfo=UTC), catalog_sha256="0" * 64,
                    probe_version=1)  # fmt: skip
    kits = with_estimates(kits_table, report={}, bundle=bundle)
    assert kits.kit_for(679, goalkeeper=False)[1] == "bundled"


def test_the_bundled_file_parses_and_every_row_is_a_known_colour() -> None:
    raw = json.loads(
        resources.files("kitshot").joinpath("data", BUNDLE_NAME).read_text(encoding="utf-8")
    )
    # Only our own estimates leave the PC: no texture names, GUIDs, bundles or images.
    assert set(raw) == {"about", "format", "source", "clubs"}
    assert set(raw["source"]) == {"catalog_sha256", "probe_version", "made"}
    assert len(raw["source"]["catalog_sha256"]) == 64
    assert len(raw["clubs"]) > 800
    for key, shirt in raw["clubs"].items():
        assert key.isdigit() and int(key) > 0
        assert isinstance(shirt, list) and 1 <= len(shirt) <= 2 and len(set(shirt)) == len(shirt)
        assert all(colour in COLOURS for colour in shirt), (key, shirt)
    bundle = load_bundle()
    assert len(bundle.clubs) == len(raw["clubs"])
    assert bundle.made.tzinfo is not None and bundle.probe_version >= 1
    for club in bundle.clubs.values():
        assert club.home.pattern in ("plain", "mixed") and club.home.shorts is None
        assert club.goalkeeper.goalkeeper and club.home.words() and club.goalkeeper.words()
        assert not COLOURS[club.goalkeeper.shirt[0]].families & families(club.home.shirt) or (
            club.goalkeeper.shirt[0] == KEEPER_COLOURS[0]
        )


def test_parse_bundle_refuses_what_it_cannot_draw() -> None:
    good = {"format": 1, "source": {"catalog_sha256": "0" * 64, "probe_version": 1,
                                    "made": "2026-11-13T07:24:21Z"},
            "clubs": {"1": ["red"], "2": ["white", "blue"]}}  # fmt: skip
    bundle = parse_bundle(good)
    assert bundle.clubs[1].home == Kit(shirt=("red",))
    assert bundle.clubs[2].home == Kit(shirt=("white", "blue"), pattern="mixed")
    assert bundle.made == datetime(2026, 11, 13, 7, 24, 21, tzinfo=UTC)
    for clubs in ({"1": ["gold"]}, {"1": []}, {"1": ["red", "red"]}, {"x": ["red"]},
                  {"1": ["red", "white", "blue"]}, {"1": "red"}):  # fmt: skip
        with pytest.raises(KitError):
            parse_bundle({**good, "clubs": clubs})
    with pytest.raises(KitError):
        parse_bundle({**good, "format": 2})
    with pytest.raises(KitError):
        parse_bundle({**good, "source": {"made": "yesterday"}})


def build_script():
    spec = importlib.util.spec_from_file_location(
        "build_kit_catalog", ROOT / "scripts" / "build_kit_catalog.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_build_script_keeps_only_colour_names() -> None:
    build = build_script()
    report = {
        "summary": {"catalog_sha256": "a" * 64, "kinds": {"home": 4},
                    "generated": "2026-12-01T09:30:00+00:00", "probe_version": 2},
        "results": [
            {**estimate(602, "red", "white"), "guid": "d" * 32, "texture_name": "arsenal home",
             "bundle": "kits2d-eng.bundle"},
            estimate(679, "sky"),
            estimate(7, "gold"),  # not in the vocabulary: left out, as the worker would
            {"key": "kit2d_1708_home", "club_id": 1708, "kind": "home",
             "status": "not_in_catalog", "primary": None, "secondary": None},
            estimate(688, "black", "white", kind="away"),
        ],
    }  # fmt: skip
    bundle = build.reduce(report)
    assert bundle["clubs"] == {602: ("red", "white"), 679: ("sky",)}
    assert bundle["source"] == {
        "catalog_sha256": "a" * 64,
        "probe_version": 2,
        "made": "2026-12-01T09:30:00Z",
    }
    text = build.dumps(bundle)
    assert "arsenal" not in text and "d" * 32 not in text and "kits2d" not in text
    assert '    "602": ["red", "white"],\n' in text
    parsed = parse_bundle(json.loads(text))
    assert dict(parsed.clubs) == parse_catalog(report)
    # A probe run for a few clubs is not the whole game: refused.
    with pytest.raises(KitError, match="--all-home"):
        build.reduce({**report, "summary": {**report["summary"], "kinds": {"home": 832}}})
    with pytest.raises(KitError):
        build.reduce({**report, "summary": {"kinds": {"home": 4}}})  # no catalog hash


def test_the_build_script_rebuilds_the_shipped_bundle_from_its_own_output(tmp_path: Path) -> None:
    """The shipped bundle is in the build script's layout (one club per line, sorted)."""
    build = build_script()
    shipped = resources.files("kitshot").joinpath("data", BUNDLE_NAME).read_text(encoding="utf-8")
    data = json.loads(shipped)
    clubs = {int(k): tuple(v) for k, v in data["clubs"].items()}
    assert list(clubs) == sorted(clubs)
    assert build.dumps({**data, "clubs": clubs}) == shipped


def test_mixed_is_allowed_in_the_table_too() -> None:
    assert PATTERNS["mixed"]
    kits = parse_kits(
        table(**{"992": {"home": {"shirt": ["red", "white"], "pattern": "mixed",
                                  "shorts": "white"},
                         "goalkeeper": {"shirt": ["green"]}}})
    )  # fmt: skip
    assert kits.clubs[992].home.words() == "red and white shirt, short sleeves, white shorts"


def test_shirt_and_jacket_words() -> None:
    kit = Kit(shirt=("red", "white"), pattern="sleeves", shorts="white", trim="black")
    assert kit.shirt_words() == "red shirt, white sleeves, black collar"
    assert kit.jacket_words() == "red track jacket, white sleeves, black collar"
    assert kit.shirt_tag_groups("jacket") == (
        frozenset({"red jacket"}),
        frozenset({"white jacket"}),
    )
    assert kit.colours() == ("red", "white", "black")
