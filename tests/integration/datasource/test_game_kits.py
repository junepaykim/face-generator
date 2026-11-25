"""FM27's kits from a made-up install folder: which database is read, the cache in the data
folder, and their place among the kit sources (the table first, the estimates after)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from support import game_db
from support.game_db import CLUBS, Club, Entry
from support.paths import ROOT

from kitshot.core.kit_values import Kit
from kitshot.datasource.game_kits import (
    CACHE_NAME,
    READER_VERSION,
    find_database,
    load_game_kits,
)
from kitshot.datasource.kits import load_kits

UNKNOWN_CLUB = 987654321
"""An ID in no kit source: not in the table, the bundle or the made-up database."""


def refuse(path: Path) -> bytes:
    raise AssertionError(f"read {path} again instead of using the cache")


def test_the_newest_database_of_this_game_is_found(tmp_path: Path) -> None:
    game_db.install(tmp_path, CLUBS, folder="2620")  # an older game's database
    assert find_database(tmp_path) is None
    game_db.install(tmp_path, CLUBS, folder="2700")
    newer = game_db.install(tmp_path, CLUBS, folder="2710")
    db = find_database(tmp_path)
    assert db is not None and db.path == newer and db.version == "27.1.0"
    assert find_database(tmp_path / "nowhere") is None
    assert find_database(None) is None


def test_the_folder_name_stands_in_for_an_unreadable_version(tmp_path: Path) -> None:
    path = game_db.install(tmp_path, CLUBS)
    path.with_name("2700_fm.xml").unlink()
    db = find_database(tmp_path)
    assert db is not None and db.version == "2700"


def test_kits_are_read_once_and_kept_in_the_data_folder(tmp_path: Path) -> None:
    install, data = tmp_path / "fm", tmp_path / "data"
    path = game_db.install(install, CLUBS)
    first = load_game_kits(install, data)
    assert first is not None and set(first.shirts) == {1708, 1736}
    cache = json.loads((data / CACHE_NAME).read_text(encoding="utf-8"))
    assert cache["reader"] == READER_VERSION and cache["database"]["size"] == path.stat().st_size
    assert cache["clubs"]["1708"][0] == [24, "BD1952", "19316B", "0042BD"]
    again = load_game_kits(install, data, read=refuse)
    assert again is not None and again.shirts == first.shirts and again.sha256 == first.sha256


def test_a_changed_database_is_read_again(tmp_path: Path) -> None:
    install, data = tmp_path / "fm", tmp_path / "data"
    path = game_db.install(install, CLUBS)
    load_game_kits(install, data)
    game_db.install(install, [Club(42, kits=(Entry(("FFFFFF", "00843A", "FFFFFF")),))])
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    changed = load_game_kits(install, data)
    assert changed is not None and set(changed.shirts) == {42}


def test_a_cache_from_another_reader_version_is_not_used(tmp_path: Path) -> None:
    install, data = tmp_path / "fm", tmp_path / "data"
    game_db.install(install, CLUBS)
    load_game_kits(install, data)
    cache = data / CACHE_NAME
    old = json.loads(cache.read_text(encoding="utf-8"))
    old["reader"] = READER_VERSION - 1
    cache.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(AssertionError, match="read"):
        load_game_kits(install, data, read=refuse)
    cache.write_text("{broken", encoding="utf-8")
    assert load_game_kits(install, data) is not None


def test_an_unreadable_database_is_left_out_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    install, data = tmp_path / "fm", tmp_path / "data"
    game_db.install(install, game_db.database(CLUBS)[:-40])
    with caplog.at_level(logging.WARNING, logger="kitshot"):
        assert load_game_kits(install, data) is None
    assert "could not be read" in caplog.text
    assert not (data / CACHE_NAME).exists()


# --- among the other kit sources ------------------------------------------------------------


def test_barcelona_and_real_madrid_get_fm27s_kits(tmp_path: Path) -> None:
    """Neither has a kit image in FM27, so before the database they drew the fallback kit."""
    install = tmp_path / "fm"
    game_db.install(install, CLUBS)
    kits = load_kits(data_dir=tmp_path / "data", install_dir=install)
    barca, source = kits.kit_for(1708, goalkeeper=False)
    assert source == "game"
    assert barca.shirt == ("navy", "red", "blue") and barca.pattern == "stripes"
    assert barca.shorts is None
    keeper, source = kits.kit_for(1708, goalkeeper=True)
    assert source == "game" and keeper.goalkeeper and keeper.shirt == ("orange",)
    real, _ = kits.kit_for(1736, goalkeeper=False)
    assert real.shirt == ("white",) and real.trim == "pink"
    real_keeper, _ = kits.kit_for(1736, goalkeeper=True)  # none in the database: a contrast
    assert real_keeper.shirt == ("green",)


def test_the_table_comes_before_fm27s_kits_and_the_bundle_after(tmp_path: Path) -> None:
    install = tmp_path / "fm"
    red = (Entry(("FFFFFF", "C50021", "FFFFFF")),)
    stripes = (Entry(("FFFFFF", "214284", "E6C521"), style=3),)
    # Newcastle is in the table; Manchester City and Colchester in the bundle as well.
    clubs = [Club(688, kits=red), Club(679, kits=red), Club(636, kits=stripes)]
    game_db.install(install, clubs)
    kits = load_kits(ROOT / "config" / "kits.toml", data_dir=tmp_path, install_dir=install)
    newcastle, source = kits.kit_for(688, goalkeeper=False)
    assert source == "table" and newcastle.pattern == "stripes"
    city, source = kits.kit_for(679, goalkeeper=False)
    assert source == "game" and city.shirt == ("red",) and city.trim == "white"
    colchester, source = kits.kit_for(636, goalkeeper=False)  # its draft left the table
    assert source == "game" and colchester == Kit(shirt=("blue", "white"), pattern="stripes")
    _, source = kits.kit_for(728, goalkeeper=False)  # Tottenham: the bundle alone
    assert source == "bundled"
    assert kits.kit_for(UNKNOWN_CLUB, goalkeeper=False) == (kits.fallback.home, "fallback")


def test_without_a_readable_database_the_other_sources_still_apply(tmp_path: Path) -> None:
    install = tmp_path / "fm"
    game_db.install(install, b"not a database")
    kits = load_kits(data_dir=tmp_path, install_dir=install)
    assert kits.game == {}
    _, source = kits.kit_for(679, goalkeeper=False)
    assert source == "bundled"
    _, source = kits.kit_for(1708, goalkeeper=False)
    assert source == "fallback"
    assert load_kits(data_dir=tmp_path).game == {}  # no install folder: not looked for
