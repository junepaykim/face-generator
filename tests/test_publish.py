"""Publishing into FM27's folder. Guards AGENTS.md hard rule 2: only graphics/facegen/."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from facegen.publish import (
    PublishError,
    config_xml,
    pack_dir,
    publish_career,
    publish_picture,
    unpublish,
    write_config,
)

SIZES = {"a": 1024, "b": 768, "c": 512}


def test_config_xml_matches_the_verified_format() -> None:
    assert config_xml([29141735, 2002069100, 29141735]) == (
        "<record>\n"
        '\t<boolean id="preload" value="false"/>\n'
        '\t<boolean id="amap" value="false"/>\n'
        '\t<list id="maps">\n'
        '\t\t<record from="29141735" to="graphics/pictures/person/29141735/portrait"/>\n'
        '\t\t<record from="2002069100" to="graphics/pictures/person/2002069100/portrait"/>\n'
        "\t</list>\n"
        "</record>\n"
    )


def test_pack_dir_stays_inside_graphics_facegen(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    assert pack_dir(fm, "ca0a7bbf751") == fm / "graphics" / "facegen" / "ca0a7bbf751"
    for bad in ("", ".", "..", "../games", "a/b", "a\\b", "c:x"):
        with pytest.raises(PublishError):
            pack_dir(fm, bad)


def test_publish_picture_refuses_paths_outside(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    master = tmp_path / "m.png"
    Image.new("RGB", (64, 64)).save(master)
    with pytest.raises(PublishError):
        publish_picture(master, fm / "games", 1, 64, fm)
    with pytest.raises(PublishError):
        publish_picture(master, fm / "graphics" / "other-pack", 1, 64, fm)
    with pytest.raises(PublishError):
        write_config(fm / "graphics", [1], fm)


def test_publish_picture_downscales_to_a_square(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    master = tmp_path / "m.png"
    Image.new("RGB", (1536, 1536), (10, 20, 30)).save(master)
    dest = publish_picture(master, pack_dir(fm, "k"), 42, 768, fm)
    assert dest == fm / "graphics" / "facegen" / "k" / "42.png"
    with Image.open(dest) as im:
        assert im.size == (768, 768) and im.format == "PNG"
    assert not list(dest.parent.glob(".*part"))  # no temporary files left


def test_non_square_master_is_refused(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    master = tmp_path / "m.png"
    Image.new("RGB", (100, 80)).save(master)
    with pytest.raises(PublishError):
        publish_picture(master, pack_dir(fm, "k"), 1, 64, fm)


def test_write_config_only_when_changed(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    folder = pack_dir(fm, "k")
    assert write_config(folder, [1, 2], fm)
    mtime = (folder / "config.xml").stat().st_mtime_ns
    assert not write_config(folder, [2, 1], fm)
    assert (folder / "config.xml").stat().st_mtime_ns == mtime
    assert write_config(folder, [1], fm)


def _career(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'k', 'K', '[]', '', '')"
    )


def _image(conn: sqlite3.Connection, uid: int, group: str, master: str, current: int = 1) -> None:
    conn.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model, prompt,"
        " negative_prompt, seed, master_path, is_current, created_at)"
        " VALUES (1, ?, (SELECT COALESCE(MAX(version), 0) + 1 FROM images WHERE unique_id = ?),"
        " 'initial', ?, 'm', 'p', 'n', 1, ?, ?, '')",
        (uid, uid, group, master, current),
    )


def test_publish_career_is_idempotent(conn: sqlite3.Connection, tmp_path: Path) -> None:
    fm, data = tmp_path / "fm", tmp_path / "data"
    (data / "masters").mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (1024, 1024), (1, 2, 3)).save(data / "masters" / name)
    _career(conn)
    _image(conn, 10, "a", "masters/a.png")
    _image(conn, 20, "b", "masters/b.png")
    _image(conn, 30, "a", "masters/missing.png")
    _image(conn, 40, "a", "masters/a.png", current=0)  # a rejected picture: never published

    first = publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    folder = fm / "graphics" / "facegen" / "k"
    assert (first.written, first.unchanged, first.missing, first.mapped) == (2, 0, (30,), 2)
    assert first.config_changed and first.changed
    assert sorted(p.name for p in folder.iterdir()) == ["10.png", "20.png", "config.xml"]
    with Image.open(folder / "20.png") as im:
        assert im.size == (768, 768)
    assert 'from="40"' not in (folder / "config.xml").read_text()

    before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    second = publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    assert (second.written, second.unchanged, second.changed) == (0, 2, False)
    assert {p.name: p.stat().st_mtime_ns for p in folder.iterdir()} == before

    # A published file deleted by hand is put back.
    (folder / "10.png").unlink()
    third = publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    assert third.written == 1 and (folder / "10.png").is_file()


def test_nothing_to_publish_writes_nothing(conn: sqlite3.Connection, tmp_path: Path) -> None:
    _career(conn)
    report = publish_career(conn, 1, "k", tmp_path / "fm", SIZES, data_dir=tmp_path)
    assert not report.changed and not (tmp_path / "fm").exists()


def test_unpublish_removes_file_and_mapping(conn: sqlite3.Connection, tmp_path: Path) -> None:
    fm, data = tmp_path / "fm", tmp_path / "data"
    (data / "masters").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(data / "masters" / "a.png")
    _career(conn)
    _image(conn, 10, "a", "masters/a.png")
    _image(conn, 20, "a", "masters/a.png")
    publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    assert unpublish(conn, 1, "k", 10, fm)
    folder = fm / "graphics" / "facegen" / "k"
    assert not (folder / "10.png").exists() and (data / "masters" / "a.png").exists()
    text = (folder / "config.xml").read_text()
    assert 'from="10"' not in text and 'from="20"' in text
    # Publishing again does not bring it back: the picture is no longer current.
    assert publish_career(conn, 1, "k", fm, SIZES, data_dir=data).written == 0
