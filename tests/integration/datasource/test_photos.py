"""photos.py: who FM27 shows a photo for, from its catalog and the user's facepacks."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from support.photos import catalog, config_xml

from kitshot.datasource import photos
from kitshot.datasource.photos import CATALOG_DIR, facepack_ids, game_photo_ids, scan
from kitshot.persistence import photo_index
from kitshot.persistence.settings import Paths

# --- parsing --------------------------------------------------------------------------------


def test_game_photo_ids_reads_person_keys_only() -> None:
    assert game_photo_ids(catalog(29084521, 2000109162, 37084661)) == {
        29084521,
        2000109162,
        37084661,
    }
    assert game_photo_ids(b"") == frozenset()


def test_facepack_ids_reads_portrait_mappings() -> None:
    assert facepack_ids(config_xml(1, 22, 2000000001)) == {1, 22, 2000000001}
    assert facepack_ids(config_xml(5, quote="'")) == {5}


def test_facepack_ids_ignores_other_mappings() -> None:
    xml = (
        b'<record from="logo" to="graphics/pictures/club/7/logo"/>'
        b'<record from="9" to="graphics/pictures/person/9/icon"/>'
    )
    assert facepack_ids(xml) == frozenset()


def test_facepack_ids_reads_the_target_not_the_file_name() -> None:
    xml = b'<record to="graphics/pictures/person/123/portrait" from="faces/abc"/>'
    assert facepack_ids(xml) == {123}


# --- scanning -------------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    p = Paths(
        fm_user_dir=tmp_path / "fm",
        data_dir=tmp_path / "data",
        fm_install_dir=tmp_path / "install",
    )
    p.graphics_dir.mkdir(parents=True)
    return p


def install_catalog(paths: Paths, *ids: int, digest: str = "abc") -> None:
    folder = paths.fm_install_dir / CATALOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "catalog_info.bin").write_bytes(catalog(*ids))
    (folder / "catalog_info.hash").write_text(digest)


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_scan_reads_the_catalog_and_every_facepack_but_ours(paths: Paths) -> None:
    install_catalog(paths, 1, 2)
    g = paths.graphics_dir
    write(g / "megapack" / "config.xml", config_xml(2, 3))
    write(g / "faces" / "newgens" / "config.xml", config_xml(4))
    write(g / "kitshot" / "career" / "config.xml", config_xml(99))  # ours: never counts
    write(g / "facegen" / "career" / "config.xml", config_xml(98))  # ours before Phase 17o
    result = scan(paths)
    assert result.index is not None
    assert result.index.game == {1, 2}
    assert result.index.facepack == {2, 3, 4}
    assert result.index.warnings == ()
    assert [c[0] for c in result.state["configs"]] == [
        "faces/newgens/config.xml",
        "megapack/config.xml",
    ]


def test_an_unchanged_scan_reads_nothing_again(paths: Paths) -> None:
    install_catalog(paths, 1)
    write(paths.graphics_dir / "pack" / "config.xml", config_xml(2))
    first = scan(paths)
    second = scan(paths, first.state)
    assert second.index is None
    assert photos.fingerprint(second.state) == photos.fingerprint(first.state)


def test_a_changed_facepack_or_catalog_is_read_again(paths: Paths) -> None:
    install_catalog(paths, 1)
    config = paths.graphics_dir / "pack" / "config.xml"
    write(config, config_xml(2))
    state = scan(paths).state

    write(config, config_xml(2, 3, 4))
    changed = scan(paths, state)
    assert changed.index is not None and changed.index.facepack == {2, 3, 4}

    install_catalog(paths, 1, 5, digest="def")
    updated = scan(paths, changed.state)
    assert updated.index is not None and updated.index.game == {1, 5}

    config.unlink()
    removed = scan(paths, updated.state)
    assert removed.index is not None and removed.index.facepack == frozenset()


def test_a_new_facepack_folder_is_found(paths: Paths) -> None:
    install_catalog(paths, 1)
    state = scan(paths).state
    write(paths.graphics_dir / "new" / "deep" / "config.xml", config_xml(7))
    result = scan(paths, state)
    assert result.index is not None and result.index.facepack == {7}


def test_an_unchanged_folder_is_not_listed_again(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_catalog(paths, 1)
    pack = paths.graphics_dir / "pack"
    write(pack / "config.xml", config_xml(2))
    for i in range(5):
        write(pack / f"{i}.png", b"")
    state = scan(paths).state

    listed: list[str] = []
    real = os.scandir

    def spy(path: os.PathLike[str] | str) -> object:
        listed.append(str(path))
        return real(path)

    monkeypatch.setattr(photos.os, "scandir", spy)
    assert scan(paths, state).index is None
    assert listed == []

    # A file added to the pack changes its folder's time: that one folder is listed again.
    write(pack / "5.png", b"")
    os.utime(pack, ns=(1, 1))
    scan(paths, state)
    assert listed == [str(pack)]


def test_a_missing_catalog_leaves_game_photos_unknown(paths: Paths) -> None:
    write(paths.graphics_dir / "pack" / "config.xml", config_xml(2))
    result = scan(paths)
    assert result.index is not None
    assert result.index.game is None
    assert result.index.facepack == {2}
    assert "could not be read" in result.index.warnings[0]


def test_a_catalog_without_person_keys_is_unknown_not_empty(paths: Paths) -> None:
    install_catalog(paths)
    result = scan(paths)
    assert result.index is not None and result.index.game is None
    assert "names no photos" in result.index.warnings[0]


def test_no_graphics_folder_means_no_facepacks(tmp_path: Path) -> None:
    paths = Paths(tmp_path / "fm", tmp_path / "data", tmp_path / "install")
    install_catalog(paths, 1)
    result = scan(paths)
    assert result.index is not None and result.index.facepack == frozenset()


# --- the database ---------------------------------------------------------------------------


def test_store_writes_the_ids_and_status(conn: sqlite3.Connection, paths: Paths) -> None:
    assert photo_index.status(conn) is None
    install_catalog(paths, 1, 2)
    write(paths.graphics_dir / "pack" / "config.xml", config_xml(2, 3))
    assert photo_index.store(conn, scan(paths))
    rows = conn.execute("SELECT unique_id, game, facepack FROM photos ORDER BY 1").fetchall()
    assert [tuple(r) for r in rows] == [(1, 1, 0), (2, 1, 1), (3, 0, 1)]
    s = photo_index.status(conn)
    assert s is not None
    assert (s.game_known, s.game, s.facepack, s.facepack_files) == (True, 2, 2, 1)


def test_an_unchanged_scan_keeps_the_stored_ids(conn: sqlite3.Connection, paths: Paths) -> None:
    install_catalog(paths, 1)
    photo_index.store(conn, scan(paths))
    changed_at = conn.execute("SELECT changed_at FROM photo_scan").fetchone()[0]
    again = scan(paths, photo_index.stored_state(conn))
    assert not photo_index.store(conn, again)
    assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 1
    assert conn.execute("SELECT changed_at FROM photo_scan").fetchone()[0] == changed_at


def test_an_unreadable_catalog_is_stored_as_unknown(conn: sqlite3.Connection, paths: Paths) -> None:
    photo_index.store(conn, scan(paths))
    s = photo_index.status(conn)
    assert s is not None and not s.game_known and s.warnings


@pytest.mark.parametrize("fail_at", ["read_bytes", "stat", "scandir"])
def test_transient_pack_failure_preserves_the_index_until_retry(
    conn: sqlite3.Connection, paths: Paths, monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    install_catalog(paths, 1)
    pack = paths.graphics_dir / "pack"
    mapping = pack / "config.xml"
    write(mapping, config_xml(2))
    assert photo_index.store(conn, scan(paths))
    before = photo_index.status(conn)
    write(mapping, config_xml(3, 4))
    os.utime(pack, ns=(1, 1))  # Exercise fresh folder enumeration as well as file reads.

    owner, name, target = (
        (os, "scandir", pack) if fail_at == "scandir" else (Path, fail_at, mapping)
    )
    original = getattr(owner, name)

    def locked(path, *args, **kwargs):
        if path == target:
            raise PermissionError(f"mapping locked: {path}")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, locked)
        failed = scan(paths, photo_index.stored_state(conn))
        assert not photo_index.store(conn, failed)
    during = photo_index.status(conn)
    assert during.game_known and (during.game, during.facepack) == (1, 1)
    assert during.changed_at == before.changed_at and "mapping locked" in during.warnings[0]
    assert [r[0] for r in conn.execute("SELECT unique_id FROM photos ORDER BY 1")] == [1, 2]

    # Removing the lock does not touch the file. The failed fingerprint must still be retried.
    assert photo_index.store(conn, scan(paths, photo_index.stored_state(conn)))
    assert [r[0] for r in conn.execute("SELECT unique_id FROM photos ORDER BY 1")] == [1, 3, 4]
    assert photo_index.status(conn).warnings == ()
    assert scan(paths, photo_index.stored_state(conn)).index is None

    mapping.unlink()
    assert photo_index.store(conn, scan(paths, photo_index.stored_state(conn)))
    assert [r[0] for r in conn.execute("SELECT unique_id FROM photos")] == [1]


def test_incomplete_first_scan_reports_unknown_until_retry(
    conn: sqlite3.Connection, paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_catalog(paths, 1)
    pack = paths.graphics_dir / "pack" / "config.xml"
    write(pack, config_xml(2))
    real_read = Path.read_bytes

    def locked(path: Path) -> bytes:
        if path == pack:
            raise PermissionError("locked mapping")
        return real_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", locked)
        assert not photo_index.store(conn, scan(paths))
    assert not photo_index.status(conn).game_known
    assert photo_index.status(conn).warnings
    assert photo_index.store(conn, scan(paths, photo_index.stored_state(conn)))
    assert photo_index.status(conn).game_known
    assert photo_index.status(conn).facepack == 1


def test_legacy_scan_rereads_generated_person_mappings(paths: Paths) -> None:
    install_catalog(paths, 1)
    write(
        paths.graphics_dir / "pack" / "config.xml",
        config_xml(2, 3).replace(b"person/3/portrait", b"person/r-3/portrait"),
    )
    legacy_state = scan(paths).state
    del legacy_state["format"]
    upgraded = scan(paths, legacy_state)
    assert upgraded.index.facepack == {2, 3}
    assert scan(paths, upgraded.state).index is None
