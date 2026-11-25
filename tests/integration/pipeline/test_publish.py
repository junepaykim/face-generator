"""Publishing into FM27's folder. Guards AGENTS.md hard rule 2: only graphics/kitshot/."""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
from PIL import Image

from kitshot.persistence.settings import Settings, load_settings
from kitshot.pipeline.publish import (
    PublishError,
    config_xml,
    deactivate_packs,
    legacy_packs_root,
    pack_dir,
    publish_active,
    publish_career,
    publish_pack,
    published_bytes,
    published_pictures,
    remove_legacy_packs,
    remove_packs,
    unpublish,
    write_config,
)

SIZES = {"a": 1024, "b": 768, "c": 512}


@pytest.fixture
def pack_settings(config_file: Path) -> Settings:
    return load_settings(defaults_path=config_file)


def _link_directory(path: Path, target: Path) -> None:
    try:
        path.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows does not grant this process symlink creation privileges")
        raise


def test_config_xml_maps_database_and_generated_people() -> None:
    assert config_xml([29141735, 2002069100, 29141735]) == (
        "<record>\n"
        '\t<boolean id="preload" value="false"/>\n'
        '\t<boolean id="amap" value="false"/>\n'
        '\t<list id="maps">\n'
        '\t\t<record from="29141735" to="graphics/pictures/person/29141735/portrait"/>\n'
        '\t\t<record from="29141735" to="graphics/pictures/person/r-29141735/portrait"/>\n'
        '\t\t<record from="2002069100" to="graphics/pictures/person/2002069100/portrait"/>\n'
        '\t\t<record from="2002069100" to="graphics/pictures/person/r-2002069100/portrait"/>\n'
        "\t</list>\n"
        "</record>\n"
    )


def test_pack_dir_stays_inside_graphics_kitshot(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    assert pack_dir(fm, "ca0a7bbf751") == fm / "graphics" / "kitshot" / "ca0a7bbf751"
    for bad in ("", ".", "..", "../games", "a/b", "a\\b", "c:x"):
        with pytest.raises(PublishError):
            pack_dir(fm, bad)


def test_publish_pack_refuses_paths_outside(
    conn: sqlite3.Connection, pack_settings: Settings, tmp_path: Path
) -> None:
    fm = tmp_path / "fm"
    master = tmp_path / "m.png"
    Image.new("RGB", (64, 64)).save(master)
    _career(conn)
    _image(conn, 1, "a", str(master))
    for key in ("../../games", "../other-pack"):
        with pytest.raises(PublishError):
            publish_pack(conn, pack_settings, 1, key)
    assert not (fm / "games").exists()
    assert not (fm / "graphics").exists()


def test_publish_pack_refuses_a_redirected_pack_root(
    conn: sqlite3.Connection, pack_settings: Settings, tmp_path: Path
) -> None:
    graphics = tmp_path / "fm" / "graphics"
    graphics.mkdir(parents=True)
    other_pack = graphics / "other-pack"
    other_pack.mkdir()
    _link_directory(graphics / "kitshot", other_pack)
    master = tmp_path / "m.png"
    Image.new("RGB", (64, 64)).save(master)
    _career(conn)
    _image(conn, 1, "a", str(master))

    with pytest.raises(PublishError):
        publish_pack(conn, pack_settings, 1, "k")
    assert not list(other_pack.iterdir())


def test_publish_pack_downscales_to_3_4(
    conn: sqlite3.Connection, pack_settings: Settings, tmp_path: Path
) -> None:
    master = tmp_path / "m.png"
    Image.new("RGB", (1296, 1728), (10, 20, 30)).save(master)
    _career(conn)
    _image(conn, 42, "b", str(master))
    report = publish_pack(conn, pack_settings, 1, "k")
    assert report.written == 1
    dest = tmp_path / "fm" / "graphics" / "kitshot" / "k" / "42.png"
    with Image.open(dest) as im:
        assert im.size == (576, 768) and im.format == "PNG"
    assert not list(dest.parent.glob(".*part"))  # no temporary files left


def test_failed_atomic_replacement_keeps_the_previous_portrait_until_retry(
    conn: sqlite3.Connection,
    pack_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = tmp_path / "m.png"
    Image.new("RGB", (48, 64)).save(master)
    _career(conn)
    _image(conn, 42, "b", str(master))
    folder = tmp_path / "fm" / "graphics" / "kitshot" / "k"
    folder.mkdir(parents=True)
    dest = folder / "42.png"
    old = master.read_bytes()
    dest.write_bytes(old)
    (folder / ".42.png.part").write_bytes(b"interrupted previous write")

    def fail_replace(src: Path, dst: Path) -> None:
        assert src.parent == dst.parent and dst == dest
        with Image.open(src) as im:
            assert im.size == (576, 768)
        raise OSError("could not replace the portrait")

    with monkeypatch.context() as patch:
        patch.setattr("kitshot.pipeline.publish.os.replace", fail_replace)
        with pytest.raises(OSError, match="could not replace the portrait"):
            publish_pack(conn, pack_settings, 1, "k")
    assert dest.read_bytes() == old
    assert list(folder.iterdir()) == [dest]
    assert conn.execute("SELECT published_path FROM images").fetchone()[0] is None

    assert publish_pack(conn, pack_settings, 1, "k").written == 1
    with Image.open(dest) as im:
        assert im.size == (576, 768)


def test_a_square_master_from_before_3_4_is_cropped_around_its_centre(
    conn: sqlite3.Connection, pack_settings: Settings, tmp_path: Path
) -> None:
    master = tmp_path / "m.png"
    img = Image.new("RGB", (1024, 1024), (255, 0, 0))  # red sides
    img.paste((0, 0, 255), (128, 0, 896, 1024))  # a blue 3:4 middle
    img.save(master)
    _career(conn)
    _image(conn, 1, "a", str(master))
    report = publish_pack(conn, pack_settings, 1, "k")
    with Image.open(report.folder / "1.png") as im:
        assert im.size == (768, 1024)
        assert im.getpixel((0, 0)) == im.getpixel((767, 1023)) == (0, 0, 255)


def test_write_config_only_when_changed(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    folder = pack_dir(fm, "k")
    assert write_config(folder, [1, 2], fm)
    mtime = (folder / "config.xml").stat().st_mtime_ns
    assert not write_config(folder, [2, 1], fm)
    assert (folder / "config.xml").stat().st_mtime_ns == mtime
    assert write_config(folder, [1], fm)


def test_write_config_refuses_paths_outside(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    with pytest.raises(PublishError):
        write_config(fm / "graphics", [1], fm)


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
    folder = fm / "graphics" / "kitshot" / "k"
    assert (first.written, first.unchanged, first.missing, first.mapped) == (2, 0, (30,), 2)
    assert first.config_changed and first.changed
    assert sorted(p.name for p in folder.iterdir()) == ["10.png", "20.png", "config.xml"]
    with Image.open(folder / "20.png") as im:
        assert im.size == (576, 768)
    assert 'from="40"' not in (folder / "config.xml").read_text()
    assert "person/r-20/portrait" in (folder / "config.xml").read_text()

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


def test_publish_repairs_old_mappings_without_redrawing_or_reencoding(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xml.etree.ElementTree as ET

    fm, data = tmp_path / "fm", tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (48, 64)).save(data / "master.png")
    _career(conn)
    # Both a low database ID and a high generated-person ID use the same publisher;
    # there need not be any player classification row (staff/media included).
    ids = (10, 2_002_200_001)
    for uid in ids:
        _image(conn, uid, "b", "master.png")
    publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    folder = pack_dir(fm, "k")
    config = folder / "config.xml"
    legacy = "\n".join(line for line in config.read_text().splitlines() if "/r-" not in line)
    config.write_text(legacy)
    before = {p.name: p.read_bytes() for p in folder.glob("*.png")}

    def no_encoding(items: list[tuple[Path, int]]) -> Iterator[bytes]:
        assert not items, "repairing a mapping must reuse the published pictures"
        return iter(())

    monkeypatch.setattr("kitshot.pipeline.publish.published_pictures", no_encoding)
    report = publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    assert (report.written, report.unchanged, report.mapped) == (0, 2, 2)
    assert report.config_changed
    maps = {r.attrib["to"]: r.attrib["from"] for r in ET.parse(config).findall("./list/record")}
    assert maps == {
        f"graphics/pictures/person/{prefix}{uid}/portrait": str(uid)
        for uid in ids
        for prefix in ("", "r-")
    }
    assert {p.name: p.read_bytes() for p in folder.glob("*.png")} == before
    assert not publish_career(conn, 1, "k", fm, SIZES, data_dir=data).changed


def test_unpublish_removes_file_and_mapping(conn: sqlite3.Connection, tmp_path: Path) -> None:
    fm, data = tmp_path / "fm", tmp_path / "data"
    (data / "masters").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(data / "masters" / "a.png")
    _career(conn)
    _image(conn, 10, "a", "masters/a.png")
    _image(conn, 20, "a", "masters/a.png")
    publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    assert unpublish(conn, 1, "k", 10, fm)
    folder = fm / "graphics" / "kitshot" / "k"
    assert not (folder / "10.png").exists() and (data / "masters" / "a.png").exists()
    text = (folder / "config.xml").read_text()
    assert 'from="10"' not in text and 'from="20"' in text
    assert "person/r-10/portrait" not in text and "person/r-20/portrait" in text
    # Publishing again does not bring it back: the picture is no longer current.
    assert publish_career(conn, 1, "k", fm, SIZES, data_dir=data).written == 0


# --- per-career packs: only the active career's pack maps anything (spec section 8) ---------


def _mapped(fm: Path, key: str) -> set[int]:
    import re

    text = (fm / "graphics" / "kitshot" / key / "config.xml").read_text()
    return {int(x) for x in re.findall(r'from="(\d+)"', text)}


def _two_careers(conn: sqlite3.Connection, data: Path) -> None:
    """Careers 1 ('k') and 2 ('k2'), each with a current picture of player 10 (newgen IDs
    repeat between careers)."""
    (data / "masters").mkdir(parents=True)
    Image.new("RGB", (64, 64), (1, 2, 3)).save(data / "masters" / "a.png")
    _career(conn)
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (2, 'k2', 'K2', '[]', '', '')"
    )
    _image(conn, 10, "a", "masters/a.png")
    _image(conn, 11, "b", "masters/a.png")
    conn.execute(
        "INSERT INTO images (career_id, unique_id, version, reason, group_name, model, prompt,"
        " negative_prompt, seed, master_path, is_current, created_at)"
        " VALUES (2, 10, 1, 'initial', 'a', 'm', 'p', 'n', 1, 'masters/a.png', 1, '')"
    )


def test_an_inactive_pack_has_its_pictures_but_maps_nothing(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    fm, data = tmp_path / "fm", tmp_path / "data"
    _two_careers(conn, data)
    report = publish_career(conn, 1, "k", fm, SIZES, data_dir=data, active=False)
    assert (report.written, report.mapped, report.active) == (2, 0, False)
    folder = fm / "graphics" / "kitshot" / "k"
    assert (folder / "10.png").is_file() and _mapped(fm, "k") == set()
    # Activating it later only rewrites config.xml.
    again = publish_career(conn, 1, "k", fm, SIZES, data_dir=data)
    assert (again.written, again.unchanged, again.mapped) == (0, 2, 2)
    assert _mapped(fm, "k") == {10, 11}


def test_deactivate_packs_unmaps_every_other_pack(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    for key in ("k", "k2", "k3"):
        write_config(pack_dir(fm, key), [1, 2], fm)
    (fm / "graphics" / "kitshot" / "no-config").mkdir()
    (fm / "graphics" / "kitshot" / "stray.txt").write_text("x")
    assert deactivate_packs(fm, keep="k2") == ("k", "k3")
    assert _mapped(fm, "k") == set() and _mapped(fm, "k2") == {1, 2}
    assert not (fm / "graphics" / "kitshot" / "no-config" / "config.xml").exists()
    assert deactivate_packs(fm, keep="k2") == ()  # nothing left to change
    assert deactivate_packs(tmp_path / "nowhere", keep=None) == ()


def test_deactivate_packs_does_not_follow_symlinks(tmp_path: Path) -> None:
    # Nothing outside graphics/kitshot/ is touched, even through a link.
    fm = tmp_path / "fm"
    folder = fm / "graphics" / "kitshot"
    folder.mkdir(parents=True)
    outside = tmp_path / "other-pack"  # stands in for someone else's pack
    outside.mkdir()
    (outside / "config.xml").write_text(config_xml([5]))
    _link_directory(folder / "link", outside)
    deactivate_packs(fm, keep=None)
    assert 'from="5"' in (outside / "config.xml").read_text()


def test_remove_packs_refuses_a_graphics_folder_that_leads_elsewhere(tmp_path: Path) -> None:
    # FM27's graphics folder moved to another disk and linked back: publishing refuses to
    # write there, so the Kitshot folder found there is someone else's.
    fm, elsewhere = tmp_path / "fm", tmp_path / "other-disk" / "graphics"
    fm.mkdir()
    (elsewhere / "kitshot" / "theirs").mkdir(parents=True)
    (elsewhere / "kitshot" / "theirs" / "1.png").write_bytes(b"png")
    _link_directory(fm / "graphics", elsewhere)
    with pytest.raises(PublishError, match="refusing to delete outside"):
        remove_packs(fm, tmp_path / "data")
    assert (elsewhere / "kitshot" / "theirs" / "1.png").is_file()
    with pytest.raises(PublishError, match="refusing to write outside"):
        pack_dir(fm, "k")  # as publishing does


def test_remove_packs_removes_a_linked_kitshot_folder_but_not_its_target(tmp_path: Path) -> None:
    fm, outside = tmp_path / "fm", tmp_path / "other-pack"
    (fm / "graphics").mkdir(parents=True)
    outside.mkdir()
    (outside / "1.png").write_bytes(b"png")
    _link_directory(fm / "graphics" / "kitshot", outside)
    assert remove_packs(fm, tmp_path / "data")
    assert not (fm / "graphics" / "kitshot").exists() and (outside / "1.png").is_file()
    assert not remove_packs(fm, tmp_path / "data")  # nothing left to remove


def test_remove_packs_deletes_our_own_folder(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    write_config(pack_dir(fm, "k"), [1], fm)
    (fm / "graphics" / "theirs").mkdir()
    assert remove_packs(fm, tmp_path / "data")
    assert not (fm / "graphics" / "kitshot").exists() and (fm / "graphics" / "theirs").is_dir()


def _old_pack(fm: Path, key: str = "k") -> Path:
    """Our pack as a version before Phase 17o published it, in graphics/facegen/."""
    folder = legacy_packs_root(fm) / key
    folder.mkdir(parents=True)
    (folder / "10.png").write_bytes(b"old png")
    (folder / "config.xml").write_text(config_xml([10]))
    return folder


def test_remove_packs_also_deletes_our_folder_from_before_the_rename(tmp_path: Path) -> None:
    fm = tmp_path / "fm"
    _old_pack(fm)
    (fm / "graphics" / "theirs").mkdir()
    assert remove_packs(fm, tmp_path / "data")  # only the old folder was there
    assert not legacy_packs_root(fm).exists() and (fm / "graphics" / "theirs").is_dir()
    write_config(pack_dir(fm, "k"), [1], fm)
    _old_pack(fm)
    assert remove_packs(fm, tmp_path / "data")
    assert not (fm / "graphics" / "kitshot").exists() and not legacy_packs_root(fm).exists()
    assert not remove_legacy_packs(fm)  # nothing left


def test_remove_legacy_packs_refuses_a_graphics_folder_that_leads_elsewhere(
    tmp_path: Path,
) -> None:
    fm, elsewhere = tmp_path / "fm", tmp_path / "other-disk" / "graphics"
    fm.mkdir()
    (elsewhere / "facegen" / "theirs").mkdir(parents=True)
    _link_directory(fm / "graphics", elsewhere)
    with pytest.raises(PublishError, match="refusing to delete outside"):
        remove_legacy_packs(fm)
    assert (elsewhere / "facegen" / "theirs").is_dir()


def test_publishing_the_active_pack_moves_it_out_of_graphics_facegen(
    conn: sqlite3.Connection, pack_settings: Settings
) -> None:
    """An installed copy updated from 0.1.x: its stored published paths point into
    graphics/facegen/. The next publish of the active pack writes every picture into
    graphics/kitshot/ and only then removes the old folder; FM27's other packs stay."""
    fm, data = pack_settings.paths.fm_user_dir, pack_settings.paths.data_dir
    assert fm is not None
    _two_careers(conn, data)
    conn.execute("UPDATE careers SET last_save_mtime = 100 WHERE id = 1")
    old = _old_pack(fm)
    conn.execute(
        "UPDATE images SET published_path = ? WHERE career_id = 1 AND unique_id = 10",
        (str(old / "10.png"),),
    )
    theirs = fm / "graphics" / "megapack"
    theirs.mkdir()
    (theirs / "config.xml").write_text(config_xml([10]))

    # Another career's pack being written (not the active one) leaves the old folder.
    inactive = publish_pack(conn, pack_settings, 2, "k2")
    assert not inactive.active and not inactive.legacy_removed and old.is_dir()

    report = publish_active(conn, pack_settings)
    assert report is not None and report.legacy_removed and report.changed
    assert report.written == 2 and _mapped(fm, "k") == {10, 11}
    assert not legacy_packs_root(fm).exists()
    assert (theirs / "config.xml").is_file()
    path = conn.execute(
        "SELECT published_path FROM images WHERE career_id = 1 AND unique_id = 10"
    ).fetchone()[0]
    assert Path(path) == fm / "graphics" / "kitshot" / "k" / "10.png"
    again = publish_active(conn, pack_settings)
    assert again is not None and not again.legacy_removed and not again.changed


def test_an_old_folder_that_cannot_be_removed_does_not_fail_the_publish(
    conn: sqlite3.Connection, pack_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kitshot.pipeline import publish

    fm, data = pack_settings.paths.fm_user_dir, pack_settings.paths.data_dir
    assert fm is not None
    _two_careers(conn, data)
    conn.execute("UPDATE careers SET last_save_mtime = 100 WHERE id = 1")
    _old_pack(fm)

    def in_use(path: object) -> None:
        raise PermissionError("[WinError 32] being used by another process")

    monkeypatch.setattr(publish.shutil, "rmtree", in_use)
    report = publish_active(conn, pack_settings)
    assert report is not None and not report.legacy_removed and _mapped(fm, "k") == {10, 11}
    assert legacy_packs_root(fm).is_dir()  # tried again at the next publish


def test_publish_pack_follows_the_newest_save(
    conn: sqlite3.Connection, pack_settings: Settings
) -> None:
    fm, data = pack_settings.paths.fm_user_dir, pack_settings.paths.data_dir
    assert fm is not None
    _two_careers(conn, data)
    conn.execute("UPDATE careers SET last_save_mtime = 100 WHERE id = 1")
    conn.execute("UPDATE careers SET last_save_mtime = 200 WHERE id = 2")

    # Career 1's worker publishes, but career 2's save is newer: pictures only, no mapping.
    one = publish_pack(conn, pack_settings, 1, "k")
    assert not one.active and one.written == 2 and _mapped(fm, "k") == set()
    two = publish_active(conn, pack_settings)
    assert two is not None and two.active and two.folder.name == "k2"
    assert _mapped(fm, "k2") == {10}

    # Career 1 is saved again: it becomes the active one and career 2's pack is unmapped.
    conn.execute("UPDATE careers SET last_save_mtime = 300 WHERE id = 1")
    switched = publish_active(conn, pack_settings)
    assert switched is not None and switched.folder.name == "k"
    assert switched.deactivated == ("k2",) and switched.written == 0 and switched.changed
    assert _mapped(fm, "k") == {10, 11} and _mapped(fm, "k2") == set()
    assert (fm / "graphics" / "kitshot" / "k2" / "10.png").is_file()  # kept for later
    assert not publish_active(conn, pack_settings).changed  # type: ignore[union-attr]


def test_publish_active_without_a_career(conn: sqlite3.Connection, pack_settings: Settings) -> None:
    assert publish_active(conn, pack_settings) is None


def test_unpublish_from_an_inactive_pack_keeps_it_unmapped(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    fm, data = tmp_path / "fm", tmp_path / "data"
    _two_careers(conn, data)
    publish_career(conn, 1, "k", fm, SIZES, data_dir=data, active=False)
    assert unpublish(conn, 1, "k", 10, fm, active=False)
    assert not (fm / "graphics" / "kitshot" / "k" / "10.png").exists()
    assert _mapped(fm, "k") == set()


def test_publish_lock_lets_one_writer_in_at_a_time(tmp_path: Path) -> None:
    import threading

    from kitshot.pipeline.publish import publish_lock

    entered = threading.Event()

    def other() -> None:
        with publish_lock(tmp_path):
            entered.set()

    with publish_lock(tmp_path):
        t = threading.Thread(target=other)
        t.start()
        assert not entered.wait(0.2)  # blocked while this thread holds it
    t.join(5)
    assert entered.is_set()


# --- the pictures are written outside the publish transaction (2026-11-25) ------------------


def _masters(data: Path, uids: Iterable[int]) -> None:
    (data / "masters").mkdir(parents=True, exist_ok=True)
    for uid in uids:
        Image.new("RGB", (48, 64), (uid, 0, 0)).save(data / "masters" / f"{uid}.png")


def _published(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM images WHERE published_path IS NOT NULL").fetchone()[
        0
    ]


def test_pictures_are_written_without_holding_the_database_write_lock(
    pack_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On 2026-11-24 a publish of 1490 pictures held SQLite's write lock for ten minutes, and
    three syncs failed behind it ("database is locked"). While pictures are encoded and
    written, another process can write at once; the mapping follows in a short transaction."""
    from kitshot.persistence import db
    from kitshot.pipeline import publish

    conn = db.connect(tmp_path / "state.db")
    other = db.connect(tmp_path / "state.db", busy_timeout_ms=0)  # no waiting for the lock
    uids = range(1, 12)  # more than one chunk
    _masters(pack_settings.paths.data_dir, uids)
    _career(conn)
    for uid in uids:
        _image(conn, uid, "c", f"masters/{uid}.png")
    real, writes = publish._write_atomic, []

    def write(path: Path, data: bytes) -> None:
        if path.suffix == ".png":
            with db.transaction(other):  # raises at once while the publish holds the lock
                other.execute("UPDATE careers SET updated_at = ?", (path.name,))
            writes.append(path.name)
        real(path, data)

    monkeypatch.setattr(publish, "_write_atomic", write)
    report = publish_pack(conn, pack_settings, 1, "k")
    fm = pack_settings.paths.fm_user_dir
    assert fm is not None and len(writes) == 11
    assert (report.written, report.unchanged, report.mapped) == (11, 0, 11)
    assert _mapped(fm, "k") == set(uids) and _published(conn) == 11
    again = publish_pack(conn, pack_settings, 1, "k")
    assert (again.written, again.unchanged, again.changed) == (0, 11, False)
    conn.close()
    other.close()


def test_a_picture_picked_while_the_pack_is_written_is_published_as_picked(
    pack_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pick and Restore change a picture without the publish lock. A picture changed while it
    is written is not recorded as published, and the same publish writes it as it is now."""
    from kitshot.persistence import db
    from kitshot.pipeline import publish

    conn = db.connect(tmp_path / "state.db")
    other = db.connect(tmp_path / "state.db")
    data = pack_settings.paths.data_dir
    _masters(data, (7,))
    Image.new("RGB", (48, 64), (0, 0, 255)).save(data / "masters" / "picked.png")
    _career(conn)
    _image(conn, 7, "c", "masters/7.png")
    real, picked = publish._write_atomic, []

    def write(path: Path, data: bytes) -> None:
        if path.name == "7.png" and not picked:
            other.execute(
                "UPDATE images SET master_path = 'masters/picked.png', published_path = NULL"
            )
            picked.append(path)
        real(path, data)

    monkeypatch.setattr(publish, "_write_atomic", write)
    report = publish_pack(conn, pack_settings, 1, "k")
    assert picked and report.written == 1 and _published(conn) == 1
    with Image.open(report.folder / "7.png") as im:
        assert im.convert("RGB").getpixel((0, 0)) == (0, 0, 255)
    conn.close()
    other.close()


def test_pictures_written_before_a_failure_stay_recorded(
    conn: sqlite3.Connection,
    pack_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each chunk is recorded as it is written: a failure later on (a full disk) does not
    leave the pictures already in place to be written again."""
    from kitshot.pipeline import publish

    uids = range(1, 11)
    _masters(pack_settings.paths.data_dir, uids)
    _career(conn)
    for uid in uids:
        _image(conn, uid, "c", f"masters/{uid}.png")
    real = publish._write_atomic

    def write(path: Path, data: bytes) -> None:
        if path.name == "10.png":
            raise OSError("no space left on the device")
        real(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(publish, "_write_atomic", write)
        with pytest.raises(OSError, match="no space"):
            publish_pack(conn, pack_settings, 1, "k")
    assert _published(conn) == publish._CHUNK == 8  # the first chunk; 9.png waits
    assert publish_pack(conn, pack_settings, 1, "k").written == 2


def test_square_pictures_are_published_square_next_to_portraits(
    conn: sqlite3.Connection, pack_settings: Settings, tmp_path: Path
) -> None:
    """Phase 16g: each picture in its own shape. A 1:1 picture is published 1:1 at its group's
    height, beside 3:4 pictures; the pre-V5 squares marked 3:4 are still cropped to 3:4."""
    portrait, square, old = tmp_path / "p.png", tmp_path / "s.png", tmp_path / "o.png"
    Image.new("RGB", (1296, 1728), (10, 20, 30)).save(portrait)
    Image.new("RGB", (1536, 1536), (40, 50, 60)).save(square)
    Image.new("RGB", (1024, 1024), (70, 80, 90)).save(old)
    _career(conn)
    _image(conn, 1, "a", str(portrait))
    _image(conn, 2, "a", str(square))
    _image(conn, 3, "b", str(old))
    conn.execute("UPDATE images SET aspect = '1:1' WHERE unique_id = 2")
    assert publish_pack(conn, pack_settings, 1, "k").written == 3
    folder = tmp_path / "fm" / "graphics" / "kitshot" / "k"
    sizes = {}
    for uid in (1, 2, 3):
        with Image.open(folder / f"{uid}.png") as im:
            sizes[uid] = im.size
    assert sizes == {1: (768, 1024), 2: (1024, 1024), 3: (576, 768)}


@pytest.mark.parametrize("rust", [False, True])
def test_published_bytes_in_each_shape(tmp_path: Path, rust: bool) -> None:
    from kitshot.platform import native

    if rust and native.ext is None:
        pytest.skip("the native extension is not built")
    master = tmp_path / "m.png"
    Image.new("RGB", (1344, 1728), (5, 6, 7)).save(master)
    for shape, size in (("3:4", (384, 512)), ("1:1", (512, 512))):
        data = published_bytes(master, 512, shape, rust=rust)
        with Image.open(io.BytesIO(data)) as im:
            assert im.size == size, shape
    assert list(published_pictures([(master, 512, "1:1"), (master, 512)], rust=rust)) == [
        published_bytes(master, 512, "1:1", rust=rust),
        published_bytes(master, 512, rust=rust),
    ]
