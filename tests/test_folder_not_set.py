"""Without FM27's user folder (neither set nor found) every reader says so instead of failing:
the sync, the save watcher, publishing, the photo scan and the CLI (Phase 8)."""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import player, snapshot

from facegen import db, photos
from facegen.cli import main
from facegen.datasource.records import Snapshot
from facegen.poller import FolderNotSetError, NoSaveError, sync_save
from facegen.publish import FolderNotSetError as PublishFolderNotSet
from facegen.publish import PublishError, publish_pack, unpublish
from facegen.service import FAILED, PollerService, SyncService
from facegen.settings import DEFAULTS_PATH, Paths, Settings, load_settings

NOT_SET = "FM27's folder is not set and was not found on this PC"


class Source:
    def load(self, path: Path) -> Snapshot:
        raise AssertionError("no save should be read")


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """The defaults as they ship (no FM27 folder) with the data folder under tmp_path."""
    text = DEFAULTS_PATH.read_text(encoding="utf-8")
    text = text.replace('data_dir = "data"', f'data_dir = "{(tmp_path / "data").as_posix()}"')
    path = tmp_path / "test.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def setup(config: Path) -> Iterator[tuple[Settings, sqlite3.Connection]]:
    settings = load_settings(defaults_path=config)
    conn = db.connect(settings.paths.db_path)
    yield settings, conn
    conn.close()


def run_cli(config: Path, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["--config", str(config), *args], source_factory=Source, out=out)
    return code, out.getvalue()


def test_the_settings_have_no_folder(setup: tuple[Settings, sqlite3.Connection]) -> None:
    settings, _ = setup
    assert settings.paths.fm_user_dir is None and settings.paths.fm_install_dir is None


def test_sync_says_the_folder_is_not_set(config: Path) -> None:
    code, text = run_cli(config, "sync", "--dry-run")
    assert code == 2
    assert NOT_SET in text and "config/local.toml" in text


def test_status_names_the_folders(config: Path) -> None:
    code, text = run_cli(config, "status")
    assert code == 0
    assert "FM27 folder   not set, and not found on this PC" in text
    assert "FM27 install  not set, and not found on this PC" in text
    assert NOT_SET in text


def test_sync_save_raises_a_no_save_error(setup: tuple[Settings, sqlite3.Connection]) -> None:
    settings, conn = setup
    with pytest.raises(FolderNotSetError) as caught:
        sync_save(conn, settings, Source())
    assert isinstance(caught.value, NoSaveError) and NOT_SET in str(caught.value)


def test_a_sync_now_reports_it(config: Path, setup: tuple[Settings, sqlite3.Connection]) -> None:
    settings, _ = setup
    syncer = SyncService(settings.paths.db_path, Source, defaults_path=config)
    assert syncer.run() == FAILED
    assert NOT_SET in (syncer.status().error or "")


def test_the_save_watcher_waits_and_says_why(
    config: Path, setup: tuple[Settings, sqlite3.Connection]
) -> None:
    settings, conn = setup
    syncer = SyncService(settings.paths.db_path, Source, defaults_path=config)
    poller = PollerService(settings.paths.db_path, syncer, defaults_path=config, conn=conn)
    assert poller.step() == settings.schedule.poll_seconds
    status = poller.status()
    assert status.state == "not set" and NOT_SET in status.detail


def test_publishing_says_the_folder_is_not_set(
    setup: tuple[Settings, sqlite3.Connection],
) -> None:
    settings, conn = setup
    with pytest.raises(PublishFolderNotSet) as caught:
        publish_pack(conn, settings, 1, "career")
    assert isinstance(caught.value, PublishError) and NOT_SET in str(caught.value)
    with pytest.raises(PublishFolderNotSet):
        unpublish(conn, 1, "career", 7, None)


def test_the_photo_scan_reads_no_folder_and_says_why(tmp_path: Path) -> None:
    paths = Paths(fm_user_dir=None, data_dir=tmp_path, fm_install_dir=None)
    scan = photos.scan(paths)
    assert scan.index is not None
    assert scan.index.game is None and scan.index.facepack == frozenset()
    assert any("install folder is not set" in w for w in scan.index.warnings)


def test_a_save_given_by_path_still_syncs_but_is_not_published(
    tmp_path: Path, config: Path
) -> None:
    save = tmp_path / "Career.fm"
    save.write_bytes(b"x")

    class One:
        def load(self, path: Path) -> Snapshot:
            return snapshot([player(1001, name="Old Hand", ca=120, pa=125)], file_name=save.name)

    out = io.StringIO()
    code = main(["--config", str(config), "sync", "--save", str(save)], source_factory=One, out=out)
    assert code == 3
    assert "Sync stored." in out.getvalue() and NOT_SET in out.getvalue()
