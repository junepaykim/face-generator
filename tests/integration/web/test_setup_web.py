"""First-run setup and hardware settings in the web UI (Phase 10), through FastAPI's client.

The exit test, without a browser: with no folder set or found, the UI asks for it, refuses a
wrong folder, takes the right one and syncs without a restart; the models download after one
confirmation; Auto-configure changes nothing on a 16 GB card, and on a faked 8 GB card applies
the low preset, which one Undo puts back.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from factories import player, snapshot
from fakes import FakeRenderer, FakeTagger
from fastapi.testclient import TestClient

from facegen.core.records import Snapshot
from facegen.datasource.kits import load_kits
from facegen.persistence import db
from facegen.persistence.settings import DEFAULTS_PATH, load_settings, set_overrides
from facegen.service import SyncService, WorkerService
from facegen.setup import models
from facegen.setup.hardware import Hardware
from facegen.setup.models import DownloadStatus, HubCache, files_for
from facegen.web.app import AppContext, create_app

BASE = "http://127.0.0.1:8765"
GIB = 2**30


class FakeSource:
    def load(self, path: Path) -> Snapshot:
        return snapshot([player(1001, name="Mine Player", age=29)], file_name=path.name)


class Poller:
    def __init__(self) -> None:
        self.woken = 0

    def wake(self) -> None:
        self.woken += 1

    def status(self) -> None:
        return None


@dataclass
class Download:
    """Stands in for models.ModelDownload."""

    started: int = 0
    now: DownloadStatus = field(default_factory=DownloadStatus)
    models: list[str] = field(default_factory=list)
    """The model of each download started."""

    def start(self, model: str) -> bool:
        if self.now.running:
            return False
        self.started += 1
        self.models.append(model)
        self.now = DownloadStatus(running=True, model=model, total_bytes=1000, files=3)
        return True

    def status(self) -> DownloadStatus:
        return self.now


@dataclass
class Env:
    client: TestClient
    config: Path
    tmp: Path
    syncer: SyncService
    poller: Poller
    asked: list[int]

    def settings(self) -> Any:
        return load_settings(defaults_path=self.config)

    def conn(self) -> sqlite3.Connection:
        return db.connect(self.settings().paths.db_path)

    def one(self, sql: str, *args: Any) -> Any:
        c = self.conn()
        try:
            row = c.execute(sql, args).fetchone()
            return None if row is None else row[0]
        finally:
            c.close()

    def follow(self, response: Any) -> str:
        assert response.status_code == 303, response.text
        page = self.client.get(response.headers["location"])
        assert page.status_code == 200
        return page.text


def write_config(tmp: Path) -> Path:
    """The defaults with FM27's user folder empty (not set), and nothing found on this PC (the
    tests' `no_folder_detection`)."""
    text = DEFAULTS_PATH.read_text(encoding="utf-8")
    text = text.replace('data_dir = "data"', f'data_dir = "{(tmp / "data").as_posix()}"')
    config = tmp / "config" / "test.toml"
    config.parent.mkdir()
    config.write_text(text, encoding="utf-8")
    return config


def fm27(root: Path) -> Path:
    """An FM27 folder with one save, written a while ago (so it has settled)."""
    folder = root / "Documents" / "Sports Interactive" / "Football Manager 27"
    (folder / "games").mkdir(parents=True)
    save = folder / "games" / "career.fm"
    save.write_bytes(b"save")
    os.utime(save, (1_700_000_000, 1_700_000_000))
    return folder


def make_env(
    tmp: Path,
    *,
    models_status: Callable[[str], models.ModelsStatus] | None = None,
    download: Download | None = None,
    card: Hardware | None = None,
    browse: Callable[[str | None], str | None] | None = None,
) -> Env:
    config = write_config(tmp)
    settings = load_settings(defaults_path=config)
    worker = WorkerService(
        settings.paths.db_path,
        renderer_factory=FakeRenderer,
        tagger_factory=FakeTagger,
        defaults_path=config,
        kits=load_kits(),
        fm_check=lambda: False,
    )
    syncer = SyncService(settings.paths.db_path, FakeSource, defaults_path=config)
    poller = Poller()
    asked: list[int] = []

    def ask() -> Hardware:
        asked.append(1)
        assert card is not None
        return card

    ctx = AppContext(
        settings.paths.db_path,
        config,
        worker,
        syncer,
        poller,  # type: ignore[arg-type]
        models=models_status,
        download=download,  # type: ignore[arg-type]
        hardware=ask if card is not None else None,
        browse=browse,
        # The folders are real temporary ones, so this system's path rules (Windows's on
        # Windows); Linux's elsewhere, WSL included, as before.
        platform="windows" if os.name == "nt" else "linux",
    )
    client = TestClient(create_app(ctx), base_url=BASE, follow_redirects=False)
    token = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/").text)
    assert token is not None
    client.headers["X-CSRF-Token"] = token[1]
    return Env(client, config, tmp, syncer, poller, asked)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return make_env(tmp_path)


# --- 10a: FM27's folder ---------------------------------------------------------------------


def test_without_a_folder_every_page_says_so_and_the_dashboard_asks(env: Env) -> None:
    text = env.client.get("/").text
    # The first-run guide (Phase 12f) asks for it as its first step.
    assert "Getting started" in text and "<strong>FM27's folder</strong>" in text
    assert 'action="/setup/folder"' in text
    assert "Browse…" not in text and "Browse cannot open a window from here" in text
    for url in ("/players", "/review", "/settings"):
        assert "FM27's folder was not found." in env.client.get(url).text, url


def test_a_wrong_folder_is_refused_and_named(env: Env) -> None:
    wrong = env.tmp / "Football Manager 26"
    wrong.mkdir()
    r = env.client.post("/setup/folder", data={"folder": str(wrong), "next": "/"})
    text = env.follow(r)
    assert "Not FM27&#39;s folder" in text and "has no games folder" in text
    assert f'value="{wrong}"' in text  # kept in the box, to correct
    assert not (env.config.parent / "local.toml").exists()
    assert env.settings().paths.fm_user_dir is None
    text = env.follow(env.client.post("/setup/folder", data={"folder": "", "next": "/"}))
    assert "Enter FM27&#39;s folder" in text


def test_the_right_folder_is_saved_and_synced_without_a_restart(env: Env) -> None:
    folder = fm27(env.tmp)
    r = env.client.post("/setup/folder", data={"folder": str(folder / "games"), "next": "/"})
    assert r.status_code == 303 and r.headers["location"].startswith("/?msg=")
    env.syncer.join(10)
    text = env.client.get(r.headers["location"]).text
    assert "That is FM27&#39;s games folder" in text and "Reading the newest save now" in text
    local = env.config.parent / "local.toml"
    assert f"fm_user_dir = '{folder}'" in local.read_text(encoding="utf-8")
    s = env.settings()
    assert s.paths.fm_user_dir == folder and s.paths.origins["paths.fm_user_dir"] == str(local)
    assert env.poller.woken == 1
    # The sync ran straight away, from the new folder, with no restart.
    status = env.syncer.status()
    assert status.error is None and (status.message or "").startswith("Synced career.fm")
    assert env.one("SELECT COUNT(*) FROM careers") == 1
    assert "Set up:" not in env.client.get("/").text
    assert "Set up:" not in env.client.get("/players").text


def test_with_the_folder_in_its_usual_place_nothing_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = fm27(tmp_path)
    monkeypatch.setattr(
        "facegen.persistence.settings.detect_folder",
        lambda key: folder if key == "paths.fm_user_dir" else None,
    )
    env = make_env(tmp_path)
    assert "Set up:" not in env.client.get("/").text
    assert "found on this PC" in env.client.get("/settings/advanced").text


def test_a_folder_that_was_moved_is_asked_for_again(env: Env) -> None:
    (env.config.parent / "local.toml").write_text(
        f"[paths]\nfm_user_dir = '{env.tmp / 'moved away'}'\n", encoding="utf-8"
    )
    assert "is not there any more" in env.client.get("/").text
    assert "is not there any more" in env.client.get("/players").text


def test_the_folder_can_be_changed_from_settings(env: Env) -> None:
    first = fm27(env.tmp / "one")
    second = fm27(env.tmp / "two")
    env.follow(env.client.post("/setup/folder", data={"folder": str(first), "next": "/settings"}))
    env.syncer.join(10)
    page = env.client.get("/settings/advanced").text
    assert f'value="{first}"' in page and 'action="/setup/folder"' in page
    r = env.client.post("/setup/folder", data={"folder": str(second), "next": "/settings"})
    assert r.headers["location"].startswith("/settings?msg=")
    env.syncer.join(10)
    assert env.settings().paths.fm_user_dir == second
    text = (env.config.parent / "local.toml").read_text(encoding="utf-8")
    assert str(second) in text and str(first) not in text


def test_the_local_files_other_keys_are_kept(env: Env) -> None:
    local = env.config.parent / "local.toml"
    local.write_text(
        "# mine\n[paths]\nfm_install_dir = '/games/FM27'\nfm_user_dir = '/old'\n", encoding="utf-8"
    )
    folder = fm27(env.tmp)
    env.follow(env.client.post("/setup/folder", data={"folder": str(folder), "next": "/"}))
    env.syncer.join(10)
    assert local.read_text(encoding="utf-8") == (
        f"# mine\n[paths]\nfm_install_dir = '/games/FM27'\nfm_user_dir = '{folder}'\n"
    )


def test_setup_posts_need_the_form_token(env: Env) -> None:
    folder = fm27(env.tmp)
    del env.client.headers["X-CSRF-Token"]
    r = env.client.post("/setup/folder", data={"folder": str(folder)})
    assert r.status_code == 403
    r = env.client.post(
        "/setup/folder",
        data={"folder": str(folder), "csrf_token": "x"},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403
    assert not (env.config.parent / "local.toml").exists()


def test_browse_opens_the_dialog_on_this_pc(tmp_path: Path) -> None:
    folder = fm27(tmp_path)
    answers = iter([None, str(folder)])
    starts: list[str | None] = []

    def browse(start: str | None) -> str | None:
        starts.append(start)
        return next(answers)

    env = make_env(tmp_path, browse=browse)
    assert "Browse…" in env.client.get("/").text
    text = env.follow(env.client.post("/setup/browse", data={"folder": "", "next": "/"}))
    assert "No folder chosen." in text and env.settings().paths.fm_user_dir is None
    text = env.follow(env.client.post("/setup/browse", data={"folder": "C:/x", "next": "/"}))
    env.syncer.join(10)
    assert env.settings().paths.fm_user_dir == folder
    assert starts == [None, "C:/x"]


def test_browse_where_no_dialog_can_open(env: Env) -> None:
    text = env.follow(env.client.post("/setup/browse", data={"next": "/"}))
    assert "Browse is not available here" in text


# --- 10b: the models --------------------------------------------------------------------------


def test_missing_models_ask_for_one_confirmation(tmp_path: Path) -> None:
    hub = HubCache(tmp_path / "hub")
    download = Download()
    folder = fm27(tmp_path)
    env = make_env(tmp_path, models_status=lambda m: models.status(m, hub), download=download)
    env.follow(env.client.post("/setup/folder", data={"folder": str(folder), "next": "/"}))
    env.syncer.join(10)

    text = env.client.get("/").text
    assert "<strong>Picture models</strong>" in text and "Download 11.6 GB" in text
    assert str(hub.root) in text and "Free space there" in text
    assert "the models are not downloaded yet (11.6 GB)" in env.client.get("/players").text
    # Nothing starts without the confirmation (AGENTS.md hard rule 4).
    text = env.follow(env.client.post("/setup/models", data={}))
    assert "Not downloaded" in text and download.started == 0
    assert 'name="model" value="animagine"' in env.client.get("/").text
    yes = {"confirm": "download", "model": "animagine"}
    text = env.follow(env.client.post("/setup/models", data=yes))
    assert "Downloading the models (11.6 GB)" in text and download.models == ["animagine"]
    assert "<progress" in env.client.get("/status").text
    text = env.follow(env.client.post("/setup/models", data=yes))
    assert "running already" in text and download.started == 1

    # Finished: the files are there, the panel says so once and the banner goes.
    for f in files_for("animagine"):
        repo = hub.repo_dir(f.repo)
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text("c0", encoding="utf-8")
        (repo / "snapshots" / "c0" / f.filename).parent.mkdir(parents=True, exist_ok=True)
        (repo / "snapshots" / "c0" / f.filename).write_bytes(b"x")
    download.now = DownloadStatus(finished=True, total_bytes=1000, done_bytes=1000)
    assert re.search(
        r"<strong>Picture models</strong>\s*Downloaded\.", env.client.get("/status").text
    )
    assert "Set up:" not in env.client.get("/players").text


def test_not_enough_space_disables_the_download(tmp_path: Path) -> None:
    def short(model: str) -> models.ModelsStatus:
        files = files_for(model)
        return models.ModelsStatus(model, tmp_path, files, files, free_bytes=10**9)

    download = Download()
    env = make_env(tmp_path, models_status=short, download=download)
    text = env.client.get("/status").text
    assert "Not enough free space" in text and "disabled" in text
    yes = {"confirm": "download", "model": "animagine"}
    text = env.follow(env.client.post("/setup/models", data=yes))
    assert "Not enough free space" in text and download.started == 0


def put_files(hub: HubCache, model: str) -> None:
    for f in files_for(model):
        target = hub.repo_dir(f.repo) / "snapshots" / "c0" / f.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        (hub.repo_dir(f.repo) / "refs").mkdir(exist_ok=True)
        (hub.repo_dir(f.repo) / "refs" / "main").write_text("c0", encoding="utf-8")


def test_only_the_selected_models_files_are_offered(tmp_path: Path) -> None:
    """Phase 15e: with Animagine's files here, choosing RealVisXL offers its files alone; the
    confirmation names the model shown, and a model changed meanwhile is not downloaded."""
    hub = HubCache(tmp_path / "hub")
    put_files(hub, "animagine")
    download = Download()
    env = make_env(tmp_path, models_status=lambda m: models.status(m, hub), download=download)
    env.follow(env.client.post("/setup/folder", data={"folder": str(fm27(tmp_path)), "next": "/"}))
    env.syncer.join(10)
    text = env.client.get("/status").text
    assert re.search(r"<strong>Picture models</strong>\s*Downloaded\.", text)  # all here
    assert "Set up:" not in env.client.get("/players").text

    conn = env.conn()
    set_overrides(conn, {"style.model": "realvis"}, env.config)
    text = env.client.get("/status").text
    assert "<strong>Picture models</strong>" in text and "Download 6.9 GB" in text
    assert "SG161222/RealVisXL_V5.0" in text and "RealVisXL V5.0, the image model" in text
    assert "h94/IP-Adapter" not in text and "The rest is here already." in text
    assert 'name="model" value="realvis"' in text
    assert "the models are not downloaded yet (6.9 GB)" in env.client.get("/players").text

    # A page shown before the model was changed: its size was another model's.
    old = {"confirm": "download", "model": "animagine"}
    text = env.follow(env.client.post("/setup/models", data=old))
    assert "The model was changed in Settings" in text and download.started == 0
    text = env.follow(env.client.post("/setup/models", data={"confirm": "download"}))
    assert "The model was changed in Settings" in text and download.started == 0
    yes = {"confirm": "download", "model": "realvis"}
    text = env.follow(env.client.post("/setup/models", data=yes))
    assert "Downloading the models (6.9 GB)" in text and download.models == ["realvis"]

    set_overrides(conn, {"style.model": "animagine"}, env.config)
    conn.close()
    assert "Set up:" not in env.client.get("/players").text  # back to the model that is here


def test_without_a_gpu_worker_the_models_are_not_part_of_the_setup(env: Env) -> None:
    assert "Picture models" not in env.client.get("/status").text
    text = env.follow(env.client.post("/setup/models", data={"confirm": "download"}))
    assert "not downloaded by this process" in text


# --- 10d: Auto-configure for my GPU -----------------------------------------------------------


def test_auto_configure_on_a_16_gb_card_changes_nothing(tmp_path: Path) -> None:
    env = make_env(tmp_path, card=Hardware("NVIDIA GeForce RTX 4080", 17_171_480_576))
    text = env.client.get("/hardware").text
    assert "NVIDIA GeForce RTX 4080" in text and "16.0 GiB" in text
    assert "12 GB or more" in text and "Nothing to change" in text
    text = env.follow(env.client.post("/settings/auto-configure"))
    assert "Nothing to change: the settings already suit the NVIDIA GeForce RTX 4080" in text
    assert env.one("SELECT COUNT(*) FROM settings_history") == 0
    assert env.asked == [1]  # the card is asked once per process


def test_auto_configure_on_an_8_gb_card_and_one_undo(tmp_path: Path) -> None:
    env = make_env(tmp_path, card=Hardware("Fake 8 GB", 8 * GIB - 2**20))
    before = env.settings()
    text = env.client.get("/hardware").text
    assert "under 10 GB" in text and "Apply these 3 changes" in text
    assert "Detailed group · Candidates per picture" in text
    text = env.follow(env.client.post("/settings/auto-configure"))
    assert "Auto-configured for the Fake 8 GB (under 10 GB)" in text
    c = env.conn()
    try:
        s = load_settings(c, env.config)  # with the overrides in the database
    finally:
        c.close()
    assert s.groups.a.candidates == 2 and s.groups.a.detail_size is None
    assert s.render.vae_tiling is True
    # Who gets a picture is untouched.
    assert (s.groups.a.ca_above, s.groups.b.any_pa_above, s.groups.c.league_ca_above) == (
        before.groups.a.ca_above,
        before.groups.b.any_pa_above,
        before.groups.c.league_ca_above,
    )
    assert env.one("SELECT COUNT(*) FROM settings_history") == 1
    assert env.one("SELECT label FROM settings_history") == "auto-configure"
    assert "auto-configured for the graphics card" in env.client.get("/settings/advanced").text

    text = env.follow(env.client.post("/settings", data={"action": "undo"}))
    assert "Undone:" in text
    assert env.one("SELECT COUNT(*) FROM settings") == 0  # every value back to its default
    assert env.asked == [1]


def test_auto_configure_asks_the_card_again_on_request(tmp_path: Path) -> None:
    env = make_env(tmp_path, card=Hardware("Fake 10 GB", 10 * GIB))
    env.client.get("/hardware")
    env.client.get("/hardware")
    assert env.asked == [1]
    assert "1152x1472" in env.client.get("/hardware?again=1").text
    assert env.asked == [1, 1]


def test_auto_configure_without_a_cuda_card_changes_nothing(tmp_path: Path) -> None:
    env = make_env(tmp_path, card=Hardware(reason="torch finds no CUDA device"))
    text = env.client.get("/hardware").text
    assert "no CUDA graphics card" in text and "torch finds no CUDA device" in text
    assert "needs an NVIDIA card with CUDA, and none was found" in text and "Apply" not in text
    text = env.follow(env.client.post("/settings/auto-configure"))
    assert "Nothing changed: no CUDA graphics card" in text
    assert env.one("SELECT COUNT(*) FROM settings_history") == 0


def test_the_settings_page_offers_auto_configure_and_the_rendering_section(env: Env) -> None:
    text = env.client.get("/settings/advanced").text
    assert 'href="/hardware"' in text
    assert '<select name="groups.a.detail_size"' in text
    assert '<option value="1152x1472"' in text
    assert 'name="render.vae_tiling"' in text
