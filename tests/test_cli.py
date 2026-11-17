from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import pytest
from factories import ABROAD, MY_CLUB, NEWGEN, player, snapshot

from facegen import native
from facegen.cli import main
from facegen.datasource.base import UnsupportedSaveError
from facegen.datasource.records import Snapshot
from facegen.settings import DEFAULTS_PATH

FM_USER_LINE = 'fm_user_dir = ""'
FM_INSTALL_LINE = 'fm_install_dir = ""'


class FakeSource:
    def __init__(self, snap: Snapshot | Exception) -> None:
        self.snap = snap
        self.paths: list[Path] = []

    def load(self, path: Path) -> Snapshot:
        self.paths.append(path)
        if isinstance(self.snap, Exception):
            raise self.snap
        return self.snap


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    fm = tmp_path / "fm"
    games = fm / "games"
    games.mkdir(parents=True)
    save = games / "Career.fm"
    save.write_bytes(b"not really a save")
    os.utime(save, (1_700_000_000, 1_700_000_000))
    text = DEFAULTS_PATH.read_text(encoding="utf-8")
    assert FM_USER_LINE in text and FM_INSTALL_LINE in text
    text = text.replace(FM_USER_LINE, f'fm_user_dir = "{fm.as_posix()}"')
    install = tmp_path / "fm-install"
    text = text.replace(FM_INSTALL_LINE, f'fm_install_dir = "{install.as_posix()}"')
    text = text.replace('data_dir = "data"', f'data_dir = "{(tmp_path / "data").as_posix()}"')
    config = tmp_path / "test.toml"
    config.write_text(text, encoding="utf-8")
    return {
        "config": config,
        "games": games,
        "save": save,
        "data": tmp_path / "data",
        "install": install,
    }


def world() -> Snapshot:
    return snapshot(
        [
            player(1001, name="Old Hand", ca=120, pa=125),
            player(NEWGEN + 1, name="Young Star", ca=160, pa=180, club_id=ABROAD),
        ],
        file_name="Career.fm",
    )


def run(env: dict[str, Path], *args: str, source: FakeSource | None = None) -> tuple[int, str]:
    out = io.StringIO()
    src = source or FakeSource(world())
    code = main(["--config", str(env["config"]), *args], source_factory=lambda: src, out=out)
    return code, out.getvalue()


def test_sync_dry_run_reads_a_copy_and_stores_nothing(env: dict[str, Path]) -> None:
    source = FakeSource(world())
    code, text = run(env, "sync", "--dry-run", source=source)
    assert code == 0
    assert "DRY RUN" in text
    assert "Groups      A 2 (1 at my club)" in text
    assert "Would queue initial A 2" in text
    assert "Old Hand" in text and "Young Star" in text
    # It parsed the snapshot copy, never the game's own file.
    assert source.paths == [(env["data"] / "snapshots" / "Career.fm").resolve()]
    _, status = run(env, "status")
    assert "No careers yet" in status


def test_sync_reads_fm27s_photo_list_and_picks_pre_existing_players_without_one(
    env: dict[str, Path],
) -> None:
    from test_photos import catalog, config_xml

    from facegen.photos import CATALOG_DIR

    snap = snapshot(
        [
            player(1001, name="Old Hand", ca=120, pa=125),
            player(1002, name="Faceless Star", ca=170, pa=175, club_id=ABROAD),
            player(1003, name="Famous Star", ca=170, pa=175, club_id=ABROAD),
            player(1004, name="Facepack Star", ca=170, pa=175, club_id=ABROAD),
        ],
        file_name="Career.fm",
    )
    folder = env["install"] / CATALOG_DIR
    folder.mkdir(parents=True)
    (folder / "catalog_info.bin").write_bytes(catalog(1003))
    (folder / "catalog_info.hash").write_text("v1")
    pack = env["games"].parent / "graphics" / "some-facepack"
    pack.mkdir(parents=True)
    (pack / "config.xml").write_bytes(config_xml(1004))
    # Our own packs never count as a facepack.
    ours = env["games"].parent / "graphics" / "facegen" / "another-career"
    ours.mkdir(parents=True)
    (ours / "config.xml").write_bytes(config_xml(1002))

    code, text = run(env, "sync", source=FakeSource(snap))
    assert code == 0
    assert "Queued      initial A 2" in text  # Old Hand (my club) and Faceless Star
    assert "Faceless Star" in text and "no photo, CA 170 > 155" in text
    assert "Famous Star" not in text and "Facepack Star" not in text
    assert "No photo    2 pre-existing players (2 have one); 1 of them in a group" in text
    code, text = run(env, "sync", source=FakeSource(snap))
    assert "no new jobs" in text and "1 of them in a group" in text


def test_sync_warns_when_fm27s_photo_list_is_missing(env: dict[str, Path]) -> None:
    code, text = run(env, "sync")
    assert code == 0
    assert "FM27's photo list" in text and "could not be read" in text
    assert "No photo" not in text


def test_sync_then_status(env: dict[str, Path]) -> None:
    code, text = run(env, "sync")
    assert code == 0 and "Sync stored." in text and "Queued      initial A 2" in text
    code, text = run(env, "sync")
    assert "no new jobs" in text
    code, status = run(env, "status")
    assert code == 0
    assert "groups  A 2  B 0  C 0  tracked 2" in status
    assert "jobs    queued 2" in status


def test_sync_writes_the_player_table(env: dict[str, Path], tmp_path: Path) -> None:
    out_csv = tmp_path / "out" / "players.csv"
    code, text = run(env, "sync", "--dry-run", "--csv", str(out_csv))
    assert code == 0 and "players to" in text
    with out_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    star = next(r for r in rows if r["name"] == "Young Star")
    assert (star["group"], star["is_newgen"], star["club"]) == ("A", "True", f"Club {ABROAD}")


def test_sync_with_an_explicit_save(env: dict[str, Path], tmp_path: Path) -> None:
    other = tmp_path / "elsewhere" / "Other.fm"
    other.parent.mkdir()
    other.write_bytes(b"x")
    source = FakeSource(world())
    code, _ = run(env, "sync", "--dry-run", "--save", str(other), source=source)
    assert code == 0
    assert source.paths == [(env["data"] / "snapshots" / "Other.fm").resolve()]


def test_snapshot_already_in_the_snapshots_folder_is_read_in_place(env: dict[str, Path]) -> None:
    snap = env["data"] / "snapshots" / "kept.fm"
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b"x")
    source = FakeSource(world())
    run(env, "sync", "--dry-run", "--save", str(snap), source=source)
    assert source.paths == [snap]


def test_unsupported_save_is_reported_and_recorded(env: dict[str, Path]) -> None:
    source = FakeSource(UnsupportedSaveError("FM28 is not supported", build="28.0.0"))
    code, text = run(env, "sync", source=source)
    assert code == 3
    assert "FM28 is not supported" in text and "unchanged" in text
    assert "bump the pin" in text  # says what would fix it
    code, status = run(env, "status")
    assert "No careers yet" in status
    assert "Last sync failed (Career.fm): FM28 is not supported" in status
    assert "bump the pin" in status


def test_a_save_that_cannot_be_read_is_not_called_unsupported(env: dict[str, Path]) -> None:
    from facegen.datasource.base import SaveReadError

    code, text = run(env, "sync", source=FakeSource(SaveReadError("damaged")))
    assert code == 3 and "damaged" in text and "bump the pin" not in text


def test_status_marks_the_active_career(env: dict[str, Path]) -> None:
    run(env, "sync")
    _, status = run(env, "status")
    assert "(active: FM27 is shown this pack)" in status


def test_status_says_whether_the_rust_extension_is_in_use(env: dict[str, Path]) -> None:
    _, status = run(env, "status")
    line = next(x for x in status.splitlines() if x.startswith("Rust code"))
    assert ("in use (facegen-native" in line) == (native.ext is not None)
    assert ("the Python versions run" in line) == (native.ext is None)


def test_missing_saves(env: dict[str, Path]) -> None:
    env["save"].unlink()
    code, text = run(env, "sync")
    assert code == 2 and "No save found" in text
    code, text = run(env, "sync", "--save", str(env["games"] / "nope.fm"))
    assert code == 2 and "No such save" in text


def test_my_club_players_are_listed_first(env: dict[str, Path]) -> None:
    _code, text = run(env, "sync", "--dry-run")
    lines = [line for line in text.splitlines() if line.strip()[:2] in ("1.", "2.")]
    assert "A, my club" in lines[0] and f"Club {MY_CLUB}" in lines[0]


def run_gpu(env: dict[str, Path], *args: str, fm: bool | None = False) -> tuple[int, str]:
    from fakes import FakeRenderer, FakeTagger

    from facegen.cli import Gpu

    out = io.StringIO()
    renderer, tagger = FakeRenderer(), FakeTagger()
    gpu = Gpu(renderer=lambda: renderer, tagger=lambda: tagger, fm_check=lambda: fm)
    code = main(
        ["--config", str(env["config"]), *args],
        source_factory=lambda: FakeSource(world()),
        gpu=gpu,
        out=out,
    )
    return code, out.getvalue()


def test_run_needs_a_career(env: dict[str, Path]) -> None:
    code, text = run_gpu(env, "run")
    assert code == 2 and "Run `facegen sync`" in text


def test_run_dry_run_prints_prompts_and_draws_nothing(env: dict[str, Path]) -> None:
    run(env, "sync")
    code, text = run_gpu(env, "run", "--group", "a", "--my-club", "--dry-run")
    assert code == 0 and "DRY RUN: 1 jobs would run" in text
    assert "Old Hand" in text and "Young Star" not in text  # my club only
    assert "+ 1girl, solo, safe, " in text and "- nsfw, explicit" in text
    _, status = run(env, "status")
    assert "jobs    queued 2" in status and "pictures 0 current" in status


def test_run_publish_gallery_and_status(env: dict[str, Path]) -> None:
    run(env, "sync")
    code, text = run_gpu(env, "run", "--group", "a", "--my-club")
    assert code == 0, text
    assert "Done        1 pictures, 0 sent to review" in text
    assert "Published   1 pictures" in text and "Reload Skin" in text
    assert "Gallery" in text
    packs = list((env["games"].parent / "graphics" / "facegen").iterdir())
    assert len(packs) == 1 and sorted(p.name for p in packs[0].iterdir()) == [
        "1001.png",
        "config.xml",
    ]
    _, status = run(env, "status")
    assert "pictures 1 current, 1 published" in status and "done 1" in status

    # Re-running changes nothing.
    code, text = run_gpu(env, "run", "--group", "a", "--my-club")
    assert code == 0 and "Done        0 pictures" in text and "nothing changed" in text
    code, text = run(env, "publish")
    assert code == 0 and "nothing changed" in text
    code, text = run(env, "gallery")
    assert code == 0 and text.strip().endswith("index.html")


def test_run_stops_while_fm27_runs(env: dict[str, Path]) -> None:
    run(env, "sync")
    code, text = run_gpu(env, "run", fm=True)
    assert code == 5 and "FM27 is running" in text and "Done        0 pictures" in text
    code, text = run_gpu(env, "run", "--no-publish", "--now", fm=True)
    assert code == 0 and "Done        2 pictures" in text and "Not published" in text
    assert not (env["games"].parent / "graphics").exists()


def test_serve_starts_the_ui_and_the_worker_and_stops_cleanly(env: dict[str, Path]) -> None:
    from fakes import FakeRenderer, FakeTagger
    from fastapi.testclient import TestClient

    from facegen.cli import Gpu

    run(env, "sync")
    seen: dict[str, object] = {}

    def fake_server(app, host: str, port: int) -> None:  # type: ignore[no-untyped-def]
        seen.update(host=host, port=port)
        with TestClient(app, base_url=f"http://{host}:{port}") as client:
            seen["page"] = client.get("/").text

    out = io.StringIO()
    gpu = Gpu(renderer=FakeRenderer, tagger=FakeTagger, fm_check=lambda: True)
    code = main(
        ["--config", str(env["config"]), "serve", "--port", "8799", "--paused"],
        source_factory=lambda: FakeSource(world()),
        gpu=gpu,
        out=out,
        run_server=fake_server,
    )
    assert code == 0
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 8799)
    assert "Old Hand" in str(seen["page"])
    assert "watching saves" in str(seen["page"])
    assert "http://127.0.0.1:8799/" in out.getvalue()
    assert f"Watching {env['games']}" in out.getvalue()
    assert "Stopping the worker" in out.getvalue()


def test_serve_without_the_save_watcher(env: dict[str, Path]) -> None:
    from fastapi.testclient import TestClient

    seen: dict[str, str] = {}

    def fake_server(app, host: str, port: int) -> None:  # type: ignore[no-untyped-def]
        with TestClient(app, base_url=f"http://{host}:{port}") as client:
            seen["page"] = client.get("/").text

    out = io.StringIO()
    code = main(
        ["--config", str(env["config"]), "serve", "--no-worker", "--no-watch"],
        source_factory=lambda: FakeSource(world()),
        out=out,
        run_server=fake_server,
    )
    assert code == 0 and "Watching" not in out.getvalue()
    assert "Not watching the saves folder" in seen["page"]


def test_sync_records_that_the_command_started_it(env: dict[str, Path]) -> None:
    import sqlite3

    run(env, "sync")
    conn = sqlite3.connect(env["data"] / "facegen.db")
    try:
        assert conn.execute("SELECT trigger FROM sync_runs").fetchone()[0] == "command"
    finally:
        conn.close()


def test_sync_of_another_careers_newer_save_switches_the_pack(env: dict[str, Path]) -> None:
    run(env, "sync")
    code, text = run_gpu(env, "run", "--group", "a", "--my-club")
    assert code == 0, text
    first = next((env["games"].parent / "graphics" / "facegen").iterdir())
    assert 'from="1001"' in (first / "config.xml").read_text()

    other = env["games"] / "Other.fm"
    other.write_bytes(b"another career")
    os.utime(other, (1_800_000_000, 1_800_000_000))
    code, text = run(
        env, "sync", source=FakeSource(snapshot([player(5001)], career="two", file_name="Other.fm"))
    )
    assert code == 0, text
    assert f"packs now map nothing: {first.name}" in text and "Published" not in text
    assert 'from="1001"' not in (first / "config.xml").read_text()
    assert (first / "1001.png").is_file()


def test_serve_refuses_to_listen_beyond_this_pc(env: dict[str, Path]) -> None:
    config = env["config"]
    config.write_text(config.read_text().replace('host = "127.0.0.1"', 'host = "0.0.0.0"'))
    code, text = run(env, "serve")
    assert code == 2 and "this PC only" in text


# --- Phase 10: the model files and the graphics card ------------------------------------------


def test_run_without_the_models_says_how_to_get_them(env: dict[str, Path]) -> None:
    from fakes import FakeRenderer, FakeTagger

    from facegen.cli import Gpu

    run(env, "sync")
    out = io.StringIO()
    renderer = FakeRenderer()
    gpu = Gpu(
        renderer=lambda: renderer, tagger=FakeTagger, fm_check=lambda: False,
        models_ready=lambda: False,
    )  # fmt: skip
    code = main(["--config", str(env["config"]), "run"], gpu=gpu, out=out)
    assert code == 2 and "facegen models --download" in out.getvalue()
    assert renderer.calls == []
    # A dry run needs no models.
    out = io.StringIO()
    code = main(["--config", str(env["config"]), "run", "--dry-run"], gpu=gpu, out=out)
    assert code == 0 and "DRY RUN" in out.getvalue()


def test_only_the_real_renderer_asks_for_the_model_files() -> None:
    from fakes import FakeRenderer

    from facegen import models
    from facegen.cli import Gpu

    assert Gpu().models_check() is models.ready_now
    assert Gpu(renderer=FakeRenderer).models_check() is None  # the tests' fakes need none
    check = lambda: True
    assert Gpu(renderer=FakeRenderer, models_ready=check).models_check() is check


def test_serve_with_the_real_worker_offers_the_model_download(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import FakeRenderer, FakeTagger
    from fastapi.testclient import TestClient

    from facegen import models
    from facegen.cli import Gpu

    run(env, "sync")
    monkeypatch.setattr(
        models, "status", lambda: models.ModelsStatus(env["data"], models.FILES, 0, 10**12)
    )
    seen: dict[str, str] = {}

    def fake_server(app, host: str, port: int) -> None:  # type: ignore[no-untyped-def]
        with TestClient(app, base_url=f"http://{host}:{port}") as client:
            seen["page"] = client.get("/").text

    gpu = Gpu(
        renderer=FakeRenderer, tagger=FakeTagger, fm_check=lambda: True,
        models_ready=lambda: False,
    )  # fmt: skip
    code = main(
        ["--config", str(env["config"]), "serve", "--paused", "--no-watch"],
        source_factory=lambda: FakeSource(world()),
        gpu=gpu,
        out=io.StringIO(),
        run_server=fake_server,
    )
    assert code == 0
    assert "Set up: the models" in seen["page"] and "Download 11.6 GB" in seen["page"]


def test_models_and_hardware_need_no_database(env: dict[str, Path], monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from facegen import cli, hardware

    monkeypatch.setattr(cli, "cmd_models", lambda args, out: 7)
    assert main(["--config", str(env["config"]), "models"], out=io.StringIO()) == 7
    card = hardware.Hardware("Fake 8 GB", 8 * 2**30)
    monkeypatch.setattr(hardware, "detect", lambda: card)
    out = io.StringIO()
    assert main(["--config", str(env["config"]), "hardware", "--json"], out=out) == 0
    assert hardware.Hardware.from_json(__import__("json").loads(out.getvalue())) == card
    assert not env["data"].exists()  # neither touched the database
    out = io.StringIO()
    assert main(["--config", str(env["config"]), "hardware"], out=out) == 0
    assert "would change groups.a.candidates: 3 -> 2" in out.getvalue()
