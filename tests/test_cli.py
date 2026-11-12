from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import pytest
from factories import ABROAD, MY_CLUB, NEWGEN, player, snapshot

from facegen.cli import main
from facegen.datasource.base import UnsupportedSaveError
from facegen.datasource.records import Snapshot
from facegen.settings import DEFAULTS_PATH

FM_USER_LINE = 'fm_user_dir = "/mnt/c/Users/<Windows user>/Documents/Sports Interactive/Football Manager 27"'


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
    assert FM_USER_LINE in text
    text = text.replace(FM_USER_LINE, f'fm_user_dir = "{fm}"')
    text = text.replace('data_dir = "data"', f'data_dir = "{tmp_path / "data"}"')
    config = tmp_path / "test.toml"
    config.write_text(text, encoding="utf-8")
    return {"config": config, "games": games, "save": save, "data": tmp_path / "data"}


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
    source = FakeSource(UnsupportedSaveError("FM28 is not supported"))
    code, text = run(env, "sync", source=source)
    assert code == 3
    assert "FM28 is not supported" in text and "unchanged" in text
    code, status = run(env, "status")
    assert "No careers yet" in status


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
    assert "http://127.0.0.1:8799/" in out.getvalue()
    assert "Stopping the worker" in out.getvalue()


def test_serve_refuses_to_listen_beyond_this_pc(env: dict[str, Path]) -> None:
    config = env["config"]
    config.write_text(config.read_text().replace('host = "127.0.0.1"', 'host = "0.0.0.0"'))
    code, text = run(env, "serve")
    assert code == 2 and "this PC only" in text
