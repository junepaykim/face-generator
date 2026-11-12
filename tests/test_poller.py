from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from facegen import poller
from facegen.datasource.base import SaveReadError


def make(path: Path, data: bytes = b"save", mtime: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_list_saves_newest_first_without_the_backup(tmp_path: Path) -> None:
    games = tmp_path / "games"
    make(games / "old.fm", mtime=1_000)
    make(games / "new.fm", mtime=3_000)
    make(games / "last save overwrite backup.fm", mtime=5_000)
    make(games / "config.xml", mtime=6_000)
    assert [p.name for p in poller.list_saves(games)] == ["new.fm", "old.fm"]
    assert poller.find_latest_save(games) == games / "new.fm"


def test_no_saves(tmp_path: Path) -> None:
    assert poller.find_latest_save(tmp_path / "missing") is None
    assert poller.find_latest_save(tmp_path) is None


def test_copy_snapshot_copies_and_keeps_the_mtime(tmp_path: Path) -> None:
    src = make(tmp_path / "games" / "My Career.fm", b"x" * 1000, mtime=1_700_000_000)
    dest = poller.copy_snapshot(src, tmp_path / "data" / "snapshots")
    assert dest == (tmp_path / "data" / "snapshots" / "My Career.fm").resolve()
    assert dest.read_bytes() == src.read_bytes()
    assert dest.stat().st_mtime == src.stat().st_mtime
    assert not list(dest.parent.glob("*.part"))
    # The source is untouched.
    assert src.read_bytes() == b"x" * 1000


def test_copy_snapshot_retries_a_save_being_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make(tmp_path / "games" / "c.fm", b"part")
    calls = {"n": 0}
    real_copy = shutil.copyfile

    def copy_while_growing(a: Path, b: Path) -> None:
        calls["n"] += 1
        real_copy(a, b)
        if calls["n"] == 1:  # the game is still writing during the first copy
            with open(src, "ab") as f:
                f.write(b" and the rest")

    monkeypatch.setattr(poller.shutil, "copyfile", copy_while_growing)
    sleeps: list[float] = []
    dest = poller.copy_snapshot(src, tmp_path / "snap", sleep=sleeps.append)
    assert dest.read_bytes() == b"part and the rest"
    assert calls["n"] == 2 and len(sleeps) == 1


def test_copy_snapshot_gives_up_on_a_save_that_keeps_changing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make(tmp_path / "games" / "c.fm")
    real_copy = shutil.copyfile

    def copy_and_grow(a: Path, b: Path) -> None:
        real_copy(a, b)
        with open(src, "ab") as f:
            f.write(b"!")

    monkeypatch.setattr(poller.shutil, "copyfile", copy_and_grow)
    with pytest.raises(SaveReadError, match="kept changing"):
        poller.copy_snapshot(src, tmp_path / "snap", attempts=2, sleep=lambda _: None)
    assert not list((tmp_path / "snap").iterdir())


def test_copy_snapshot_refuses_to_overwrite_its_source(tmp_path: Path) -> None:
    src = make(tmp_path / "snap" / "c.fm")
    with pytest.raises(ValueError):
        poller.copy_snapshot(src, tmp_path / "snap")
