from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from facegen.datasource import snapshots
from facegen.datasource.base import SaveReadError
from facegen.runtime import poller


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


def test_copy_snapshot_keeps_the_time_it_checked_not_a_later_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make(tmp_path / "games" / "c.fm", b"first", mtime=1_700_000_000)
    calls = {"n": 0}
    real_signature = snapshots._signature

    def autosave_right_after_the_check(path: Path) -> tuple[int, int]:
        signature = real_signature(path)
        calls["n"] += 1
        if calls["n"] == 2:  # the check after the copy has passed; then the game saves again
            src.write_bytes(b"second")
            os.utime(src, (1_700_000_100, 1_700_000_100))
        return signature

    monkeypatch.setattr(snapshots, "_signature", autosave_right_after_the_check)
    dest = poller.copy_snapshot(src, tmp_path / "snap")
    assert dest.read_bytes() == b"first"
    # The copy carries the first save's time, so the newer save is still seen as new.
    assert dest.stat().st_mtime == 1_700_000_000


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

    monkeypatch.setattr(snapshots.shutil, "copyfile", copy_while_growing)
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

    monkeypatch.setattr(snapshots.shutil, "copyfile", copy_and_grow)
    with pytest.raises(SaveReadError, match="kept changing"):
        poller.copy_snapshot(src, tmp_path / "snap", attempts=2, sleep=lambda _: None)
    assert not list((tmp_path / "snap").iterdir())


def test_copy_snapshot_refuses_to_overwrite_its_source(tmp_path: Path) -> None:
    src = make(tmp_path / "snap" / "c.fm")
    with pytest.raises(ValueError):
        poller.copy_snapshot(src, tmp_path / "snap")


# --- the watcher -----------------------------------------------------------------------------


def version(path: Path, size: int = 10, mtime: int = 1_000) -> poller.SaveVersion:
    return poller.SaveVersion(path, size, mtime * 10**9, float(mtime))


def test_watcher_waits_for_a_changed_save_to_settle(tmp_path: Path) -> None:
    w = poller.SaveWatcher(settle_seconds=10)
    save = tmp_path / "c.fm"
    d = w.look(version(save), already_synced=False, now=0, interval=60)
    assert (d.due, d.state, d.wait) == (None, "settling", 10)
    # Still being written: size changed, so the wait starts again.
    d = w.look(version(save, size=20), already_synced=False, now=10, interval=60)
    assert (d.due, d.wait) == (None, 10)
    d = w.look(version(save, size=20), already_synced=False, now=15, interval=60)
    assert (d.due, d.wait) == (None, 5)
    d = w.look(version(save, size=20), already_synced=False, now=20, interval=60)
    assert d.due == version(save, size=20) and d.state == "due" and d.wait == 0
    w.done()
    d = w.look(version(save, size=20), already_synced=True, now=21, interval=60)
    assert (d.due, d.state, d.wait) == (None, "synced", 60)
    assert "c.fm" in d.detail and "synced" in d.detail


def test_watcher_decisions_carry_a_code_and_values_for_the_log_page(tmp_path: Path) -> None:
    """Phase 17i: the Log page words the watcher from a code and values, not the English."""
    from datetime import datetime

    w = poller.SaveWatcher(settle_seconds=10)
    save = tmp_path / "Sam Lee - 레드우드.fm"
    d = w.look(None, already_synced=False, now=0, interval=60)
    assert (d.reason, d.values) == ("no_save", {})
    d = w.look(version(save), already_synced=False, now=0, interval=60)
    assert (d.reason, d.values) == ("settling", {"name": save.name})
    d = w.look(version(save), already_synced=False, now=5, interval=60)
    assert (d.reason, d.values) == ("settling", {"name": save.name})
    d = w.look(version(save), already_synced=False, now=10, interval=60)
    assert (d.state, d.reason, d.values) == ("due", "reading", {"name": save.name})
    w.done()
    d = w.look(version(save), already_synced=True, now=11, interval=60)
    saved = datetime.fromtimestamp(1_000).strftime("%Y-%m-%d %H:%M")
    assert (d.reason, d.values) == ("synced", {"name": save.name, "saved": saved})
    assert d.detail == f"newest save {save.name} (saved {saved}) is synced"


def test_watcher_ignores_a_save_already_synced(tmp_path: Path) -> None:
    w = poller.SaveWatcher(settle_seconds=10)
    for now in (0, 60, 120):
        d = w.look(version(tmp_path / "c.fm"), already_synced=True, now=now, interval=60)
        assert d.due is None and d.state == "synced"


def test_watcher_with_no_save(tmp_path: Path) -> None:
    d = poller.SaveWatcher().look(None, already_synced=False, now=0, interval=60)
    assert (d.due, d.state, d.wait) == (None, "no save", 60)


def test_watcher_follows_a_newer_save_of_another_name(tmp_path: Path) -> None:
    w = poller.SaveWatcher(settle_seconds=10)
    w.look(version(tmp_path / "a.fm"), already_synced=False, now=0, interval=60)
    # Another save became the newest before the first settled: start over with it.
    d = w.look(version(tmp_path / "b.fm", mtime=2_000), already_synced=False, now=10, interval=60)
    assert d.due is None and d.state == "settling"
    d = w.look(version(tmp_path / "b.fm", mtime=2_000), already_synced=False, now=20, interval=60)
    assert d.due is not None and d.due.path.name == "b.fm"


def test_save_version_reads_the_file(tmp_path: Path) -> None:
    path = make(tmp_path / "c.fm", b"12345", mtime=1_700_000_000)
    v = poller.SaveVersion.of(path)
    assert (v.size, v.mtime, v.mtime_ns) == (5, 1_700_000_000.0, 1_700_000_000 * 10**9)


def test_copy_snapshot_reports_a_busy_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A save that keeps changing is a busy save: the watcher tries it again later."""
    from facegen.datasource.base import SaveBusyError

    src = make(tmp_path / "games" / "c.fm")
    real_copy = shutil.copyfile

    def copy_and_grow(a: Path, b: Path) -> None:
        real_copy(a, b)
        with open(src, "ab") as f:
            f.write(b"!")

    monkeypatch.setattr(snapshots.shutil, "copyfile", copy_and_grow)
    with pytest.raises(SaveBusyError):
        poller.copy_snapshot(src, tmp_path / "snap", attempts=1)
