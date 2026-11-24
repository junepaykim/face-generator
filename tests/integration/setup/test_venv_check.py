"""Packages left half-installed by an interrupted `uv sync` (venv_check.py, Start.bat)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kitshot.persistence.settings import PROJECT_ROOT
from kitshot.setup import venv_check


def dist(site: Path, folder: str, *files: str) -> None:
    info = site / folder
    info.mkdir(parents=True)
    for name in files:
        (info / name).write_text("x", encoding="utf-8")


def test_a_dist_info_without_metadata_or_record_is_half_installed(tmp_path: Path) -> None:
    dist(tmp_path, "six-1.17.0.dist-info", "METADATA", "RECORD", "WHEEL")
    # What the console closing during the first uv sync left of torch (2026-11-16).
    dist(tmp_path, "torch-2.15.0+cu132.dist-info", "entry_points.txt")
    dist(tmp_path, "kitshot_native-0.1.0.dist-info", "METADATA")  # RECORD not written yet
    (tmp_path / "stray.dist-info").write_text("not a folder", encoding="utf-8")
    assert venv_check.broken([tmp_path, tmp_path]) == ["kitshot_native", "torch"]
    assert venv_check.reinstall_options(["torch"]) == "--reinstall-package torch"
    assert venv_check.broken([tmp_path / "missing"]) == []


def test_main_prints_the_uv_options_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dist(tmp_path, "torch-2.15.0+cu132.dist-info", "entry_points.txt")
    monkeypatch.setattr(venv_check, "site_dirs", lambda: [tmp_path])
    assert venv_check.main() == venv_check.BROKEN
    out = capsys.readouterr()
    assert out.out == "--reinstall-package torch\n" and "half-installed: torch" in out.err
    monkeypatch.setattr(venv_check, "site_dirs", lambda: [tmp_path / "empty"])
    assert venv_check.main() == 0 and capsys.readouterr().out == ""


def test_this_environment_is_whole() -> None:
    assert venv_check.broken(venv_check.site_dirs()) == []


def test_start_bat_runs_the_check_after_uv_sync() -> None:
    text = (PROJECT_ROOT / "Start.bat").read_bytes().decode("ascii")
    assert text.count("\r\n") == text.count("\n")  # CRLF throughout
    sync = text.index('"%UV%" sync --locked --inexact --no-group native --no-group dev\r\n')
    check = text.index("""('call "%PY%" -m kitshot.venv_check')""")
    repair = text.index('"%UV%" sync --locked --inexact --no-group native --no-group dev %REPAIR%')
    launch = text.index('"%PY%" -m kitshot.cli launch --log')
    assert sync < check < repair < launch
