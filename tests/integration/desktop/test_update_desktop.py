"""Phase 13 on the desktop: the tray's update notice, Update now in the app, the installer's
shortcuts pointed at the app after the first setup, and removing the picture pack."""

from __future__ import annotations

import base64
import io
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from support.desktop import FakeSyncer, FakeWorker

from facegen.cli import main
from facegen.desktop import shortcuts
from facegen.desktop.app_controller import NOTICE_TITLE_CHARS, UPDATE_LOOK_SECONDS, AppController
from facegen.persistence.settings import load_settings
from facegen.pipeline.publish import packs_root
from facegen.platform import procs
from facegen.runtime.update_service import UpdateStatus
from facegen.setup import launch


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Updates:
    def __init__(self) -> None:
        self.s = UpdateStatus(enabled=True, current="0.9.0")
        self.asked = 0

    def status(self) -> UpdateStatus:
        self.asked += 1
        return self.s


def offer(version: str) -> UpdateStatus:
    return UpdateStatus(enabled=True, current="0.9.0", latest=version, available=True, offered=True)


def test_the_tray_says_once_per_version_that_a_newer_one_is_out() -> None:
    clock, updates = Clock(), Updates()
    c = AppController(
        worker=FakeWorker(),
        syncer=FakeSyncer(),
        open_dashboard=lambda: None,
        updates=updates,
        clock=clock,
    )
    assert [n for n in c.notices() if n.kind == "update"] == []
    updates.s = offer("1.0.0")
    assert [n for n in c.notices() if n.kind == "update"] == []  # looked a moment ago
    clock.now += UPDATE_LOOK_SECONDS
    (notice,) = [n for n in c.notices() if n.kind == "update"]
    assert notice.title == "facegen 1.0.0 is available"
    assert notice.message == "You have 0.9.0. Open the dashboard to update."
    assert len(notice.title) <= NOTICE_TITLE_CHARS
    clock.now += UPDATE_LOOK_SECONDS
    assert [n for n in c.notices() if n.kind == "update"] == []  # said once
    updates.s = offer("1.1.0")
    clock.now += UPDATE_LOOK_SECONDS
    assert [n.title for n in c.notices() if n.kind == "update"] == ["facegen 1.1.0 is available"]
    # Put off with Later: not offered, not said.
    updates.s = UpdateStatus(enabled=True, current="0.9.0", latest="1.2.0", available=True)
    clock.now += UPDATE_LOOK_SECONDS
    assert [n for n in c.notices() if n.kind == "update"] == []


def test_a_broken_update_check_does_not_stop_the_tray() -> None:
    class Broken:
        def status(self) -> UpdateStatus:
            raise OSError("disk gone")

    c = AppController(
        worker=FakeWorker(), syncer=FakeSyncer(), open_dashboard=lambda: None, updates=Broken()
    )
    assert [n for n in c.notices() if n.kind == "update"] == []


def test_the_hook_is_only_set_where_an_update_in_place_works(tmp_path: Path) -> None:
    from facegen.desktop import shell as desktop

    (tmp_path / "Start.bat").write_bytes(b"@echo off\r\n")
    assert desktop._can_update_here("nt", tmp_path)  # an installed or unzipped facegen
    assert not desktop._can_update_here("posix", tmp_path)  # Linux and WSL: `facegen serve`
    (tmp_path / ".git").mkdir()
    assert not desktop._can_update_here("nt", tmp_path)  # a git checkout: git pull
    assert not desktop._can_update_here("nt", tmp_path / "elsewhere")  # no Start.bat
    with pytest.raises(OSError, match="Windows only"):
        procs.start_console("Start.bat", "--update", cwd=".", os_name="posix")
    calls: list[tuple[Any, ...]] = []
    procs.start_console(
        r"C:\facegen\Start.bat",
        "--update",
        cwd=r"C:\facegen",
        os_name="nt",
        startfile=lambda *a: calls.append(a),
    )
    assert calls == [(r"C:\facegen\Start.bat", "open", "--update", r"C:\facegen")]


def test_the_app_mutex_is_windows_only() -> None:
    from facegen.desktop import shell as desktop

    release = desktop.hold_app_mutex("posix")
    release()  # nothing to let go


# --- the installer's shortcuts -------------------------------------------------------------

FOLDERS = {
    "desktop": r"C:\Users\me\OneDrive\바탕 화면",
    "programs": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs",
}
PYW = r"C:\Users\me\AppData\Local\facegen\.venv\Scripts\pythonw.exe"
APP = r"C:\Users\me\AppData\Local\facegen\.venv\Scripts\facegen-app.exe"
STARTUP = r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
ROOT = r"C:\Users\me\AppData\Local\facegen"
ICON = ROOT + r"\data\app\facegen.ico"
START_BAT = ROOT + r"\Start.bat"


def decoded(args: Sequence[str]) -> str:
    return base64.b64decode(args[args.index("-EncodedCommand") + 1]).decode("utf-16-le")


def test_refresh_points_only_our_earlier_shortcuts_at_the_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    links = shortcuts.app_shortcuts(FOLDERS, program=APP, root=ROOT, icon=ICON)
    monkeypatch.setattr(procs, "windows_tool", lambda *a, **k: "powershell.exe")
    calls: list[Sequence[str]] = []

    def run(args: Sequence[str], **kw: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, links[1].path + "\r\n", "")

    # Only the Start Menu's is there: only it is looked at.
    replacing = [START_BAT, PYW]
    done = shortcuts.refresh(links, replacing, run=run, exists=lambda p: p == links[1].path)
    assert done == [links[1].path]
    script = decoded(calls[0])
    assert links[0].path not in script and "Test-Path -LiteralPath" in script
    assert f"if (@('{START_BAT}', '{PYW}') -icontains $s.TargetPath)" in script
    assert f"$s.TargetPath = '{APP}'" in script
    assert "$s.Arguments = '-m facegen.desktop app'" in script
    assert "WindowStyle" not in script
    # None there, or nothing to replace: PowerShell is not even started.
    assert shortcuts.refresh(links, replacing, run=run, exists=lambda p: False) == []
    assert shortcuts.refresh(links, [], run=run, exists=lambda p: True) == []
    assert len(calls) == 1

    def fails(args: Sequence[str], **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "Access is denied.\n")

    with pytest.raises(shortcuts.ShortcutError, match="Access is denied"):
        shortcuts.refresh(links, replacing, run=fails, exists=lambda p: True)


def test_the_refreshed_startup_shortcut_stays_minimised() -> None:
    link = shortcuts.startup_shortcut(STARTUP, program=APP, root=ROOT, icon=ICON)
    script = shortcuts.refresh_script([link], [PYW])
    assert "$s.Arguments = '-m facegen.desktop app --minimized'" in script
    assert f"$s.WindowStyle = {shortcuts.WINDOW_MINIMIZED}" in script


def test_paths_with_quotes_stay_whole_in_the_refresh() -> None:
    link = shortcuts.Shortcut(r"C:\Users\O'Neil\Desktop\facegen.lnk", PYW, "-m x", ROOT, ICON)
    script = shortcuts.refresh_script([link], [r"C:\Users\O'Neil\facegen\Start.bat"])
    assert r"'C:\Users\O''Neil\Desktop\facegen.lnk'" in script
    assert r"@('C:\Users\O''Neil\facegen\Start.bat') -icontains" in script


@pytest.mark.skipif(os.name == "nt", reason="on Windows it would change this PC's shortcuts")
def test_launch_leaves_shortcuts_alone_off_windows(tmp_path: Path) -> None:
    out = io.StringIO()
    launch.refresh_shortcuts(tmp_path, None, out)
    assert out.getvalue() == ""


# --- removing the picture pack -------------------------------------------------------------


def test_remove_pack_deletes_only_facegens_graphics(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    fm = settings.paths.fm_user_dir
    assert fm is not None
    pack = packs_root(fm) / "career1"
    pack.mkdir(parents=True)
    (pack / "1.png").write_bytes(b"png")
    (pack / "config.xml").write_text("<record/>", encoding="utf-8")
    other = fm / "graphics" / "sortitoutsi" / "faces"
    other.mkdir(parents=True)
    (other / "2.png").write_bytes(b"png")
    out = io.StringIO()
    assert main(["--config", str(config_file), "remove-pack", "--yes"], out=out) == 0
    assert not packs_root(fm).exists() and (other / "2.png").is_file()
    assert "Removed" in out.getvalue()
    out = io.StringIO()
    assert main(["--config", str(config_file), "remove-pack", "--yes"], out=out) == 0
    assert "no picture pack to remove" in out.getvalue()


def test_remove_pack_asks_first(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(defaults_path=config_file)
    assert settings.paths.fm_user_dir is not None
    root = packs_root(settings.paths.fm_user_dir)
    root.mkdir(parents=True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    out = io.StringIO()
    assert main(["--config", str(config_file), "remove-pack"], out=out) == 1
    assert root.is_dir() and "Nothing was removed" in out.getvalue()
