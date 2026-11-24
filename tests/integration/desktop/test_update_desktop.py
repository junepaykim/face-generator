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

from kitshot.cli import main
from kitshot.desktop import shortcuts
from kitshot.desktop.app_controller import NOTICE_TITLE_CHARS, UPDATE_LOOK_SECONDS, AppController
from kitshot.persistence.settings import load_settings
from kitshot.pipeline.publish import packs_root
from kitshot.platform import procs
from kitshot.runtime.update_service import UpdateStatus
from kitshot.setup import launch


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
    assert notice.title == "Kitshot 1.0.0 is available"
    assert notice.message == "You have 0.9.0. Open the dashboard to update."
    assert len(notice.title) <= NOTICE_TITLE_CHARS
    clock.now += UPDATE_LOOK_SECONDS
    assert [n for n in c.notices() if n.kind == "update"] == []  # said once
    updates.s = offer("1.1.0")
    clock.now += UPDATE_LOOK_SECONDS
    assert [n.title for n in c.notices() if n.kind == "update"] == ["Kitshot 1.1.0 is available"]
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
    from kitshot.desktop import shell as desktop

    (tmp_path / "Start.bat").write_bytes(b"@echo off\r\n")
    assert desktop._can_update_here("nt", tmp_path)  # an installed or unzipped Kitshot
    assert not desktop._can_update_here("posix", tmp_path)  # Linux and WSL: `kitshot serve`
    (tmp_path / ".git").mkdir()
    assert not desktop._can_update_here("nt", tmp_path)  # a git checkout: git pull
    assert not desktop._can_update_here("nt", tmp_path / "elsewhere")  # no Start.bat
    with pytest.raises(OSError, match="Windows only"):
        procs.start_console("Start.bat", "--update", cwd=".", os_name="posix")
    calls: list[tuple[Any, ...]] = []
    procs.start_console(
        r"C:\kitshot\Start.bat",
        "--update",
        cwd=r"C:\kitshot",
        os_name="nt",
        startfile=lambda *a: calls.append(a),
    )
    assert calls == [(r"C:\kitshot\Start.bat", "open", "--update", r"C:\kitshot")]


def test_the_app_mutex_is_windows_only() -> None:
    from kitshot.desktop import shell as desktop

    release = desktop.hold_app_mutex("posix")
    release()  # nothing to let go


# --- the installer's shortcuts -------------------------------------------------------------

FOLDERS = {
    "desktop": r"C:\Users\me\OneDrive\바탕 화면",
    "programs": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs",
}
PYW = r"C:\Users\me\AppData\Local\kitshot\.venv\Scripts\pythonw.exe"
APP = r"C:\Users\me\AppData\Local\kitshot\.venv\Scripts\kitshot-app.exe"
STARTUP = r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
ROOT = r"C:\Users\me\AppData\Local\kitshot"
ICON = ROOT + r"\data\app\kitshot-0123456789.ico"
OLD_ICON = ROOT + r"\data\app\facegen.ico"
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
    assert "$s.Arguments = '-m kitshot.desktop app'" in script
    assert "WindowStyle" not in script
    # None there under either name: PowerShell is not even started.
    assert shortcuts.refresh(links, replacing, run=run, exists=lambda p: False) == []
    assert len(calls) == 1

    def fails(args: Sequence[str], **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "Access is denied.\n")

    with pytest.raises(shortcuts.ShortcutError, match="Access is denied"):
        shortcuts.refresh(links, replacing, run=fails, exists=lambda p: True)


def test_refresh_renames_our_shortcuts_from_before_kitshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """facegen.lnk (before Phase 17c) beside a link: looked at when only it is there, made
    into Kitshot.lnk when it starts Start.bat, pythonw.exe or the app, then deleted."""
    links = shortcuts.app_shortcuts(FOLDERS, program=APP, root=ROOT, icon=ICON)
    old = FOLDERS["programs"] + r"\facegen.lnk"
    assert links[1].path == FOLDERS["programs"] + r"\Kitshot.lnk"
    monkeypatch.setattr(procs, "windows_tool", lambda *a, **k: "powershell.exe")
    calls: list[Sequence[str]] = []

    def run(args: Sequence[str], **kw: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        out = f"{links[1].path}\r\n{links[1].path}\r\n"  # said twice: reported once
        return subprocess.CompletedProcess(args, 0, out, "")

    done = shortcuts.refresh(links, [START_BAT, PYW], run=run, exists=lambda p: p == old)
    assert done == [links[1].path]
    script = decoded(calls[0])
    assert f"if (Test-Path -LiteralPath '{old}') {{" in script
    assert f"$o = $shell.CreateShortcut('{old}')" in script
    assert f"if (@('{START_BAT}', '{PYW}', '{APP}') -icontains $o.TargetPath) {{" in script
    assert f"if (-not (Test-Path -LiteralPath '{links[1].path}')) {{" in script
    assert f"Remove-Item -LiteralPath '{old}' -Force" in script
    # The new one is written in full, before the old one goes.
    made = script.index(f"$s = $shell.CreateShortcut('{links[1].path}')")
    assert made < script.index("$s.Save()", made) < script.index("Remove-Item")


def test_refresh_gives_the_apps_shortcuts_the_new_icon() -> None:
    """A shortcut that starts the app already gets the current icon and description only:
    its arguments stay as they are."""
    link = shortcuts.startup_shortcut(STARTUP, program=APP, root=ROOT, icon=ICON)
    script = shortcuts.refresh_script([link], [START_BAT, PYW])
    branch = script[script.index("} elseif") :]
    assert branch.startswith(
        f"}} elseif (($s.TargetPath -ieq '{APP}') -and (($s.IconLocation -ine '{ICON},0')"
        f" -or ($s.Description -cne {shortcuts.ps_string(shortcuts.DESCRIPTION)}))) {{"
    )
    branch = branch[: branch.index("\n  }")]
    assert f"$s.IconLocation = '{ICON},0'" in branch and "$s.Save()" in branch
    assert "Arguments" not in branch and "TargetPath =" not in branch


def test_the_refreshed_startup_shortcut_stays_minimised() -> None:
    link = shortcuts.startup_shortcut(STARTUP, program=APP, root=ROOT, icon=ICON)
    script = shortcuts.refresh_script([link], [PYW])
    assert "$s.Arguments = '-m kitshot.desktop app --minimized'" in script
    assert f"$s.WindowStyle = {shortcuts.WINDOW_MINIMIZED}" in script


def test_paths_with_quotes_stay_whole_in_the_refresh() -> None:
    link = shortcuts.Shortcut(r"C:\Users\O'Neil\Desktop\facegen.lnk", PYW, "-m x", ROOT, ICON)
    script = shortcuts.refresh_script([link], [r"C:\Users\O'Neil\kitshot\Start.bat"])
    assert r"'C:\Users\O''Neil\Desktop\facegen.lnk'" in script
    assert r"@('C:\Users\O''Neil\kitshot\Start.bat') -icontains" in script


@pytest.mark.skipif(os.name == "nt", reason="on Windows it would change this PC's shortcuts")
def test_launch_leaves_shortcuts_alone_off_windows(tmp_path: Path) -> None:
    out = io.StringIO()
    launch.refresh_shortcuts(tmp_path, None, out)
    assert out.getvalue() == ""


# --- removing the picture pack -------------------------------------------------------------


def test_remove_pack_deletes_only_kitshots_graphics(config_file: Path) -> None:
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


def test_remove_pack_refuses_a_redirected_graphics_folder(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    fm = settings.paths.fm_user_dir
    assert fm is not None
    elsewhere = config_file.parent / "other-disk" / "graphics"
    (elsewhere / "kitshot").mkdir(parents=True)
    (elsewhere / "kitshot" / "1.png").write_bytes(b"png")
    fm.mkdir(parents=True, exist_ok=True)
    try:
        (fm / "graphics").symlink_to(elsewhere, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows does not grant this process symlink creation privileges")
        raise
    out = io.StringIO()
    assert main(["--config", str(config_file), "remove-pack", "--yes"], out=out) == 1
    assert "Not removed: refusing to delete outside" in out.getvalue()
    assert (elsewhere / "kitshot" / "1.png").is_file()


def test_remove_pack_asks_first(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(defaults_path=config_file)
    assert settings.paths.fm_user_dir is not None
    root = packs_root(settings.paths.fm_user_dir)
    root.mkdir(parents=True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    out = io.StringIO()
    assert main(["--config", str(config_file), "remove-pack"], out=out) == 1
    assert root.is_dir() and "Nothing was removed" in out.getvalue()
