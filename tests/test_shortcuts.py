"""Windows shortcuts to the desktop app (Phase 11): what they run, the PowerShell that writes
them, and Start with Windows."""

from __future__ import annotations

import base64
import io
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from facegen import procs, shortcuts
from facegen.cli import main

FOLDERS = {
    "desktop": r"C:\Users\me\OneDrive\바탕 화면",
    "programs": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs",
    "startup": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup",
}
PYW = r"C:\facegen\.venv\Scripts\pythonw.exe"
ROOT = r"C:\facegen"
ICON = r"C:\facegen\data\app\facegen.ico"


def test_the_app_command_line() -> None:
    assert shortcuts.app_arguments() == "-m facegen.desktop app"
    assert shortcuts.app_arguments(minimized=True) == "-m facegen.desktop app --minimized"
    assert shortcuts.app_arguments(config=r"C:\my configs\a.toml") == (
        r'-m facegen.desktop --config "C:\my configs\a.toml" app'
    )
    assert shortcuts.pythonw_beside(r"C:\facegen\.venv\Scripts\python.exe") == PYW


def test_desktop_and_start_menu_shortcuts() -> None:
    links = shortcuts.app_shortcuts(FOLDERS, pythonw=PYW, root=ROOT, icon=ICON)
    assert [link.path for link in links] == [
        r"C:\Users\me\OneDrive\바탕 화면\facegen.lnk",
        r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\facegen.lnk",
    ]
    assert all(link.target == PYW and link.working_dir == ROOT for link in links)
    assert all(link.arguments == "-m facegen.desktop app" and not link.minimized for link in links)
    only = shortcuts.app_shortcuts(
        {"programs": FOLDERS["programs"]}, pythonw=PYW, root=ROOT, icon=ICON
    )
    assert len(only) == 1  # a folder Windows did not name is left out


def test_the_startup_shortcut_starts_in_the_tray() -> None:
    link = shortcuts.startup_shortcut(FOLDERS["startup"], pythonw=PYW, root=ROOT, icon=ICON)
    assert link.path.endswith(r"Startup\facegen.lnk")
    assert link.arguments == "-m facegen.desktop app --minimized" and link.minimized


def test_the_powershell_quotes_every_path_as_it_is() -> None:
    odd = shortcuts.Shortcut(
        r"C:\Users\O'Brien\Desktop\facegen.lnk", PYW, "-m facegen.desktop app", ROOT, ICON
    )
    script = shortcuts.powershell_script(
        [odd, shortcuts.startup_shortcut(FOLDERS["startup"], pythonw=PYW, root=ROOT, icon=ICON)]
    )
    assert "$s = $shell.CreateShortcut('C:\\Users\\O''Brien\\Desktop\\facegen.lnk')" in script
    assert f"$s.IconLocation = '{ICON},0'" in script
    assert script.count("$s.Save()") == 2
    assert script.count("$s.WindowStyle = 7") == 1  # the Startup one only
    assert script.startswith("$ErrorActionPreference = 'Stop'")
    command = shortcuts.powershell_command("powershell.exe", script)
    assert command[:-1] == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
    ]
    assert base64.b64decode(command[-1]).decode("utf-16-le") == script


def test_windows_folders_ask_windows_then_fall_back_to_the_profile() -> None:
    env = {"USERPROFILE": r"C:\Users\me", "APPDATA": r"C:\Users\me\AppData\Roaming"}
    asked = shortcuts.windows_folders(env, ask=lambda fid: None)
    assert asked == {
        "desktop": r"C:\Users\me\Desktop",
        "programs": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs",
        "startup": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup",
    }
    known = {shortcuts.FOLDER_IDS["desktop"]: FOLDERS["desktop"]}
    assert shortcuts.windows_folders(env, ask=known.get)["desktop"] == FOLDERS["desktop"]

    def broken(fid: str) -> str | None:
        raise OSError("no shell32")

    assert shortcuts.windows_folders({}, ask=broken) == {}


def test_writing_reports_powershell_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    link = shortcuts.startup_shortcut(FOLDERS["startup"], pythonw=PYW, root=ROOT, icon=ICON)
    monkeypatch.setattr(procs, "windows_tool", lambda *a, **k: "powershell.exe")
    calls: list[Sequence[str]] = []

    def fails(args: Sequence[str], **kw: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert kw["timeout"] == shortcuts.POWERSHELL_SECONDS
        return subprocess.CompletedProcess(args, 1, "", "Access is denied.\n")

    with pytest.raises(shortcuts.ShortcutError, match="Access is denied"):
        shortcuts.write([link], run=fails)
    assert calls and calls[0][0] == "powershell.exe"
    shortcuts.write([], run=fails)  # nothing to write: nothing run
    assert len(calls) == 1
    monkeypatch.setattr(procs, "windows_tool", lambda *a, **k: None)
    with pytest.raises(shortcuts.ShortcutError, match="PowerShell was not found"):
        shortcuts.write([link], run=fails)


def test_start_with_windows_adds_and_removes_only_our_shortcut(tmp_path: Path) -> None:
    other = tmp_path / "OneDrive.lnk"
    other.write_text("someone else's", encoding="utf-8")
    written: list[shortcuts.Shortcut] = []

    def writer(links: Sequence[shortcuts.Shortcut]) -> None:
        written.extend(links)
        for link in links:
            Path(link.path).write_text("lnk", encoding="utf-8")

    app = shortcuts.AppPaths(ROOT, PYW, ICON)
    switch = shortcuts.StartupShortcut(str(tmp_path), app, writer=writer)
    assert switch.path == tmp_path / "facegen.lnk" and not switch.is_on()
    switch.set(True)
    assert switch.is_on() and written[0].minimized
    switch.set(False)
    assert not switch.is_on() and other.is_file()
    switch.set(False)  # off already: nothing to do


def test_the_shortcuts_command_is_for_windows(config_file: Path) -> None:
    if os.name == "nt":
        pytest.skip("this checks the message elsewhere")
    out = io.StringIO()
    assert main(["--config", str(config_file), "shortcuts"], out=out) == 2
    assert "Shortcuts are for Windows" in out.getvalue()


def test_this_app_writes_the_icon_once(tmp_path: Path) -> None:
    app = shortcuts.this_app(tmp_path)
    icon = Path(app.icon)
    assert icon == tmp_path / "app" / "facegen.ico" and icon.is_file()
    assert app.pythonw.endswith("pythonw.exe") and app.config is None
    stamp = icon.stat().st_mtime_ns
    shortcuts.this_app(tmp_path)
    assert icon.stat().st_mtime_ns == stamp
