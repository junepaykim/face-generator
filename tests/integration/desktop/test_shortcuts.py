"""Windows shortcuts to the desktop app (Phase 11): what they run, the PowerShell that writes
them, and Start with Windows."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import kitshot
from kitshot.cli import main
from kitshot.desktop import app_exe, app_icon, shortcuts
from kitshot.platform import procs

FOLDERS = {
    "desktop": r"C:\Users\me\OneDrive\바탕 화면",
    "programs": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs",
    "startup": r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup",
}
PYW = r"C:\kitshot\.venv\Scripts\pythonw.exe"
ROOT = r"C:\kitshot"
ICON = r"C:\kitshot\data\app\kitshot-0123456789.ico"


def test_the_app_command_line() -> None:
    assert shortcuts.app_arguments() == "-m kitshot.desktop app"
    assert shortcuts.app_arguments(minimized=True) == "-m kitshot.desktop app --minimized"
    assert shortcuts.app_arguments(config=r"C:\my configs\a.toml") == (
        r'-m kitshot.desktop --config "C:\my configs\a.toml" app'
    )
    assert shortcuts.pythonw_beside(r"C:\kitshot\.venv\Scripts\python.exe") == PYW


def test_desktop_and_start_menu_shortcuts() -> None:
    links = shortcuts.app_shortcuts(FOLDERS, program=PYW, root=ROOT, icon=ICON)
    assert [link.path for link in links] == [
        r"C:\Users\me\OneDrive\바탕 화면\Kitshot.lnk",
        r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Kitshot.lnk",
    ]
    assert all(link.target == PYW and link.working_dir == ROOT for link in links)
    assert all(link.arguments == "-m kitshot.desktop app" and not link.minimized for link in links)
    assert all(link.description.startswith("Kitshot: ") for link in links)
    only = shortcuts.app_shortcuts(
        {"programs": FOLDERS["programs"]}, program=PYW, root=ROOT, icon=ICON
    )
    assert len(only) == 1  # a folder Windows did not name is left out


def test_the_startup_shortcut_starts_in_the_tray() -> None:
    link = shortcuts.startup_shortcut(FOLDERS["startup"], program=PYW, root=ROOT, icon=ICON)
    assert link.path.endswith(r"Startup\Kitshot.lnk")
    assert link.arguments == "-m kitshot.desktop app --minimized" and link.minimized


def test_the_powershell_quotes_every_path_as_it_is() -> None:
    odd = shortcuts.Shortcut(
        r"C:\Users\O'Brien\Desktop\facegen.lnk", PYW, "-m kitshot.desktop app", ROOT, ICON
    )
    script = shortcuts.powershell_script(
        [odd, shortcuts.startup_shortcut(FOLDERS["startup"], program=PYW, root=ROOT, icon=ICON)]
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
    link = shortcuts.startup_shortcut(FOLDERS["startup"], program=PYW, root=ROOT, icon=ICON)
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
    assert switch.path == tmp_path / "Kitshot.lnk" and not switch.is_on()
    switch.set(True)
    assert switch.is_on() and written[0].minimized
    switch.set(False)
    assert not switch.is_on() and other.is_file()
    switch.set(False)  # off already: nothing to do


def test_start_with_windows_knows_the_name_before_kitshot(tmp_path: Path) -> None:
    """An earlier version's facegen.lnk counts as on; turning it on again writes Kitshot.lnk
    in its place, and turning it off removes either."""
    old = tmp_path / "facegen.lnk"
    old.write_text("lnk", encoding="utf-8")

    def writer(links: Sequence[shortcuts.Shortcut]) -> None:
        for link in links:
            Path(link.path).write_text("lnk", encoding="utf-8")

    switch = shortcuts.StartupShortcut(
        str(tmp_path), shortcuts.AppPaths(ROOT, PYW, ICON), writer=writer
    )
    assert switch.legacy == [old] and switch.is_on()
    switch.set(True)
    assert switch.path.is_file() and not old.exists()
    old.write_text("lnk", encoding="utf-8")
    switch.set(False)
    assert not switch.path.exists() and not old.exists() and not switch.is_on()


def test_the_earlier_name_is_beside_the_shortcut() -> None:
    link = shortcuts.startup_shortcut(FOLDERS["startup"], program=PYW, root=ROOT, icon=ICON)
    assert shortcuts.legacy_paths(link.path) == [FOLDERS["startup"] + r"\facegen.lnk"]
    assert shortcuts.legacy_paths(r"C:\x\FACEGEN.lnk") == []  # it is the earlier name itself
    # Not a drive path, so it is this OS's own: Windows writes it with backslashes.
    assert shortcuts.legacy_paths("/tmp/links/Kitshot.lnk") == [str(Path("/tmp/links/facegen.lnk"))]


def test_the_shortcuts_command_is_for_windows(config_file: Path) -> None:
    if os.name == "nt":
        pytest.skip("this checks the message elsewhere")
    out = io.StringIO()
    assert main(["--config", str(config_file), "shortcuts"], out=out) == 2
    assert "Shortcuts are for Windows" in out.getvalue()


def test_this_app_writes_the_icon_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[Path, Path, str]] = []

    def no_copy(scripts: Path, icon: Path, version: str) -> Path | None:
        asked.append((scripts, icon, version))
        return None  # as off Windows, or when it could not be made

    monkeypatch.setattr(app_exe, "ensure", no_copy)  # not this venv's Scripts folder
    app = shortcuts.this_app(tmp_path)
    icon = Path(app.icon)
    assert icon == tmp_path / "app" / app_icon.ico_name() and icon.is_file()
    assert icon.name.startswith("kitshot-")
    assert app.program.endswith("pythonw.exe") and app.config is None
    assert asked == [(Path(sys.executable).parent, icon, kitshot.__version__)]
    stamp = icon.stat().st_mtime_ns
    shortcuts.this_app(tmp_path)
    assert icon.stat().st_mtime_ns == stamp


def test_the_shortcuts_run_the_copy_once_it_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "Scripts" / app_exe.APP_EXE
    monkeypatch.setattr(app_exe, "ensure", lambda scripts, icon, version: copy)
    app = shortcuts.this_app(tmp_path)
    assert app.program == str(copy)
    [link] = shortcuts.app_shortcuts(
        {"programs": FOLDERS["programs"]}, program=app.program, root=ROOT, icon=app.icon
    )
    assert link.target == str(copy) and link.arguments == "-m kitshot.desktop app"


def test_a_changed_icon_reaches_the_shortcuts_and_the_app_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new icon is a new .ico file: the shortcuts and the window are given its path, and
    kitshot-app.exe is made again from it (its stamp holds the .ico's bytes)."""
    from PIL import Image

    asked: list[Path] = []
    monkeypatch.setattr(app_exe, "ensure", lambda scripts, icon, version: asked.append(icon))
    before = Path(shortcuts.this_app(tmp_path).icon)
    original = app_icon.bundled_png

    def recoloured(size: int) -> bytes:
        with Image.open(io.BytesIO(original(size))) as img:
            out = io.BytesIO()
            img.convert("RGBA").rotate(90).save(out, format="PNG")
            return out.getvalue()

    monkeypatch.setattr(app_icon, "bundled_png", recoloured)
    app_icon._decoded.cache_clear()
    app_icon.icon_key.cache_clear()
    try:
        after = Path(shortcuts.this_app(tmp_path).icon)
    finally:
        app_icon._decoded.cache_clear()
        app_icon.icon_key.cache_clear()
    assert after != before and after.is_file() and asked == [before, after]
    key = app_exe.stamp_key(b"launcher", before.read_bytes(), "0.1.7")
    assert key != app_exe.stamp_key(b"launcher", after.read_bytes(), "0.1.7")
