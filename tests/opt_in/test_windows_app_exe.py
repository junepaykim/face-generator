"""kitshot-app.exe made from this venv's real pythonw.exe launcher with the Win32 resource
functions: Windows reads Kitshot's name and icon from it, and it starts Python in the venv."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kitshot.desktop import app_exe
from kitshot.desktop.app_icon import ICO_SIZES, write_ico

pytestmark = pytest.mark.windows_only


def file_description(path: Path) -> str:
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version")  # type: ignore[attr-defined]
    version.GetFileVersionInfoSizeW.argtypes = (wintypes.LPCWSTR, wintypes.LPVOID)
    version.GetFileVersionInfoW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
    )
    version.VerQueryValueW.argtypes = (
        wintypes.LPCVOID,
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_wchar_p),
        ctypes.POINTER(wintypes.UINT),
    )
    size = version.GetFileVersionInfoSizeW(str(path), None)
    assert size
    block = ctypes.create_string_buffer(size)
    assert version.GetFileVersionInfoW(str(path), 0, size, block)
    text, length = ctypes.c_wchar_p(), wintypes.UINT()
    key = r"\StringFileInfo\040904B0\FileDescription"
    assert version.VerQueryValueW(block, key, ctypes.byref(text), ctypes.byref(length))
    return text.value or ""


@pytest.mark.skipif(os.name != "nt", reason="Win32 resources of a real venv launcher")
def test_the_launcher_copy_says_kitshot_and_starts_python(tmp_path: Path) -> None:
    scripts = Path(sys.executable).parent
    if not (scripts.parent / "pyvenv.cfg").is_file() or not (scripts / "pythonw.exe").is_file():
        pytest.skip("not run from a venv with a pythonw.exe launcher")
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    shutil.copy(scripts.parent / "pyvenv.cfg", venv / "pyvenv.cfg")
    shutil.copy(scripts / "pythonw.exe", venv / "Scripts" / "pythonw.exe")
    icon = write_ico(tmp_path / "app")

    exe = app_exe.ensure(venv / "Scripts", icon, "1.2.3")
    assert exe == venv / "Scripts" / app_exe.APP_EXE
    assert file_description(exe) == "Kitshot"
    kernel32 = app_exe._kernel32()
    found = app_exe._resources(kernel32, exe, (app_exe.RT_ICON, app_exe.RT_GROUP_ICON))
    assert sorted(name for kind, name, _ in found if kind == app_exe.RT_ICON) == list(
        range(1, len(ICO_SIZES) + 1)
    )
    assert [name for kind, name, _ in found if kind == app_exe.RT_GROUP_ICON] == [1]
    done = subprocess.run(
        [str(exe), "-c", "import sys; print(sys.executable); print(sys.prefix)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    executable, prefix = done.stdout.split("\n")[:2]
    assert Path(executable) == exe and Path(prefix) == venv
    assert app_exe.ensure(venv / "Scripts", icon, "1.2.3") == exe  # up to date: kept
