"""facegen-app.exe in a venv's Scripts folder: when it is made, kept or given up
(desktop/app_exe.py). The Win32 resource step itself runs in opt_in/test_windows_app_exe.py."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from support.pe import IMAGE_SIZE, synthetic_pe

from facegen.desktop import app_exe
from facegen.desktop.app_exe import APP_EXE, STAMP, AppExeError


class Maker:
    """Stands in for `make`: writes the copy, or fails as told."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, bytes, bytes, str]] = []
        self.fail: Exception | None = None

    def __call__(self, target: Path, launcher: bytes, ico: bytes, version: str) -> None:
        self.calls.append((target, launcher, ico, version))
        if self.fail is not None:
            raise self.fail
        target.write_bytes(b"copy of " + launcher)


@pytest.fixture
def venv(tmp_path: Path) -> Path:
    (tmp_path / "Scripts").mkdir()
    (tmp_path / "pyvenv.cfg").write_text("home = C:\\Python312\n", encoding="utf-8")
    (tmp_path / "Scripts" / "pythonw.exe").write_bytes(b"launcher")
    (tmp_path / "facegen.ico").write_bytes(b"icon")
    return tmp_path


def ensure(venv: Path, maker: Maker, version: str = "0.1.2", **kw: object) -> Path | None:
    kw.setdefault("os_name", "nt")
    return app_exe.ensure(venv / "Scripts", venv / "facegen.ico", version, maker=maker, **kw)  # type: ignore[arg-type]


def leftovers(venv: Path) -> list[str]:
    return sorted(p.name for p in (venv / "Scripts").iterdir() if p.name.startswith("."))


def test_the_copy_is_made_once_and_stamped(venv: Path) -> None:
    maker = Maker()
    exe = ensure(venv, maker)
    assert exe == venv / "Scripts" / APP_EXE and exe.read_bytes() == b"copy of launcher"
    [(part, launcher, ico, version)] = maker.calls
    assert part.parent == venv / "Scripts" and part.suffix == ".exe"
    assert (launcher, ico, version) == (b"launcher", b"icon", "0.1.2")
    assert (venv / "Scripts" / STAMP).read_text(encoding="ascii").strip() == (
        app_exe.stamp_key(b"launcher", b"icon", "0.1.2")
    )
    assert leftovers(venv) == []
    assert ensure(venv, maker) == exe and len(maker.calls) == 1  # up to date: kept


@pytest.mark.parametrize("change", ["launcher", "icon", "version", "stamp", "copy"])
def test_the_copy_is_made_again_when_what_it_came_from_changed(venv: Path, change: str) -> None:
    maker = Maker()
    ensure(venv, maker)
    version = "0.1.2"
    if change == "launcher":
        (venv / "Scripts" / "pythonw.exe").write_bytes(b"new launcher")
    elif change == "icon":
        (venv / "facegen.ico").write_bytes(b"new icon")
    elif change == "version":
        version = "0.1.3"
    elif change == "stamp":
        (venv / "Scripts" / STAMP).unlink()
    else:
        (venv / "Scripts" / APP_EXE).unlink()
    assert ensure(venv, maker, version) == venv / "Scripts" / APP_EXE
    assert len(maker.calls) == 2


def test_nothing_is_made_off_windows_or_outside_a_venv(venv: Path) -> None:
    maker = Maker()
    assert ensure(venv, maker, os_name="posix") is None
    (venv / "pyvenv.cfg").unlink()  # the base interpreter's own folder, say
    assert ensure(venv, maker) is None
    (venv / "pyvenv.cfg").write_text("home = C:\\Python312\n", encoding="utf-8")
    (venv / "Scripts" / "pythonw.exe").unlink()
    assert ensure(venv, maker) is None
    assert maker.calls == [] and not (venv / "Scripts" / APP_EXE).exists()


def test_a_copy_that_cannot_be_made_leaves_pythonw(
    venv: Path, caplog: pytest.LogCaptureFixture
) -> None:
    maker = Maker()
    maker.fail = AppExeError("facegen-app.exe did not start Python")
    with caplog.at_level(logging.WARNING, logger="facegen"):
        assert ensure(venv, maker) is None
    assert "did not start Python" in caplog.text and leftovers(venv) == []
    assert not (venv / "Scripts" / STAMP).exists()


def test_a_failed_remake_keeps_the_working_copy(venv: Path) -> None:
    maker = Maker()
    exe = ensure(venv, maker)
    maker.fail = OSError("disk full")
    assert ensure(venv, maker, "0.1.3") == exe and exe is not None and exe.is_file()
    assert leftovers(venv) == []


def test_a_copy_in_use_is_kept_until_a_later_start(
    venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maker = Maker()
    exe = ensure(venv, maker)
    old_stamp = (venv / "Scripts" / STAMP).read_text(encoding="ascii")
    replace = os.replace

    def in_use(src: object, dst: object) -> None:
        if Path(str(dst)).name == APP_EXE:
            raise PermissionError(13, "The process cannot access the file")
        replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", in_use)
    assert ensure(venv, maker, "0.1.3") == exe
    assert (venv / "Scripts" / STAMP).read_text(encoding="ascii") == old_stamp
    assert leftovers(venv) == []


def test_a_copy_already_running_is_not_made_again_first(
    venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maker = Maker()
    exe = ensure(venv, maker)
    monkeypatch.setattr(app_exe, "in_use", lambda path: path == exe)
    assert ensure(venv, maker, "0.1.3") == exe
    assert len(maker.calls) == 1  # no copy made only to find it cannot replace this one


def test_a_file_nobody_holds_is_not_in_use(tmp_path: Path) -> None:
    free = tmp_path / "facegen-app.exe"
    free.write_bytes(b"copy")
    assert not app_exe.in_use(free)
    assert not app_exe.in_use(tmp_path / "missing.exe")


def test_make_appends_what_followed_the_sections_and_checks_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed: list[tuple[bytes, bytes, str]] = []

    def replace_resources(path: Path, ico: bytes, version: str) -> None:
        changed.append((path.read_bytes(), ico, version))

    monkeypatch.setattr(app_exe, "replace_resources", replace_resources)
    target = tmp_path / ".facegen-app-1.exe"
    launcher = synthetic_pe(extra=b"metadata", cert=b"S" * 16)
    app_exe.make(target, launcher, b"icon", "0.1.2", starts=lambda exe: exe == target)
    [(image, ico, version)] = changed
    assert len(image) == IMAGE_SIZE and (ico, version) == (b"icon", "0.1.2")
    assert target.read_bytes() == image + b"metadata"

    with pytest.raises(AppExeError, match="did not start Python"):
        app_exe.make(target, launcher, b"icon", "0.1.2", starts=lambda exe: False)
    assert not target.exists()
