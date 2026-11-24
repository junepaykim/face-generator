"""Phase 17o: an installed 0.1.x (package `facegen`) updated in place to this source.

The update runs the installed version's code. `update.py`, `release_payload.py` and
`install_transaction.py` did not change between 0.1.6 and the rename except for names, so the
current updater stands in for 0.1.x's here (research.md records the same run with 0.1.6's own
code). The release is this repository's payload, taken from git's list of files.
"""

from __future__ import annotations

import ast
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from kitshot.desktop import shell
from kitshot.desktop.app_exe import APP_EXE, LEGACY_APP_EXE
from kitshot.setup import update
from kitshot.setup.update import Updater

ROOT = Path(__file__).resolve().parents[3]
COMPAT = {
    "src/facegen/__init__.py",
    "src/facegen/cli.py",
    "src/facegen/desktop/__init__.py",
    "src/facegen/desktop/__main__.py",
}
"""Everything `src/facegen/` holds after the rename: entry points older files start."""


def _listed() -> list[str]:
    """This repository's files as a release would take them: tracked and new, not ignored."""
    try:
        listed = subprocess.run(
            ["git", "--git-dir", str(ROOT / ".git"), "--work-tree", str(ROOT), "ls-files",
             "--cached", "--others", "--exclude-standard"],
            cwd=ROOT.parent,  # Windows Git cannot use a WSL UNC path as its working directory.
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available here")
    return [p for p in listed if (ROOT / p).is_file()]


def _release_zip(path: Path, version: str) -> Path:
    """This source as GitHub's archive of tag v<version>: one top folder."""
    with zipfile.ZipFile(path, "w") as zf:
        for rel in _listed():
            if not update.in_payload(PurePosixPath(rel)):
                continue
            data = (ROOT / rel).read_bytes()
            if rel == "pyproject.toml":
                current = update.project_version(data.decode("utf-8"))
                data = data.replace(
                    f'version = "{current}"'.encode(), f'version = "{version}"'.encode()
                )
            zf.writestr(f"kitshot-{version}/{rel}", data)
    return path


def _old_install(root: Path) -> Path:
    """An installed 0.1.5 as its files were before the rename (Phase 17o)."""
    files = {
        "pyproject.toml": '[project]\nname = "facegen"\nversion = "0.1.5"\n'
        '[project.scripts]\nfacegen = "facegen.cli:main"\n',
        "Start.bat": '@echo off\r\n"%PY%" -m facegen.cli launch --log "%LOG%"\r\n',
        "uv.lock": 'name = "facegen"\n',
        "src/facegen/__init__.py": '"""FM27 Face Generator."""\n',
        "src/facegen/cli.py": "def main(): return 0\n",
        "src/facegen/db.py": "OLD = True\n",
        "src/facegen/desktop/__init__.py": "",
        "src/facegen/desktop/__main__.py": "from facegen.desktop.shell import main\n",
        "src/facegen/desktop/shell.py": "def main(): return 0\n",
        "src/facegen/web/templates/base.html": "<html>\n",
        "src/facegen/data/icon/kitshot-16.png": "png",
        "native/Cargo.toml": '[package]\nname = "facegen-native"\n',
        "native/facegen_native.pyi": "",
        "native/wheels.sha256": "",
        "config/default.toml": "[ui]\n",
        "config/local.toml": "[paths]\nfm_user_dir = 'C:\\\\FM'\n",
        "data/facegen.db": "the database",
        "data/app/facegen.ico": "an icon a pin may use",
        "data/logs/facegen.log": "old log\n",
        ".venv/Scripts/python.exe": "python",
        f".venv/Scripts/{LEGACY_APP_EXE}": "the old app program",
        ".bin/uv.exe": "uv",
        "facegen.ico": "installer icon",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return root


def _files(root: Path, under: str) -> set[str]:
    base = root / under
    return {
        p.relative_to(root).as_posix()
        for p in base.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


@pytest.fixture
def updated(tmp_path: Path) -> Path:
    root = _old_install(tmp_path / "facegen")
    archive = _release_zip(tmp_path / "release.zip", "0.1.6")

    def fetch_archive(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, dest)

    out = io.StringIO()
    steps = Updater(
        root=root,
        fetch=lambda url: [
            {
                "tag_name": "v0.1.6",
                "html_url": "https://github.com/junepaykim/kitshot/releases/tag/v0.1.6",
            }
        ],
        fetch_archive=fetch_archive,
        running=lambda data: False,
        sleep=lambda s: None,
        now=lambda: 5000.0,
    )
    code = update.run_update(out, root / "data", steps=steps)
    assert code == update.UPDATED, out.getvalue()
    assert "Updated Kitshot 0.1.5 to 0.1.6" in out.getvalue()
    return root


def test_an_update_from_0_1_x_replaces_the_package_and_keeps_this_pcs_files(
    updated: Path,
) -> None:
    root = updated
    assert update.installed_version(root) == "0.1.6"
    # The old package is gone but for the entry points older files still start.
    assert _files(root, "src/facegen") == COMPAT
    assert (root / "src/kitshot/cli.py").is_file()
    assert (root / "src/kitshot/data/icon/kitshot-16.png").is_file()
    assert not (root / "native/facegen_native.pyi").exists()
    assert (root / "native/kitshot_native.pyi").is_file()
    assert 'name = "kitshot"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    start = (root / "Start.bat").read_text(encoding="utf-8")
    assert "-m kitshot.cli launch" in start and "facegen" not in start.lower()
    # This PC's own: the folder keeps its name, data, settings, venv and the old program.
    assert (root / "data/facegen.db").read_text(encoding="utf-8") == "the database"
    assert (root / "data/logs/facegen.log").is_file()
    assert (root / "config/local.toml").read_text(encoding="utf-8").startswith("[paths]")
    assert (root / f".venv/Scripts/{LEGACY_APP_EXE}").is_file()
    assert (root / "data/app/facegen.ico").is_file()
    # The replaced files are kept, the old package among them.
    backup = root / "data/update/backup-0.1.5"
    assert (backup / "src/facegen/db.py").read_text(encoding="utf-8") == "OLD = True\n"


def _python(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """This venv's Python with the updated folder's src/ first, as its editable .pth would
    give it (the .pth names src/, so `kitshot` imports before uv sync runs again)."""
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_after_the_update_old_entry_points_reach_kitshot(updated: Path) -> None:
    """The 0.1.x uninstaller runs `-m facegen.cli remove-pack --yes`; a taskbar pin runs
    `facegen-app.exe -m facegen.desktop app`; the new Start.bat runs `-m kitshot.cli`."""
    root = updated
    where = _python(
        root, "-c", "import kitshot, facegen; print(kitshot.__file__, facegen.__file__)"
    )
    assert where.returncode == 0, where.stderr
    assert all(Path(p).is_relative_to(root / "src") for p in where.stdout.split())
    for module, usage in (
        ("facegen.cli", "usage: kitshot"),
        ("kitshot.cli", "usage: kitshot"),
        ("facegen.desktop", "usage: python -m kitshot.desktop"),
    ):
        done = _python(root, "-m", module, "--help")
        assert done.returncode == 0, (module, done.stderr)
        assert done.stdout.startswith(usage), (module, done.stdout[:200])
    for command in ("remove-pack", "launch", "update"):
        done = _python(root, "-m", "facegen.cli", command, "--help")
        assert done.returncode == 0 and f"kitshot {command}" in done.stdout, done.stderr
    # --help creates no runtime state.
    assert not (root / "data/kitshot.db").exists() and not (root / "data/logs/kitshot.log").exists()


def test_the_compat_package_is_only_entry_points() -> None:
    """src/facegen/ holds the forwarding entry points and nothing imports it from kitshot."""
    assert _files(ROOT, "src/facegen") == COMPAT
    for rel in COMPAT:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module is not None and node.module.startswith("kitshot"), rel
            elif isinstance(node, ast.Import):
                pytest.fail(f"{rel} imports {[a.name for a in node.names]}")
    for path in (ROOT / "src/kitshot").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(n == "facegen" or n.startswith("facegen.") for n in names), path


def test_the_app_holds_both_mutexes_and_the_installer_waits_for_either() -> None:
    assert shell.APP_MUTEXES == ("kitshot-app", "facegen-app")
    iss = (ROOT / "installer" / "kitshot.iss").read_text(encoding="utf-8-sig")
    assert "AppMutex=kitshot-app,facegen-app" in iss
    # A new install goes to Kitshot; an upgrade keeps the folder it is in.
    assert r"DefaultDirName={localappdata}\Kitshot" in iss and "UsePreviousAppDir=yes" in iss
    assert "-m kitshot.cli remove-pack --yes" in iss
    assert APP_EXE == "kitshot-app.exe" and LEGACY_APP_EXE == "facegen-app.exe"


def test_launch_points_shortcuts_to_the_old_program_at_the_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shortcut an older version wrote (facegen-app.exe -m facegen.desktop app, Startup
    included) is among those `refresh` points at the app."""
    from kitshot.desktop import shortcuts
    from kitshot.setup import launch

    scripts = r"C:\Users\x\AppData\Local\facegen\.venv\Scripts"
    monkeypatch.setattr(sys, "executable", scripts + r"\python.exe")
    folders = {"desktop": r"C:\D", "programs": r"C:\P", "startup": r"C:\S"}
    app = shortcuts.AppPaths(
        r"C:\Users\x\AppData\Local\facegen", scripts + "\\" + APP_EXE, r"C:\i.ico"
    )
    seen: dict[str, Any] = {}

    def refresh(links: Any, replacing: Any) -> list[str]:
        seen["links"], seen["replacing"] = list(links), list(replacing)
        return []

    monkeypatch.setattr(shortcuts, "windows_folders", lambda: folders)
    monkeypatch.setattr(shortcuts, "this_app", lambda data_dir, config: app)
    monkeypatch.setattr(shortcuts, "refresh", refresh)
    launch.refresh_shortcuts(tmp_path, None, io.StringIO(), os_name="nt")
    assert scripts + "\\" + LEGACY_APP_EXE in seen["replacing"]
    startup = [s for s in seen["links"] if s.path.startswith(folders["startup"])]
    assert startup and startup[0].arguments == "-m kitshot.desktop app --minimized"
    script = shortcuts.refresh_script(seen["links"], seen["replacing"])
    assert LEGACY_APP_EXE in script and "-m kitshot.desktop app" in script
