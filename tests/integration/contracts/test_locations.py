# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from facegen import locations
from facegen.datasource import kits
from facegen.desktop import shell as desktop
from facegen.persistence import settings
from facegen.platform import procs
from facegen.setup import native_wheel


def test_resource_loaders_ignore_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "custom.toml"
    config.write_text(settings.DEFAULTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "local.toml").write_text('[paths]\ndata_dir = "relative-data"\n')
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)

    loaded = settings.load_settings(defaults_path=config)
    assert loaded.paths.data_dir == locations.PROJECT_ROOT / "relative-data"
    assert settings.local_path(config) == tmp_path / "local.toml"
    assert desktop.FALLBACK_LOG == locations.PROJECT_ROOT / "data" / "logs" / "facegen.log"
    table = kits.load_kits(data_dir=tmp_path)
    assert table.kit_for(636, goalkeeper=False)[1] == "table"
    assert table.kit_for(679, goalkeeper=False)[1] == "bundled"
    planned = native_wheel.plan(native_wheel.Host("win32", "AMD64", "cpython", (3, 12)))
    assert planned.wheel is not None and planned.sha256 is not None


def test_desktop_fallback_paths_survive_without_installed_dependencies(tmp_path: Path) -> None:
    """A moved checkout can still report startup errors before loading any dependencies."""
    checkout = tmp_path / "moved-checkout"
    package = checkout / "src" / "facegen"
    package.mkdir(parents=True)
    source = Path(locations.__file__).parent
    for name in ("__init__.py", "locations.py"):
        shutil.copyfile(source / name, package / name)
    shutil.copytree(source / "desktop", package / "desktop")
    script = """
import json
import sys
from facegen import locations
from facegen.desktop import shell as desktop
assert 'facegen.persistence.settings' not in sys.modules
print(json.dumps([str(locations.CONFIG_DIR), str(locations.NATIVE_DIR), str(desktop.FALLBACK_LOG)]))
"""
    result = procs.run_quiet(
        [sys.executable, "-S", "-c", script],
        timeout=10,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(checkout / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        str(checkout / "config"),
        str(checkout / "native"),
        str(checkout / "data" / "logs" / "facegen.log"),
    ]
