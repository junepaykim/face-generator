# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Package boundaries and the public imports used by launchers and the web app."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest
from support.paths import ROOT

from facegen.platform import procs


def run_python(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    source = ROOT / "src"
    return procs.run_quiet(
        [sys.executable, "-S", *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_package_initializers_and_startup_need_only_the_standard_library(tmp_path: Path) -> None:
    result = run_python(
        tmp_path,
        "-c",
        """
import importlib
import importlib.abc
import sys

class NoAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            'sqlite3', 'PIL', 'fastapi', 'fmsave', 'facegen_native', 'torch',
            'diffusers', 'transformers', 'timm', 'huggingface_hub', 'webview', 'pystray',
            'facegen.persistence.settings', 'facegen.runtime.composition',
        }
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('startup loaded adapter: ' + fullname)

sys.meta_path.insert(0, NoAdapters())
for name in ('core', 'datasource', 'persistence', 'pipeline', 'render', 'runtime',
             'desktop', 'setup', 'platform'):
    importlib.import_module('facegen.' + name)

from facegen.desktop import shell
from facegen.setup import venv_check
assert callable(shell.main)
assert callable(venv_check.main)
""",
    )
    assert result.returncode == 0, result.stderr


def test_package_version_tracks_the_checkout_and_installed_distribution(tmp_path: Path) -> None:
    import tomllib

    expected = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    result = run_python(tmp_path, "-c", "from facegen import __version__; print(__version__)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected

    # A regular wheel includes distribution metadata and no source-tree pyproject.toml.
    metadata = tmp_path / "facegen-2.0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: facegen\nVersion: 2.0.1\n", encoding="utf-8")
    result = run_python(
        tmp_path,
        "-c",
        "from pathlib import Path\n"
        "from facegen import locations\n"
        "locations.PROJECT_ROOT = Path.cwd()\n"
        "from facegen import __version__\n"
        "print(__version__)",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2.0.1"


@pytest.mark.parametrize("module", ["facegen.cli", "facegen.desktop", "facegen.app_window"])
def test_published_module_help_without_installed_dependencies(tmp_path: Path, module: str) -> None:
    result = run_python(tmp_path, "-m", module, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())


def test_environment_check_wrapper_preserves_its_exit_and_output(tmp_path: Path) -> None:
    package = tmp_path / "example-1.0.dist-info"
    package.mkdir()
    result = run_python(
        tmp_path,
        "-c",
        """
import runpy
from pathlib import Path
from facegen.setup import venv_check

venv_check.site_dirs = lambda: [Path.cwd()]
runpy.run_module('facegen.venv_check', run_name='__main__')
""",
    )
    assert result.returncode == 5, result.stderr
    assert result.stdout.strip() == "--reinstall-package example"
    assert result.stderr.strip() == "half-installed: example"
    assert list(tmp_path.iterdir()) == [package]
    assert not list(package.iterdir())


WEB_EXPORTS = [
    ("age", "core.age", "WORDINGS audience table_name"),
    ("classify", "core.classify", "Group"),
    ("db", "persistence.db", "ACTIVE_CAREER_ORDER connect"),
    (
        "folder_setup",
        "setup.folder_setup",
        "LocalFileError RealDisk check_folder value_to_write write_local_path",
    ),
    ("hardware", "setup.hardware", "Hardware changes recommend"),
    (
        "manual",
        "pipeline.manual",
        (
            "DEFAULT_GROUP ManualError cancel generate new_face new_picture new_pictures "
            "remove_picture restorable restore_picture retry set_pin"
        ),
    ),
    ("models", "setup.models", "DownloadStatus ModelDownload ModelsStatus REPOS gigabytes"),
    ("native", "platform.native", "status"),
    ("photos", "datasource.photos", "player_counts status"),
    ("planner", "core.planner", "REFERENCE_KINDS reference_scale"),
    ("platform_info", "platform.platform_info", "Kind current"),
    ("poller", "runtime.poller", "UNSUPPORTED_ADVICE"),
    ("publish", "pipeline.publish", "PublishError RELOAD_HINT"),
    (
        "settings",
        "persistence.settings",
        (
            "DETAIL_SIZES Settings SettingsError folder_not_set_message is_fixed "
            "load_defaults load_settings local_label local_path"
        ),
    ),
    (
        "settings_history",
        "pipeline.settings_history",
        "Outcome can_redo can_undo entries redo save save_defaults undo",
    ),
    (
        "traits",
        "core.traits",
        "CharacterSheet TRAIT_TABLES eyes_drawn_round personality_from_json personality_leans",
    ),
    ("worker", "pipeline.worker", "Career load_career"),
    ("service", "runtime.policy", "BUSY_TIMEOUT_MS"),
    ("service", "runtime.worker_service", "GPU_FAULT_ADVICE"),
    ("service", "runtime.poller_service", "PollerService"),
    ("service", "runtime.sync_service", "SyncService"),
    ("service", "runtime.contracts", "WorkerControl"),
]


@pytest.mark.parametrize(("legacy", "canonical", "names"), WEB_EXPORTS)
def test_web_compatibility_exports_keep_the_canonical_objects(
    legacy: str, canonical: str, names: str
) -> None:
    facade = importlib.import_module("facegen." + legacy)
    implementation = importlib.import_module("facegen." + canonical)
    for name in names.split():
        assert getattr(facade, name) is getattr(implementation, name), name


def test_app_context_reexport_preserves_positional_constructor() -> None:
    from facegen.web.app import AppContext
    from facegen.web.context import AppContext as CanonicalContext

    assert AppContext is CanonicalContext
    names = (
        "db_path",
        "defaults_path",
        "worker",
        "syncer",
        "poller",
        "models",
        "download",
        "hardware",
        "browse",
        "platform",
        # Phase 13, appended so the positions above are unchanged.
        "updates",
        "install_update",
    )
    parameters = inspect.signature(AppContext).parameters
    assert tuple(parameters) == names
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters.values())
    assert parameters["db_path"].default is inspect.Parameter.empty
    assert all(parameters[name].default is None for name in names[1:])
    values = (Path("temporary.sqlite"), *(object() for _ in names[1:]))
    positional = AppContext(*values)
    assert all(
        getattr(positional, name) is value for name, value in zip(names, values, strict=True)
    )
    minimal = AppContext(values[0])
    assert all(getattr(minimal, name) is None for name in names[1:])


def test_web_imports_canonical_modules_and_keeps_composition_out_of_handlers() -> None:
    """Include imports inside functions and TYPE_CHECKING, including relative imports."""
    web = ROOT / "src" / "facegen" / "web"
    facades = {f"facegen.{legacy}" for legacy, _, _ in WEB_EXPORTS}
    violations = []
    for path in sorted(web.rglob("*.py")):
        package = ".".join(path.parent.relative_to(ROOT / "src").parts)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = resolve_name("." * node.level + base, package)
                names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            for name in names:
                forbidden = any(
                    name == facade or name.startswith(facade + ".") for facade in facades
                )
                if path != web / "app.py":
                    forbidden |= name == "facegen.web.app" or name.startswith("facegen.web.app.")
                    if "routes" not in path.relative_to(web).parts:
                        forbidden |= name == "facegen.web.routes" or name.startswith(
                            "facegen.web.routes."
                        )
                if forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: imports {name}")
    assert not violations, "Web imports must follow the dependency boundaries:\n" + "\n".join(
        violations
    )
