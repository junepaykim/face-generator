"""Runtime contracts and dependency defaults are usable before adapters are loaded."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from facegen.platform import procs


def test_runtime_boundaries_import_without_entrypoints_or_gpu(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys

class NoAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            'facegen.cli', 'facegen.pipeline.worker', 'facegen.render.renderer', 'facegen.render.tagger',
            'facegen.datasource.fmsave_source', 'facegen.persistence.db', 'sqlite3',
            'torch', 'timm', 'fmsave', 'webview', 'pystray',
        }
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('runtime boundary loaded adapter: ' + fullname)

sys.meta_path.insert(0, NoAdapters())
from facegen.desktop import shell as desktop
from facegen.setup import launch
from facegen.platform import output
from facegen.runtime import contracts as service_contracts
from facegen.runtime.dependencies import Gpu, default_source, default_renderer, default_tagger

gpu = Gpu()
assert gpu.renderer is default_renderer
assert gpu.tagger is default_tagger
assert Gpu(renderer=lambda: None).models_check() is None
"""
    result = procs.run_quiet(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
