# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Runtime contracts and dependency defaults are usable before adapters are loaded."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from support.paths import ROOT

from kitshot.platform import procs


def test_runtime_boundaries_import_without_entrypoints_or_gpu(tmp_path: Path) -> None:
    source = ROOT / "src"
    script = """
import importlib.abc
import sys

class NoAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            'kitshot.cli', 'kitshot.pipeline.worker', 'kitshot.render.renderer', 'kitshot.render.tagger',
            'kitshot.datasource.fmsave_source', 'kitshot.persistence.db', 'sqlite3',
            'torch', 'timm', 'fmsave', 'webview', 'pystray',
        }
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('runtime boundary loaded adapter: ' + fullname)

sys.meta_path.insert(0, NoAdapters())
from kitshot.desktop import shell as desktop
from kitshot.setup import launch
from kitshot.platform import output
from kitshot.runtime import contracts as service_contracts
from kitshot.runtime.dependencies import Gpu, default_source, default_renderer, default_tagger

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


def test_default_renderer_builds_the_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each `style.model` key names a renderer; the classes are stand-ins, so nothing loads."""
    from kitshot.core.settings_values import MODEL_CHOICES
    from kitshot.render import renderer, zimage
    from kitshot.runtime.dependencies import default_renderer

    monkeypatch.setattr(renderer, "AnimagineRenderer", lambda: "animagine")
    monkeypatch.setattr(renderer, "SdxlRenderer", lambda spec: spec.repo)
    monkeypatch.setattr(zimage, "ZImageRenderer", lambda: "zimage")
    assert default_renderer() == default_renderer("realvis") == renderer.REALVIS_REPO
    assert default_renderer("animagine") == "animagine"
    assert default_renderer("zimage_turbo") == "zimage"
    assert all(default_renderer(key) for key in MODEL_CHOICES)
    with pytest.raises(ValueError, match="no image model called 'flux'"):
        default_renderer("flux")
