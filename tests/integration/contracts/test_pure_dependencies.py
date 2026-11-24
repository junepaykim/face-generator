# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Policy code remains usable without storage, discovery, or model dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from support.paths import ROOT

from kitshot.platform import procs


def test_policy_imports_do_not_load_adapters(tmp_path: Path) -> None:
    source = ROOT / "src"
    script = """
import importlib.abc
import sys

class NoAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            'kitshot.persistence.settings', 'kitshot.persistence.settings_store', 'kitshot.persistence.sheet_store',
            'kitshot.datasource.kits', 'kitshot.qc', 'kitshot.render.tagger', 'kitshot.render.renderer',
            'kitshot.persistence.db', 'kitshot.platform.fm_paths', 'kitshot.platform.platform_info',
            'sqlite3', 'fmsave', 'torch', 'timm', 'huggingface_hub',
        }
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('policy loaded adapter: ' + fullname)

sys.meta_path.insert(0, NoAdapters())
from kitshot.core import age
from kitshot.core import classify
from kitshot.core import planner
from kitshot.core import prompt
from kitshot.core import shoot
from kitshot.core import wardrobe
from kitshot.core.kit_values import Kit
from kitshot.render.quality_checks import TagResult, check, expectation
from kitshot.core.settings_values import MyClubSettings
from kitshot.core.traits import sample_sheet

sheet = sample_sheet('test-career', 123, 1)
kit = Kit(('blue',))
assert kit.words() == 'blue shirt, short sleeves'
assert MyClubSettings(frozenset(), 0).round_eyes == 0.0
assert TagResult({'general': 1.0}, {}).rating == 'general'
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


def test_compatibility_exports_keep_the_same_values_and_types() -> None:
    from kitshot import qc
    from kitshot.core import kit_values, settings_values
    from kitshot.datasource import kits
    from kitshot.persistence import settings
    from kitshot.render import quality_checks, tagger

    assert settings.Settings is settings_values.Settings
    assert settings.Paths is settings_values.Paths
    assert settings.DETAIL_SIZES is settings_values.DETAIL_SIZES
    assert kits.Kit is kit_values.Kit
    assert kits.KitTable is kit_values.KitTable
    assert kits.COLOURS is kit_values.COLOURS
    assert qc.TagResult is quality_checks.TagResult
    assert qc.check is quality_checks.check
    assert qc.Eva02Tagger is tagger.Eva02Tagger
    assert qc.TAGGER_REPO == tagger.TAGGER_REPO
