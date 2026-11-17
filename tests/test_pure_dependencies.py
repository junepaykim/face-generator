"""Policy code remains usable without storage, discovery, or model dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from facegen.platform import procs


def test_policy_imports_do_not_load_adapters(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys

class NoAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            'facegen.persistence.settings', 'facegen.persistence.settings_store', 'facegen.persistence.sheet_store',
            'facegen.datasource.kits', 'facegen.qc', 'facegen.render.tagger', 'facegen.render.renderer',
            'facegen.persistence.db', 'facegen.platform.fm_paths', 'facegen.platform.platform_info',
            'sqlite3', 'fmsave', 'torch', 'timm', 'huggingface_hub',
        }
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('policy loaded adapter: ' + fullname)

sys.meta_path.insert(0, NoAdapters())
from facegen.core import age
from facegen.core import classify
from facegen.core import planner
from facegen.core import prompt
from facegen.core import shoot
from facegen.core import wardrobe
from facegen.core.kit_values import Kit
from facegen.render.quality_checks import TagResult, check, expectation
from facegen.core.settings_values import MyClubSettings
from facegen.core.traits import sample_sheet

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
    from facegen import qc
    from facegen.core import kit_values, settings_values
    from facegen.datasource import kits
    from facegen.persistence import settings
    from facegen.render import quality_checks, tagger

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
