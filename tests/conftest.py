from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from support.config import write_config
from support.paths import TESTS

from facegen.persistence import db
from facegen.persistence.settings import Settings, load_settings


@pytest.fixture(autouse=True)
def no_folder_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """FM27's folders are not looked for on the PC running the tests: a folder that no file
    sets is "not set". Tests of detection replace this or call fm_paths directly."""
    monkeypatch.setattr("facegen.persistence.settings.detect_folder", lambda key: None)


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = db.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Defaults with FM27's folder at tmp_path/fm and the data folder at tmp_path/data."""
    return write_config(tmp_path)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign one tier from the path, plus the architecture contract selector."""
    for item in items:
        parts = item.path.relative_to(TESTS).parts
        tier = parts[0]
        if tier not in {"unit", "integration", "parity", "opt_in"}:
            raise pytest.UsageError(f"Place collected tests in a tier directory: {item.path}")
        item.add_marker(getattr(pytest.mark, tier))
        if parts[:2] == ("integration", "contracts"):
            item.add_marker(pytest.mark.di)
