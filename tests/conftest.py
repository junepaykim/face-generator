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
def no_folder_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ignore the checkout's local paths and real FM folders; tests can supply their own."""
    monkeypatch.setattr(
        "facegen.persistence.settings.LOCAL_PATH", tmp_path / "config" / "local.toml"
    )
    monkeypatch.setattr("facegen.persistence.settings.detect_folder", lambda key: None)


@pytest.fixture(autouse=True)
def no_release_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing asks GitHub for facegen's releases or downloads one: tests that need answers
    pass their own fetchers (setup/update.py)."""

    def refuse(url: str, *args: object, **kwargs: object) -> None:
        from facegen.setup.update import UpdateError

        raise UpdateError(f"tests do not reach the network: {url}")

    monkeypatch.setattr("facegen.setup.update.fetch_json", refuse)
    monkeypatch.setattr("facegen.setup.update.download", refuse)


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
