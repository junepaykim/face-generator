from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from facegen import db
from facegen.settings import DEFAULTS_PATH, Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
FM_USER_LINE = (
    'fm_user_dir = "/mnt/c/Users/<Windows user>/Documents/Sports Interactive/Football Manager 27"'
)
FM_INSTALL_LINE = (
    'fm_install_dir = "/mnt/c/Program Files (x86)/Steam/steamapps/common/Football Manager 27"'
)


def write_config(tmp_path: Path) -> Path:
    """A defaults file whose FM27 user, FM27 install and data folders are under tmp_path."""
    text = DEFAULTS_PATH.read_text(encoding="utf-8")
    assert FM_USER_LINE in text and FM_INSTALL_LINE in text
    text = text.replace(FM_USER_LINE, f'fm_user_dir = "{(tmp_path / "fm").as_posix()}"')
    text = text.replace(
        FM_INSTALL_LINE, f'fm_install_dir = "{(tmp_path / "fm-install").as_posix()}"'
    )
    text = text.replace('data_dir = "data"', f'data_dir = "{(tmp_path / "data").as_posix()}"')
    config = tmp_path / "test.toml"
    config.write_text(text, encoding="utf-8")
    return config


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
