"""Temporary settings files with all writable paths inside the test directory."""

from pathlib import Path

from kitshot.persistence.settings import DEFAULTS_PATH

FM_USER_LINE = 'fm_user_dir = ""'
FM_INSTALL_LINE = 'fm_install_dir = ""'


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
