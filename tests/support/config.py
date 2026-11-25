"""Temporary settings files with all writable paths inside the test directory."""

from pathlib import Path

from kitshot.persistence.settings import DEFAULTS_PATH

FM_USER_LINE = 'fm_user_dir = ""'
FM_INSTALL_LINE = 'fm_install_dir = ""'
STYLE_LINES = {'model = "realvis"': 'model = "animagine"', 'gender = "man"': 'gender = "woman"'}
"""The picture style the tests' small made-up world was written for: Animagine women, the
defaults before 0.1.8 (user, 2026-11-25). The fake tagger answers as for one girl unless told
otherwise. The contract tests check the real defaults."""


def old_style(text: str) -> str:
    """A defaults file's text with STYLE_LINES: Animagine women."""
    for line, old in STYLE_LINES.items():
        assert text.count(line) == 1, line
        text = text.replace(line, old)
    return text


def write_config(tmp_path: Path, *, style: bool = False) -> Path:
    """A defaults file whose FM27 user, FM27 install and data folders are under tmp_path. It
    draws Animagine women (`old_style`); `style=True` keeps the real picture style."""
    text = DEFAULTS_PATH.read_text(encoding="utf-8")
    assert FM_USER_LINE in text and FM_INSTALL_LINE in text
    text = text.replace(FM_USER_LINE, f'fm_user_dir = "{(tmp_path / "fm").as_posix()}"')
    text = text.replace(
        FM_INSTALL_LINE, f'fm_install_dir = "{(tmp_path / "fm-install").as_posix()}"'
    )
    text = text.replace('data_dir = "data"', f'data_dir = "{(tmp_path / "data").as_posix()}"')
    if not style:
        text = old_style(text)
    config = tmp_path / "test.toml"
    config.write_text(text, encoding="utf-8")
    return config
