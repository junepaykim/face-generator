"""Choosing the language and filling in texts (Phase 12a)."""

from __future__ import annotations

import pytest

from kitshot.i18n import DEFAULT, Translator, negotiate


@pytest.mark.parametrize(
    ("setting", "header", "expected"),
    [
        ("auto", "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7", "ko"),  # Korean Windows' WebView2
        ("auto", "en-US,en;q=0.9,ko;q=0.8", "en"),
        ("auto", "fr-FR,fr;q=0.9", "en"),  # nothing held: English
        ("auto", "fr-FR, ko;q=0.5", "ko"),  # the first held one, whatever its rank
        ("auto", "en;q=0.2, ko;q=0.8", "ko"),  # by quality, not position
        ("auto", "ko;q=0, en", "en"),  # q=0 means "not this one"
        ("auto", "ko;q=bad, en", "en"),
        ("auto", "KO", "ko"),
        ("auto", "", "en"),
        ("auto", None, "en"),
        ("ko", "en-US", "ko"),  # the switch wins over the window
        ("en", "ko-KR", "en"),
    ],
)
def test_negotiate(setting: str, header: str | None, expected: str) -> None:
    assert negotiate(setting, header) == expected


def test_values_are_filled_in_and_plurals_follow_the_language() -> None:
    en, ko = Translator("en"), Translator("ko")
    assert en("hardware.apply", count=1) == "Apply this 1 change"
    assert en("hardware.apply", count=3) == "Apply these 3 changes"
    assert ko("hardware.apply", count=1) == ko("hardware.apply", count=3).replace("3", "1")
    assert en("players.total", count=33012) == "33,012 players"
    assert ko("common.group", group="A") == "A 그룹"


def test_unknown_language_and_key() -> None:
    assert Translator("fr").lang == DEFAULT
    t = Translator("ko")
    assert t("no.such.key") == "no.such.key"  # shows rather than failing the page
    assert not t.has("no.such.key") and t.has("nav.players")
