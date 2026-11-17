"""Korean and English UI text (Phase 12a): one message catalog per language.

Templates and routes hold keys such as ``"nav.players"``; ``en.toml`` and ``ko.toml`` beside
this file hold the words. A text may name values in braces (``"{count} queued"``), filled in
with ``str.format``. A text that depends on a number is a table with ``one`` and ``other``
(English) or a single text (Korean, which does not inflect), chosen by the ``count`` value.

Standard library only, so the tray and the window can use it as well as the web pages.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Any

LANGUAGES: dict[str, str] = {"en": "English", "ko": "한국어"}
"""Each language the catalog holds, by code, named in itself."""
DEFAULT = "en"
AUTO = "auto"
"""`display.language`: the language the window or browser asks for."""

_PLURAL_FORMS = ("one", "other")


def _flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path + "."))
        elif isinstance(value, str):
            flat[path] = value
        else:
            raise TypeError(f"message {path!r} is not text")
    return flat


@cache
def catalog(lang: str) -> Mapping[str, str]:
    """Every text of one language by dotted key."""
    if lang not in LANGUAGES:
        raise KeyError(f"no message catalog for {lang!r}")
    text = resources.files(__name__).joinpath(f"{lang}.toml").read_text(encoding="utf-8")
    return MappingProxyType(_flatten(tomllib.loads(text)))


def negotiate(setting: str, accept_language: str | None) -> str:
    """The language to show: the setting's, or for "auto" the most preferred one in an
    Accept-Language header that the catalog holds (WebView2 sends the Windows display
    language there). English when none is held."""
    if setting in LANGUAGES:
        return setting
    ranked: list[tuple[float, int, str]] = []
    for position, part in enumerate((accept_language or "").split(",")):
        tag, _, params = part.partition(";")
        primary = tag.strip().lower().split("-")[0]
        if primary not in LANGUAGES:
            continue
        quality = _quality(params)
        if quality is not None and quality > 0:
            ranked.append((-quality, position, primary))
    return min(ranked)[2] if ranked else DEFAULT


def _quality(params: str) -> float | None:
    """A language range's q value: 1 when none is given, None when it cannot be read."""
    for param in params.split(";"):
        name, _, value = param.partition("=")
        if name.strip().lower() == "q":
            try:
                return float(value.strip())
            except ValueError:
                return None
    return 1.0


def _plural(lang: str, count: Any) -> str:
    """English: "one" for exactly 1. Korean has one form, so its texts are not tables."""
    return "one" if lang == "en" and count == 1 else "other"


GROUPS = ("a", "b", "c")
"""Any text may name a picture group as {a}, {b} or {c}: filled in with its name ("group.a",
"Detailed" or 정밀), or on a page with a coloured chip of it."""


class Translator:
    """Texts in one language, with English where that language lacks one."""

    __slots__ = ("lang", "values")

    def __init__(self, lang: str = DEFAULT) -> None:
        self.lang = lang if lang in LANGUAGES else DEFAULT
        self.values: dict[str, str] = {}
        """Filled in wherever a text names them and the caller gives none: the group names."""
        self.values = {g: self(f"group.{g}") for g in GROUPS}

    def template(self, key: str, count: Any = None) -> str:
        """The text for `key` before its values are filled in. A key no catalog holds comes
        back as itself, so a missing text shows rather than failing the page."""
        for lang in dict.fromkeys((self.lang, DEFAULT)):
            texts = catalog(lang)
            if key in texts:
                return texts[key]
            form = _plural(lang, count)
            for name in (form, *_PLURAL_FORMS):
                if f"{key}.{name}" in texts:
                    return texts[f"{key}.{name}"]
        return key

    def has(self, key: str) -> bool:
        return self.template(key) != key

    def __call__(self, key: str, /, **values: Any) -> str:
        return self.template(key, values.get("count")).format(**{**self.values, **values})


ENGLISH = Translator(DEFAULT)
"""For code with no request to take a language from (logs, the command line)."""
