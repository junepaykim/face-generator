"""A log sink for tests (runtime.contracts.LogFn): the lines' texts as a list, with levels."""

from __future__ import annotations


class Lines(list[str]):
    """Each line's text, in order; `levels` holds each one's level."""

    def __init__(self) -> None:
        super().__init__()
        self.levels: list[str] = []

    def __call__(self, line: str, level: str = "info", /) -> None:
        self.append(line)
        self.levels.append(level)

    def at(self, level: str) -> list[str]:
        """The lines logged at `level`."""
        return [text for text, lv in zip(self, self.levels, strict=True) if lv == level]
