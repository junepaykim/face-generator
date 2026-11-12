"""A fake renderer and tagger, so the worker runs without a GPU."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageDraw

from facegen.prompt import Prompt
from facegen.qc import TagResult
from facegen.traits import ACCESSORY, EYE_COLOUR, FIELDS, HAIR_COLOUR, MARK

GENERAL = {"general": 0.95, "sensitive": 0.04, "questionable": 0.0, "explicit": 0.0}
SENSITIVE = {"general": 0.30, "sensitive": 0.65, "questionable": 0.04, "explicit": 0.0}
_COLOUR_TAGS = {t for table in (HAIR_COLOUR, EYE_COLOUR) for o in table.values() for t in o.tags}
_TRAIT_TAGS = {
    t
    for name, table in FIELDS.items()
    if table not in (ACCESSORY, MARK)
    for o in table.values()
    for t in o.tags
}

# What the fake tagger says about a picture.
VERDICTS: dict[str, TagResult] = {
    # Every hair and eye colour tag present, so any sheet passes.
    "pass": TagResult(GENERAL, {"solo": 0.99, "1girl": 0.99, **dict.fromkeys(_COLOUR_TAGS, 0.6)}),
    # Also confident in every trait and shirt colour: ranks above "pass".
    "good": TagResult(
        GENERAL,
        {
            "solo": 0.99,
            "1girl": 0.99,
            **dict.fromkeys(_COLOUR_TAGS | _TRAIT_TAGS, 0.95),
            **dict.fromkeys(
                ["red shirt", "blue shirt", "white shirt", "green shirt", "yellow shirt"], 0.9
            ),
        },
    ),
    "sensitive": TagResult(
        SENSITIVE, {"solo": 0.99, "1girl": 0.99, **dict.fromkeys(_COLOUR_TAGS, 0.6)}
    ),
    "two": TagResult(
        GENERAL, {"solo": 0.1, "2girls": 0.9, "1girl": 0.2, **dict.fromkeys(_COLOUR_TAGS, 0.6)}
    ),
}


def figure(size: int = 64, top: float = 0.04) -> Image.Image:
    img = Image.new("RGB", (size, size), (200, 200, 205))
    draw = ImageDraw.Draw(img)
    y = int(top * size)
    draw.ellipse((size * 0.38, y, size * 0.62, y + size * 0.24), fill=(240, 205, 180))
    draw.rectangle((size * 0.2, y + size * 0.25, size * 0.8, size), fill=(30, 60, 160))
    return img


class FakeRenderer:
    model = "fake/animagine"

    def __init__(self, fail_on_call: int | None = None, exc: BaseException | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.prompts: list[Prompt] = []
        self.fail_on_call = fail_on_call
        self.exc = exc or KeyboardInterrupt()

    def render(self, prompt: Prompt, seed: int) -> Image.Image:
        self.calls.append(("render", seed))
        self.prompts.append(prompt)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise self.exc
        img = figure()
        img.info["seed"] = seed
        return img

    def detail(self, image: Image.Image, prompt: Prompt, seed: int) -> Image.Image:
        self.calls.append(("detail", seed))
        out = image.resize((96, 96))
        out.info["seed"] = seed
        return out

    def settings(self) -> dict[str, Any]:
        return {"size": 64, "steps": 1, "cfg": 5.5, "sampler": "fake"}

    def detail_settings(self) -> dict[str, Any]:
        return {"scale": 1.5, "strength": 0.35}

    @property
    def renders(self) -> int:
        return sum(1 for kind, _ in self.calls if kind == "render")

    @property
    def details(self) -> int:
        return sum(1 for kind, _ in self.calls if kind == "detail")


class FakeTagger:
    """Answers with the given verdicts in order, then "pass"."""

    name = "fake/tagger"

    def __init__(self, verdicts: Sequence[str] = ()) -> None:
        self.verdicts = list(verdicts)
        self.seen: list[tuple[int | None, str]] = []

    def __call__(self, images: Sequence[Image.Image]) -> list[TagResult]:
        out = []
        for im in images:
            verdict = self.verdicts.pop(0) if self.verdicts else "pass"
            self.seen.append((im.info.get("seed"), verdict))
            out.append(VERDICTS[verdict])
        return out
