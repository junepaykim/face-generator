"""A fake renderer and tagger, so the worker runs without a GPU."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from PIL import Image, ImageDraw

from facegen.core.prompt import Prompt
from facegen.core.traits import ACCESSORY, EYE_COLOUR, FIELDS, HAIR_COLOUR, MARK
from facegen.render.quality_checks import TagResult

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
    # One man, as the tagger saw the made-up men of Phase 15 (docs/research.md section 20).
    "boy": TagResult(
        GENERAL,
        {"solo": 0.98, "1boy": 0.99, "male focus": 0.9, **dict.fromkeys(_COLOUR_TAGS, 0.6)},
    ),
    # One man with no shirt: rated general, but the men's content tags catch it.
    "shirtless": TagResult(
        GENERAL,
        {"solo": 0.98, "1boy": 0.99, "shirtless": 0.8, "topless male": 0.7},
    ),
}


BACKDROPS = ("grey", "vertical", "horizontal", "split", "diagonal")
"""The backdrops `figure` can draw: the plain grey studio backdrop; a gradient top to bottom
or left to right; two tones split down the middle or along a diagonal (shoot.py)."""


def backdrop(width: int, height: int, kind: str = "grey") -> Image.Image:
    """A backdrop with no figure: one of BACKDROPS."""
    if kind == "grey":
        return Image.new("RGB", (width, height), (200, 200, 205))
    img = Image.new("RGB", (width, height))
    a, b = (150, 190, 240), (20, 60, 150)  # light blue to navy
    for y in range(height):
        for x in range(width):
            match kind:
                case "vertical":
                    t = y / max(1, height - 1)
                case "horizontal":
                    t = x / max(1, width - 1)
                case "split":
                    t = 0.0 if x < width / 2 else 1.0
                case "diagonal":
                    t = 0.0 if x / width + y / height < 0.9 else 1.0
                case _:
                    raise ValueError(kind)
            if kind in ("split", "diagonal"):
                colour = (200, 200, 205) if t == 0 else (235, 200, 60)  # grey and yellow
            else:
                colour = tuple(round(p + (q - p) * t) for p, q in zip(a, b, strict=True))
            img.putpixel((x, y), colour)
    return img


def figure(width: int = 56, height: int = 72, top: float = 0.04, kind: str = "grey") -> Image.Image:
    """A head and shoulders on a backdrop (BACKDROPS; the grey studio backdrop by default).
    The default size is the model's 7:9 portrait render (896x1152) at a small size; the
    worker crops it to 3:4."""
    img = backdrop(width, height, kind)
    draw = ImageDraw.Draw(img)
    y = int(top * height)
    head = 0.24 * width
    draw.ellipse(
        (width / 2 - head / 2, y, width / 2 + head / 2, y + 0.19 * height), fill=(240, 205, 180)
    )
    draw.rectangle((width * 0.2, y + 0.2 * height, width * 0.8, height), fill=(30, 60, 160))
    return img


class FakeRenderer:
    """Built by the worker as `renderer_factory(model)`; `model` reads "fake/<key>"."""

    def __init__(
        self,
        key: str = "animagine",
        fail_on_call: int | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self.key = key
        self.model = f"fake/{key}"
        self.calls: list[tuple[str, int]] = []
        self.prompts: list[Prompt] = []
        self.references: list[Any] = []
        """The reference given to each render and detail call, None for none."""
        self.fail_on_call = fail_on_call
        self.exc = exc or KeyboardInterrupt()
        self.texts: set[str] = set()
        """Prompt texts encoded so far, as the real renderer's cache holds them."""
        self.trips: list[list[Prompt]] = []
        """The prompts of each trip of the text encoders, in order."""
        self.detail_sizes: list[tuple[int, int]] = []
        """The target size of each detail pass, in order."""
        self.vae_tiling = False

    # The two stages of prompt encoding (facegen.prompt_cache), without the cache's size limit.

    def prepare(self, prompts: Iterable[Prompt]) -> int:
        prompts = list(prompts)
        new = {t for p in prompts for t in (p.positive, p.negative)} - self.texts
        if new:
            self.trips.append(prompts)
            self.texts |= new
        return len(new)

    def encoded(self, prompt: Prompt) -> bool:
        return {prompt.positive, prompt.negative} <= self.texts

    def render(self, prompt: Prompt, seed: int, reference: Any = None) -> Image.Image:
        self.prepare([prompt])  # as the real renderer encodes a prompt nobody prepared
        self.calls.append(("render", seed))
        self.prompts.append(prompt)
        self.references.append(reference)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise self.exc
        img = figure()
        img.info["seed"] = seed
        return img

    def detail(
        self,
        image: Image.Image,
        prompt: Prompt,
        seed: int,
        reference: Any = None,
        *,
        size: tuple[int, int],
    ) -> Image.Image:
        """Scaled as the real renderer's target size scales the 896x1152 render: the fake's
        56x72 picture is that render divided by 16."""
        self.prepare([prompt])
        self.calls.append(("detail", seed))
        self.references.append(reference)
        self.detail_sizes.append(size)
        out = image.resize((image.width * size[0] // 896, image.height * size[1] // 1152))
        out.info["seed"] = seed
        return out

    def set_vae_tiling(self, on: bool) -> None:
        self.vae_tiling = on

    def settings(self) -> dict[str, Any]:
        return {
            "width": 56,
            "height": 72,
            "steps": 1,
            "cfg": 5.5,
            "sampler": "fake",
            "vae_tiling": self.vae_tiling,
        }

    closed = 0

    def close(self) -> None:
        self.closed += 1

    def detail_settings(self) -> dict[str, Any]:
        return {"strength": 0.35}

    def reference_settings(self) -> dict[str, Any]:
        return {"adapter": "fake/ip-adapter"}

    @property
    def renders(self) -> int:
        return sum(1 for kind, _ in self.calls if kind == "render")

    @property
    def details(self) -> int:
        return sum(1 for kind, _ in self.calls if kind == "detail")


class FakeZImageRenderer(FakeRenderer):
    """As Z-Image-Turbo: no reference adapter (a reference is an error), no negative prompt,
    and the prompt's length in tokens rather than SDXL's chunks."""

    reference_adapter = False
    negative_prompt = False

    def __init__(self, key: str = "zimage_turbo", **kw: Any) -> None:
        super().__init__(key, **kw)

    def render(self, prompt: Prompt, seed: int, reference: Any = None) -> Image.Image:
        if reference is not None:
            raise ValueError("Z-Image-Turbo has no reference adapter")
        return super().render(prompt, seed)

    def prompt_tokens(self, prompt: Prompt) -> int:
        return len(prompt.positive.split())

    def reference_settings(self) -> dict[str, Any]:
        raise AssertionError("Z-Image-Turbo has no reference settings")


class FakeTagger:
    """Answers with the given verdicts in order, then `default` ("pass"; "boy" for men)."""

    name = "fake/tagger"

    def __init__(self, verdicts: Sequence[str] = (), default: str = "pass") -> None:
        self.verdicts = list(verdicts)
        self.default = default
        self.seen: list[tuple[int | None, str]] = []

    closed = 0

    def close(self) -> None:
        self.closed += 1

    def __call__(self, images: Sequence[Image.Image]) -> list[TagResult]:
        out = []
        for im in images:
            verdict = self.verdicts.pop(0) if self.verdicts else self.default
            self.seen.append((im.info.get("seed"), verdict))
            out.append(VERDICTS[verdict])
        return out
