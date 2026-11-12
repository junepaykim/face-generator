"""The real model and tagger. Opt-in: FACEGEN_TEST_GPU=1 uv run pytest tests/test_gpu.py

Needs the GPU, the models in the Hugging Face cache (Animagine, the EVA02 tagger and, for the
reference test, h94/IP-Adapter), and FM27 closed. Draws one made-up player; the pictures stay
in memory. Run with -s to see the timings and memory.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FACEGEN_TEST_GPU") != "1", reason="set FACEGEN_TEST_GPU=1 to use the GPU"
)


@pytest.fixture(scope="module")
def gpu() -> Iterator[tuple[Any, Any]]:
    from facegen.qc import Eva02Tagger
    from facegen.worker import AnimagineRenderer

    renderer = AnimagineRenderer()
    tagger = Eva02Tagger()
    yield renderer, tagger
    print("memory:", renderer.memory())
    renderer.close()
    tagger.close()


@pytest.fixture(scope="module")
def case() -> tuple[Any, Any, Any]:
    from facegen.kits import Kit
    from facegen.prompt import build_prompt
    from facegen.traits import sample_sheet

    sheet = sample_sheet("gpu-test", 1, 1)
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    return sheet, kit, build_prompt(sheet, "W07", kit, "crossed")


def _difference(a: Any, b: Any) -> float:
    from PIL import ImageChops, ImageStat

    return max(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean)


def test_render_check_and_detail(gpu: tuple[Any, Any], case: tuple[Any, Any, Any]) -> None:
    from facegen import qc
    from facegen.picture import crop_to_aspect
    from facegen.worker import HEIGHT, WIDTH

    renderer, tagger = gpu
    sheet, kit, prompt = case
    assert renderer.prompt_chunks(prompt) >= 2  # the long prompt is not cut at 77 tokens

    image = renderer.render(prompt, sheet.base_seed)
    assert image.size == (WIDTH, HEIGHT) == (896, 1152)
    tags = tagger([image])[0]
    result = qc.check(tags, qc.expectation(sheet, kit), qc.head_top(image))
    print("portrait:", result.to_json(tags) | {"tags": None})
    assert result.rating == "general", result.failures
    # Seeded: the same seed draws the same picture (allowing for GPU rounding).
    assert _difference(renderer.render(prompt, sheet.base_seed), image) < 2.0

    detailed = renderer.detail(image, prompt, sheet.base_seed)
    assert detailed.size == (1344, 1728)
    assert crop_to_aspect(detailed).size == (1296, 1728)  # the 3:4 master


def test_reference_picture(gpu: tuple[Any, Any], case: tuple[Any, Any, Any]) -> None:
    """IP-Adapter loads on the first reference, draws with it, and switches out again so a
    picture without a reference is the plain model's."""
    from facegen import qc
    from facegen.picture import crop_to_aspect, face_box
    from facegen.worker import Reference

    renderer, tagger = gpu
    sheet, kit, prompt = case
    seed = sheet.base_seed
    plain = renderer.render(prompt, seed)
    existing = crop_to_aspect(renderer.render(prompt, seed + 1))
    reference = Reference("gpu-test", existing.crop(face_box(existing)), 0.1)

    t0 = time.perf_counter()
    referenced = renderer.render(prompt, seed, reference)
    print(
        f"first render with a reference (adapter load included): {time.perf_counter() - t0:.1f} s"
    )
    t0 = time.perf_counter()
    again = renderer.render(prompt, seed, reference)
    print(f"render with a reference: {time.perf_counter() - t0:.1f} s")
    assert _difference(again, referenced) < 2.0  # seeded with a reference too
    assert _difference(referenced, plain) > 0.5  # the reference changes the picture

    tags = tagger([referenced])[0]
    result = qc.check(tags, qc.expectation(sheet, kit), qc.head_top(referenced))
    print("with reference:", result.to_json(tags) | {"tags": None})
    assert result.rating == "general", result.failures

    detailed = renderer.detail(referenced, prompt, seed, reference)
    assert detailed.size == (1344, 1728)

    # Switched out: the UNet is the plain model again.
    assert _difference(renderer.render(prompt, seed), plain) < 2.0
