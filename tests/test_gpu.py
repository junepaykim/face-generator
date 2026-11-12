"""The real model and tagger. Opt-in: FACEGEN_TEST_GPU=1 uv run pytest tests/test_gpu.py

Needs the GPU, the models in the Hugging Face cache, and FM27 closed. Draws one made-up
player; the picture stays in pytest's temporary folder.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FACEGEN_TEST_GPU") != "1", reason="set FACEGEN_TEST_GPU=1 to use the GPU"
)


def test_render_check_and_detail() -> None:
    from facegen import qc
    from facegen.kits import Kit
    from facegen.prompt import build_prompt
    from facegen.qc import Eva02Tagger
    from facegen.traits import sample_sheet
    from facegen.worker import SIZE, AnimagineRenderer

    sheet = sample_sheet("gpu-test", 1, 1)
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    prompt = build_prompt(sheet, "W07", kit, "crossed")
    renderer = AnimagineRenderer()
    assert renderer.prompt_chunks(prompt) >= 2  # the long prompt is not cut at 77 tokens
    tagger = Eva02Tagger()

    image = renderer.render(prompt, sheet.base_seed)
    assert image.size == (SIZE, SIZE)
    tags = tagger([image])[0]
    result = qc.check(tags, qc.expectation(sheet, kit), qc.head_top(image))
    assert result.rating == "general", result.failures
    # Seeded: the same seed draws the same picture (allowing for GPU rounding).
    from PIL import ImageChops, ImageStat

    again = renderer.render(prompt, sheet.base_seed)
    assert max(ImageStat.Stat(ImageChops.difference(again, image)).mean) < 2.0

    detailed = renderer.detail(image, prompt, sheet.base_seed)
    assert detailed.size == (1536, 1536)
