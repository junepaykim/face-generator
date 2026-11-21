# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Z-Image-Turbo with NF4 transformer blocks and an INT8 text encoder, and the tagger. Opt-in:
FACEGEN_TEST_GPU=1 uv run --no-sync pytest tests/opt_in/test_gpu_zimage.py

Needs the GPU, Tongyi-MAI/Z-Image-Turbo (all of it) and the EVA02 tagger in the Hugging Face
cache, and FM27 closed. Draws one made-up man; the pictures stay in memory. Run with -s to see
the timings and memory.

The prompt is a fixed text in run `realistic`'s template (docs/research.md section 20) until
Phase 15f's prompt builder replaces it. Z-Image-Turbo has no negative prompt.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("FACEGEN_TEST_GPU") != "1", reason="set FACEGEN_TEST_GPU=1 to use the GPU"
    ),
]

SEED = 7_654_321


def _prompt(age: int) -> Any:
    from facegen.core.prompt import Prompt

    return Prompt(
        f"Professional studio headshot photograph of a {age}-year-old West African male football"
        " player, head and shoulders, centred, facing the camera and looking into the lens. He"
        " has dark brown skin, short black hair in a high-top fade, a short trimmed beard, dark"
        " brown eyes and a confident smile. He wears an orange football shirt with black trim,"
        " fully clothed. A plain off-white studio background, soft even lighting, sharp focus,"
        " natural skin texture, realistic photo. He is the only person in the picture.",
        "",
    )


@pytest.fixture(scope="module")
def gpu() -> Iterator[tuple[Any, Any]]:
    from facegen.render.tagger import Eva02Tagger
    from facegen.render.zimage import ZImageRenderer

    renderer = ZImageRenderer()
    tagger = Eva02Tagger()
    yield renderer, tagger
    print("memory:", renderer.memory())
    renderer.close()
    tagger.close()


def _difference(a: Any, b: Any) -> float:
    from PIL import ImageChops, ImageStat

    return max(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean)


def _one_man(tags: Any) -> dict[str, Any]:
    return {
        "rating": tags.rating,
        **{t: round(tags.p(t), 3) for t in ("1boy", "solo", "1girl", "realistic")},
    }


def test_render_and_detail(gpu: tuple[Any, Any]) -> None:
    import torch

    from facegen.render.picture import crop_to_aspect
    from facegen.render.renderer import HEIGHT, WIDTH
    from facegen.render.zimage import MAX_TOKENS, ZIMAGE_REPO

    renderer, tagger = gpu
    assert renderer.model == ZIMAGE_REPO
    assert renderer.nf4_layers == 238
    assert renderer.pipe.transformer.dtype == renderer.pipe.vae.dtype == torch.bfloat16
    settings = renderer.settings()
    assert (settings["steps"], settings["cfg"]) == (9, 0.0)
    prompt = _prompt(27)
    assert 50 < renderer.prompt_tokens(prompt) < MAX_TOKENS

    image = renderer.render(prompt, SEED)
    assert image.size == (WIDTH, HEIGHT) == (896, 1152)
    tags = tagger([image])[0]
    print("portrait:", _one_man(tags))
    assert tags.rating == "general"
    # Seeded: the same seed draws the same picture (allowing for GPU rounding).
    assert _difference(renderer.render(prompt, SEED), image) < 2.0

    detailed = renderer.detail(image, prompt, SEED, size=(1344, 1728))
    assert detailed.size == (1344, 1728)
    assert crop_to_aspect(detailed).size == (1296, 1728)  # the 3:4 master
    assert tagger([detailed])[0].rating == "general"
    assert renderer.detail_settings()["vae_tiling"] is True
    assert renderer.vae_tiling is False  # the detail pass's tiling does not stay on


def test_no_reference_and_no_overlong_prompt(gpu: tuple[Any, Any]) -> None:
    """No IP-Adapter for Z-Image: a reference is refused, never silently dropped. A prompt the
    pipeline would cut at 512 tokens is refused too."""
    from PIL import Image

    from facegen.core.prompt import Prompt
    from facegen.render.renderer import Reference

    renderer, _ = gpu
    assert renderer.reference_adapter is False
    reference = Reference("gpu-test-zimage", Image.new("RGB", (64, 64)), 0.1)
    with pytest.raises(ValueError, match="no reference adapter"):
        renderer.render(_prompt(27), SEED, reference)
    with pytest.raises(ValueError, match="reads 512"):
        renderer.prepare([Prompt("football " * 600, "")])


def test_group_a_job_keeps_peak_vram_under_8_gib(gpu: tuple[Any, Any]) -> None:
    """A group A job as the worker runs it, with the encode trip for the next jobs: PyTorch's
    peak reserved memory stays under 8 GiB (user, 2026-11-11). The text encoder and the
    tagger are back in system memory afterwards, the transformer on the card."""
    import torch

    renderer, tagger = gpu
    prompt = _prompt(19)  # new prompts: the encoder must load
    coming = [_prompt(age) for age in (23, 30, 34)]
    torch.cuda.reset_peak_memory_stats()
    assert renderer.prepare([prompt, *coming]) == 4  # the worker's trip for the next jobs
    candidates = [renderer.render(prompt, SEED + k) for k in range(3)]
    tagger(candidates)
    detailed = renderer.detail(candidates[0], prompt, SEED, size=(1344, 1728))
    tagger([detailed])
    reserved = torch.cuda.max_memory_reserved() / 2**30
    allocated = torch.cuda.max_memory_allocated() / 2**30
    print(f"Z-Image-Turbo group A job: peak reserved {reserved:.2f} GiB, allocated {allocated:.2f}")
    assert reserved < 8.0
    for model in (renderer.encoder.text_encoder, tagger.model):
        assert next(model.parameters()).device.type == "cpu"
    assert renderer.pipe.transformer.device.type == "cuda"


def test_cached_prompts_draw_bit_for_bit_what_direct_encoding_draws(
    gpu: tuple[Any, Any],
) -> None:
    """The prompt cache changes no pixel: an embedding prepared in one trip, kept in system
    memory and copied back equals the pipeline's own encoding, and draws the same bytes."""
    import torch

    from facegen.render.gpu import on_gpu
    from facegen.render.renderer import HEIGHT, WIDTH
    from facegen.render.zimage import GUIDANCE, MAX_TOKENS, STEPS

    renderer, _ = gpu
    prompt = _prompt(31)
    renderer.prepare([prompt])
    assert renderer.encoded(prompt)
    with torch.no_grad(), on_gpu(renderer.encoder.text_encoder):
        (expected,) = renderer.encoder._encode_prompt(
            [prompt.positive], device=torch.device("cuda"), max_sequence_length=MAX_TOKENS
        )
    (got,) = renderer._embeddings(prompt)
    assert got.device == expected.device and got.dtype == expected.dtype
    assert torch.equal(got, expected)

    image = renderer.render(prompt, SEED)
    with torch.no_grad():
        direct = renderer.pipe(
            prompt_embeds=[expected],
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            cfg_normalization=False,
            generator=torch.Generator("cuda").manual_seed(SEED),
        ).images[0]
    assert image.tobytes() == direct.tobytes()
