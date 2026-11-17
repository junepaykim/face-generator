# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""The real model and tagger. Opt-in: FACEGEN_TEST_GPU=1 uv run --no-sync pytest tests/opt_in/test_gpu.py

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

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("FACEGEN_TEST_GPU") != "1", reason="set FACEGEN_TEST_GPU=1 to use the GPU"
    ),
]


@pytest.fixture(scope="module")
def gpu() -> Iterator[tuple[Any, Any]]:
    from facegen.pipeline.worker import AnimagineRenderer
    from facegen.qc import Eva02Tagger

    renderer = AnimagineRenderer()
    tagger = Eva02Tagger()
    yield renderer, tagger
    print("memory:", renderer.memory())
    renderer.close()
    tagger.close()


@pytest.fixture(scope="module")
def case() -> tuple[Any, Any, Any]:
    from facegen.core.prompt import build_prompt
    from facegen.core.traits import sample_sheet
    from facegen.datasource.kits import Kit

    sheet = sample_sheet("gpu-test", 1, 1)
    kit = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
    return sheet, kit, build_prompt(sheet, "W07", kit, "crossed")


def _difference(a: Any, b: Any) -> float:
    from PIL import ImageChops, ImageStat

    return max(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean)


def test_render_check_and_detail(gpu: tuple[Any, Any], case: tuple[Any, Any, Any]) -> None:
    from facegen import qc
    from facegen.pipeline.worker import HEIGHT, WIDTH
    from facegen.render.picture import crop_to_aspect

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

    detailed = renderer.detail(image, prompt, sheet.base_seed, size=(1344, 1728))
    assert detailed.size == (1344, 1728)
    assert crop_to_aspect(detailed).size == (1296, 1728)  # the 3:4 master


def test_reference_picture(gpu: tuple[Any, Any], case: tuple[Any, Any, Any]) -> None:
    """IP-Adapter loads on the first reference, draws with it, and switches out again so a
    picture without a reference is the plain model's."""
    from facegen import qc
    from facegen.pipeline.worker import Reference
    from facegen.render.picture import crop_to_aspect, face_box

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

    detailed = renderer.detail(referenced, prompt, seed, reference, size=(1344, 1728))
    assert detailed.size == (1344, 1728)

    # Switched out: the UNet is the plain model again.
    assert _difference(renderer.render(prompt, seed), plain) < 2.0


def test_offloaded_models_keep_peak_vram_under_8_gib(
    gpu: tuple[Any, Any], case: tuple[Any, Any, Any]
) -> None:
    """A group A job, with a reference, as the worker runs it: PyTorch's peak reserved memory
    stays under 8 GiB (user, 2026-11-11), and the text encoders, the tagger and the image
    encoder are back in system memory afterwards."""
    import torch

    from facegen.core.prompt import build_prompt
    from facegen.pipeline.worker import Reference
    from facegen.render.picture import face_box

    renderer, tagger = gpu
    sheet, kit, _ = case
    prompt = build_prompt(sheet, "W03", kit, "crossed")  # new prompts: the encoders must load
    coming = [build_prompt(sheet, w, kit, "crossed") for w in ("W04", "W06", "W09")]
    torch.cuda.reset_peak_memory_stats()
    assert renderer.prepare([prompt, *coming]) > 0  # the worker's trip for the next jobs
    candidates = [renderer.render(prompt, sheet.base_seed + k) for k in range(3)]
    tagger(candidates)
    reference = Reference("gpu-test-vram", candidates[0].crop(face_box(candidates[0])), 0.1)
    redraw = renderer.render(prompt, sheet.base_seed, reference)
    tagger([redraw])
    detailed = renderer.detail(redraw, prompt, sheet.base_seed, reference, size=(1344, 1728))
    tagger([detailed])
    reserved = torch.cuda.max_memory_reserved() / 2**30
    allocated = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"group A job with a reference: peak reserved {reserved:.2f} GiB, allocated {allocated:.2f}"
    )
    assert reserved < 8.0
    pipe = renderer.pipe
    for model in (pipe.text_encoder, pipe.text_encoder_2, pipe.image_encoder, tagger.model):
        assert next(model.parameters()).device.type == "cpu"
    assert pipe.unet.device.type == "cuda"


def test_pre_encoded_prompts_draw_bit_for_bit_what_encoding_per_job_drew(
    gpu: tuple[Any, Any], case: tuple[Any, Any, Any]
) -> None:
    """The prompt cache changes no pixel (user, 2026-11-12). Prompts prepared in one trip,
    kept in system memory and copied back give the very embeddings the renderer encoded per
    job before the cache, and the same picture to the byte."""
    import torch
    from encoding_reference import encode_pair_before_cache

    from facegen.core.prompt import build_prompt
    from facegen.render.gpu import on_gpu
    from facegen.render.renderer import CFG, HEIGHT, STEPS, WIDTH

    renderer, _ = gpu
    sheet, kit, _ = case
    # A young and an adult wording: two different negative prompts.
    prompts = [build_prompt(sheet, w, kit, "crossed") for w in ("W02", "W05", "W08")]
    renderer.prepare(prompts)
    assert all(renderer.encoded(p) for p in prompts)
    pipe = renderer.pipe
    for prompt in prompts:
        with torch.no_grad(), on_gpu(pipe.text_encoder, pipe.text_encoder_2):
            expected = encode_pair_before_cache(pipe, prompt.positive, prompt.negative)
        got = renderer._embeddings(prompt)
        assert got[4] == expected[4] >= 2
        for a, b in zip(got[:4], expected[:4], strict=True):
            assert a.device == b.device and a.dtype == b.dtype and torch.equal(a, b)

    # The picture, against the pipeline given the per-job encoding directly.
    seed = sheet.base_seed
    image = renderer.render(prompts[2], seed)
    pe, pp, ne, npool, _ = expected
    with torch.no_grad(), renderer._fp16_accumulation():
        direct = pipe(
            prompt_embeds=pe,
            pooled_prompt_embeds=pp,
            negative_prompt_embeds=ne,
            negative_pooled_prompt_embeds=npool,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
    assert image.tobytes() == direct.tobytes()
