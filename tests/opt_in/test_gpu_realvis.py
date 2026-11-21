# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""RealVisXL V5.0 with INT8 UNet blocks, the realistic SDXL model, and the tagger. Opt-in:
FACEGEN_TEST_GPU=1 uv run --no-sync pytest tests/opt_in/test_gpu_realvis.py

Needs the GPU, the models in the Hugging Face cache (RealVisXL, the EVA02 tagger and, for the
reference test, h94/IP-Adapter), and FM27 closed. Draws one made-up man; the pictures stay in
memory. Run with -s to see the timings and memory.

The prompt is a fixed text in run `realistic`'s template (docs/research.md section 20) until
Phase 15f's prompt builder replaces it; the negative is the content negative the user approved
on 2026-11-20 with that run's composition and style terms.
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

NEGATIVE = (
    "nsfw, nude, naked, shirtless, bare chest, topless, underwear, swimwear, revealing clothes, "
    "suggestive, multiple people, two people, text, watermark, signature, logo, frame, cartoon, "
    "anime, illustration, painting, drawing, 3d render, cgi, plastic skin, deformed, "
    "bad anatomy, asymmetric eyes, blurry, lowres"
)
SEED = 1_234_567


def _prompt(age: int) -> Any:
    from facegen.core.prompt import Prompt

    return Prompt(
        f"Professional studio headshot photograph of a {age}-year-old Mediterranean male football"
        " player, head and shoulders, centred, facing the camera and looking into the lens. He"
        " has olive skin, dark brown wavy hair swept back, light stubble, brown eyes and a"
        " neutral expression. He wears a claret football shirt with light blue sleeves, fully"
        " clothed. A dark navy blue studio background, soft even lighting, sharp focus, natural"
        " skin texture, realistic photo. He is the only person in the picture.",
        NEGATIVE,
    )


@pytest.fixture(scope="module")
def gpu() -> Iterator[tuple[Any, Any]]:
    from facegen.render.renderer import REALVIS, SdxlRenderer
    from facegen.render.tagger import Eva02Tagger

    renderer = SdxlRenderer(REALVIS)
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


def test_the_vae_tiles_as_animagines(gpu: tuple[Any, Any]) -> None:
    """512-pixel tiles (64 latents), not its config's 1024 (Phase 15i)."""
    renderer, _ = gpu
    vae = renderer.pipe.vae
    assert (vae.tile_sample_min_size, vae.tile_latent_min_size) == (512, 64)
    assert renderer.img2img.vae is vae
    assert "512-pixel tiles" in renderer.settings()["vae"]


def test_render_and_detail(gpu: tuple[Any, Any]) -> None:
    import torch

    from facegen.render.picture import crop_to_aspect
    from facegen.render.renderer import HEIGHT, REALVIS_REPO, WIDTH

    renderer, tagger = gpu
    # Its own VAE (the fp16-fix weights) decodes in fp16, not upcast to fp32.
    assert renderer.pipe.vae.config.force_upcast is False
    assert renderer.pipe.vae.dtype == torch.float16
    assert renderer.model == REALVIS_REPO
    settings = renderer.settings()
    assert (settings["sampler"], settings["steps"], settings["cfg"]) == (
        "dpmpp_2m_sde_karras",
        30,
        5.0,
    )
    prompt = _prompt(22)
    assert renderer.prompt_chunks(prompt) >= 2  # the long prompt is not cut at 77 tokens

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


def test_reference_picture(gpu: tuple[Any, Any]) -> None:
    """IP-Adapter plus-face loads into RealVisXL's UNet (SDXL's architecture), draws with a
    reference, and switches out again so a picture without one is the plain model's."""
    from facegen.render.picture import crop_to_aspect, face_box
    from facegen.render.renderer import Reference

    renderer, tagger = gpu
    prompt = _prompt(22)
    plain = renderer.render(prompt, SEED)
    existing = crop_to_aspect(renderer.render(prompt, SEED + 1))
    reference = Reference("gpu-test-realvis", existing.crop(face_box(existing)), 0.1)

    t0 = time.perf_counter()
    referenced = renderer.render(prompt, SEED, reference)
    print(
        f"first render with a reference (adapter load included): {time.perf_counter() - t0:.1f} s"
    )
    again = renderer.render(prompt, SEED, reference)
    assert _difference(again, referenced) < 2.0  # seeded with a reference too
    assert _difference(referenced, plain) > 0.5  # the reference changes the picture
    tags = tagger([referenced])[0]
    print("with reference:", _one_man(tags))
    assert tags.rating == "general"

    detailed = renderer.detail(referenced, prompt, SEED, reference, size=(1344, 1728))
    assert detailed.size == (1344, 1728)

    # Switched out: the UNet is the plain model again.
    assert _difference(renderer.render(prompt, SEED), plain) < 2.0


def test_group_a_job_with_a_reference_keeps_peak_vram_under_8_gib(gpu: tuple[Any, Any]) -> None:
    """A group A job with a reference, as the worker runs it: PyTorch's peak reserved memory
    stays under 8 GiB (user, 2026-11-11), and the text encoders, the tagger and the image
    encoder are back in system memory afterwards."""
    import torch

    from facegen.render.picture import face_box
    from facegen.render.renderer import Reference

    renderer, tagger = gpu
    prompt = _prompt(19)  # new prompts: the encoders must load
    coming = [_prompt(age) for age in (24, 29, 33)]
    torch.cuda.reset_peak_memory_stats()
    assert renderer.prepare([prompt, *coming]) > 0  # the worker's trip for the next jobs
    candidates = [renderer.render(prompt, SEED + k) for k in range(3)]
    tagger(candidates)
    reference = Reference("gpu-test-realvis-vram", candidates[0].crop(face_box(candidates[0])), 0.1)
    redraw = renderer.render(prompt, SEED, reference)
    tagger([redraw])
    detailed = renderer.detail(redraw, prompt, SEED, reference, size=(1344, 1728))
    tagger([detailed])
    reserved = torch.cuda.max_memory_reserved() / 2**30
    allocated = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"RealVisXL group A job with a reference: peak reserved {reserved:.2f} GiB,"
        f" allocated {allocated:.2f}"
    )
    assert reserved < 8.0
    pipe = renderer.pipe
    for model in (pipe.text_encoder, pipe.text_encoder_2, pipe.image_encoder, tagger.model):
        assert next(model.parameters()).device.type == "cpu"
    assert pipe.unet.device.type == "cuda"
