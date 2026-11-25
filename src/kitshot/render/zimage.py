"""Z-Image-Turbo, the realistic diffusion transformer: NF4 blocks, an INT8 Qwen3 text encoder.

Everything comes from one repository, `Tongyi-MAI/Z-Image-Turbo`. The transformer is
published in FP32; it is built on the GPU tensor by tensor, its 238 block Linear layers as NF4
(render/quantise.py) and the rest as BF16, so system memory never holds it whole. Qwen3-4B's
decoder layers become INT8 on the GPU one by one, for each trip that encodes prompts; between
trips it is not kept (user, 2026-11-24: 4.2 GiB of system memory otherwise), only the
embeddings are.
The encoder (4.5 GiB on the card) and the transformer (3.4 GiB) together would pass 8 GiB, so
the transformer leaves the card for the encoder's trip.

The card's sampling: 9 steps (8 transformer passes) without classifier-free guidance, so the
prompt has no negative: the positive prompt states the clothing and the check is the gate.
diffusers has no IP-Adapter for Z-Image: a re-draw cannot take a reference picture.
Measurements: docs/research.md section 20.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from kitshot.core.prompt import Prompt
from kitshot.core.settings_values import IMAGE_MODELS
from kitshot.platform.memory import release_free_memory
from kitshot.render.gpu import on_gpu
from kitshot.render.prompt_cache import EmbeddingCache
from kitshot.render.prompt_cache import prepare as prepare_texts
from kitshot.render.renderer import (
    DETAIL_STRENGTH,
    HEIGHT,
    TEXT_CACHE_SIZE,
    WIDTH,
    Reference,
    collect_freed,
)

if TYPE_CHECKING:
    from PIL.Image import Image

ZIMAGE_REPO = IMAGE_MODELS["zimage_turbo"].repo
STEPS = 9
"""The card's 9 steps: 8 transformer passes."""
GUIDANCE = 0.0
"""Distilled without classifier-free guidance: no negative prompt."""
SAMPLER = "flow_match_euler"
MAX_TOKENS = 512
"""The pipeline's limit; it would cut a longer prompt silently, so a longer one is an error."""
TEXT_SCOPE = "layers."
"""Qwen3-4B's Linear layers stored as INT8: all of them sit in its 36 decoder layers."""


def load_transformer_nf4(device: str = "cuda") -> tuple[Any, int]:
    """The transformer on `device`, built tensor by tensor from its FP32 shards: the block
    Linear layers (`quantise.block_linears`) in NF4, everything else in BF16. Returns it and
    how many layers are NF4.

    `from_pretrained` would first hold all 12 GB of BF16 weights in system memory (a private
    peak of 12.2 GiB on this PC, docs/research.md section 20); here one tensor at a time is.
    Each weight is rounded to BF16 as `from_pretrained(dtype=torch.bfloat16)` rounds it, so
    the model is the same.
    """
    import json
    from pathlib import Path

    import torch
    from accelerate import init_empty_weights
    from diffusers import ZImageTransformer2DModel
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from kitshot.render.quantise import NF4_GROUP, Nf4Linear, block_linears

    def fetch(name: str) -> str:
        return hf_hub_download(ZIMAGE_REPO, f"transformer/{name}", local_files_only=True)

    config = ZImageTransformer2DModel.load_config(
        ZIMAGE_REPO, subfolder="transformer", local_files_only=True
    )
    with init_empty_weights():  # parameters on the meta device; buffers are real
        model = ZImageTransformer2DModel.from_config(config)
    weight_map = json.loads(
        Path(fetch("diffusion_pytorch_model.safetensors.index.json")).read_text("utf-8")
    )["weight_map"]
    names = {name for name, _ in model.named_parameters()}
    if set(weight_map) != names:
        raise RuntimeError(
            f"the checkpoint does not match the model: {len(set(weight_map) ^ names)} names differ"
        )
    nf4 = set(block_linears(model, NF4_GROUP))
    for shard in sorted(set(weight_map.values())):
        with safe_open(fetch(shard), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
                owner, _, attr = key.rpartition(".")
                module = model.get_submodule(owner)
                tensor = f.get_tensor(key).to(torch.bfloat16).to(device)
                module._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
                if owner in nf4 and not any(p.is_meta for p in module.parameters()):
                    parent, _, child = owner.rpartition(".")
                    setattr(model.get_submodule(parent), child, Nf4Linear(module, device))
    model.to(device)  # the buffers
    left = [n for n, p in model.named_parameters() if p.is_meta]
    if left:
        raise RuntimeError(f"{len(left)} weights were not in the checkpoint, such as {left[0]}")
    return model.eval(), len(nf4)


class ZImageRenderer:
    """Z-Image-Turbo with NF4 transformer blocks, BF16 compute and a BF16 VAE.

    `prepare` encodes upcoming prompts in one trip of the text encoder; the embeddings (one
    variable-length tensor per text) wait in system memory, up to TEXT_CACHE_SIZE texts.
    The detail pass always decodes in tiles: untiled, 1344x1728 passes 9 GiB.

    With `release_text_encoder` (the default), each trip reads the text encoder from disk and
    makes it INT8 on the card, and lets go of it afterwards; off, the INT8 encoder waits in
    system memory (`offload_text_encoder`) or on the card. Its weights are the same either way.
    """

    model = ZIMAGE_REPO
    reference_adapter = False
    """No IP-Adapter for Z-Image: re-draws are drawn without a reference."""
    negative_prompt = False
    """Without guidance the negative prompt is not used."""

    def __init__(
        self,
        detail_strength: float = DETAIL_STRENGTH,
        *,
        offload_text_encoder: bool = True,
        release_text_encoder: bool = True,
        transformer_off_for_encoding: bool = True,
        detail_vae_tiling: bool = True,
    ) -> None:
        from kitshot.render.gpu import prepare_environment

        prepare_environment()
        import torch
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImageImg2ImgPipeline,
            ZImagePipeline,
        )

        self._torch = torch
        bf16 = torch.bfloat16
        # The tokenizer and the scheduler, without a transformer, VAE or text encoder.
        # local_files_only: sharded weights look up the Hub otherwise, even with HF_HUB_OFFLINE.
        self.encoder = ZImagePipeline.from_pretrained(
            ZIMAGE_REPO,
            transformer=None,
            vae=None,
            text_encoder=None,
            dtype=bf16,
            local_files_only=True,
        )
        self.encoder.set_progress_bar_config(disable=True)
        self.offload_text_encoder = offload_text_encoder
        self.release_text_encoder = release_text_encoder and offload_text_encoder
        self.int8_text_layers = 0
        """Counted as the text encoder loads. A released one first loads for the first prompt's
        trip, before anything is drawn, so a picture's settings always have the count."""
        if not self.release_text_encoder:
            # INT8 on the card before the transformer arrives there; then it waits.
            text_encoder = self._load_text_encoder()
            text_encoder.to("cpu" if offload_text_encoder else "cuda")
            self.encoder.text_encoder = text_encoder
            gc.collect()
            torch.cuda.empty_cache()

        transformer, self.nf4_layers = load_transformer_nf4("cuda")
        gc.collect()

        vae = AutoencoderKL.from_pretrained(
            ZIMAGE_REPO, subfolder="vae", dtype=bf16, local_files_only=True
        ).to("cuda")
        parts = {"vae": vae, "text_encoder": None, "tokenizer": None, "transformer": transformer}
        self.pipe = ZImagePipeline(scheduler=self.encoder.scheduler, **parts)
        self.pipe.set_progress_bar_config(disable=True)
        # A scheduler of its own: the detail pass sets its timesteps from the strength.
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.encoder.scheduler.config)
        self.img2img = ZImageImg2ImgPipeline(scheduler=scheduler, **parts)
        self.img2img.set_progress_bar_config(disable=True)
        if transformer.dtype != bf16 or vae.dtype != bf16:
            raise RuntimeError(f"model not in bfloat16: transformer {transformer.dtype}")

        self.transformer_off_for_encoding = transformer_off_for_encoding and offload_text_encoder
        self.detail_strength = detail_strength
        self.detail_vae_tiling = detail_vae_tiling
        self.vae_tiling = False
        self._texts: EmbeddingCache[Any] = EmbeddingCache(TEXT_CACHE_SIZE)
        # The current job's embedding on the GPU, so its renders do not copy it again.
        self._embeds: dict[str, Any] = {}

    def _load_text_encoder(self) -> Any:
        """Qwen3-4B read from disk as the whole repository is read (memory-mapped, so it costs
        little private memory), its decoder layers made INT8 on the card one by one; the rest
        stays in system memory. Only the encoder loads: the tokenizer and scheduler are passed."""
        from diffusers import ZImagePipeline

        from kitshot.render.quantise import quantise_linears

        part = ZImagePipeline.from_pretrained(
            ZIMAGE_REPO,
            transformer=None,
            vae=None,
            tokenizer=self.encoder.tokenizer,
            scheduler=self.encoder.scheduler,
            dtype=self._torch.bfloat16,
            local_files_only=True,
        )
        text_encoder = part.text_encoder.eval()
        self.int8_text_layers = quantise_linears(text_encoder, TEXT_SCOPE, "cuda")
        return text_encoder

    @contextmanager
    def _text_encoder_loaded(self) -> Iterator[None]:
        """A released text encoder, read for one trip onto the card and let go of after."""
        try:
            self.encoder.text_encoder = self._load_text_encoder().to("cuda")
            yield
        finally:
            self.encoder.text_encoder = None
            collect_freed(self._torch)

    @contextmanager
    def _encoder_on_gpu(self) -> Iterator[None]:
        """The text encoder on the card for one trip; the transformer waits in system memory
        meanwhile, so the two never sit on the card together."""
        transformer = self.pipe.transformer
        with self._torch.no_grad():
            if self.transformer_off_for_encoding:
                transformer.to("cpu")
            try:
                if self.release_text_encoder:
                    with self._text_encoder_loaded():
                        yield
                else:
                    with on_gpu(self.encoder.text_encoder, enabled=self.offload_text_encoder):
                        yield
            finally:
                if self.transformer_off_for_encoding:
                    transformer.to("cuda")
                release_free_memory()  # the transformer's copy in system memory

    def prompt_tokens(self, prompt: Prompt) -> int:
        """The prompt's length in Qwen3 tokens, chat template included; needs no encoder."""
        tokenizer = self.encoder.tokenizer
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.positive}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return len(tokenizer(text).input_ids)

    def _encode(self, text: str) -> Any:
        """One text's embedding (penultimate hidden state, one row per token), in system
        memory. The pipeline's own encoding; it pads to MAX_TOKENS and drops the padding."""
        tokens = self.prompt_tokens(Prompt(text, ""))
        if tokens > MAX_TOKENS:
            raise ValueError(f"the prompt is {tokens} tokens; Z-Image-Turbo reads {MAX_TOKENS}")
        (embedding,) = self.encoder._encode_prompt(
            [text], device=self._torch.device("cuda"), max_sequence_length=MAX_TOKENS
        )
        return embedding.cpu()

    def prepare(self, prompts: Iterable[Prompt]) -> int:
        """Encode every prompt not cached yet, in one trip of the text encoder; returns how
        many texts were encoded (with 0 nothing moved). The negative is not used."""
        texts = [p.positive for p in prompts]
        return prepare_texts(self._texts, texts, self._encode, self._encoder_on_gpu)

    def encoded(self, prompt: Prompt) -> bool:
        """Whether the prompt is cached, so drawing it moves no text encoder."""
        return not self._texts.missing([prompt.positive])

    def _embeddings(self, prompt: Prompt) -> list[Any]:
        """The prompt's embedding on the GPU, as the pipeline takes it; encoded if needed."""
        text = prompt.positive
        if text not in self._embeds:
            self.prepare([prompt])
            self._embeds = {text: self._texts.get(text).to("cuda")}
        return [self._embeds[text]]

    def set_vae_tiling(self, on: bool) -> None:
        """Decode in tiles and one picture at a time (`render.vae_tiling`), for renders; the
        detail pass tiles regardless (`detail_vae_tiling`)."""
        if on == self.vae_tiling:
            return
        vae = self.pipe.vae
        if on:
            vae.enable_tiling()
            vae.enable_slicing()
        else:
            vae.disable_tiling()
            vae.disable_slicing()
        self.vae_tiling = on

    @contextmanager
    def _detail_tiling(self) -> Iterator[None]:
        before = self.vae_tiling
        self.set_vae_tiling(before or self.detail_vae_tiling)
        try:
            yield
        finally:
            self.set_vae_tiling(before)

    @staticmethod
    def _no_reference(reference: Reference | None) -> None:
        if reference is not None:
            raise ValueError("Z-Image-Turbo has no reference adapter; draw without a reference")

    def render(
        self,
        prompt: Prompt,
        seed: int,
        reference: Reference | None = None,
        *,
        size: tuple[int, int] = (WIDTH, HEIGHT),
    ) -> Image:
        self._no_reference(reference)
        embeds = self._embeddings(prompt)
        generator = self._torch.Generator("cuda").manual_seed(seed)
        with self._torch.no_grad():
            return self.pipe(
                prompt_embeds=embeds,
                width=size[0],
                height=size[1],
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                cfg_normalization=False,
                generator=generator,
            ).images[0]

    def detail(
        self,
        image: Image,
        prompt: Prompt,
        seed: int,
        reference: Reference | None = None,
        *,
        size: tuple[int, int],
    ) -> Image:
        """Upscale to `size` (width, height; multiples of 64) and re-draw lightly, with the
        same prompt and seed; the VAE works in tiles."""
        from PIL import Image as PILImage

        self._no_reference(reference)
        width, height = size
        if width % 64 or height % 64:
            raise ValueError(f"the detail size must be multiples of 64, got {width}x{height}")
        big = image.resize((width, height), PILImage.Resampling.LANCZOS)
        embeds = self._embeddings(prompt)
        generator = self._torch.Generator("cuda").manual_seed(seed)
        with self._torch.no_grad(), self._detail_tiling():
            return self.img2img(
                prompt_embeds=embeds,
                image=big,
                strength=self.detail_strength,
                width=width,
                height=height,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                cfg_normalization=False,
                generator=generator,
            ).images[0]

    def settings(self) -> dict[str, Any]:
        return {
            "width": WIDTH,
            "height": HEIGHT,
            "steps": STEPS,
            "cfg": GUIDANCE,
            "sampler": SAMPLER,
            "dtype": "bfloat16",
            "negative_prompt": "not used (no guidance)",
            "transformer_weights": (
                f"nf4 (NormalFloat, one absmax per 64 inputs) in {self.nf4_layers} block"
                " Linear layers, rebuilt in bfloat16 per call"
            ),
            "text_encoder": (
                f"Qwen3-4B, int8 per output channel in {self.int8_text_layers} Linear layers"
            ),
            "vae": f"{ZIMAGE_REPO}/vae (bfloat16)",
            "vae_tiling": self.vae_tiling,
            "prompt_encoding": f"Qwen3 chat template, penultimate layer, up to {MAX_TOKENS} tokens",
        }

    def detail_settings(self) -> dict[str, Any]:
        return {
            "strength": self.detail_strength,
            "steps": STEPS,
            "vae_tiling": self.vae_tiling or self.detail_vae_tiling,
        }

    def close(self) -> None:
        """Free the GPU memory the model holds. Never raises (see SdxlRenderer.close)."""
        self.pipe = self.img2img = self.encoder = None
        self._texts.clear()
        self._embeds = {}
        gc.collect()
        with contextlib.suppress(Exception):
            self._torch.cuda.empty_cache()

    def memory(self) -> dict[str, float]:
        cuda = self._torch.cuda
        return {
            "peak_allocated_gib": round(cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(cuda.max_memory_reserved() / 2**30, 2),
        }
