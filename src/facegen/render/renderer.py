"""GPU rendering and its small CPU-testable interfaces; models load only on use."""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from facegen.core.prompt import Prompt
from facegen.render import quality_checks as qc
from facegen.render.gpu import on_gpu
from facegen.render.prompt_cache import EmbeddingCache
from facegen.render.prompt_cache import prepare as prepare_texts

if TYPE_CHECKING:
    from PIL.Image import Image

MODEL_REPO = "cagliostrolab/animagine-xl-4.0"
WIDTH, HEIGHT = 896, 1152
"""The render size: Animagine's portrait size nearest 3:4 (its model card lists 896x1152),
about SDXL's native million pixels. The master is cropped to exactly 3:4 (864x1152)."""
STEPS = 28
CFG = 5.5
SAMPLER = "euler_a"

DETAIL_STRENGTH = 0.35

IPA_REPO = "h94/IP-Adapter"
IPA_SUBFOLDER = "sdxl_models"
IPA_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IPA_ENCODER = "models/image_encoder"


@dataclass(frozen=True, slots=True)
class Reference:
    """The player's existing picture, as a reference for a re-draw that keeps the sheet."""

    key: str
    """Names the picture and crop, so the renderer embeds it once per job."""
    image: Image
    """The crop of the existing picture's face."""
    scale: float
    """The adapter's strength (`reference.scale`)."""


class Renderer(Protocol):
    model: str

    def render(self, prompt: Prompt, seed: int, reference: Reference | None = None) -> Image: ...

    def detail(
        self,
        image: Image,
        prompt: Prompt,
        seed: int,
        reference: Reference | None = None,
        *,
        size: tuple[int, int],
    ) -> Image:
        """The picture upscaled to `size` (width, height) and re-drawn lightly."""
        ...

    def settings(self) -> dict[str, Any]: ...


class Tagger(Protocol):
    name: str

    def __call__(self, images: Sequence[Image]) -> list[qc.TagResult]: ...


def _token_chunks(tokenizer: Any, text: str) -> list[list[int]]:
    # verbose=False: the "longer than 77 tokens" warning is expected; we chunk on purpose.
    ids = tokenizer(text, add_special_tokens=False, verbose=False).input_ids
    size = tokenizer.model_max_length - 2
    return [ids[i : i + size] for i in range(0, len(ids), size)] or [[]]


@dataclass(frozen=True, slots=True)
class TextEmbedding:
    """One prompt text through both text encoders, kept in system memory (fp16)."""

    hidden: Any
    """[1, 77 * chunks, 2048]: each chunk's penultimate layer, both encoders side by side."""
    pooled: Any
    """[1, 1280]: the second encoder's pooled output of the first chunk."""
    chunks: int


def _encode_text(pipe: Any, text: str) -> TextEmbedding:
    """SDXL prompt embedding without the 77-token cut-off, copied to system memory.

    Encode 75-token chunks independently through both encoders to retain kit and content
    terms beyond token 77. Embeddings do not depend on the paired text; `pair_embeddings`
    pads the shorter prompt with empty-text chunks afterwards.
    """
    import torch

    per_encoder = []
    pooled = None
    counts = []
    for tok, enc in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)):
        chunks = _token_chunks(tok, text)
        counts.append(len(chunks))
        hidden = []
        for k, chunk in enumerate(chunks):
            row = [tok.bos_token_id, *chunk, tok.eos_token_id]
            row += [tok.pad_token_id] * (tok.model_max_length - len(row))
            out = enc(torch.tensor([row], device=enc.device), output_hidden_states=True)
            hidden.append(out.hidden_states[-2])
            if enc is pipe.text_encoder_2 and k == 0:
                pooled = out[0]
        per_encoder.append(torch.cat(hidden, dim=1))
    if counts[0] != counts[1]:
        raise RuntimeError(f"the tokenizers split the prompt into {counts} chunks")
    return TextEmbedding(torch.cat(per_encoder, dim=-1).cpu(), pooled.cpu(), counts[0])


def pair_embeddings(
    positive: TextEmbedding, negative: TextEmbedding, empty: TextEmbedding
) -> tuple[Any, Any, Any, Any, int]:
    """A positive and a negative prompt as the pipeline takes them, with their chunk count.

    Pad both to the same chunk count using `empty` (start, end and padding tokens).
    This is bit-identical to encoding each text with the pair's chunk count.
    """
    import torch

    n = max(positive.chunks, negative.chunks)

    def padded(text: TextEmbedding) -> Any:
        return torch.cat([text.hidden, *[empty.hidden] * (n - text.chunks)], dim=1)

    return padded(positive), positive.pooled, padded(negative), negative.pooled, n


INT8_SCOPE = "transformer_blocks"
"""The UNet's Linear layers stored as INT8: the 700 in its transformer blocks (attention and
feed-forward), 4.1 of its 4.8 GiB in fp16."""

TEXT_CACHE_SIZE = 128
"""Prompt texts whose embeddings the renderer keeps in system memory (`prompt_cache`). One
75-token chunk is 77 x 2048 fp16 values, 0.3 MiB; with the usual two or three chunks per text
the full cache holds 80 to 120 MiB."""


class AnimagineRenderer:
    """Animagine XL 4.0 Opt in fp16 with its fp16-fix VAE; no compile warm-up.

    Optional fp16 accumulation and INT8 transformer weights change pictures slightly;
    disabling both gives the fp16 baseline. INT8 weights are rebuilt per layer. Text
    encoders can wait in CPU memory between encoding sessions without changing output.

    `prepare` batches upcoming prompts into one GPU visit. Up to TEXT_CACHE_SIZE texts
    stay cached in CPU memory; unprepared prompts are encoded on first use. Cached and
    uncached paths supply identical embeddings to the pipeline.
    """

    model = MODEL_REPO

    def __init__(
        self,
        detail_strength: float = DETAIL_STRENGTH,
        *,
        fp16_accumulation: bool = True,
        int8_weights: bool = True,
        offload_text_encoders: bool = True,
    ) -> None:
        from facegen.render.gpu import prepare_environment

        prepare_environment()
        import torch
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLPipeline,
        )

        self._torch = torch
        self.fp16_accumulation = fp16_accumulation
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_REPO, dtype=torch.float16, use_safetensors=True
        )
        self.int8_layers = 0
        if int8_weights:
            from facegen.render.quantise import quantise_linears

            # Layer by layer straight onto the GPU, before the rest of the model follows.
            self.int8_layers = quantise_linears(pipe.unet, INT8_SCOPE, "cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        self.offload_text_encoders = offload_text_encoders
        if offload_text_encoders:
            # The text encoders stay in system memory. diffusers 0.42 reports a split
            # pipeline's device as that of its GPU components, so it still runs on the GPU.
            pipe.unet.to("cuda")
            pipe.vae.to("cuda")
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        # diffusers 0.42's from_pipe casts the shared UNet to float32 without an explicit
        # dtype, doubling its weights and spilling into system RAM. Reuse fp16 components.
        self.img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        self.img2img.set_progress_bar_config(disable=True)
        if pipe.unet.dtype != torch.float16 or pipe.vae.dtype != torch.float16:
            raise RuntimeError(f"model not in float16: unet {pipe.unet.dtype}")
        self.detail_strength = detail_strength
        self.vae_tiling = False
        self._texts: EmbeddingCache[TextEmbedding] = EmbeddingCache(TEXT_CACHE_SIZE)
        # The current job's prompt on the GPU, so its renders do not copy it again.
        self._embeds: dict[tuple[str, str], tuple[Any, ...]] = {}
        # The IP-Adapter, loaded at the first reference: (its attention processors, its image
        # projection), the UNet's own processors, and whether the adapter is switched in.
        self._adapter: tuple[dict[str, Any], Any] | None = None
        self._plain_processors: dict[str, Any] = {}
        self._adapter_on = False
        self._reference_embeds: dict[str, Any] = {}

    @staticmethod
    def _texts_of(prompts: Iterable[Prompt]) -> list[str]:
        # The empty text pads the shorter prompt of a pair (pair_embeddings).
        return ["", *(t for p in prompts for t in (p.positive, p.negative))]

    @contextmanager
    def _encoders_on_gpu(self) -> Iterator[None]:
        encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
        with self._torch.no_grad(), on_gpu(*encoders, enabled=self.offload_text_encoders):
            yield

    def prepare(self, prompts: Iterable[Prompt]) -> int:
        """Encode every text of these prompts not cached yet, in one trip of the text encoders.

        The first stage of drawing (`prompt_cache`): the worker passes the next jobs' prompts,
        so their renders and detail passes need no trip of their own. Returns how many texts
        were encoded; with 0 the encoders did not move.
        """
        return prepare_texts(
            self._texts,
            self._texts_of(prompts),
            lambda text: _encode_text(self.pipe, text),
            self._encoders_on_gpu,
        )

    def encoded(self, prompt: Prompt) -> bool:
        """Whether the prompt is cached, so drawing it moves no text encoder."""
        return not self._texts.missing(self._texts_of([prompt]))

    def _embeddings(self, prompt: Prompt) -> tuple[Any, ...]:
        """The prompt's embeddings on the GPU, from the cache; encoded first if not there."""
        key = (prompt.positive, prompt.negative)
        if key not in self._embeds:
            self.prepare([prompt])
            positive, negative, empty = (self._texts.get(t) for t in (*key, ""))
            *tensors, n = pair_embeddings(positive, negative, empty)
            on_card = tuple(t.to("cuda") for t in tensors)
            self._embeds = {key: (*on_card, n)}
        return self._embeds[key]

    def prompt_chunks(self, prompt: Prompt) -> int:
        """How many 75-token chunks the prompt is encoded in; needs no encoder."""
        return max(
            len(_token_chunks(self.pipe.tokenizer, t)) for t in (prompt.positive, prompt.negative)
        )

    def _load_adapter(self) -> None:
        """Load IP-Adapter into the shared UNet; the detail pipeline shares it too.

        The image encoder (ViT-H, 1.3 GB) is needed once per job, to embed the reference, so
        it waits in system memory in between: with it on the card, a detail pass reached
        12.9 GiB of the 16 GB that Windows also uses (docs/research.md section 5).
        """
        unet = self.pipe.unet
        self._plain_processors = dict(unet.attn_processors)
        self.pipe.load_ip_adapter(
            IPA_REPO,
            subfolder=IPA_SUBFOLDER,
            weight_name=IPA_WEIGHT,
            image_encoder_folder=IPA_ENCODER,
        )
        self.pipe.image_encoder.to("cpu")
        self._adapter = (dict(unet.attn_processors), unet.encoder_hid_proj)
        self._adapter_on = True

    def _switch_adapter(self, on: bool) -> None:
        """Switch the loaded adapter in or out of the UNet without reloading it.

        Out, the UNet is exactly the plain model again, so pictures drawn without a
        reference are the same as before the adapter was loaded. This is what diffusers'
        `unload_ip_adapter` does, minus dropping the weights.
        """
        if self._adapter is None or on == self._adapter_on:
            return
        unet = self.pipe.unet
        processors, projection = self._adapter
        # The plus adapter's image projection has attention layers of its own, counted among
        # the UNet's: attach or detach it before the processors are swapped. And
        # set_attn_processor takes the processors out of the dict it is given: pass a copy.
        unet.encoder_hid_proj = projection if on else None
        unet.config.encoder_hid_dim_type = "ip_image_proj" if on else None
        unet.set_attn_processor(dict(processors if on else self._plain_processors))
        self._adapter_on = on

    def _conditioning(self, reference: Reference | None) -> dict[str, Any]:
        """Extra pipeline arguments for a reference picture; switches the adapter in or out."""
        if reference is None:
            self._switch_adapter(False)
            return {}
        if self._adapter is None:
            self._load_adapter()
        self._switch_adapter(True)
        self.pipe.set_ip_adapter_scale(reference.scale)
        if reference.key not in self._reference_embeds:
            with self._torch.no_grad(), on_gpu(self.pipe.image_encoder):
                embeds = self.pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=reference.image.convert("RGB"),
                    ip_adapter_image_embeds=None,
                    device=self._torch.device("cuda"),
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
            self._reference_embeds = {reference.key: embeds}
        return {"ip_adapter_image_embeds": self._reference_embeds[reference.key]}

    def set_vae_tiling(self, on: bool) -> None:
        """Decode in tiles and one picture at a time (`render.vae_tiling`): less
        graphics memory on a small card, a little slower. The worker sets it before each job
        from the settings, so a change applies to the next picture."""
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
    def _fp16_accumulation(self) -> Iterator[None]:
        """Let fp16 matmuls accumulate in fp16 while the pipeline runs.

        GeForce cards run fp16 tensor cores at half rate when they accumulate in fp32,
        PyTorch's default: one UNet call goes from 184 to 145 ms, and the noise prediction
        moves less than bf16 would move it (docs/research.md, run `inference`). The flag is
        process-wide, so it is set only around the pipeline: the text encoders, the
        IP-Adapter image encoder and the tagger keep fp32 accumulation. With
        `fp16_accumulation` off the flag is left as it is.
        """
        if not self.fp16_accumulation:
            yield
            return
        matmul = self._torch.backends.cuda.matmul
        before = matmul.allow_fp16_accumulation
        matmul.allow_fp16_accumulation = True
        try:
            yield
        finally:
            matmul.allow_fp16_accumulation = before

    def render(self, prompt: Prompt, seed: int, reference: Reference | None = None) -> Image:
        pe, pp, ne, npool, _ = self._embeddings(prompt)
        extra = self._conditioning(reference)
        generator = self._torch.Generator("cuda").manual_seed(seed)
        with self._torch.no_grad(), self._fp16_accumulation():
            return self.pipe(
                prompt_embeds=pe,
                pooled_prompt_embeds=pp,
                negative_prompt_embeds=ne,
                negative_pooled_prompt_embeds=npool,
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                generator=generator,
                **extra,
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
        same prompt, seed and reference."""
        from PIL import Image as PILImage

        width, height = size
        if width % 64 or height % 64:
            raise ValueError(f"the detail size must be multiples of 64, got {width}x{height}")
        big = image.resize((width, height), PILImage.Resampling.LANCZOS)
        pe, pp, ne, npool, _ = self._embeddings(prompt)
        extra = self._conditioning(reference)
        generator = self._torch.Generator("cuda").manual_seed(seed)
        with self._torch.no_grad(), self._fp16_accumulation():
            return self.img2img(
                prompt_embeds=pe,
                pooled_prompt_embeds=pp,
                negative_prompt_embeds=ne,
                negative_pooled_prompt_embeds=npool,
                image=big,
                strength=self.detail_strength,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                generator=generator,
                **extra,
            ).images[0]

    def settings(self) -> dict[str, Any]:
        return {
            "width": WIDTH,
            "height": HEIGHT,
            "steps": STEPS,
            "cfg": CFG,
            "sampler": SAMPLER,
            "dtype": "float16",
            "matmul_accumulation": "float16" if self.fp16_accumulation else "float32",
            "unet_weights": (
                f"int8 per output channel in {self.int8_layers} {INT8_SCOPE} Linear layers,"
                " rebuilt in float16 per call"
                if self.int8_layers
                else "float16"
            ),
            "vae": f"{MODEL_REPO}/vae (fp16-fix)",
            "vae_tiling": self.vae_tiling,
            "prompt_encoding": "75-token chunks",
        }

    def detail_settings(self) -> dict[str, Any]:
        return {"strength": self.detail_strength, "steps": STEPS}

    def reference_settings(self) -> dict[str, Any]:
        return {
            "adapter": f"{IPA_REPO}/{IPA_SUBFOLDER}/{IPA_WEIGHT}",
            "image_encoder": f"{IPA_REPO}/{IPA_ENCODER}",
        }

    def close(self) -> None:
        """Free the GPU memory the model holds. Never raises: after a CUDA fault even
        `empty_cache` fails, and the references are dropped all the same."""
        self.pipe = self.img2img = None
        self._texts.clear()
        self._embeds = {}
        self._adapter = None
        self._plain_processors = {}
        self._reference_embeds = {}
        gc.collect()
        with contextlib.suppress(Exception):
            self._torch.cuda.empty_cache()

    def memory(self) -> dict[str, float]:
        cuda = self._torch.cuda
        return {
            "peak_allocated_gib": round(cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(cuda.max_memory_reserved() / 2**30, 2),
        }
