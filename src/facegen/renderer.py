"""GPU rendering and its small CPU-testable interfaces; models load only on use."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from facegen import qc
from facegen.gpu import on_gpu
from facegen.prompt import Prompt
from facegen.prompt_cache import EmbeddingCache
from facegen.prompt_cache import prepare as prepare_texts

if TYPE_CHECKING:
    from PIL.Image import Image

MODEL_REPO = "cagliostrolab/animagine-xl-4.0"
WIDTH, HEIGHT = 896, 1152
"""The render size: Animagine's portrait size nearest 3:4 (its model card lists 896x1152),
about SDXL's native million pixels. The master is cropped to exactly 3:4 (864x1152)."""
STEPS = 28
CFG = 5.5
SAMPLER = "euler_a"

# Group A's detail pass: the chosen picture upscaled and re-drawn lightly at a higher
# resolution, then downscaled when published (decisions.md, proposed 2026-11-11).
DETAIL_SCALE = 1.5
DETAIL_STRENGTH = 0.35

# The reference picture (spec section 5; bake-off run r4): IP-Adapter plus-face for SDXL with
# its ViT-H image encoder, from the Hugging Face cache.
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
        self, image: Image, prompt: Prompt, seed: int, reference: Reference | None = None
    ) -> Image: ...

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

    Splits the text into 75-token chunks, encodes each through both text encoders and joins
    them, as ComfyUI and A1111 do. The stock pipeline silently drops everything after token
    77, which would cut the kit and the content terms from a long prompt. Each chunk is
    encoded on its own, so a text's embedding does not depend on the text it is paired with:
    `pair_embeddings` pads the shorter one with the empty text's chunk afterwards.
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

    Both are padded to the same number of chunks with `empty`, the empty text's embedding (one
    chunk of nothing but start, end and padding tokens). That is exactly what encoding each
    text with the pair's chunk count would give, chunk for chunk, so the result is the same
    to the bit as encoding the pair together.
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
    """Animagine XL 4.0 Opt in fp16 with its own fp16-fix VAE (docs/research.md section 5).

    Three speed and memory settings, all on by default (user, 2026-11-11; docs/research.md,
    run `inference`), with no torch.compile and so no warm-up:
    - `fp16_accumulation`: fp16 matmuls accumulate in fp16 while the pipeline runs.
    - `int8_weights`: the UNet's transformer-block Linear layers are stored as INT8 and
      rebuilt in fp16 per layer (`facegen.quantise`): the UNet's weights go from 4.8 to
      2.8 GiB.
    - `offload_text_encoders`: the text encoders (1.5 GiB) wait in system memory and come
      onto the GPU only to encode prompts that are not cached yet. With the tagger doing the
      same, a group A job's peak reserved memory stays under 8 GiB, so FM27 and a browser
      fit beside it.
    The first two change the pictures slightly; offloading does not. With the first two off
    the renderer draws exactly as before; the spikes use that for their baseline.

    Prompt embeddings are kept in system memory, TEXT_CACHE_SIZE texts at most
    (`facegen.prompt_cache`). The worker calls `prepare` with the next jobs' prompts, so one
    trip of the text encoders serves many jobs; a prompt nobody prepared is encoded when it
    is first drawn. The cache changes no pixel: the pipeline gets the same embeddings.
    """

    model = MODEL_REPO

    def __init__(
        self,
        detail_scale: float = DETAIL_SCALE,
        detail_strength: float = DETAIL_STRENGTH,
        *,
        fp16_accumulation: bool = True,
        int8_weights: bool = True,
        offload_text_encoders: bool = True,
    ) -> None:
        from facegen.gpu import prepare_environment

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
            from facegen.quantise import quantise_linears

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
        # Share the loaded modules. Not `from_pipe`: in diffusers 0.42 it casts the shared
        # UNet to float32 when no dtype is given, which doubles the weights to 12.9 GiB and
        # pushes the card into system RAM (docs/research.md section 5).
        self.img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        self.img2img.set_progress_bar_config(disable=True)
        if pipe.unet.dtype != torch.float16 or pipe.vae.dtype != torch.float16:
            raise RuntimeError(f"model not in float16: unet {pipe.unet.dtype}")
        self.detail_scale = detail_scale
        self.detail_strength = detail_strength
        self._texts: EmbeddingCache[TextEmbedding] = EmbeddingCache(TEXT_CACHE_SIZE)
        # The current job's prompt on the GPU, so its renders do not copy it again.
        self._embeds: dict[tuple[str, str], tuple[Any, ...]] = {}
        # The IP-Adapter, loaded at the first reference: (its attention processors, its image
        # projection), the UNet's own processors, and whether the adapter is switched in.
        self._adapter: tuple[dict[str, Any], Any] | None = None
        self._plain_processors: dict[str, Any] = {}
        self._adapter_on = False
        self._reference_embeds: dict[str, Any] = {}

    # -- the prompt --

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
            self.prepare([prompt])  # nothing to do once the worker has prepared it
            positive, negative, empty = (self._texts.get(t) for t in (*key, ""))
            *tensors, n = pair_embeddings(positive, negative, empty)
            on_card = tuple(t.to("cuda") for t in tensors)
            self._embeds = {key: (*on_card, n)}  # one prompt per job
        return self._embeds[key]

    def prompt_chunks(self, prompt: Prompt) -> int:
        """How many 75-token chunks the prompt is encoded in; needs no encoder."""
        return max(
            len(_token_chunks(self.pipe.tokenizer, t)) for t in (prompt.positive, prompt.negative)
        )

    # -- the reference picture --

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
            self._reference_embeds = {reference.key: embeds}  # one reference per job
        return {"ip_adapter_image_embeds": self._reference_embeds[reference.key]}

    # -- drawing --

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
        self, image: Image, prompt: Prompt, seed: int, reference: Reference | None = None
    ) -> Image:
        """Upscale and re-draw lightly, with the same prompt, seed and reference."""
        from PIL import Image as PILImage

        size = tuple(round(side * self.detail_scale / 64) * 64 for side in image.size)
        big = image.resize(size, PILImage.Resampling.LANCZOS)
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
            "prompt_encoding": "75-token chunks",
        }

    def detail_settings(self) -> dict[str, Any]:
        return {"scale": self.detail_scale, "strength": self.detail_strength, "steps": STEPS}

    def reference_settings(self) -> dict[str, Any]:
        return {
            "adapter": f"{IPA_REPO}/{IPA_SUBFOLDER}/{IPA_WEIGHT}",
            "image_encoder": f"{IPA_REPO}/{IPA_ENCODER}",
        }

    def close(self) -> None:
        """Free the GPU memory the model holds."""
        self.pipe = self.img2img = None
        self._texts.clear()
        self._embeds = {}
        self._adapter = None
        self._plain_processors = {}
        self._reference_embeds = {}
        gc.collect()
        self._torch.cuda.empty_cache()

    def memory(self) -> dict[str, float]:
        cuda = self._torch.cuda
        return {
            "peak_allocated_gib": round(cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(cuda.max_memory_reserved() / 2**30, 2),
        }
