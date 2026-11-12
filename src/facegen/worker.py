"""GPU job runner (docs/architecture.md section 6).

For each queued job: load or sample the character sheet, build the prompt, render the
group's candidates, check each one, keep the best that passed, run group A's detail pass,
save the master under data/masters/<career key>/<unique_id>/, and record the picture. A job
whose candidates all fail the check three times goes to review (status "failed") and
nothing of it is published.

Pictures are 3:4 portraits (picture.py): the model renders at its portrait size near 3:4
and the master is cropped to exactly 3:4.

The model and the tagger sit behind the small `Renderer` and `Tagger` interfaces; the
runner itself needs neither a GPU nor a save, so the tests drive it with fakes.

Rules carried from the spec:
- the GPU works only while FM27 (fm.exe) is not running, unless told to generate now;
- jobs are idempotent: an interrupted job goes back to the queue and is redone from the
  same seeds;
- a re-draw that keeps the character sheet uses the player's current picture as a reference
  where `planner.reference_scale` says so (spec section 5): IP-Adapter's face model on a
  crop of that picture's face. Without a picture file to use, it is drawn without one.
"""

from __future__ import annotations

import fcntl
import gc
import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from facegen import qc
from facegen.age import age_wording
from facegen.db import ACTIVE_CAREER_ORDER, transaction
from facegen.gpu import on_gpu
from facegen.kits import KitTable
from facegen.picture import crop_to_aspect, face_box
from facegen.planner import reference_scale
from facegen.prompt import Prompt, build_prompt
from facegen.publish import PublishReport, publish_pack
from facegen.settings import Settings
from facegen.traits import (
    CharacterSheet,
    load_sheet,
    pick_pose,
    picture_seed,
    sample_sheet,
    sheet_for,
)

if TYPE_CHECKING:
    from PIL.Image import Image

MODEL_REPO = "cagliostrolab/animagine-xl-4.0"
WIDTH, HEIGHT = 896, 1152
"""The render size: Animagine's portrait size nearest 3:4 (its model card lists 896x1152),
about SDXL's native million pixels. The master is cropped to exactly 3:4 (864x1152)."""
STEPS = 28
CFG = 5.5
SAMPLER = "euler_a"
MAX_ATTEMPTS = 3
"""Rounds of candidates before a job goes to review (spec: three failed checks)."""

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


# --- FM27 ------------------------------------------------------------------------------------


def fm_running() -> bool | None:
    """Whether FM27 (fm.exe) runs on Windows; None if it cannot be told."""
    try:
        out = subprocess.run(
            ["tasklist.exe", "/FI", "IMAGENAME eq fm.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return b'"fm.exe"' in out.lower()


# --- the diffusers renderer (GPU) --------------------------------------------------------------


def _token_chunks(tokenizer: Any, text: str) -> list[list[int]]:
    # verbose=False: the "longer than 77 tokens" warning is expected; we chunk on purpose.
    ids = tokenizer(text, add_special_tokens=False, verbose=False).input_ids
    size = tokenizer.model_max_length - 2
    return [ids[i : i + size] for i in range(0, len(ids), size)] or [[]]


def _encode(pipe: Any, text: str, n_chunks: int) -> tuple[Any, Any]:
    """SDXL prompt embedding without the 77-token cut-off.

    Splits the prompt into 75-token chunks, encodes each through both text encoders and
    joins them, as ComfyUI and A1111 do. The stock pipeline silently drops everything after
    token 77, which would cut the kit and the content terms from a long prompt.
    """
    import torch

    per_encoder = []
    pooled = None
    for tok, enc in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)):
        chunks = _token_chunks(tok, text)
        chunks += [[]] * (n_chunks - len(chunks))
        hidden = []
        for k, chunk in enumerate(chunks):
            row = [tok.bos_token_id, *chunk, tok.eos_token_id]
            row += [tok.pad_token_id] * (tok.model_max_length - len(row))
            out = enc(torch.tensor([row], device=enc.device), output_hidden_states=True)
            hidden.append(out.hidden_states[-2])
            if enc is pipe.text_encoder_2 and k == 0:
                pooled = out[0]
        per_encoder.append(torch.cat(hidden, dim=1))
    return torch.cat(per_encoder, dim=-1), pooled


INT8_SCOPE = "transformer_blocks"
"""The UNet's Linear layers stored as INT8: the 700 in its transformer blocks (attention and
feed-forward), 4.1 of its 4.8 GiB in fp16."""


class AnimagineRenderer:
    """Animagine XL 4.0 Opt in fp16 with its own fp16-fix VAE (docs/research.md section 5).

    Three speed and memory settings, all on by default (user, 2026-11-11; docs/research.md,
    run `inference`), with no torch.compile and so no warm-up:
    - `fp16_accumulation`: fp16 matmuls accumulate in fp16 while the pipeline runs.
    - `int8_weights`: the UNet's transformer-block Linear layers are stored as INT8 and
      rebuilt in fp16 per layer (`facegen.quantise`): the UNet's weights go from 4.8 to
      2.8 GiB.
    - `offload_text_encoders`: the text encoders (1.5 GiB) wait in system memory and come
      onto the GPU only to encode a job's prompt. With the tagger doing the same, a group A
      job's peak reserved memory stays under 8 GiB, so FM27 and a browser fit beside it.
    The first two change the pictures slightly; offloading does not. With the first two off
    the renderer draws exactly as before; the spikes use that for their baseline.
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
        self._embeds: dict[tuple[str, str], tuple[Any, ...]] = {}
        # The IP-Adapter, loaded at the first reference: (its attention processors, its image
        # projection), the UNet's own processors, and whether the adapter is switched in.
        self._adapter: tuple[dict[str, Any], Any] | None = None
        self._plain_processors: dict[str, Any] = {}
        self._adapter_on = False
        self._reference_embeds: dict[str, Any] = {}

    def _embeddings(self, prompt: Prompt) -> tuple[Any, ...]:
        key = (prompt.positive, prompt.negative)
        if key not in self._embeds:
            encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
            with self._torch.no_grad(), on_gpu(*encoders, enabled=self.offload_text_encoders):
                n = max(len(_token_chunks(self.pipe.tokenizer, t)) for t in key)
                pe, pp = _encode(self.pipe, prompt.positive, n)
                ne, npool = _encode(self.pipe, prompt.negative, n)
            self._embeds = {key: (pe, pp, ne, npool, n)}  # one prompt per job
        return self._embeds[key]

    def prompt_chunks(self, prompt: Prompt) -> int:
        return self._embeddings(prompt)[4]

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


# --- the runner --------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class WorkerBusy(RuntimeError):
    """Another worker holds the lock."""


class JobGone(RuntimeError):
    """The job was cancelled or taken between choosing it and starting it."""


@contextmanager
def worker_lock(data_dir: Path) -> Iterator[None]:
    """One worker at a time: two would fight over the GPU and the same jobs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "worker.lock").open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerBusy("another facegen worker is running") from exc
        f.write(str(os.getpid()))
        f.flush()
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class Career:
    id: int
    key: str
    display_name: str


@dataclass(frozen=True, slots=True)
class Job:
    id: int
    unique_id: int
    kind: str
    group: str
    target_club_id: int | None
    target_band: str | None
    name: str
    at_my_club: bool
    goalkeeper: bool
    height_cm: int | None
    strength: int | None
    age: int | None
    manual: bool = False


@dataclass(frozen=True, slots=True)
class JobFilter:
    groups: frozenset[str] | None = None
    """Only jobs in these groups; None for all."""
    my_club_only: bool = False
    manual_always: bool = False
    """Manual requests from the UI pass whatever the other conditions say."""

    def accepts(self, job: Job) -> bool:
        if self.manual_always and job.manual:
            return True
        return (self.groups is None or job.group in self.groups) and (
            job.at_my_club or not self.my_club_only
        )


@dataclass
class Candidate:
    attempt: int
    index: int
    seed: int
    image: Image
    tags: qc.TagResult
    result: qc.CheckResult
    path: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "index": self.index,
            "seed": self.seed,
            "path": self.path,
            **self.result.to_json(self.tags),
        }


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job: Job
    status: str  # "done" or "failed"
    image_id: int
    version: int
    seconds: float
    message: str


@dataclass(frozen=True, slots=True)
class Preview:
    job: Job
    sheet: CharacterSheet
    wording: str | None
    kit: str
    kit_known: bool
    pose: str
    prompt: Prompt | None
    reference_scale: float | None = None
    """The strength of the reference picture this job would use, None for none."""


@dataclass
class RunSummary:
    done: list[JobOutcome] = field(default_factory=list)
    failed: list[JobOutcome] = field(default_factory=list)
    stopped: str | None = None
    publish: PublishReport | None = None


def load_career(conn: sqlite3.Connection, key: str | None = None) -> Career | None:
    """The named career, or the active one: the career whose save was written last."""
    if key is not None:
        row = conn.execute(
            "SELECT id, key, display_name FROM careers WHERE key = ?", (key,)
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT id, key, display_name FROM careers ORDER BY {ACTIVE_CAREER_ORDER} LIMIT 1"
        ).fetchone()
    return Career(row["id"], row["key"], row["display_name"]) if row else None


def queued_jobs(conn: sqlite3.Connection, career_id: int, f: JobFilter) -> list[Job]:
    """Queued jobs that match the filter, in queue order."""
    rows = conn.execute(
        "SELECT j.id, j.unique_id, j.kind, j.group_name, j.target_club_id, j.target_band,"
        " j.manual, p.name, p.at_my_club, p.is_goalkeeper, p.height_cm, p.strength, p.age"
        " FROM jobs j JOIN players p ON p.career_id = j.career_id AND p.unique_id = j.unique_id"
        " WHERE j.career_id = ? AND j.status = 'queued' ORDER BY j.tier, j.rank, j.id",
        (career_id,),
    ).fetchall()
    jobs = [
        Job(
            id=r["id"],
            unique_id=r["unique_id"],
            kind=r["kind"],
            group=r["group_name"],
            target_club_id=r["target_club_id"],
            target_band=r["target_band"],
            name=r["name"],
            at_my_club=bool(r["at_my_club"]),
            goalkeeper=bool(r["is_goalkeeper"]),
            height_cm=r["height_cm"],
            strength=r["strength"],
            age=r["age"],
            manual=bool(r["manual"]),
        )
        for r in rows
    ]
    return [j for j in jobs if f.accepts(j)]


def master_dir(data_dir: Path, career_key: str, unique_id: int) -> Path:
    return data_dir / "masters" / career_key / str(unique_id)


class Worker:
    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        kits: KitTable,
        career: Career,
        *,
        renderer_factory: Callable[[], Renderer],
        tagger_factory: Callable[[], Tagger],
        fm_check: Callable[[], bool | None] = fm_running,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.conn = conn
        self.settings = settings
        self.kits = kits
        self.career = career
        self._renderer_factory = renderer_factory
        self._tagger_factory = tagger_factory
        self._renderer: Renderer | None = None
        self._tagger: Tagger | None = None
        self.fm_check = fm_check
        self.log = log
        self.clock = clock
        self.data_dir = settings.paths.data_dir

    # -- models, loaded on first use --

    @property
    def renderer(self) -> Renderer:
        if self._renderer is None:
            t0 = self.clock()
            self.log("Loading the image model...")
            self._renderer = self._renderer_factory()
            self.log(f"  loaded in {self.clock() - t0:.0f} s")
        return self._renderer

    @property
    def tagger(self) -> Tagger:
        if self._tagger is None:
            self._tagger = self._tagger_factory()
        return self._tagger

    @property
    def models_loaded(self) -> bool:
        return self._renderer is not None or self._tagger is not None

    def unload(self) -> bool:
        """Free the GPU: drop the model and the tagger; they load again on the next job.

        Returns whether anything was loaded.
        """
        loaded = self.models_loaded
        for model in (self._renderer, self._tagger):
            close = getattr(model, "close", None)
            if close is not None:
                close()
        self._renderer = self._tagger = None
        return loaded

    # -- queue --

    def runnable(self, f: JobFilter) -> list[Job]:
        """Queued jobs that match the filter, in queue order."""
        return queued_jobs(self.conn, self.career.id, f)

    def reference_scale(self, job: Job) -> float | None:
        """The reference strength for a job, judged where the player is now (spec section 5)."""
        return reference_scale(job.kind, job.at_my_club, self.settings.reference)

    def preview(self, f: JobFilter, limit: int | None = None) -> list[Preview]:
        """What the next jobs would draw, without a GPU and without storing anything.

        A player with no sheet yet is shown with the sheet they would get.
        """
        out = []
        for job in self.runnable(f)[:limit]:
            found = load_sheet(self.conn, self.career.id, job.unique_id)
            if found is None or job.kind == "reroll":
                sheet_version = 1 if found is None else found[0] + 1
                sheet = sample_sheet(
                    self.career.key,
                    job.unique_id,
                    sheet_version,
                    height_cm=job.height_cm,
                    strength=job.strength,
                    body=self.settings.body,
                )
            else:
                sheet_version, sheet = found
            version = self._next_version(job.unique_id)
            wording = job.target_band or age_wording(
                job.age, job.group, job.at_my_club, self.settings.age
            )
            kit, kit_known = self.kits.kit_for(job.target_club_id, job.goalkeeper)
            pose = pick_pose(sheet.base_seed, version)
            my_club = self.settings.my_club if job.at_my_club else None
            prompt = build_prompt(sheet, wording, kit, pose, my_club) if wording else None
            out.append(
                Preview(
                    job,
                    sheet,
                    wording,
                    kit.describe(),
                    kit_known,
                    pose,
                    prompt,
                    self.reference_scale(job),
                )
            )
        return out

    def requeue_stale(self) -> int:
        """Jobs left 'running' by a worker that died go back to the queue, in any career.

        Called while holding the worker lock, so no other worker can be running them.
        """
        with transaction(self.conn):
            cur = self.conn.execute(
                "UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'running'",
                (_now(),),
            )
        return cur.rowcount

    def run(
        self,
        f: JobFilter,
        *,
        limit: int | None = None,
        publish: bool = True,
        generate_now: bool = False,
    ) -> RunSummary:
        summary = RunSummary()
        with worker_lock(self.data_dir):
            stale = self.requeue_stale()
            if stale:
                self.log(f"Put {stale} interrupted job(s) back in the queue.")
            tried: set[int] = set()
            total = len(self.runnable(f))
            if limit is not None:
                total = min(total, limit)
            try:
                while limit is None or len(summary.done) + len(summary.failed) < limit:
                    ready = [j for j in self.runnable(f) if j.id not in tried]
                    if not ready:
                        break
                    if not generate_now and not self.settings.schedule.generate_while_fm_running:
                        running = self.fm_check()
                        if running is None:
                            summary.stopped = (
                                "cannot tell whether FM27 is running (tasklist.exe failed); "
                                "close it and pass --now to generate anyway"
                            )
                            break
                        if running:
                            summary.stopped = "FM27 is running; close it first (the queue is kept)"
                            break
                    job = ready[0]
                    tried.add(job.id)
                    n = len(summary.done) + len(summary.failed) + 1
                    total = max(total, n)
                    self.log(f"[{n}/{total}] {job.name} ({job.unique_id}), {job.kind} {job.group}")
                    outcome = self.run_job(job)
                    (summary.done if outcome.status == "done" else summary.failed).append(outcome)
                    self.log(f"  {outcome.message} ({outcome.seconds:.1f} s)")
            finally:
                if publish:
                    summary.publish = self.publish()
        return summary

    def publish(self) -> PublishReport:
        """Publish this career's pictures; FM27 is shown them only if it is the active career."""
        return publish_pack(self.conn, self.settings, self.career.id, self.career.key)

    # -- one job --

    def _claim(self, job: Job) -> None:
        with transaction(self.conn):
            cur = self.conn.execute(
                "UPDATE jobs SET status = 'running', error = NULL, updated_at = ?"
                " WHERE id = ? AND status = 'queued'",
                (_now(), job.id),
            )
        if cur.rowcount != 1:
            raise JobGone(f"job {job.id} is no longer queued")

    def _release(self, job: Job, error: str) -> None:
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE jobs SET status = 'queued', error = ?, updated_at = ?"
                " WHERE id = ? AND status = 'running'",
                (error, _now(), job.id),
            )

    def run_job(self, job: Job) -> JobOutcome:
        self._claim(job)
        try:
            return self._run_job(job)
        except BaseException as exc:
            # Interrupted, out of memory, model failure: the job is redone next time.
            self._release(job, f"interrupted: {type(exc).__name__}: {exc}"[:500])
            raise

    def _group(self, group: str) -> Any:
        return getattr(self.settings.groups, group)

    def _run_job(self, job: Job) -> JobOutcome:
        t0 = self.clock()
        s = self.settings
        group = self._group(job.group)
        with transaction(self.conn):
            sheet_version, sheet = sheet_for(
                self.conn,
                self.career.id,
                self.career.key,
                job.unique_id,
                height_cm=job.height_cm,
                strength=job.strength,
                body=s.body,
                new_face=job.kind == "reroll",
            )
        version = self._next_version(job.unique_id)
        wording = job.target_band or age_wording(job.age, job.group, job.at_my_club, s.age)
        kit, kit_known = self.kits.kit_for(job.target_club_id, job.goalkeeper)
        pose = pick_pose(sheet.base_seed, version)
        my_club = s.my_club if job.at_my_club else None
        if wording is None:
            return self._finish_failed(job, version, sheet_version, None, [], t0, "age unknown")
        prompt = build_prompt(sheet, wording, kit, pose, my_club)
        expect = qc.expectation(sheet, kit, my_club)

        folder = master_dir(self.data_dir, self.career.key, job.unique_id)
        reference, reference_info = None, None
        scale = self.reference_scale(job)
        if scale is not None:
            reference, reference_info = self._reference(job, scale, folder, version)
        cand_dir = folder / f"v{version:03d}-candidates"
        candidates: list[Candidate] = []
        winner: Candidate | None = None
        for attempt in range(MAX_ATTEMPTS):
            batch = []
            for k in range(group.candidates):
                seed = picture_seed(sheet.base_seed, version, attempt, k)
                image = self.renderer.render(prompt, seed, reference)
                batch.append((k, seed, image))
            tag_results = self.tagger([img for _, _, img in batch])
            for (k, seed, image), tags in zip(batch, tag_results, strict=True):
                result = qc.check(tags, expect, qc.head_top(image))
                cand = Candidate(attempt, k, seed, image, tags, result)
                cand.path = self._save_candidate(cand_dir, cand)
                candidates.append(cand)
            this_round = candidates[-len(batch) :]
            pick = qc.best([c.result for c in this_round])
            if pick is not None:
                winner = this_round[pick]
                break
            reasons = "; ".join(sorted({r for c in this_round for r in c.result.failures}))
            self.log(f"  attempt {attempt + 1} failed the check: {reasons}")

        context = {
            "wording": wording,
            "pose": pose,
            "kit": kit.describe(),
            "kit_known": kit_known,
            "goalkeeper": job.goalkeeper,
            "my_club": job.at_my_club,
            "sheet_version": sheet_version,
            "reference": reference_info,
            "prompt": prompt,
        }
        if winner is None:
            return self._finish_failed(job, version, sheet_version, context, candidates, t0)

        final, detail_info = winner.image, None
        if group.detail_pass:
            final, detail_info = self._detail(winner, prompt, expect, reference)
        master = folder / f"v{version:03d}.png"
        self._save_master(master, crop_to_aspect(final), prompt, winner.seed)
        render = {**self.renderer.settings(), **_model_extras(self.renderer, prompt)}
        render["detail"] = detail_info
        render.update({k: v for k, v in context.items() if k != "prompt"})
        check_result = {
            "passed": True,
            "attempts": winner.attempt + 1,
            "chosen": {"attempt": winner.attempt, "index": winner.index, "seed": winner.seed},
            "result": winner.result.to_json(winner.tags),
            "candidates": [c.to_json() for c in candidates],
            "tagger": getattr(self.tagger, "name", "tagger"),
        }
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE images SET is_current = 0"
                " WHERE career_id = ? AND unique_id = ? AND is_current = 1",
                (self.career.id, job.unique_id),
            )
            image_id = self._insert_image(
                job,
                version,
                sheet_version,
                wording,
                prompt,
                winner.seed,
                master,
                render,
                check_result,
                current=True,
            )
            self.conn.execute(
                "UPDATE jobs SET status = 'done', attempts = ?, error = NULL, updated_at = ?"
                " WHERE id = ?",
                (winner.attempt + 1, _now(), job.id),
            )
        passed = sum(1 for c in candidates if c.result.passed)
        detail_note = ""
        if detail_info is not None:
            detail_note = ", detail pass" if detail_info["used"] else ", detail pass rejected"
        reference_note = ""
        if reference_info is not None:
            reference_note = (
                f", reference v{reference_info['version']} at {reference_info['scale']}"
                if reference_info["used"]
                else f", no reference ({reference_info['why']})"
            )
        message = (
            f"v{version} {wording}, {kit.describe()}, {pose}: {passed}/{len(candidates)} passed, "
            f"chose seed {winner.seed} (score {winner.result.score:.2f}){detail_note}"
            f"{reference_note}"
        )
        return JobOutcome(job, "done", image_id, version, self.clock() - t0, message)

    def _detail(
        self,
        winner: Candidate,
        prompt: Prompt,
        expect: qc.Expectation,
        reference: Reference | None,
    ) -> tuple[Image, dict[str, Any]]:
        """Group A's detail pass. Its result is checked again; a failure keeps the original."""
        t0 = self.clock()
        detailed = self.renderer.detail(winner.image, prompt, winner.seed, reference)
        tags = self.tagger([detailed])[0]
        result = qc.check(tags, expect, qc.head_top(detailed))
        info: dict[str, Any] = {
            **_detail_settings(self.renderer),
            "used": result.passed,
            "seconds": round(self.clock() - t0, 2),
            "size": list(detailed.size),
            "check": result.to_json(tags),
        }
        return (detailed if result.passed else winner.image), info

    def _finish_failed(
        self,
        job: Job,
        version: int,
        sheet_version: int,
        context: dict[str, Any] | None,
        candidates: list[Candidate],
        t0: float,
        why: str | None = None,
    ) -> JobOutcome:
        """Nothing passed: keep the best try for the review list; publish nothing."""
        best = max(candidates, key=lambda c: c.result.score, default=None)
        reasons = why or "; ".join(sorted({r for c in candidates for r in c.result.failures}))
        error = f"failed the quality check {MAX_ATTEMPTS} times: {reasons}" if not why else why
        image_id = 0
        with transaction(self.conn):
            if best is not None and context is not None:
                folder = master_dir(self.data_dir, self.career.key, job.unique_id)
                master = folder / f"v{version:03d}-rejected.png"
                prompt: Prompt = context["prompt"]
                self._save_master(master, crop_to_aspect(best.image), prompt, best.seed)
                render = {**self.renderer.settings(), **_model_extras(self.renderer, prompt)}
                render.update({k: v for k, v in context.items() if k != "prompt"})
                image_id = self._insert_image(
                    job,
                    version,
                    sheet_version,
                    context["wording"],
                    prompt,
                    best.seed,
                    master,
                    render,
                    {
                        "passed": False,
                        "attempts": MAX_ATTEMPTS,
                        "failures": reasons,
                        "candidates": [c.to_json() for c in candidates],
                    },
                    current=False,
                )
            self.conn.execute(
                "UPDATE jobs SET status = 'failed', attempts = ?, error = ?, updated_at = ?"
                " WHERE id = ?",
                (MAX_ATTEMPTS if candidates else 0, error[:500], _now(), job.id),
            )
        return JobOutcome(
            job, "failed", image_id, version, self.clock() - t0, f"to review: {error}"
        )

    # -- storage --

    def _next_version(self, unique_id: int) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM images WHERE career_id = ? AND unique_id = ?",
            (self.career.id, unique_id),
        ).fetchone()
        return int(row[0])

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.data_dir))

    def _insert_image(
        self,
        job: Job,
        version: int,
        sheet_version: int,
        wording: str,
        prompt: Prompt,
        seed: int,
        master: Path,
        render: dict[str, Any],
        check_result: dict[str, Any],
        *,
        current: bool,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO images (career_id, unique_id, version, reason, group_name, club_id,"
            " age_band, traits_version, model, prompt, negative_prompt, seed, master_path,"
            " published_path, check_result, is_current, created_at, render)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
            (
                self.career.id,
                job.unique_id,
                version,
                job.kind,
                job.group,
                job.target_club_id,
                wording,
                sheet_version,
                self.renderer.model,
                prompt.positive,
                prompt.negative,
                seed,
                self._rel(master),
                json.dumps(check_result),
                int(current),
                _now(),
                json.dumps(render),
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def _save_candidate(self, folder: Path, cand: Candidate) -> str:
        """The candidate as it would be published: cropped to 3:4, as WebP."""
        path = folder / f"a{cand.attempt}c{cand.index}-{cand.seed}.webp"
        self._save_webp(path, crop_to_aspect(cand.image))
        return self._rel(path)

    def _save_webp(self, path: Path, image: Image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        image.convert("RGB").save(tmp, format="WEBP", quality=90)
        os.replace(tmp, path)

    def _reference(
        self, job: Job, scale: float, folder: Path, version: int
    ) -> tuple[Reference | None, dict[str, Any]]:
        """The player's current picture as a reference, and what to record about it.

        The face is cropped from the current master (`picture.face_box`) and kept next to the
        new picture as vNNN-reference.webp. With no current picture or its file gone, the job
        is drawn without a reference and the record says why.
        """
        info: dict[str, Any] = {"scale": scale, **_reference_settings(self.renderer)}
        row = self.conn.execute(
            "SELECT id, version, master_path FROM images"
            " WHERE career_id = ? AND unique_id = ? AND is_current = 1",
            (self.career.id, job.unique_id),
        ).fetchone()
        if row is None or not row["master_path"]:
            return None, {"used": False, "why": "no current picture", **info}
        master = self.data_dir / row["master_path"]
        if not master.is_file():
            why = f"picture file missing: {row['master_path']}"
            self.log(f"  drawn without a reference: {why}")
            return None, {"used": False, "why": why, **info}
        from PIL import Image as PILImage

        with PILImage.open(master) as im:
            picture = im.convert("RGB")
        box = face_box(picture)
        crop = picture.crop(box)
        path = folder / f"v{version:03d}-reference.webp"
        self._save_webp(path, crop)
        info.update(
            used=True,
            image_id=row["id"],
            version=row["version"],
            master=row["master_path"],
            box=list(box),
            crop=self._rel(path),
        )
        return Reference(f"{row['master_path']}|{box}", crop, scale), info

    def _save_master(self, path: Path, image: Image, prompt: Prompt, seed: int) -> None:
        from PIL.PngImagePlugin import PngInfo

        path.parent.mkdir(parents=True, exist_ok=True)
        meta = PngInfo()
        meta.add_text(
            "facegen",
            json.dumps(
                {
                    "model": self.renderer.model,
                    "prompt": prompt.positive,
                    "negative": prompt.negative,
                    "seed": seed,
                    **self.renderer.settings(),
                }
            ),
        )
        tmp = path.with_name(f".{path.name}.part")
        image.convert("RGB").save(tmp, format="PNG", pnginfo=meta)
        os.replace(tmp, path)


def _model_extras(renderer: Renderer, prompt: Prompt) -> dict[str, Any]:
    chunks = getattr(renderer, "prompt_chunks", None)
    return {"prompt_chunks": chunks(prompt)} if chunks else {}


def _detail_settings(renderer: Renderer) -> dict[str, Any]:
    fn = getattr(renderer, "detail_settings", None)
    return fn() if fn else {}


def _reference_settings(renderer: Renderer) -> dict[str, Any]:
    fn = getattr(renderer, "reference_settings", None)
    return fn() if fn else {}
