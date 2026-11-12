"""GPU job runner (docs/architecture.md section 6).

For each queued job: load or sample the character sheet, build the prompt, render the
group's candidates, check each one, keep the best that passed, run group A's detail pass,
save the master under data/masters/<career key>/<unique_id>/, and record the picture. A job
whose candidates all fail the check three times goes to review (status "failed") and
nothing of it is published.

The model and the tagger sit behind the small `Renderer` and `Tagger` interfaces; the
runner itself needs neither a GPU nor a save, so the tests drive it with fakes.

Rules carried from the spec:
- the GPU works only while FM27 (fm.exe) is not running, unless told to generate now;
- jobs are idempotent: an interrupted job goes back to the queue and is redone from the
  same seeds;
- re-draws that should use a reference picture (planner.reference_scale) wait for Phase 5.
"""

from __future__ import annotations

import fcntl
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
from facegen.db import transaction
from facegen.kits import KitTable
from facegen.planner import reference_scale
from facegen.prompt import Prompt, build_prompt
from facegen.publish import PublishReport, publish_career
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
SIZE = 1024  # SDXL's native square size
STEPS = 28
CFG = 5.5
SAMPLER = "euler_a"
MAX_ATTEMPTS = 3
"""Rounds of candidates before a job goes to review (spec: three failed checks)."""

# Group A's detail pass: the chosen picture upscaled and re-drawn lightly at a higher
# resolution, then downscaled when published (decisions.md, proposed 2026-11-11).
DETAIL_SCALE = 1.5
DETAIL_STRENGTH = 0.35


class Renderer(Protocol):
    model: str

    def render(self, prompt: Prompt, seed: int) -> Image: ...

    def detail(self, image: Image, prompt: Prompt, seed: int) -> Image: ...

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


class AnimagineRenderer:
    """Animagine XL 4.0 Opt in fp16 with its own fp16-fix VAE (docs/research.md section 5)."""

    model = MODEL_REPO

    def __init__(
        self, detail_scale: float = DETAIL_SCALE, detail_strength: float = DETAIL_STRENGTH
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
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_REPO, dtype=torch.float16, use_safetensors=True
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
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

    def _embeddings(self, prompt: Prompt) -> tuple[Any, ...]:
        key = (prompt.positive, prompt.negative)
        if key not in self._embeds:
            with self._torch.no_grad():
                n = max(len(_token_chunks(self.pipe.tokenizer, t)) for t in key)
                pe, pp = _encode(self.pipe, prompt.positive, n)
                ne, npool = _encode(self.pipe, prompt.negative, n)
            self._embeds = {key: (pe, pp, ne, npool, n)}  # one prompt per job
        return self._embeds[key]

    def prompt_chunks(self, prompt: Prompt) -> int:
        return self._embeddings(prompt)[4]

    def render(self, prompt: Prompt, seed: int) -> Image:
        pe, pp, ne, npool, _ = self._embeddings(prompt)
        generator = self._torch.Generator("cuda").manual_seed(seed)
        with self._torch.no_grad():
            return self.pipe(
                prompt_embeds=pe,
                pooled_prompt_embeds=pp,
                negative_prompt_embeds=ne,
                negative_pooled_prompt_embeds=npool,
                width=SIZE,
                height=SIZE,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                generator=generator,
            ).images[0]

    def detail(self, image: Image, prompt: Prompt, seed: int) -> Image:
        from PIL import Image as PILImage

        side = round(image.width * self.detail_scale / 64) * 64
        big = image.resize((side, side), PILImage.Resampling.LANCZOS)
        pe, pp, ne, npool, _ = self._embeddings(prompt)
        generator = self._torch.Generator("cuda").manual_seed(seed)
        with self._torch.no_grad():
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
            ).images[0]

    def settings(self) -> dict[str, Any]:
        return {
            "size": SIZE,
            "steps": STEPS,
            "cfg": CFG,
            "sampler": SAMPLER,
            "dtype": "float16",
            "vae": f"{MODEL_REPO}/vae (fp16-fix)",
            "prompt_encoding": "75-token chunks",
        }

    def detail_settings(self) -> dict[str, Any]:
        return {"scale": self.detail_scale, "strength": self.detail_strength, "steps": STEPS}

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


@dataclass(frozen=True, slots=True)
class JobFilter:
    groups: frozenset[str] | None = None
    """Only jobs in these groups; None for all."""
    my_club_only: bool = False


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


@dataclass
class RunSummary:
    done: list[JobOutcome] = field(default_factory=list)
    failed: list[JobOutcome] = field(default_factory=list)
    waiting_for_reference: int = 0
    stopped: str | None = None
    publish: PublishReport | None = None


def load_career(conn: sqlite3.Connection, key: str | None = None) -> Career | None:
    """The named career, or the one synced last."""
    if key is not None:
        row = conn.execute(
            "SELECT id, key, display_name FROM careers WHERE key = ?", (key,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, key, display_name FROM careers ORDER BY updated_at DESC, id DESC LIMIT 1"
        ).fetchone()
    return Career(row["id"], row["key"], row["display_name"]) if row else None


def queued_jobs(conn: sqlite3.Connection, career_id: int, f: JobFilter) -> list[Job]:
    """Queued jobs that match the filter, in queue order."""
    rows = conn.execute(
        "SELECT j.id, j.unique_id, j.kind, j.group_name, j.target_club_id, j.target_band,"
        " p.name, p.at_my_club, p.is_goalkeeper, p.height_cm, p.strength, p.age"
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
        )
        for r in rows
    ]
    return [
        j
        for j in jobs
        if (f.groups is None or j.group in f.groups) and (j.at_my_club or not f.my_club_only)
    ]


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

    # -- queue --

    def runnable(self, f: JobFilter) -> tuple[list[Job], int]:
        """Jobs this phase can run, and how many wait for reference pictures (Phase 5)."""
        ready, waiting = [], 0
        for job in queued_jobs(self.conn, self.career.id, f):
            if reference_scale(job.kind, job.at_my_club, self.settings.reference) is not None:
                waiting += 1
            else:
                ready.append(job)
        return ready, waiting

    def preview(self, f: JobFilter, limit: int | None = None) -> list[Preview]:
        """What the next jobs would draw, without a GPU and without storing anything.

        A player with no sheet yet is shown with the sheet they would get.
        """
        out = []
        for job in self.runnable(f)[0][:limit]:
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
            out.append(Preview(job, sheet, wording, kit.describe(), kit_known, pose, prompt))
        return out

    def requeue_stale(self) -> int:
        """Jobs left 'running' by a worker that died go back to the queue."""
        with transaction(self.conn):
            cur = self.conn.execute(
                "UPDATE jobs SET status = 'queued', updated_at = ?"
                " WHERE career_id = ? AND status = 'running'",
                (_now(), self.career.id),
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
            total = len(self.runnable(f)[0])
            if limit is not None:
                total = min(total, limit)
            try:
                while limit is None or len(summary.done) + len(summary.failed) < limit:
                    ready, summary.waiting_for_reference = self.runnable(f)
                    ready = [j for j in ready if j.id not in tried]
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
        sizes = {
            "a": self.settings.groups.a.published_long_edge,
            "b": self.settings.groups.b.published_long_edge,
            "c": self.settings.groups.c.published_long_edge,
        }
        with transaction(self.conn):
            return publish_career(
                self.conn,
                self.career.id,
                self.career.key,
                self.settings.paths.fm_user_dir,
                sizes,
                data_dir=self.data_dir,
            )

    # -- one job --

    def _claim(self, job: Job) -> None:
        with transaction(self.conn):
            cur = self.conn.execute(
                "UPDATE jobs SET status = 'running', error = NULL, updated_at = ?"
                " WHERE id = ? AND status = 'queued'",
                (_now(), job.id),
            )
        if cur.rowcount != 1:
            raise RuntimeError(f"job {job.id} is no longer queued")

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
        cand_dir = folder / f"v{version:03d}-candidates"
        candidates: list[Candidate] = []
        winner: Candidate | None = None
        for attempt in range(MAX_ATTEMPTS):
            batch = []
            for k in range(group.candidates):
                seed = picture_seed(sheet.base_seed, version, attempt, k)
                image = self.renderer.render(prompt, seed)
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
            "prompt": prompt,
        }
        if winner is None:
            return self._finish_failed(job, version, sheet_version, context, candidates, t0)

        final, detail_info = winner.image, None
        if group.detail_pass:
            final, detail_info = self._detail(winner, prompt, expect)
        master = folder / f"v{version:03d}.png"
        self._save_master(master, final, prompt, winner.seed)
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
        message = (
            f"v{version} {wording}, {kit.describe()}, {pose}: {passed}/{len(candidates)} passed, "
            f"chose seed {winner.seed} (score {winner.result.score:.2f}){detail_note}"
        )
        return JobOutcome(job, "done", image_id, version, self.clock() - t0, message)

    def _detail(
        self, winner: Candidate, prompt: Prompt, expect: qc.Expectation
    ) -> tuple[Image, dict[str, Any]]:
        """Group A's detail pass. Its result is checked again; a failure keeps the original."""
        t0 = self.clock()
        detailed = self.renderer.detail(winner.image, prompt, winner.seed)
        tags = self.tagger([detailed])[0]
        result = qc.check(tags, expect, qc.head_top(detailed))
        info: dict[str, Any] = {
            **_detail_settings(self.renderer),
            "used": result.passed,
            "seconds": round(self.clock() - t0, 2),
            "size": detailed.width,
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
                self._save_master(master, best.image, prompt, best.seed)
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
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"a{cand.attempt}c{cand.index}-{cand.seed}.webp"
        tmp = path.with_name(f".{path.name}.part")
        cand.image.convert("RGB").save(tmp, format="WEBP", quality=90)
        os.replace(tmp, path)
        return self._rel(path)

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
