"""Generate, check and store 3:4 portraits through injected Renderer/Tagger interfaces.

Each job keeps the best passing candidate and optionally adds a checked detail pass.
Masters live under data/masters/<career key>/<unique_id>/. After three failed attempts,
the job goes to review and publishes nothing. Interrupted jobs retry with the same seeds.

Generation waits for FM27 to stop unless Generate now overrides it. Keep-face redraws use
the current portrait's face crop when `reference_scale` permits, the file is available and
the current picture shows the same gender drawn by the same model; Z-Image-Turbo takes no
reference. Player shoots use the career's in-game season; staff and media use their role's
wardrobe and poses on the grey backdrop. Everyone is drawn as `style.gender` by
`style.model` (core/style.py); a man also has a men's sheet stored with his sheet version.
Fakes can run the orchestration without a GPU or save.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kitshot.core.age import age_wording
from kitshot.core.domain import FinishedStatus
from kitshot.core.job_records import Career, Job, JobFilter
from kitshot.core.kit_values import Kit, KitSource, KitTable
from kitshot.core.man_traits import ManSheet, sample_man_sheet
from kitshot.core.planner import reference_scale
from kitshot.core.prompt import Prompt, Wear
from kitshot.core.records import is_first_team_manager
from kitshot.core.settings_values import (
    ASPECTS,
    IMAGE_MODELS,
    MAN,
    RENDER_SIZES,
    GroupSpec,
    ImageModel,
    MyClubSettings,
    Settings,
)
from kitshot.core.shoot import DEFAULT_BACKDROP, Backdrop, Pose, Shoot, shoot_for
from kitshot.core.style import portrait_prompt, shoot_hair
from kitshot.core.traits import (
    CharacterSheet,
    pick_pose,
    picture_seed,
    sample_sheet,
)
from kitshot.core.wardrobe import Outfit, media_pose, outfit_for
from kitshot.persistence import jobs
from kitshot.persistence.db import transaction
from kitshot.persistence.image_store import ImageStore, master_dir, reference_problem
from kitshot.persistence.jobs import JobGone, load_career, queued_jobs
from kitshot.persistence.sheet_store import (
    load_man_sheet,
    load_sheet,
    next_sheet_version,
    select_sheet,
)
from kitshot.pipeline.publish import PublishReport, publish_pack
from kitshot.pipeline.reclassification import reconcile_facepacks, reconcile_human_managers
from kitshot.platform import platform_info, procs
from kitshot.platform.locking import LockBusy, file_lock
from kitshot.platform.memory import release_free_memory
from kitshot.render import quality_checks as qc
from kitshot.render.picture import crop_to_aspect, face_box
from kitshot.render.renderer import (
    HEIGHT,
    INT8_SCOPE,
    MODEL_REPO,
    WIDTH,
    AnimagineRenderer,
    Reference,
    Renderer,
    Tagger,
)
from kitshot.runtime.contracts import LogFn, print_line

MAX_ATTEMPTS = 3
PRE_ENCODE_JOBS = 32
"""How many jobs' prompts one trip of the text encoders encodes: the job about to run and
the next ones in the queue (`Renderer.prepare`, `prompt_cache`)."""

# Compatibility exports for CLI, tests and experiments.
__all__ = [
    "HEIGHT",
    "INT8_SCOPE",
    "MAX_ATTEMPTS",
    "MODEL_REPO",
    "PRE_ENCODE_JOBS",
    "WIDTH",
    "AnimagineRenderer",
    "Career",
    "Job",
    "JobFilter",
    "JobGone",
    "JobOutcome",
    "Preview",
    "Reference",
    "Renderer",
    "RunSummary",
    "Tagger",
    "Worker",
    "WorkerBusy",
    "fm_running",
    "load_career",
    "master_dir",
    "queued_jobs",
    "worker_lock",
]

if TYPE_CHECKING:
    from PIL.Image import Image


def fm_running() -> bool | None:
    """Whether FM27 (fm.exe) runs; None if it cannot be told.

    On Windows and WSL Windows's `tasklist.exe` is asked (from System32 when it is not on
    PATH), with no console window. On another Linux FM27 can only run under Wine or Proton,
    where its process keeps its name.
    """
    kind = platform_info.current()
    if kind == "linux":
        return procs.linux_process_running(_process_names())
    tasklist = procs.windows_tool("tasklist.exe", kind, os.environ)
    if tasklist is None:
        return None
    try:
        out = procs.run_quiet(procs.tasklist_command(tasklist), capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return procs.tasklist_says_running(out.stdout)


def _process_names() -> list[str]:
    names = []
    for comm in Path("/proc").glob("[0-9]*/comm"):
        try:
            names.append(comm.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return names


class WorkerBusy(RuntimeError):
    """Another worker holds the lock."""


@contextmanager
def worker_lock(data_dir: Path) -> Iterator[None]:
    """One worker at a time: two would fight over the GPU and the same jobs."""
    try:
        with file_lock(data_dir / "worker.lock", blocking=False):
            yield
    except LockBusy as exc:
        raise WorkerBusy("another Kitshot worker is running") from exc


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
    status: FinishedStatus
    image_id: int
    version: int
    seconds: float
    message: str


@dataclass(frozen=True, slots=True)
class Setup:
    """What one picture shows besides the character sheet: clothes, pose and backdrop."""

    wear: Wear
    pose: str | Pose
    backdrop: Backdrop
    kit_source: KitSource
    """Where the kit's colours came from: config/kits.toml, the kit catalog, the bundled
    catalog or the fallback."""
    shoot: Shoot | None = None
    """A player's shoot; None for staff."""

    def pose_text(self) -> str:
        if isinstance(self.pose, Pose):
            return self.pose.describe()
        return self.pose or "no arm pose"

    def describe(self) -> str:
        """Clothes, pose and backdrop in a few words, for logs and previews."""
        return f"{self.wear.describe()}, {self.pose_text()}, {self.backdrop.describe()} backdrop"


@dataclass(frozen=True, slots=True)
class _PreparedPortrait:
    version: int
    wording: str | None
    setup: Setup
    my_club: MyClubSettings | None
    prompt: Prompt | None
    gender: str
    """Who the picture shows (`style.gender`), recorded with it."""


@dataclass(frozen=True, slots=True)
class Preview:
    job: Job
    sheet: CharacterSheet
    wording: str | None
    kit: str
    kit_source: KitSource
    pose: str
    prompt: Prompt | None
    reference_scale: float | None = None
    """The strength of the reference picture this job would use, None for none."""
    man: ManSheet | None = None
    """The men's sheet the job would draw, while everyone is drawn as a man."""


@dataclass
class RunSummary:
    done: list[JobOutcome] = field(default_factory=list)
    failed: list[JobOutcome] = field(default_factory=list)
    stopped: str | None = None
    publish: PublishReport | None = None


class Worker:
    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        kits: KitTable,
        career: Career,
        *,
        renderer_factory: Callable[[str], Renderer],
        tagger_factory: Callable[[], Tagger],
        fm_check: Callable[[], bool | None] = fm_running,
        log: LogFn = print_line,
        clock: Callable[[], float] = time.perf_counter,
        read_settings: Callable[[], Settings] | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings
        self._read_settings = read_settings
        self.kits = kits
        self.career = career
        self._renderer_factory = renderer_factory
        self._tagger_factory = tagger_factory
        self._renderer: Renderer | None = None
        self._renderer_model: str | None = None
        """The `style.model` the loaded renderer was built for."""
        self._tagger: Tagger | None = None
        self.fm_check = fm_check
        self.log = log
        self.clock = clock
        self.data_dir = settings.paths.data_dir

    @property
    def images(self) -> ImageStore:
        """Use the current career when the service reuses this worker's loaded models."""
        return ImageStore(self.conn, self.data_dir, self.career)

    @property
    def renderer(self) -> Renderer:
        """The selected model's renderer, loaded on first use. After `style.model` changes
        (settings are read again before each job), the old model is unloaded first."""
        model = self.settings.style.model
        if self._renderer is not None and self._renderer_model != model:
            old, self._renderer, self._renderer_model = self._renderer, None, None
            self.log(f"The image model is now {model}: unloading {old.model}.")
            self._close(old)
            release_free_memory()
        if self._renderer is None:
            t0 = self.clock()
            self.log("Loading the image model...")
            self._renderer = self._renderer_factory(model)
            self._renderer_model = model
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

        Returns whether anything was loaded. Never raises: the models are let go of first,
        so a `close` that fails (after a CUDA fault every GPU call does) cannot leave a
        closed model behind for the next job.
        """
        models = (self._renderer, self._tagger)
        self._renderer = self._tagger = self._renderer_model = None
        for model in models:
            self._close(model)
        release_free_memory()
        return any(m is not None for m in models)

    def _close(self, model: object) -> None:
        """Free a model already let go of. Never raises."""
        close = getattr(model, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - report it; the model is gone either way
            self.log(f"  could not free the GPU cleanly: {type(exc).__name__}: {exc}", "warning")

    def runnable(self, f: JobFilter, *, limit: int | None = None) -> list[Job]:
        """Queued jobs that match the filter, in queue order."""
        return queued_jobs(self.conn, self.career.id, f, limit=limit)

    def reference_scale(self, job: Job) -> float | None:
        """The reference strength for a job, judged where the player is now (spec section 5)."""
        return reference_scale(job.kind, job.at_my_club, self.settings.reference)

    @property
    def image_model(self) -> ImageModel:
        """What the selected model (`style.model`) draws and takes."""
        return IMAGE_MODELS[self.settings.style.model]

    def preview(self, f: JobFilter, limit: int | None = None) -> list[Preview]:
        """What the next jobs would draw, without a GPU and without storing anything.

        A player with no sheet yet is shown with the sheet they would get.
        """
        if limit is not None and limit < 0:
            # Preserve the earlier Python slice for callers that omit the last N jobs.
            limit = max(0, sum(jobs.queued_counts(self.conn, self.career.id, f).values()) + limit)
        return [self._preview(job) for job in self.runnable(f, limit=limit)]

    def _preview(self, job: Job, *, check_reference: bool = True) -> Preview:
        """What one queued job would draw; reads only."""
        found = load_sheet(self.conn, self.career.id, job.unique_id, job.traits_version)
        man = None
        if found is None or (job.kind == "reroll" and job.traits_version is None):
            sheet_version = next_sheet_version(self.conn, self.career.id, job.unique_id)
            sheet = sample_sheet(
                self.career.key,
                job.unique_id,
                sheet_version,
                height_cm=job.height_cm,
                strength=job.strength,
                body=self.settings.body,
                personality=job.personality,
                personality_weight=self.settings.personality.weight,
                manager=job.staff and is_first_team_manager(job.staff_job),
            )
        else:
            sheet_version, sheet = found
            if self.settings.style.gender == MAN:
                man = load_man_sheet(self.conn, self.career.id, job.unique_id, sheet_version)
        if self.settings.style.gender == MAN and man is None:
            man = sample_man_sheet(
                self.career.key,
                job.unique_id,
                sheet_version,
                nation=job.nation,
                personality=job.personality,
                personality_weight=self.settings.personality.weight,
            )
        portrait = self._prepare_portrait(job, sheet, man)
        setup = portrait.setup
        scale = (
            self.reference_scale(job) if check_reference and self.image_model.reference else None
        )
        if scale is not None and reference_problem(
            self.settings, self.images.current_reference(job.unique_id)
        ):
            scale = None
        return Preview(
            job,
            sheet,
            portrait.wording,
            setup.wear.describe(),
            setup.kit_source,
            f"{setup.pose_text()}, {setup.backdrop.describe()} backdrop",
            portrait.prompt,
            scale,
            man,
        )

    def requeue_stale(self) -> int:
        """Jobs left 'running' by a worker that died go back to the queue, in any career.

        Called while holding the worker lock, so no other worker can be running them.
        """
        stale = jobs.requeue_stale(self.conn)
        with transaction(self.conn):
            settings = self._read_settings() if self._read_settings else self.settings
            reconcile_human_managers(self.conn, settings)
            reconcile_facepacks(self.conn, settings)
        return stale

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
            total = sum(jobs.queued_counts(self.conn, self.career.id, f).values())
            if limit is not None:
                total = min(total, limit)
            try:
                while limit is None or len(summary.done) + len(summary.failed) < limit:
                    n = len(summary.done) + len(summary.failed) + 1
                    batch = (
                        PRE_ENCODE_JOBS if limit is None else min(PRE_ENCODE_JOBS, limit - n + 1)
                    )
                    ready = self.runnable(f, limit=batch)
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
                    total = max(total, n)
                    self.log(f"[{n}/{total}] {job.name} ({job.unique_id}), {job.kind} {job.group}")
                    try:
                        outcome = self.run_job(job, ready[1:])
                    except JobGone:
                        continue
                    (summary.done if outcome.status == "done" else summary.failed).append(outcome)
                    self.log(
                        f"  {outcome.message} ({outcome.seconds:.1f} s)",
                        "info" if outcome.status == "done" else "warning",
                    )
            finally:
                if publish:
                    summary.publish = self.publish()
        return summary

    def publish(self) -> PublishReport:
        """Publish this career's pictures; FM27 is shown them only if it is the active career."""
        return publish_pack(self.conn, self.settings, self.career.id, self.career.key)

    def run_job(self, job: Job, upcoming: Sequence[Job] = ()) -> JobOutcome:
        """Run one job. `upcoming` are the queued jobs expected next, in order: when this
        job's prompt is not encoded yet, theirs are encoded in the same trip."""
        job = jobs.claim(self.conn, self.career.id, job)
        try:
            return self._run_job(job, upcoming)
        except BaseException as exc:
            # Interrupted, out of memory, model failure: the job is redone next time.
            jobs.release(self.conn, job.id, f"interrupted: {type(exc).__name__}: {exc}"[:500])
            raise

    def _group(self, group: str) -> GroupSpec:
        return getattr(self.settings.groups, group)

    def _run_job(self, job: Job, upcoming: Sequence[Job] = ()) -> JobOutcome:
        t0 = self.clock()
        group = self._group(job.group)
        sheet_version, sheet = jobs.allocate_sheet(
            self.conn, self.career, job, self.settings.body, self.settings.personality.weight
        )
        man = None
        if self.settings.style.gender == MAN:
            man = jobs.allocate_man_sheet(
                self.conn, self.career, job, sheet_version, self.settings.personality.weight
            )
        portrait = self._prepare_portrait(job, sheet, man)
        version, wording, prompt = portrait.version, portrait.wording, portrait.prompt
        if wording is None or prompt is None:
            return self._finish_failed(
                job, version, sheet_version, None, [], t0, "age unknown", portrait.gender
            )
        setup = portrait.setup
        wear = setup.wear
        if man is not None and job.age is not None:  # a man has a prompt only with his age
            expect = qc.man_expectation(man, job.age, wear)
        else:
            expect = qc.expectation(sheet, wear, portrait.my_club)
        self._pre_encode(prompt, upcoming)

        folder = master_dir(self.data_dir, self.career.key, job.unique_id)
        reference, reference_info = None, None
        scale = self.reference_scale(job)
        if scale is not None:
            reference, reference_info = self._reference(job, scale, folder, version)
        cand_dir = folder / f"v{version:03d}-candidates"
        candidates: list[Candidate] = []
        winner: Candidate | None = None
        self._configure_renderer()
        # The shape of pictures drawn from now on (Phase 16g); the picture keeps it.
        aspect = self.settings.picture.aspect
        ratio, size = ASPECTS[aspect], RENDER_SIZES[aspect]
        for attempt in range(MAX_ATTEMPTS):
            batch = []
            for k in range(group.candidates):
                seed = picture_seed(sheet.base_seed, version, attempt, k)
                image = self.renderer.render(prompt, seed, reference, size=size)
                batch.append((k, seed, image))
            tag_results = self.tagger([img for _, _, img in batch])
            for (k, seed, image), tags in zip(batch, tag_results, strict=True):
                result = qc.check(tags, expect, qc.head_top(image))
                cand = Candidate(attempt, k, seed, image, tags, result)
                cand.path = self.images.save_candidate(cand_dir, cand, ratio)
                candidates.append(cand)
            this_round = candidates[-len(batch) :]
            pick = qc.best([c.result for c in this_round])
            if pick is not None:
                winner = this_round[pick]
                break
            reasons = "; ".join(sorted({r for c in this_round for r in c.result.failures}))
            self.log(f"  attempt {attempt + 1} failed the check: {reasons}", "warning")

        context = {
            "wording": wording,
            "pose": setup.pose_text(),
            "kit": wear.describe(),
            "kit_known": setup.kit_source != "fallback",
            "kit_source": setup.kit_source,
            "backdrop": setup.backdrop.describe(),
            "shoot": setup.shoot.context() if setup.shoot else None,
            "goalkeeper": job.goalkeeper,
            "staff": job.staff,
            "media": job.media_role,
            "outfit": wear.look if isinstance(wear, Outfit) else None,
            "my_club": job.at_my_club,
            "sheet_version": sheet_version,
            "man_sheet": man.describe() if man is not None else None,
            "reference": reference_info,
            "prompt": prompt,
            "aspect": aspect,
        }
        if winner is None:
            return self._finish_failed(
                job, version, sheet_version, context, candidates, t0, gender=portrait.gender
            )

        final, detail_info = winner.image, None
        detail_size = group.detail_size(aspect)
        if detail_size is not None:
            final, detail_info = self._detail(winner, prompt, expect, reference, detail_size)
        master = folder / f"v{version:03d}.png"
        drawn = self._render_settings(size)
        self.images.save_master(
            master,
            crop_to_aspect(final, ratio),
            prompt,
            winner.seed,
            self.renderer.model,
            drawn,
        )
        render = {**drawn, **_model_extras(self.renderer, prompt)}
        render["detail"] = detail_info
        render.update({k: v for k, v in context.items() if k != "prompt"})
        render["seconds"] = round(self.clock() - t0, 2)  # the dashboard's estimate (Phase 12c)
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
            image_id = self.images.insert_image(
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
                model=self.renderer.model,
                gender=portrait.gender,
                aspect=aspect,
            )
            select_sheet(self.conn, self.career.id, job.unique_id, sheet_version)
            jobs.finish(self.conn, job.id, "done", winner.attempt + 1)
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
            f"v{version} {wording}, {setup.describe()}: {passed}/{len(candidates)} passed, "
            f"chose seed {winner.seed} (score {winner.result.score:.2f}){detail_note}"
            f"{reference_note}"
        )
        return JobOutcome(job, "done", image_id, version, self.clock() - t0, message)

    def _prepare_portrait(
        self, job: Job, sheet: CharacterSheet, man: ManSheet | None = None
    ) -> _PreparedPortrait:
        """Choose the next picture's setup and prompt without assigning a sheet or rendering.

        The age wording is the picture's age band for everyone: a man is re-drawn when it
        changes too, and his prompt states his age in years (user, 2026-11-20)."""
        style = self.settings.style
        version = self.images.next_version(job.unique_id)
        wording = job.target_band or age_wording(
            job.age,
            job.group,
            job.at_my_club,
            self.settings.age,
            staff=job.staff,
            media=job.media_role is not None,
        )
        setup = self._setup(job, sheet, version, man)
        my_club = self.settings.my_club if job.at_my_club else None
        prompt = None
        if wording is not None:
            prompt = portrait_prompt(
                style.model,
                style.gender,
                sheet=sheet,
                man=man,
                wording=wording,
                age=job.age,
                wear=setup.wear,
                pose=setup.pose,
                backdrop=setup.backdrop,
                my_club=my_club,
            )
        return _PreparedPortrait(version, wording, setup, my_club, prompt, style.gender)

    def _setup(
        self, job: Job, sheet: CharacterSheet, version: int, man: ManSheet | None = None
    ) -> Setup:
        """Select a player shoot or a staff/media outfit with its role's pose and backdrop."""
        if job.media_role:
            home, source = self.kits.kit_for(job.target_club_id, False)
            outfit = outfit_for(home, job.media_role, job.age, sheet.base_seed, version)
            pose = media_pose(outfit, sheet.base_seed, version)
            return Setup(outfit, pose, DEFAULT_BACKDROP, source)
        if job.staff:
            home, source = self.kits.kit_for(job.target_club_id, False)
            outfit = outfit_for(home, job.staff_job, job.age, sheet.base_seed, version)
            pose = (
                media_pose(outfit, sheet.base_seed, version)
                if is_first_team_manager(job.staff_job)
                else pick_pose(sheet.base_seed, version)
            )
            return Setup(outfit, pose, DEFAULT_BACKDROP, source)
        kit, source = self.kits.kit_for(job.target_club_id, job.goalkeeper)
        home: Kit = kit
        if job.goalkeeper:
            home, _ = self.kits.kit_for(job.target_club_id, False)
        shoot = shoot_for(
            kit,
            shoot_hair(sheet, man, job.age),
            sheet.base_seed,
            version,
            home=home,
            age=job.age,
            game_date=self.career.game_date,
        )
        return Setup(shoot.wear, shoot.pose, shoot.backdrop, source, shoot)

    def _pre_encode(self, prompt: Prompt, upcoming: Sequence[Job]) -> None:
        """Encode this job's prompt and the next jobs' in one trip of the text encoders.

        Only when this job's prompt is not encoded yet: so once per PRE_ENCODE_JOBS jobs, or
        for a job queued after the last trip. The next jobs' prompts are the ones `preview`
        shows, which the jobs draw unless something changes first; a prompt that changed is
        encoded when its job draws it. A renderer without `prepare` encodes as it draws.
        """
        renderer = self.renderer
        prepare = getattr(renderer, "prepare", None)
        encoded = getattr(renderer, "encoded", None)
        if prepare is None or encoded is None or encoded(prompt):
            return
        prompts = [prompt]
        for job in upcoming[: PRE_ENCODE_JOBS - 1]:
            try:
                p = self._preview(job, check_reference=False).prompt
            except Exception as exc:  # noqa: BLE001 - the job reports it again when it runs
                self.log(
                    f"  not encoded ahead: {job.name} ({type(exc).__name__}: {exc})", "warning"
                )
                continue
            if p is not None:
                prompts.append(p)
        texts = prepare(prompts)
        if len(prompts) > 1:
            self.log(f"  encoded the prompts of the next {len(prompts)} jobs ({texts} texts)")

    def _render_settings(self, size: tuple[int, int]) -> dict[str, Any]:
        """The renderer's settings as recorded with a picture, at the size it was drawn."""
        return {**self.renderer.settings(), "width": size[0], "height": size[1]}

    def _configure_renderer(self) -> None:
        """Apply VAE tiling per job rather than only when the renderer loads."""
        set_tiling = getattr(self.renderer, "set_vae_tiling", None)
        if set_tiling is not None:
            set_tiling(self.settings.render.vae_tiling)

    def _detail(
        self,
        winner: Candidate,
        prompt: Prompt,
        expect: qc.Expectation,
        reference: Reference | None,
        size: tuple[int, int],
    ) -> tuple[Image, dict[str, Any]]:
        """The detail pass at the group's `detail_size`. Its result is checked again; a failure
        keeps the original."""
        t0 = self.clock()
        detailed = self.renderer.detail(winner.image, prompt, winner.seed, reference, size=size)
        tags = self.tagger([detailed])[0]
        result = qc.check(tags, expect, qc.head_top(detailed))
        info: dict[str, Any] = {
            **_detail_settings(self.renderer),
            "target": list(size),
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
        gender: str = "woman",
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
                aspect = context["aspect"]
                drawn = self._render_settings(RENDER_SIZES[aspect])
                self.images.save_master(
                    master,
                    crop_to_aspect(best.image, ASPECTS[aspect]),
                    prompt,
                    best.seed,
                    self.renderer.model,
                    drawn,
                )
                render = {**drawn, **_model_extras(self.renderer, prompt)}
                render.update({k: v for k, v in context.items() if k != "prompt"})
                render["seconds"] = round(self.clock() - t0, 2)
                image_id = self.images.insert_image(
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
                    model=self.renderer.model,
                    gender=gender,
                    aspect=aspect,
                )
            jobs.finish(self.conn, job.id, "failed", MAX_ATTEMPTS if candidates else 0, error[:500])
        return JobOutcome(
            job, "failed", image_id, version, self.clock() - t0, f"to review: {error}"
        )

    def _reference(
        self, job: Job, scale: float, folder: Path, version: int
    ) -> tuple[Reference | None, dict[str, Any]]:
        """The player's current picture as a reference, and what to record about it.

        The face is cropped from the current master (`picture.face_box`) and kept next to the
        new picture as vNNN-reference.webp. The job is drawn without a reference, and the
        record says why, when the model takes none (Z-Image-Turbo), with no current picture
        or its file gone, or when the current picture shows another gender or was drawn by
        another model.
        """
        renderer = self.renderer
        adapter = self.image_model.reference and getattr(renderer, "reference_adapter", True)
        info: dict[str, Any] = {"scale": scale}
        row = None
        if adapter:
            info.update(_reference_settings(renderer))
            row = self.images.current_reference(job.unique_id)
        why = reference_problem(self.settings, row, renderer_model=renderer.model, adapter=adapter)
        if why is not None:
            self.log(f"  drawn without a reference: {why}")
            return None, {"used": False, "why": why, **info}
        assert row is not None
        master = self.data_dir / row["master_path"]
        from PIL import Image as PILImage

        with PILImage.open(master) as im:
            picture = im.convert("RGB")
        box = face_box(picture)
        crop = picture.crop(box)
        path = folder / f"v{version:03d}-reference.webp"
        self.images.save_webp(path, crop)
        info.update(
            used=True,
            image_id=row["id"],
            version=row["version"],
            master=row["master_path"],
            box=list(box),
            crop=self.images.rel(path),
        )
        return Reference(f"{row['master_path']}|{box}", crop, scale), info


def _model_extras(renderer: Renderer, prompt: Prompt) -> dict[str, Any]:
    """The prompt's length as the model reads it: SDXL's 75-token chunks, or Z-Image-Turbo's
    tokens."""
    chunks = getattr(renderer, "prompt_chunks", None)
    if chunks:
        return {"prompt_chunks": chunks(prompt)}
    tokens = getattr(renderer, "prompt_tokens", None)
    return {"prompt_tokens": tokens(prompt)} if tokens else {}


def _detail_settings(renderer: Renderer) -> dict[str, Any]:
    fn = getattr(renderer, "detail_settings", None)
    return fn() if fn else {}


def _reference_settings(renderer: Renderer) -> dict[str, Any]:
    fn = getattr(renderer, "reference_settings", None)
    return fn() if fn else {}
