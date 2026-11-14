"""GPU job runner (docs/architecture.md section 6).

For each queued job: load or sample the character sheet, build the prompt, render the
group's candidates, check each one, keep the best that passed, run group A's detail pass,
save the master under data/masters/<career key>/<unique_id>/, and record the picture. A job
whose candidates all fail the check three times goes to review (status "failed") and
nothing of it is published.

Pictures are 3:4 portraits (picture.py): the model renders at its portrait size near 3:4
and the master is cropped to exactly 3:4.

A player's picture is a shoot (shoot.py): the kit in one of its looks, a pose of arms and
angle, and a backdrop, with the season from the career's in-game date. My club's staff are
dressed by wardrobe.py, with a pose from traits.POSE on the grey backdrop; journalists and
press officers too, with a calm pose or none (a prop in hand). Someone whose age wording is
a man's (gender.py) is drawn and checked as a man.

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

import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from facegen import jobs, qc
from facegen.age import age_wording, is_male
from facegen.db import transaction
from facegen.domain import FinishedStatus
from facegen.image_store import ImageStore, master_dir
from facegen.jobs import Career, Job, JobFilter, JobGone, load_career, queued_jobs
from facegen.kits import Kit, KitSource, KitTable
from facegen.locking import LockBusy, file_lock
from facegen.picture import crop_to_aspect, face_box
from facegen.planner import reference_scale
from facegen.prompt import Prompt, Wear, build_prompt
from facegen.publish import PublishReport, publish_pack
from facegen.renderer import (
    HEIGHT,
    INT8_SCOPE,
    MODEL_REPO,
    WIDTH,
    AnimagineRenderer,
    Reference,
    Renderer,
    Tagger,
)
from facegen.settings import GroupA, GroupB, GroupC, MyClubSettings, Settings
from facegen.shoot import DEFAULT_BACKDROP, Backdrop, Pose, Shoot, shoot_for
from facegen.traits import (
    CharacterSheet,
    load_sheet,
    next_sheet_version,
    pick_pose,
    picture_seed,
    sample_sheet,
    select_sheet,
)
from facegen.wardrobe import Outfit, media_pose, outfit_for

MAX_ATTEMPTS = 3
PRE_ENCODE_JOBS = 32
"""How many jobs' prompts one trip of the text encoders encodes: the job about to run and
the next ones in the queue (`Renderer.prepare`, `prompt_cache`)."""

# Keep existing CLI, tests and experiment imports working after the module split.
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
    """Whether FM27 (fm.exe) runs on Windows; None if it cannot be told."""
    try:
        out = subprocess.run(
            ["tasklist.exe", "/FI", "IMAGENAME eq fm.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return b'"fm.exe"' in out.stdout.lower()


class WorkerBusy(RuntimeError):
    """Another worker holds the lock."""


@contextmanager
def worker_lock(data_dir: Path) -> Iterator[None]:
    """One worker at a time: two would fight over the GPU and the same jobs."""
    try:
        with file_lock(data_dir / "worker.lock", blocking=False):
            yield
    except LockBusy as exc:
        raise WorkerBusy("another facegen worker is running") from exc


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
    """Where the kit's colours came from: config/kits.toml, the kit catalog or the fallback."""
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

    @property
    def images(self) -> ImageStore:
        """Use the current career when the service reuses this worker's loaded models."""
        return ImageStore(self.conn, self.data_dir, self.career)

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

        Returns whether anything was loaded. Never raises: the models are let go of first,
        so a `close` that fails (after a CUDA fault every GPU call does) cannot leave a
        closed model behind for the next job.
        """
        models = (self._renderer, self._tagger)
        self._renderer = self._tagger = None
        for model in models:
            close = getattr(model, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - report it; the model is gone either way
                self.log(f"  could not free the GPU cleanly: {type(exc).__name__}: {exc}")
        return any(m is not None for m in models)

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
        return [self._preview(job) for job in self.runnable(f)[:limit]]

    def _preview(self, job: Job) -> Preview:
        """What one queued job would draw; reads only."""
        found = load_sheet(self.conn, self.career.id, job.unique_id, job.traits_version)
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
            )
        else:
            sheet_version, sheet = found
        version = self.images.next_version(job.unique_id)
        wording = job.target_band or age_wording(
            job.age, job.group, job.at_my_club, self.settings.age, **_audience(job)
        )
        setup = self._setup(job, sheet, version)
        my_club = self.settings.my_club if job.at_my_club else None
        prompt = self._prompt(sheet, wording, setup, my_club) if wording else None
        return Preview(
            job,
            sheet,
            wording,
            setup.wear.describe(),
            setup.kit_source,
            f"{setup.pose_text()}, {setup.backdrop.describe()} backdrop",
            prompt,
            self.reference_scale(job),
        )

    def requeue_stale(self) -> int:
        """Jobs left 'running' by a worker that died go back to the queue, in any career.

        Called while holding the worker lock, so no other worker can be running them.
        """
        return jobs.requeue_stale(self.conn)

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
                    upcoming = ready[1:]
                    if limit is not None:
                        upcoming = upcoming[: limit - n]
                    try:
                        outcome = self.run_job(job, upcoming)
                    except JobGone:
                        tried.discard(job.id)
                        continue
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

    def _group(self, group: str) -> GroupA | GroupB | GroupC:
        return getattr(self.settings.groups, group)

    def _run_job(self, job: Job, upcoming: Sequence[Job] = ()) -> JobOutcome:
        t0 = self.clock()
        s = self.settings
        group = self._group(job.group)
        sheet_version, sheet = jobs.allocate_sheet(
            self.conn, self.career, job, s.body, s.personality.weight
        )
        version = self.images.next_version(job.unique_id)
        wording = job.target_band or age_wording(
            job.age, job.group, job.at_my_club, s.age, **_audience(job)
        )
        setup = self._setup(job, sheet, version)
        wear = setup.wear
        my_club = s.my_club if job.at_my_club else None
        if wording is None:
            return self._finish_failed(job, version, sheet_version, None, [], t0, "age unknown")
        prompt = self._prompt(sheet, wording, setup, my_club)
        expect = qc.expectation(sheet, wear, my_club, male=is_male(wording))
        self._pre_encode(prompt, upcoming)

        folder = master_dir(self.data_dir, self.career.key, job.unique_id)
        reference, reference_info = None, None
        scale = self.reference_scale(job)
        if scale is not None:
            reference, reference_info = self._reference(job, scale, folder, version, wording)
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
                cand.path = self.images.save_candidate(cand_dir, cand)
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
            "pose": setup.pose_text(),
            "kit": wear.describe(),
            "kit_known": setup.kit_source != "fallback",
            "kit_source": setup.kit_source,
            "backdrop": setup.backdrop.describe(),
            "shoot": setup.shoot.context() if setup.shoot else None,
            "goalkeeper": job.goalkeeper,
            "staff": job.staff,
            "media": job.media_role,
            "male": is_male(wording),
            "outfit": wear.look if isinstance(wear, Outfit) else None,
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
        self.images.save_master(
            master,
            crop_to_aspect(final),
            prompt,
            winner.seed,
            self.renderer.model,
            self.renderer.settings(),
        )
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

    def _setup(self, job: Job, sheet: CharacterSheet, version: int) -> Setup:
        """A player's shoot (shoot.py), or for my club's staff an outfit in the home colours
        (wardrobe.py) with a pose from the old library on the grey backdrop; for a journalist
        or press officer, clothes for their work with a calm pose or a prop in hand."""
        if job.media_role:
            home, source = self.kits.kit_for(job.target_club_id, False)
            outfit = outfit_for(home, job.media_role, job.age, sheet.base_seed, version)
            pose = media_pose(outfit, sheet.base_seed, version)
            return Setup(outfit, pose, DEFAULT_BACKDROP, source)
        if job.staff:
            home, source = self.kits.kit_for(job.target_club_id, False)
            outfit = outfit_for(home, job.staff_job, job.age, sheet.base_seed, version)
            pose = pick_pose(sheet.base_seed, version)
            return Setup(outfit, pose, DEFAULT_BACKDROP, source)
        kit, source = self.kits.kit_for(job.target_club_id, job.goalkeeper)
        home: Kit = kit
        if job.goalkeeper:
            home, _ = self.kits.kit_for(job.target_club_id, False)
        shoot = shoot_for(
            kit,
            sheet.hair_colour,
            sheet.base_seed,
            version,
            home=home,
            age=job.age,
            game_date=self.career.game_date,
        )
        return Setup(shoot.wear, shoot.pose, shoot.backdrop, source, shoot)

    @staticmethod
    def _prompt(
        sheet: CharacterSheet, wording: str, setup: Setup, my_club: MyClubSettings | None
    ) -> Prompt:
        return build_prompt(
            sheet, wording, setup.wear, setup.pose, my_club, backdrop=setup.backdrop
        )

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
                p = self._preview(job).prompt
            except Exception as exc:  # noqa: BLE001 - the job reports it again when it runs
                self.log(f"  not encoded ahead: {job.name} ({type(exc).__name__}: {exc})")
                continue
            if p is not None:
                prompts.append(p)
        texts = prepare(prompts)
        if len(prompts) > 1:
            self.log(f"  encoded the prompts of the next {len(prompts)} jobs ({texts} texts)")

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
                self.images.save_master(
                    master,
                    crop_to_aspect(best.image),
                    prompt,
                    best.seed,
                    self.renderer.model,
                    self.renderer.settings(),
                )
                render = {**self.renderer.settings(), **_model_extras(self.renderer, prompt)}
                render.update({k: v for k, v in context.items() if k != "prompt"})
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
                )
            jobs.finish(self.conn, job.id, "failed", MAX_ATTEMPTS if candidates else 0, error[:500])
        return JobOutcome(
            job, "failed", image_id, version, self.clock() - t0, f"to review: {error}"
        )

    # -- storage --

    def _reference(
        self, job: Job, scale: float, folder: Path, version: int, wording: str
    ) -> tuple[Reference | None, dict[str, Any]]:
        """The player's current picture as a reference, and what to record about it.

        The face is cropped from the current master (`picture.face_box`) and kept next to the
        new picture as vNNN-reference.webp. With no current picture or its file gone, or a
        current picture of the other gender than `wording`'s, the job is drawn without a
        reference and the record says why.
        """
        info: dict[str, Any] = {"scale": scale, **_reference_settings(self.renderer)}
        row = self.conn.execute(
            "SELECT id, version, master_path, COALESCE(restored_band, age_band) AS band"
            " FROM images WHERE career_id = ? AND unique_id = ? AND is_current = 1",
            (self.career.id, job.unique_id),
        ).fetchone()
        if row is None or not row["master_path"]:
            return None, {"used": False, "why": "no current picture", **info}
        if row["band"] is not None and is_male(row["band"]) != is_male(wording):
            why = "the current picture is of the other gender"
            self.log(f"  drawn without a reference: {why}")
            return None, {"used": False, "why": why, **info}
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


def _audience(job: Job) -> dict[str, bool]:
    """`age_wording`'s staff, media and male flags for a job's person."""
    return {"staff": job.staff, "media": job.media_role is not None, "male": job.drawn_male}


def _model_extras(renderer: Renderer, prompt: Prompt) -> dict[str, Any]:
    chunks = getattr(renderer, "prompt_chunks", None)
    return {"prompt_chunks": chunks(prompt)} if chunks else {}


def _detail_settings(renderer: Renderer) -> dict[str, Any]:
    fn = getattr(renderer, "detail_settings", None)
    return fn() if fn else {}


def _reference_settings(renderer: Renderer) -> dict[str, Any]:
    fn = getattr(renderer, "reference_settings", None)
    return fn() if fn else {}
