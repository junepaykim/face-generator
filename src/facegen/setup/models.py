"""Model cache inspection and resumable downloads, after explicit user consent.

MODELS lists, for each image model (`style.model`), every file it needs: its own, IP-Adapter's
where it takes a reference picture, and the tagger's. Only the selected model's files are
checked and offered. HubCache checks `refs/main` without importing huggingface_hub, accepting
both symlinks and Windows copies. Rendering stays offline; only `facegen models --child` clears
the offline flags. The child emits JSON lines on stdout: plan, current file, byte progress and
completion or error.

The caller must obtain consent in the UI or CLI (including `--yes`) before starting a
download, for the model it names. Finished files are skipped; huggingface_hub resumes partial
files.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from facegen.platform import procs
from facegen.platform.pipes import JsonWriter, drain_lines

logger = logging.getLogger("facegen")

REVISION = "main"
"""Downloads must populate `refs/main`, which the offline renderer and tagger load."""


@dataclass(frozen=True, slots=True)
class ModelFile:
    repo: str
    filename: str
    size: int
    """Bytes at the commit noted beside the list (Hub API), for consent, free-space checks and
    progress; a later revision may differ a little."""

    @property
    def label(self) -> str:
        return f"{self.repo}/{self.filename}"


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
REALVIS = "SG161222/RealVisXL_V5.0"
ZIMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"
IP_ADAPTER = "h94/IP-Adapter"
TAGGER = "SmilingWolf/wd-eva02-large-tagger-v3"

REPOS = {
    ANIMAGINE: "Animagine XL 4.0, the image model",
    REALVIS: "RealVisXL V5.0, the image model",
    ZIMAGE_TURBO: "Z-Image-Turbo, the image model",
    IP_ADAPTER: "IP-Adapter plus-face, for re-draws that keep the face",
    TAGGER: "WD EVA02 tagger, for the quality check",
}
"""Every repository a model can need, in the order they are listed."""


def _files(repo: str, sizes: Mapping[str, int]) -> tuple[ModelFile, ...]:
    return tuple(ModelFile(repo, name, size) for name, size in sizes.items())


ANIMAGINE_FILES = _files(  # main at 2b7c1b39
    ANIMAGINE,
    {
        "model_index.json": 671,
        "scheduler/scheduler_config.json": 613,
        "text_encoder/config.json": 560,
        "text_encoder/model.safetensors": 246_144_152,
        "text_encoder_2/config.json": 570,
        "text_encoder_2/model.safetensors": 1_389_382_176,
        "tokenizer/merges.txt": 524_619,
        "tokenizer/special_tokens_map.json": 472,
        "tokenizer/tokenizer_config.json": 704,
        "tokenizer/vocab.json": 1_059_962,
        "tokenizer_2/merges.txt": 524_619,
        "tokenizer_2/special_tokens_map.json": 460,
        "tokenizer_2/tokenizer_config.json": 855,
        "tokenizer_2/vocab.json": 1_059_962,
        "unet/config.json": 1_772,
        "unet/diffusion_pytorch_model.safetensors": 5_135_149_760,
        "vae/config.json": 782,
        "vae/diffusion_pytorch_model.safetensors": 167_335_342,
    },
)
REALVIS_FILES = _files(  # main at ac93e0dd; the fp16 variant, its own VAE
    REALVIS,
    {
        "model_index.json": 708,
        "scheduler/scheduler_config.json": 526,
        "text_encoder/config.json": 584,
        "text_encoder/model.fp16.safetensors": 246_144_152,
        "text_encoder_2/config.json": 594,
        "text_encoder_2/model.fp16.safetensors": 1_389_382_176,
        "tokenizer/merges.txt": 573_514,
        "tokenizer/special_tokens_map.json": 496,
        "tokenizer/tokenizer_config.json": 734,
        "tokenizer/vocab.json": 1_109_372,
        "tokenizer_2/merges.txt": 573_514,
        "tokenizer_2/special_tokens_map.json": 484,
        "tokenizer_2/tokenizer_config.json": 893,
        "tokenizer_2/vocab.json": 1_109_372,
        "unet/config.json": 1_850,
        "unet/diffusion_pytorch_model.fp16.safetensors": 5_135_149_760,
        "vae/config.json": 807,
        "vae/diffusion_pytorch_model.fp16.safetensors": 167_335_342,
    },
)
ZIMAGE_TURBO_FILES = _files(  # main at f332072a; everything from this one repository
    ZIMAGE_TURBO,
    {
        "model_index.json": 467,
        "scheduler/scheduler_config.json": 173,
        "text_encoder/config.json": 726,
        "text_encoder/model-00001-of-00003.safetensors": 3_957_900_840,
        "text_encoder/model-00002-of-00003.safetensors": 3_987_450_520,
        "text_encoder/model-00003-of-00003.safetensors": 99_630_640,
        "text_encoder/model.safetensors.index.json": 32_819,
        "tokenizer/merges.txt": 1_671_853,
        "tokenizer/tokenizer.json": 11_422_654,
        "tokenizer/tokenizer_config.json": 9_732,
        "tokenizer/vocab.json": 2_776_833,
        "transformer/config.json": 473,
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": 9_973_693_184,
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": 9_973_714_824,
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": 4_672_282_880,
        "transformer/diffusion_pytorch_model.safetensors.index.json": 48_969,
        "vae/config.json": 805,
        "vae/diffusion_pytorch_model.safetensors": 167_666_902,
    },
)
IP_ADAPTER_FILES = _files(  # main at 018e4027
    IP_ADAPTER,
    {
        "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors": 847_517_512,
        "models/image_encoder/config.json": 560,
        "models/image_encoder/model.safetensors": 2_528_373_448,
    },
)
TAGGER_FILES = _files(  # main at b25b82a0
    TAGGER,
    {
        "config.json": 634,
        "model.safetensors": 1_260_796_004,
        "selected_tags.csv": 308_468,
    },
)

DEFAULT_MODEL = "animagine"
MODELS: dict[str, tuple[ModelFile, ...]] = {
    "animagine": (*ANIMAGINE_FILES, *IP_ADAPTER_FILES, *TAGGER_FILES),
    "realvis": (*REALVIS_FILES, *IP_ADAPTER_FILES, *TAGGER_FILES),
    "zimage_turbo": (*ZIMAGE_TURBO_FILES, *TAGGER_FILES),
}
"""The files each `style.model` loads (sizes from the Hub API, 2026-11-20). Z-Image-Turbo takes
no reference picture, so it needs no IP-Adapter."""


def files_for(model: str) -> tuple[ModelFile, ...]:
    """The files the image model `model` (a `style.model` key) needs."""
    try:
        return MODELS[model]
    except KeyError:
        raise ValueError(f"no image model called {model!r}") from None


SPACE_MARGIN = 1 << 30
"""Free space wanted beyond the files themselves: huggingface_hub writes each file beside its
final place before moving it, and the disk should not be left full."""


def total_size(files: Iterable[ModelFile]) -> int:
    return sum(f.size for f in files)


def missing(files: Iterable[ModelFile], has: Callable[[ModelFile], bool]) -> list[ModelFile]:
    """The files `has` does not find, in the order given. Pure."""
    return [f for f in files if not has(f)]


def gigabytes(n: int) -> str:
    """Bytes for people: "11.6 GB" (decimal, as disks are sold), "640 MB" below one."""
    if n >= 10**9:
        return f"{n / 10**9:.1f} GB"
    return f"{max(n, 0) / 10**6:.0f} MB"


def hub_cache_root(env: Mapping[str, str] | None = None) -> Path:
    """The Hugging Face cache folder, as huggingface_hub works it out (HF_HUB_CACHE, HF_HOME,
    then ~/.cache/huggingface/hub). Without huggingface_hub (an install without the GPU group,
    such as CI's), the same rules from its documentation, so the status still answers."""
    if env is None:
        try:
            from huggingface_hub import constants
        except ImportError:
            pass
        else:
            return Path(constants.HF_HUB_CACHE)
        env = os.environ
    cache = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache).expanduser()
    home = env.get("HF_HOME")
    if not home:
        base = env.get("XDG_CACHE_HOME") or str(Path("~/.cache").expanduser())
        home = str(Path(base) / "huggingface")
    return Path(home).expanduser() / "hub"


def _snapshot_has(snapshot: Path | None, filename: str) -> bool:
    try:
        return snapshot is not None and (snapshot / filename).is_file()
    except OSError:
        return False


class HubCache:
    """The Hugging Face cache at `root`, read directly (layout: models--<org>--<name>/refs/main,
    snapshots/<commit>/<file>, blobs/)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def repo_dir(self, repo: str) -> Path:
        return self.root / ("models--" + repo.replace("/", "--"))

    def snapshot(self, repo: str) -> Path | None:
        try:
            commit = (self.repo_dir(repo) / "refs" / REVISION).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return self.repo_dir(repo) / "snapshots" / commit if commit else None

    def has(self, f: ModelFile) -> bool:
        return _snapshot_has(self.snapshot(f.repo), f.filename)

    def missing_files(self, files: Iterable[ModelFile]) -> list[ModelFile]:
        """Resolve each repository's main reference once for this check."""
        snapshot = functools.cache(self.snapshot)
        return missing(files, lambda f: _snapshot_has(snapshot(f.repo), f.filename))

    def partial_bytes(self, repo: str) -> int:
        """Bytes in the repository's unfinished downloads (`blobs/*.incomplete`)."""
        total = 0
        try:
            with os.scandir(self.repo_dir(repo) / "blobs") as entries:
                for e in entries:
                    if e.name.endswith(".incomplete"):
                        try:
                            total += e.stat().st_size
                        except OSError:
                            continue
        except OSError:
            return 0
        return total


def free_space(path: Path) -> int | None:
    """Free bytes on the disk that holds `path`, or the nearest folder above it that exists."""
    for folder in (path, *path.parents):
        try:
            if folder.exists():
                return shutil.disk_usage(folder).free
        except OSError:
            continue
    return None


@dataclass(frozen=True, slots=True)
class ModelsStatus:
    model: str
    """The `style.model` key whose files these are."""
    cache: Path
    files: tuple[ModelFile, ...]
    """All the model's files, missing or not."""
    missing: tuple[ModelFile, ...]
    free_bytes: int | None

    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def total_bytes(self) -> int:
        return total_size(self.files)

    @property
    def missing_bytes(self) -> int:
        return total_size(self.missing)

    @property
    def enough_space(self) -> bool:
        return self.free_bytes is None or self.free_bytes >= self.missing_bytes + SPACE_MARGIN

    def missing_repos(self) -> list[str]:
        """The repositories with a file missing, in the list's order."""
        return list(dict.fromkeys(f.repo for f in self.missing))

    def describe(self) -> str:
        n = len(self.files)
        if self.ready:
            return f"{self.model}: all {n} files are in {self.cache}"
        return (
            f"{self.model}: {len(self.missing)} of {n} files missing"
            f" ({gigabytes(self.missing_bytes)}), to go into {self.cache}"
        )


def status(model: str, cache: HubCache | None = None) -> ModelsStatus:
    """The files of the image model `model` (a `style.model` key) in the cache: `cache`, or
    this PC's Hugging Face cache."""
    files = files_for(model)
    cache = cache or HubCache(hub_cache_root())
    return ModelsStatus(
        model=model,
        cache=cache.root,
        files=files,
        missing=tuple(cache.missing_files(files)),
        free_bytes=free_space(cache.root),
    )


def ready_now(model: str) -> bool:
    """Whether the selected model's files are all in this PC's cache: the worker asks before
    it loads the models."""
    try:
        return status(model).ready
    except Exception:  # noqa: BLE001 - a cache it cannot read: loading reports the real error
        return True


CHILD_ENV_DROP = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
CHILD_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
PROGRESS_SECONDS = 1.0


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The download process's environment: online (the offline switches dropped), quiet."""
    env = dict(os.environ if base is None else base)
    for name in CHILD_ENV_DROP:
        env.pop(name, None)
    env.update(CHILD_ENV)
    return env


def hf_fetch(f: ModelFile) -> None:
    """Download one file into the cache with huggingface_hub (resumes a partial one)."""
    for name in CHILD_ENV_DROP:  # read when huggingface_hub is imported
        os.environ.pop(name, None)
    from huggingface_hub import hf_hub_download

    hf_hub_download(f.repo, f.filename, revision=REVISION)


def run_child(
    out: TextIO,
    *,
    files: Sequence[ModelFile],
    cache: HubCache | None = None,
    fetch: Callable[[ModelFile], None] = hf_fetch,
    interval: float = PROGRESS_SECONDS,
) -> int:
    """`facegen models --child --model KEY`: download what is missing of `files` (that model's
    list), one file at a time, telling the parent on `out`. Returns the exit code: 0 when every
    file is there, 1 otherwise."""
    cache = cache or HubCache(hub_cache_root())
    writer = JsonWriter()

    def send(message: dict[str, Any]) -> None:
        writer.send(out, message)

    todo = missing(files, cache.has)
    total = total_size(todo)
    send({"type": "plan", "files": len(todo), "bytes": total})
    finished = 0
    current: list[ModelFile] = []
    stop = threading.Event()

    def bytes_now() -> int:
        if not current:
            return finished
        f = current[0]
        return finished + min(cache.partial_bytes(f.repo), f.size)

    def report() -> None:
        while not stop.wait(interval):
            send({"type": "progress", "done": bytes_now()})

    reporter = threading.Thread(target=report, name="facegen-models-progress", daemon=True)
    reporter.start()
    try:
        for i, f in enumerate(todo, start=1):
            current[:] = [f]
            send({"type": "file", "index": i, "name": f.label, "size": f.size})
            fetch(f)
            finished += f.size
            current.clear()
            send({"type": "progress", "done": finished})
    except BaseException as exc:  # noqa: BLE001 - say why, then end
        stop.set()
        message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        send({"type": "error", "message": message[:1000]})
        return 1
    finally:
        stop.set()
        reporter.join(interval + 1)
    left = missing(files, cache.has)
    if left:
        names = ", ".join(f.label for f in left[:3])
        send({"type": "error", "message": f"still missing after the download: {names}"})
        return 1
    send({"type": "done", "bytes": total})
    return 0


def child_command(program: Sequence[str] | None = None) -> list[str]:
    """The command line that starts the download process: this Python, unless `program`.
    `ModelDownload.start` adds `--model KEY`."""
    base = list(program) if program else [sys.executable, "-m", "facegen.cli"]
    return [*base, "models", "--child"]


@dataclass
class DownloadStatus:
    running: bool = False
    model: str | None = None
    """The `style.model` key whose files are (or were last) being downloaded."""
    started: str | None = None
    done_bytes: int = 0
    total_bytes: int = 0
    files: int = 0
    index: int = 0
    current: str | None = None
    error: str | None = None
    finished: bool = False
    """The last download ended with every file there."""
    log: list[str] = field(default_factory=list)

    @property
    def percent(self) -> int:
        if self.finished:
            return 100
        if not self.total_bytes:
            return 0
        return min(100, int(100 * self.done_bytes / self.total_bytes))


class ModelDownload:
    """The parent's side of the download: starts `facegen models --child --model KEY`, reads
    its messages, keeps the status for the dashboard. One download at a time."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        on_done: Callable[[], Any] = lambda: None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = list(command) if command else child_command()
        self._on_done = on_done
        self._env = env
        self._lock = threading.Lock()
        self._status = DownloadStatus()
        self._child: subprocess.Popen[str] | None = None
        self._stopping = False
        self._thread: threading.Thread | None = None

    def status(self) -> DownloadStatus:
        with self._lock:
            s = self._status
            return DownloadStatus(**{**vars(s), "log": list(s.log)})

    def start(self, model: str) -> bool:
        """Start downloading the files of `model` (a `style.model` key, the one the user agreed
        to) in the background; False if a download is running already."""
        files_for(model)  # a model with no list is refused before anything starts
        with self._lock:
            if self._status.running:
                return False
            self._status = DownloadStatus(
                running=True, model=model, started=datetime.now().strftime("%H:%M:%S")
            )
            self._stopping = False
        command = [*self._command, "--model", model]
        self._thread = threading.Thread(
            target=self._run, args=(command,), name="facegen-models", daemon=True
        )
        self._thread.start()
        return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self) -> None:
        """End a running download (facegen is quitting); it resumes at the next start."""
        with self._lock:
            self._stopping = True  # a child not started yet is ended as soon as it is
            child = self._child
        if child is not None and child.poll() is None:
            child.kill()

    def _update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._status, key, value)

    def receive(self, message: dict[str, Any]) -> None:
        """One message from the download process."""
        match message.get("type"):
            case "plan":
                self._update(total_bytes=int(message["bytes"]), files=int(message["files"]))
            case "file":
                self._update(index=int(message["index"]), current=str(message["name"]))
                logger.info("models: downloading %s", message["name"])
            case "progress":
                self._update(done_bytes=int(message["done"]))
            case "done":
                self._update(finished=True, current=None, done_bytes=int(message["bytes"]))
            case "error":
                self._update(error=str(message["message"]))
            case _:
                logger.warning("models: unknown message %r", message)

    def _run(self, command: list[str]) -> None:
        try:
            child = procs.popen_json(
                command,
                detached=False,
                stdin=subprocess.DEVNULL,
                env=child_env(self._env),
            )
        except OSError as exc:
            self._update(running=False, error=f"the download could not be started: {exc}")
            return
        with self._lock:
            self._child = child
            if self._stopping:
                child.kill()
        tail: list[str] = []

        def drain() -> None:
            assert child.stderr is not None

            def receive(line: str) -> None:
                tail[:] = [*tail[-19:], line]
                logger.info("models: %s", line)

            drain_lines(child.stderr, receive)

        reader = threading.Thread(target=drain, name="facegen-models-stderr", daemon=True)
        reader.start()
        assert child.stdout is not None
        for raw in child.stdout:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.info("models: %s", raw.rstrip())
                continue
            if isinstance(message, dict):
                try:
                    self.receive(message)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("models: bad message %r (%s)", raw[:200], exc)
        code = child.wait()
        reader.join(5)
        with self._lock:
            self._child = None
            s = self._status
            if code != 0 and not s.error:
                last = tail[-1] if tail else f"exit code {code}"
                s.error = f"the download stopped ({last})"
            s.running = False
            s.log = tail[-5:]
        self._on_done()


def wait_and_print(
    download: ModelDownload, model: str, out: TextIO, *, every: float = 2.0
) -> DownloadStatus:
    """`facegen models --download`: start it for `model` and print its progress until it
    ends."""
    download.start(model)
    shown = None
    while True:
        s = download.status()
        line = (
            f"  {s.percent:3d}%  {gigabytes(s.done_bytes)} of {gigabytes(s.total_bytes)}"
            f"  file {s.index} of {s.files}: {s.current}"
            if s.current
            else None
        )
        if line and line != shown:
            print(line, file=out, flush=True)
            shown = line
        if not s.running:
            return s
        time.sleep(every)
