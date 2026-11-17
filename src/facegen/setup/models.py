"""The model files facegen draws and checks with, and their one-time download (Phase 10b).

The files are those the README's three `hf download` commands fetch, as data (`FILES`): the
image model, IP-Adapter's face model with its image encoder, and the tagger behind the quality
check, about 11.6 GB together. They go into the Hugging Face cache, where the renderer and the
tagger load them with no network (gpu.py sets `HF_HUB_OFFLINE`).

- `missing(files, has)` is pure: the files `has` does not find. `HubCache` answers `has` for
  the cache on disk without importing huggingface_hub: a file is there when the snapshot that
  the repository's `refs/main` names holds it (a symlink into `blobs/`, or a copy on Windows
  without Developer Mode).
- `status()` is what the CLI and the dashboard show: what is missing, its size, and the free
  space where the cache is.
- The download runs in a process of its own (`facegen models --child`), with `HF_HUB_OFFLINE`
  unset only there, and reports on its stdout one JSON object per line: the plan, each file it
  starts, the bytes so far about once a second, and the end. `ModelDownload` is the parent's
  side, for the web UI and `facegen models --download`. A download that stops is taken up again
  where it stopped: finished files are skipped and huggingface_hub resumes a partial one.

AGENTS.md hard rule 4: nothing is downloaded until the user has confirmed it once, in the UI
(`POST /setup/models`) or at `facegen models --download`'s question (or with `--yes`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from facegen.platform import procs
from facegen.platform.pipes import JsonWriter, drain_lines

logger = logging.getLogger("facegen")

REVISION = "main"
"""The README's commands fetch each repository's main branch, and the renderer and the tagger
load `refs/main` from the cache: the download must write that ref, so it asks for main too."""


@dataclass(frozen=True, slots=True)
class ModelFile:
    repo: str
    filename: str
    size: int
    """Bytes, as fetched on this PC in November 2026: for the size shown before the download,
    the disk space check and the progress bar. A later revision may differ a little."""

    @property
    def label(self) -> str:
        return f"{self.repo}/{self.filename}"


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
IP_ADAPTER = "h94/IP-Adapter"
TAGGER = "SmilingWolf/wd-eva02-large-tagger-v3"

REPOS = {
    ANIMAGINE: "Animagine XL 4.0, the image model",
    IP_ADAPTER: "IP-Adapter plus-face, for re-draws that keep the face",
    TAGGER: "WD EVA02 tagger, for the quality check",
}

FILES: tuple[ModelFile, ...] = (
    # uv run hf download cagliostrolab/animagine-xl-4.0 --include "model_index.json"
    #   --include "scheduler/*" --include "text_encoder*/*" --include "tokenizer*/*"
    #   --include "unet/*" --include "vae/*"
    ModelFile(ANIMAGINE, "model_index.json", 671),
    ModelFile(ANIMAGINE, "scheduler/scheduler_config.json", 613),
    ModelFile(ANIMAGINE, "text_encoder/config.json", 560),
    ModelFile(ANIMAGINE, "text_encoder/model.safetensors", 246_144_152),
    ModelFile(ANIMAGINE, "text_encoder_2/config.json", 570),
    ModelFile(ANIMAGINE, "text_encoder_2/model.safetensors", 1_389_382_176),
    ModelFile(ANIMAGINE, "tokenizer/merges.txt", 524_619),
    ModelFile(ANIMAGINE, "tokenizer/special_tokens_map.json", 472),
    ModelFile(ANIMAGINE, "tokenizer/tokenizer_config.json", 704),
    ModelFile(ANIMAGINE, "tokenizer/vocab.json", 1_059_962),
    ModelFile(ANIMAGINE, "tokenizer_2/merges.txt", 524_619),
    ModelFile(ANIMAGINE, "tokenizer_2/special_tokens_map.json", 460),
    ModelFile(ANIMAGINE, "tokenizer_2/tokenizer_config.json", 855),
    ModelFile(ANIMAGINE, "tokenizer_2/vocab.json", 1_059_962),
    ModelFile(ANIMAGINE, "unet/config.json", 1_772),
    ModelFile(ANIMAGINE, "unet/diffusion_pytorch_model.safetensors", 5_135_149_760),
    ModelFile(ANIMAGINE, "vae/config.json", 782),
    ModelFile(ANIMAGINE, "vae/diffusion_pytorch_model.safetensors", 167_335_342),
    # uv run hf download h94/IP-Adapter sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors
    #   models/image_encoder/config.json models/image_encoder/model.safetensors
    ModelFile(IP_ADAPTER, "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", 847_517_512),
    ModelFile(IP_ADAPTER, "models/image_encoder/config.json", 560),
    ModelFile(IP_ADAPTER, "models/image_encoder/model.safetensors", 2_528_373_448),
    # uv run hf download SmilingWolf/wd-eva02-large-tagger-v3 config.json model.safetensors
    #   selected_tags.csv
    ModelFile(TAGGER, "config.json", 634),
    ModelFile(TAGGER, "model.safetensors", 1_260_796_004),
    ModelFile(TAGGER, "selected_tags.csv", 308_468),
)

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


def hub_cache_root() -> Path:
    """The Hugging Face cache folder, as huggingface_hub works it out (HF_HUB_CACHE, HF_HOME,
    then ~/.cache/huggingface/hub)."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


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
        snapshot = self.snapshot(f.repo)
        try:
            return snapshot is not None and (snapshot / f.filename).is_file()
        except OSError:
            return False

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
    cache: Path
    missing: tuple[ModelFile, ...]
    total_bytes: int
    """All the files, missing or not."""
    free_bytes: int | None

    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def missing_bytes(self) -> int:
        return total_size(self.missing)

    @property
    def enough_space(self) -> bool:
        return self.free_bytes is None or self.free_bytes >= self.missing_bytes + SPACE_MARGIN

    def missing_repos(self) -> list[str]:
        return sorted({f.repo for f in self.missing}, key=list(REPOS).index)

    def describe(self) -> str:
        if self.ready:
            return f"all {len(FILES)} files are in {self.cache}"
        return (
            f"{len(self.missing)} of {len(FILES)} files missing ({gigabytes(self.missing_bytes)}),"
            f" to go into {self.cache}"
        )


def status(cache: HubCache | None = None, files: Sequence[ModelFile] = FILES) -> ModelsStatus:
    cache = cache or HubCache(hub_cache_root())
    return ModelsStatus(
        cache=cache.root,
        missing=tuple(missing(files, cache.has)),
        total_bytes=total_size(files),
        free_bytes=free_space(cache.root),
    )


def ready_now() -> bool:
    """Whether every file is in this PC's cache: the worker asks before it loads the models."""
    try:
        return status().ready
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
    cache: HubCache | None = None,
    files: Sequence[ModelFile] = FILES,
    fetch: Callable[[ModelFile], None] = hf_fetch,
    interval: float = PROGRESS_SECONDS,
) -> int:
    """`facegen models --child`: download what is missing, one file at a time, telling the
    parent on `out`. Returns the exit code: 0 when every file is there, 1 otherwise."""
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
    """The command line that starts the download process: this Python, unless `program`."""
    base = list(program) if program else [sys.executable, "-m", "facegen.cli"]
    return [*base, "models", "--child"]


@dataclass
class DownloadStatus:
    running: bool = False
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
    """The parent's side of the download: starts `facegen models --child`, reads its messages,
    keeps the status for the dashboard. One download at a time."""

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

    def start(self) -> bool:
        """Start the download in the background; False if one is running already."""
        with self._lock:
            if self._status.running:
                return False
            self._status = DownloadStatus(running=True, started=datetime.now().strftime("%H:%M:%S"))
            self._stopping = False
        self._thread = threading.Thread(target=self._run, name="facegen-models", daemon=True)
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

    def _run(self) -> None:
        try:
            child = procs.popen_json(
                self._command,
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


def wait_and_print(download: ModelDownload, out: TextIO, *, every: float = 2.0) -> DownloadStatus:
    """`facegen models --download`: start it and print its progress until it ends."""
    download.start()
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
