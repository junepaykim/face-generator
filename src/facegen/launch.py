"""What Start.bat asks of facegen once `uv sync` has installed it (Phase 11): `facegen launch`.

Start.bat stays thin: it finds or installs uv, runs `uv sync`, then this, then starts the
desktop app with pythonw.exe. The decisions are made here, in Python, where they are tested:

- `facegen launch --open-running`, run before `uv sync`: is a facegen app running already?
  Then it is asked to open its window (instance.py) and Start.bat ends: nothing is synced
  while facegen's own files are in use. Exit 0: it was running and took the request; 1: none
  runs; 4: one runs but did not answer.
- `facegen launch`: the prebuilt Rust extension (`facegen native --install`; a failure is a
  warning, as the Python versions do the same work), then the model files: when some are
  missing, their size is shown and the download starts only after one "y" (`facegen models
  --download`, AGENTS.md rule 4); "no" is fine, the dashboard offers it again. Exit 0: the
  app starts without either, so neither step's failure stops it (an unexpected error in a
  step is shown as a warning too).

Settings it cannot read are said in plain words, not as a traceback (exit 2). One case is put
right after a question: a local file copied from WSL names a folder as WSL sees it
(`/mnt/c/...`), which Windows refuses (fm_paths.native_path); with a "y" the line is written in
Windows's form (folder_setup.write_local_path keeps the file's other lines), as facegen has
written it since Phase 10, and that form works on WSL too.

Everything it prints also goes to `--log` (Start.bat's data/logs/start.log), except the
download's progress lines.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from facegen import fm_paths, instance, platform_info
from facegen.platform_info import Kind

LOG_LIMIT = 1_000_000
"""start.log is renamed to start.log.1 when it is larger than this."""
_PROGRESS = re.compile(r"^\s+\d+%")

OPENED, NOT_RUNNING, NO_ANSWER = 0, 1, 4


class Tee:
    """A text stream that writes to the console and appends to the log (progress lines to
    the console only)."""

    def __init__(self, console: TextIO | None, log: TextIO | None) -> None:
        self.console = console
        self.log = log
        self._line = ""

    def write(self, text: str) -> int:
        if self.console is not None:
            self.console.write(text)
        if self.log is not None:
            self._line += text
            *lines, self._line = self._line.split("\n")
            for line in lines:
                if not _PROGRESS.match(line):
                    self.log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
        return len(text)

    def flush(self) -> None:
        if self.console is not None:
            self.console.flush()
        if self.log is not None:
            self.log.flush()


def open_log(path: Path | None) -> TextIO | None:
    """Append to the log, after setting a large one aside; None if it cannot be written."""
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > LOG_LIMIT:
            path.replace(path.with_name(path.name + ".1"))
        return path.open("a", encoding="utf-8")
    except OSError:
        return None


def open_running(data_dir: Path, out: TextIO, *, timeout: float | None = None) -> int:
    """If a facegen app runs, ask it to open its window (exit codes in the module docstring)."""
    if not instance.held_elsewhere(data_dir):
        return NOT_RUNNING
    instance.request_open(data_dir)
    if instance.wait_taken(data_dir, timeout):
        print("facegen is running already: its window opens.", file=out, flush=True)
        return OPENED
    print(
        "facegen is running already, but it did not answer. If it is still starting, wait a"
        " moment and start it again; if it seems stuck, end it in Task Manager.",
        file=out,
        flush=True,
    )
    return NO_ANSWER


def prepare(
    out: TextIO,
    *,
    native_step: Callable[[TextIO], int],
    models_step: Callable[[TextIO], int],
) -> int:
    """The steps after `uv sync`; 0, as the app can start without either (docstring)."""
    print("facegen: checking the Rust extension and the model files.", file=out, flush=True)
    if _step(native_step, out) != 0:
        print(
            "Warning: the prebuilt Rust extension is not installed; facegen runs the same"
            " without it, a little slower to publish.",
            file=out,
            flush=True,
        )
    if _step(models_step, out) != 0:
        print(
            "The models are not downloaded yet. The dashboard offers the download; until then"
            " no pictures are drawn.",
            file=out,
            flush=True,
        )
    return 0


def _step(step: Callable[[TextIO], int], out: TextIO) -> int:
    """A step's exit code; an unexpected error is shown and counts as a failed step."""
    try:
        return step(out)
    except Exception as exc:  # noqa: BLE001 - the app can start without this step
        print(f"Warning: {type(exc).__name__}: {exc}", file=out, flush=True)
        return 1


def asker(out: TextIO, read: Callable[[], str] = input) -> Callable[[str], str]:
    """The model download's question, asked on the console and written to the log too."""

    def ask(question: str) -> str:
        out.write(question)
        out.flush()
        answer = read()
        if getattr(out, "log", None) is not None:
            out.log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} answer: {answer!r}\n")  # type: ignore[attr-defined]
        return answer

    return ask


def wsl_paths(local: Path, kind: Kind) -> dict[str, tuple[str, str]]:
    """On Windows, the local file's `[paths]` keys written as WSL sees them (`/mnt/c/...`),
    each with its Windows form. Empty elsewhere, and when the file is missing or unreadable."""
    if kind != "windows":
        return {}
    try:
        data = tomllib.loads(local.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    paths = data.get("paths")
    found: dict[str, tuple[str, str]] = {}
    for key, value in paths.items() if isinstance(paths, dict) else []:
        if isinstance(value, str) and (windows := fm_paths.windows_path_of_wsl(value)):
            found[key] = (value, windows)
    return found


def data_dir_of(
    config: Path | None,
    out: TextIO,
    *,
    kind: Kind,
    ask: Callable[[str], str],
    interactive: Callable[[], bool],
) -> Path | None:
    """The data folder from the settings; None, after saying why, when they cannot be read.
    A WSL path in the local file is offered in Windows's form first (module docstring)."""
    from facegen import folder_setup
    from facegen.settings import SettingsError, load_settings, local_label, local_path

    try:
        return load_settings(defaults_path=config).paths.data_dir
    except SettingsError as exc:
        error = exc
    local = local_path(config)
    found = wsl_paths(local, kind)
    if found and interactive():
        print(f"{local_label(config)} names a folder as WSL sees it:", file=out)
        for key, (old, new) in found.items():
            print(f"  {key} = {old}\n  on Windows: {new}", file=out)
        answer = ask("Write it in Windows's form (it works on WSL too)? [Y/n] ")
        if answer.strip().lower() in ("", "y", "yes"):
            try:
                for key, (_, new) in found.items():
                    folder_setup.write_local_path(local, key, new)
                data_dir = load_settings(defaults_path=config).paths.data_dir
            except (folder_setup.LocalFileError, OSError, SettingsError) as exc:
                error = exc
            else:
                print(f"Written in {local_label(config)}.", file=out, flush=True)
                return data_dir
    print(f"facegen's settings cannot be read: {error}", file=out, flush=True)
    return None


def cmd_launch(args: argparse.Namespace, out: TextIO) -> int:
    """`facegen launch` (module docstring)."""
    from facegen import cli

    log = open_log(args.log)
    tee = Tee(out, log)

    def interactive() -> bool:
        return sys.stdin is not None and sys.stdin.isatty()

    try:
        if args.open_running:
            # Settings it cannot read are the next step's to explain: only the log hears it.
            data_dir = data_dir_of(
                args.config,
                Tee(None, log),
                kind=platform_info.current(),
                ask=input,
                interactive=lambda: False,
            )
            return NOT_RUNNING if data_dir is None else open_running(data_dir, tee)
        data_dir = data_dir_of(
            args.config,
            tee,
            kind=platform_info.current(),
            ask=asker(tee),
            interactive=interactive,
        )
        if data_dir is None:
            return 2
        native_args = argparse.Namespace(install=True, wheel_dir=None)
        model_args = argparse.Namespace(download=True, yes=False, child=False)
        return prepare(
            tee,
            native_step=lambda o: cli.cmd_native(native_args, o),
            models_step=lambda o: cli.cmd_models(
                model_args, o, ask=asker(o), interactive=interactive
            ),
        )
    finally:
        tee.flush()
        if log is not None:
            log.close()
