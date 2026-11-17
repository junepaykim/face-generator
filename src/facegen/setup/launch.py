"""Start.bat's activation and setup decisions, exposed as `facegen launch`.

`--open-running` runs before uv sync to avoid replacing files in use: exit 0 means the
running app accepted activation, 1 means no app, and 4 means an app did not answer.
Normal launch installs the optional native wheel, then offers missing models after
showing their size and obtaining consent. Either step may fail without blocking startup
(exit 0); declining models leaves the dashboard's download option available.

Unreadable settings produce a plain error (exit 2). Converting a copied WSL folder path
to Windows form requires confirmation and preserves the local file's other text.
Output is also written to `--log`, except download progress.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TextIO

from facegen.desktop import instance
from facegen.platform import fm_paths, platform_info
from facegen.platform.platform_info import Kind

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
    from facegen.persistence.settings import SettingsError, load_settings, local_label, local_path
    from facegen.setup import folder_setup

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
    from facegen.setup import models
    from facegen.setup import operations as setup_operations

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
        return prepare(
            tee,
            native_step=lambda o: setup_operations.prepare_native(
                install=True, report=partial(print, file=o)
            ),
            models_step=lambda o: setup_operations.prepare_models(
                download=True,
                confirmed=False,
                report=partial(print, file=o),
                ask=asker(o),
                interactive=interactive,
                run_download=lambda: models.wait_and_print(models.ModelDownload(), o),
            ),
        )
    finally:
        tee.flush()
        if log is not None:
            log.close()
