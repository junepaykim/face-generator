"""Reusable setup operations with caller-supplied reporting, consent and download adapters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from facegen.platform import native
from facegen.setup import models, native_wheel


class Report(Protocol):
    def __call__(self, line: str, *, flush: bool = False) -> None: ...


def prepare_native(
    *,
    install: bool,
    source: Path | None = None,
    report: Report,
    host: native_wheel.Host | None = None,
    native_dir: Path | None = None,
    fetch: Callable[[str, Path], None] = native_wheel.download,
    installer: Callable[[Path], None] = native_wheel.uv_install,
    loads: Callable[[], bool] = native_wheel.imports_cleanly,
) -> int:
    """Inspect or install the verified wheel; failure leaves Python fallbacks available."""
    now = native.status()
    report(f"Rust code   {now.describe()}")
    try:
        chosen = native_wheel.plan(host, native_dir=native_dir)
    except (OSError, native_wheel.WheelError) as exc:
        report(f"Wheel       {exc}")
        return 3 if install else 0
    report(f"Wheel       {chosen.describe()}")
    if not install:
        if not now.in_use and chosen.wheel is not None:
            report("Run `facegen native --install` to install it.")
        return 0
    if now.in_use:
        report("Nothing to install.")
        return 0
    try:
        name = native_wheel.install(chosen, source=source, fetch=fetch, installer=installer)
    except native_wheel.WheelError as exc:
        report(f"Not installed: {exc}")
        report("facegen goes on with the Python versions.")
        return 3
    if not loads():
        report(f"Installed {name}, but it does not load here; the Python versions run.")
        return 3
    report(f"Installed {name} (SHA-256 checked): in use from facegen's next start.")
    return 0


def prepare_models(
    *,
    model: str,
    download: bool,
    confirmed: bool,
    report: Report,
    ask: Callable[[str], str],
    interactive: Callable[[], bool],
    run_download: Callable[[str], models.DownloadStatus],
    cache: models.HubCache | None = None,
) -> int:
    """Check the files of `model` (the selected `style.model`) and run their download, only
    theirs, after consent and a free-space check."""
    now = models.status(model, cache)
    report(f"Image model  {model}")
    for repo in dict.fromkeys(f.repo for f in now.files):
        left = [f for f in now.missing if f.repo == repo]
        size = models.total_size(f for f in now.files if f.repo == repo)
        state = "ready" if not left else f"{len(left)} file(s) missing"
        report(f"{repo:<38} {models.gigabytes(size):>8}  {state}  ({models.REPOS[repo]})")
    report(f"Cache        {now.cache}")
    free = "unknown" if now.free_bytes is None else models.gigabytes(now.free_bytes)
    report(f"Free space   {free}")
    if now.ready:
        report("All model files are here; nothing to download.")
        return 0
    size = models.gigabytes(now.missing_bytes)
    if not download:
        report(f"To download ({size}): facegen models --download")
        return 1
    if not now.enough_space:
        report(f"Not enough free space for {size} (and 1 GB to spare) in {now.cache}.")
        return 3
    if not confirmed:
        if not interactive():
            report("Not downloaded: confirm with --yes, or run this in a console.")
            return 3
        answer = ask(f"Download {size} from Hugging Face into {now.cache}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            report("Not downloaded.")
            return 3
    report(f"Downloading {size}; it can be stopped and started again.", flush=True)
    result = run_download(model)
    if result.error or not result.finished:
        report(f"The download did not finish: {result.error or 'stopped'}")
        report("Run the same command again. Completed files are kept.")
        return 3
    report("All model files are here.")
    return 0
