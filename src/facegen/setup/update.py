"""facegen's releases: the update check and the update in place (Phase 13).

The check is one HTTPS request for this repository's releases list on GitHub, sent with a
generic client name and nothing about this PC or its user; `updates.check` turns it off. App
releases are tagged `v<major>.<minor>.<patch>`. Drafts, pre-releases and the Rust wheels'
`native-v...` releases are not app releases.

A release's files (its payload) are its tagged source minus what only developers use: the
same selection makes the installer's folder and an update. An update in place (`facegen
update`, run by `Start.bat --update`) fetches the tag's source archive, checks it, and puts its
payload over this folder. data/, .venv/, .bin/ and config/local.toml are never written; files
under src/ and native/ that the release no longer has are removed. Every file replaced or
removed is copied into data/update/backup-<old version>/ first, and put back if anything
fails. A git checkout is not updated this way: `git pull` is its update.

Only the standard library is used, so the update runs whatever the venv holds; everything it
needs is imported before the first file is replaced.
"""

from __future__ import annotations

import filecmp
import json
import os
import re
import shutil
import stat
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from facegen.locations import PROJECT_ROOT

REPOSITORY = "junepaykim/face-generator"
RELEASES_API = f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=30"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases"
ARCHIVE_URL = "https://github.com/{repository}/archive/refs/tags/{tag}.zip"
USER_AGENT = "facegen"
"""The only header that says anything: the program's name, not its version or the PC's."""
CHECK_TIMEOUT = 20.0
DOWNLOAD_TIMEOUT = 120.0
MAX_LIST_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_UNPACKED_BYTES = 400 * 1024 * 1024
MAX_ENTRIES = 20_000

TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

LEFT_OUT = frozenset(
    {".github", "tests", "scripts", "docs", "installer", ".gitignore", ".gitattributes",
     "AGENTS.md", "CLAUDE.md"}
)  # fmt: skip
"""Top-level entries of the source that a release does not ship: developer files."""
NEVER_WRITTEN = (".git", ".venv", ".bin", "data", "config/local.toml")
"""This PC's own: an archive that had them is refused rather than written over them."""
MANAGED_DIRS = ("src", "native")
"""Folders whose files are the release's alone: a file the new release lacks is removed."""
SKIPPED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})
SKIPPED_UNDER = (PurePosixPath("native/target"),)
"""Build output, never part of a payload and never removed by an update."""

Version = tuple[int, int, int]


class UpdateError(Exception):
    """The check or the update could not be done (the message says why)."""


def parse_version(text: str | None) -> Version | None:
    """(1, 2, 3) for "1.2.3" or "v1.2.3"; None for anything else (a pre-release, say)."""
    if not text:
        return None
    m = TAG.match(text if text.startswith("v") else f"v{text}")
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def project_version(pyproject_text: str) -> str:
    try:
        return str(tomllib.loads(pyproject_text)["project"]["version"])
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise UpdateError(f"pyproject.toml has no project version: {exc}") from exc


def installed_version(root: Path = PROJECT_ROOT) -> str:
    """This folder's version, from its pyproject.toml (what an update replaces)."""
    try:
        return project_version((root / "pyproject.toml").read_text(encoding="utf-8"))
    except OSError as exc:
        raise UpdateError(f"pyproject.toml could not be read: {exc}") from exc


def is_checkout(root: Path = PROJECT_ROOT) -> bool:
    """A git checkout (a developer's), updated with git rather than in place."""
    return (root / ".git").exists()


@dataclass(frozen=True, slots=True)
class Release:
    version: str
    tag: str
    page: str
    """The release's page on GitHub: what is new."""


def choose_release(items: Iterable[Any]) -> Release | None:
    """The newest app release in GitHub's releases list (module docstring)."""
    best: tuple[Version, Release] | None = None
    for item in items:
        if not isinstance(item, dict) or item.get("draft") or item.get("prerelease"):
            continue
        tag = item.get("tag_name")
        version = parse_version(tag) if isinstance(tag, str) and tag.startswith("v") else None
        if version is None:
            continue
        page = item.get("html_url")
        if not isinstance(page, str) or not page.startswith("https://github.com/"):
            page = f"{RELEASES_PAGE}/tag/{tag}"
        release = Release(".".join(map(str, version)), str(tag), page)
        if best is None or version > best[0]:
            best = (version, release)
    return best[1] if best else None


def is_newer(release: Release | None, current: str) -> bool:
    new, now = parse_version(release.version if release else None), parse_version(current)
    return new is not None and (now is None or new > now)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    )


def fetch_json(url: str, *, timeout: float = CHECK_TIMEOUT) -> Any:
    """GET `url` as JSON. Raises UpdateError (no network, an error status, not JSON)."""
    try:
        with urllib.request.urlopen(_request(url), timeout=timeout) as response:
            body = response.read(MAX_LIST_BYTES + 1)
    except OSError as exc:  # urllib's errors are OSErrors
        raise UpdateError(f"could not reach GitHub: {exc}") from exc
    if len(body) > MAX_LIST_BYTES:
        raise UpdateError("GitHub's answer was too large")
    try:
        return json.loads(body)
    except ValueError as exc:
        raise UpdateError(f"GitHub's answer was not JSON: {exc}") from exc


def latest_release(fetch: Callable[[str], Any] | None = None) -> Release | None:
    """The newest app release; one request (`fetch_json` unless given). None when there is
    none yet."""
    items = (fetch or fetch_json)(RELEASES_API)
    if not isinstance(items, list):
        raise UpdateError("GitHub's answer was not a list of releases")
    return choose_release(items)


# --- the check's record --------------------------------------------------------------------

CHECK_FILE = "update-check.json"
"""In data/app/: the last check, so a restart does not ask GitHub again."""


@dataclass(frozen=True, slots=True)
class CheckRecord:
    checked_at: float
    latest: Release | None
    error: str | None = None
    dismissed: str | None = None
    """The version whose offer was put off with Later."""


def check_path(data_dir: Path) -> Path:
    return data_dir / "app" / CHECK_FILE


def load_record(data_dir: Path) -> CheckRecord | None:
    try:
        raw = json.loads(check_path(data_dir).read_text(encoding="utf-8"))
        latest = raw.get("latest")
        return CheckRecord(
            checked_at=float(raw["checked_at"]),
            latest=Release(**latest) if isinstance(latest, dict) else None,
            error=raw.get("error"),
            dismissed=raw.get("dismissed"),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_record(data_dir: Path, record: CheckRecord) -> None:
    path = check_path(data_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
        tmp.write_text(json.dumps(asdict(record), indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # only saves asking again after a restart


# --- the payload ---------------------------------------------------------------------------


def _skipped(rel: PurePosixPath) -> bool:
    return any(part in SKIPPED_PARTS for part in rel.parts) or any(
        rel == under or under in rel.parents for under in SKIPPED_UNDER
    )


def never_written(rel: PurePosixPath) -> bool:
    return any(rel == PurePosixPath(p) or PurePosixPath(p) in rel.parents for p in NEVER_WRITTEN)


def in_payload(rel: PurePosixPath) -> bool:
    """Whether a file of the source (relative, '/'-separated) is shipped in a release."""
    if not rel.parts or rel.parts[0] in LEFT_OUT or _skipped(rel):
        return False
    return not never_written(rel)


def safe_relative(name: str) -> PurePosixPath:
    """An archive member's path, refused when it would land outside the folder."""
    if "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise UpdateError(f"the archive has an unsafe path: {name!r}")
    rel = PurePosixPath(name)
    if any(part in ("..", "") for part in rel.parts):
        raise UpdateError(f"the archive has an unsafe path: {name!r}")
    return rel


def unpack_payload(archive: Path, dest: Path) -> list[PurePosixPath]:
    """Unpack the release payload of a GitHub source archive (one top folder) into `dest`.
    Returns the payload's files. Raises UpdateError for anything unexpected."""
    try:
        zf = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdateError(f"the downloaded archive cannot be read: {exc}") from exc
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if not members or len(members) > MAX_ENTRIES:
            raise UpdateError("the archive is empty or has too many files")
        if sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
            raise UpdateError("the archive unpacks to more than facegen's source can be")
        tops = {safe_relative(m.filename).parts[0] for m in members}
        if len(tops) != 1:
            raise UpdateError("the archive is not one folder of facegen's source")
        files: list[PurePosixPath] = []
        for m in members:
            rel = PurePosixPath(*safe_relative(m.filename).parts[1:])
            if stat.S_ISLNK(m.external_attr >> 16):
                raise UpdateError(f"the archive has a link: {m.filename!r}")
            if not rel.parts:
                continue
            if never_written(rel):
                raise UpdateError(f"the archive has {rel}, which is this PC's own")
            if not in_payload(rel):
                continue
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            files.append(rel)
    if PurePosixPath("pyproject.toml") not in files or PurePosixPath("Start.bat") not in files:
        raise UpdateError("the archive is not facegen's source (no pyproject.toml or Start.bat)")
    return sorted(files)


def payload_files(root: Path) -> list[PurePosixPath]:
    """The payload of a source folder on disk (the installer's staging, a test's tree)."""
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rel = PurePosixPath(path.relative_to(root).as_posix())
            if in_payload(rel):
                out.append(rel)
    return out


def managed_files(root: Path) -> list[PurePosixPath]:
    """This folder's files that belong to the release alone (MANAGED_DIRS)."""
    out = []
    for top in MANAGED_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            rel = PurePosixPath(path.relative_to(root).as_posix())
            if path.is_file() and not _skipped(rel):
                out.append(rel)
    return out


@dataclass(frozen=True, slots=True)
class Applied:
    written: int
    unchanged: int
    removed: int
    backup: Path


def apply_payload(
    root: Path, staged: Path, files: Sequence[PurePosixPath], backup: Path
) -> Applied:
    """Put the staged payload over `root` (module docstring). On any failure, every file
    already replaced or removed is put back, and UpdateError is raised."""
    new = set(files)
    if any(never_written(rel) for rel in new):
        raise UpdateError("the new files include this PC's own folders")
    gone = [rel for rel in managed_files(root) if rel not in new]
    if backup.exists():
        shutil.rmtree(backup)
    backed_up: list[PurePosixPath] = []
    created: list[PurePosixPath] = []
    written = unchanged = 0

    def keep(rel: PurePosixPath) -> None:
        target = backup.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root.joinpath(*rel.parts), target)
        backed_up.append(rel)

    try:
        for rel in sorted(new):
            source = staged.joinpath(*rel.parts)
            target = root.joinpath(*rel.parts)
            if target.is_file() and filecmp.cmp(source, target, shallow=False):
                unchanged += 1
                continue
            if target.is_file():
                keep(rel)
            else:
                created.append(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(f".{target.name}.update-part")
            shutil.copyfile(source, part)
            os.replace(part, target)
            written += 1
        for rel in gone:
            keep(rel)
            root.joinpath(*rel.parts).unlink()
        _remove_empty_dirs(root, gone)
    except (OSError, UpdateError) as exc:
        _put_back(root, backup, backed_up, created)
        raise UpdateError(f"the files could not be replaced, so nothing changed: {exc}") from exc
    return Applied(written, unchanged, len(gone), backup)


def _put_back(
    root: Path, backup: Path, backed_up: Sequence[PurePosixPath], created: Sequence[PurePosixPath]
) -> None:
    for rel in created:
        root.joinpath(*rel.parts).unlink(missing_ok=True)
    for rel in backed_up:
        target = root.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup.joinpath(*rel.parts), target)


def _remove_empty_dirs(root: Path, removed: Iterable[PurePosixPath]) -> None:
    dirs = {rel.parent for rel in removed}
    for rel in sorted(dirs, key=lambda d: len(d.parts), reverse=True):
        path = root.joinpath(*rel.parts)
        while rel.parts and rel.parts[0] in MANAGED_DIRS and len(rel.parts) > 1:
            try:
                path.rmdir()  # only when empty
            except OSError:
                break
            rel, path = rel.parent, path.parent


def download(url: str, dest: Path, *, timeout: float = DOWNLOAD_TIMEOUT) -> None:
    """Fetch the release's source archive into `dest` (at most MAX_ARCHIVE_BYTES)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            urllib.request.urlopen(_request(url), timeout=timeout) as response,
            dest.open("wb") as f,
        ):
            total = 0
            while block := response.read(1 << 16):
                total += len(block)
                if total > MAX_ARCHIVE_BYTES:
                    raise UpdateError("the archive is larger than facegen's source can be")
                f.write(block)
    except OSError as exc:
        raise UpdateError(f"could not download {url}: {exc}") from exc


def archive_url(tag: str, repository: str = REPOSITORY) -> str:
    return ARCHIVE_URL.format(repository=repository, tag=tag)


# --- `facegen update` ----------------------------------------------------------------------

UPDATED, CHECKOUT, STILL_RUNNING, FAILED = 0, 3, 4, 5
WAIT_SECONDS = 180.0
"""How long the update waits for facegen to quit: the current picture (a minute at most),
the worker and the web server."""


@dataclass
class Updater:
    """The steps of an update, each replaceable (the tests give fakes)."""

    root: Path = PROJECT_ROOT
    fetch: Callable[[str], Any] | None = None
    """The releases list (`fetch_json` unless given)."""
    fetch_archive: Callable[[str, Path], None] | None = None
    """The source archive (`download` unless given)."""
    running: Callable[[Path], bool] | None = None
    """Whether a facegen app holds the data folder's lock (desktop.instance)."""
    sleep: Callable[[float], None] | None = None
    clock: Callable[[], float] | None = None
    now: Callable[[], float] | None = None


def run_update(
    out: Any, data_dir: Path, *, wait: float = WAIT_SECONDS, steps: Updater | None = None
) -> int:
    """`facegen update`: wait for facegen to quit, then put the newest release over this
    folder (module docstring). Exit codes: UPDATED (also when already the newest), CHECKOUT,
    STILL_RUNNING, FAILED."""
    import time

    from facegen.desktop import instance

    steps = steps or Updater()
    running = steps.running or instance.held_elsewhere
    sleep = steps.sleep or time.sleep
    clock = steps.clock or time.monotonic
    now = steps.now or time.time

    def say(text: str) -> None:
        print(text, file=out, flush=True)

    root = steps.root
    if is_checkout(root):
        say(
            "This folder is a git checkout: update it with `git pull` and `uv sync` instead."
            " Nothing was changed."
        )
        return CHECKOUT
    try:
        current = installed_version(root)
    except UpdateError as exc:
        say(f"Cannot update: {exc}")
        return FAILED
    end = clock() + wait
    if running(data_dir):
        say("Waiting for facegen to quit (Quit is in its tray icon's menu)...")
        while running(data_dir):
            if clock() >= end:
                say(
                    "facegen is still running, so its files cannot be replaced. Quit it from"
                    " the tray icon and start the update again."
                )
                return STILL_RUNNING
            sleep(1.0)
    say(f"facegen {current}: looking for a newer version on GitHub...")
    try:
        latest = latest_release(steps.fetch)
    except UpdateError as exc:
        say(f"Could not look for a newer version: {exc}")
        return FAILED
    if not is_newer(latest, current):
        say(f"facegen {current} is the newest version.")
        old = load_record(data_dir)
        save_record(data_dir, CheckRecord(now(), latest, None, old.dismissed if old else None))
        return UPDATED
    assert latest is not None
    work = data_dir / "update"
    staged = work / latest.tag
    archive = work / f"facegen-{latest.tag}.zip"
    backup = work / f"backup-{current}"
    try:
        for old in (staged, archive):
            if old.is_dir():
                shutil.rmtree(old)
            elif old.exists():
                old.unlink()
        say(f"Downloading facegen {latest.version} ({archive_url(latest.tag)})...")
        (steps.fetch_archive or download)(archive_url(latest.tag), archive)
        files = unpack_payload(archive, staged)
        found = project_version((staged / "pyproject.toml").read_text(encoding="utf-8"))
        if found != latest.version:
            raise UpdateError(f"the {latest.tag} archive says it is version {found}")
        applied = apply_payload(root, staged, files, backup)
    except (UpdateError, OSError) as exc:
        say(f"The update did not work, and facegen {current} is left as it was: {exc}")
        return FAILED
    finally:
        shutil.rmtree(staged, ignore_errors=True)
        archive.unlink(missing_ok=True)
    for other in work.glob("backup-*"):
        if other != backup and other.is_dir():
            shutil.rmtree(other, ignore_errors=True)
    save_record(data_dir, CheckRecord(now(), latest))
    say(
        f"Updated facegen {current} to {latest.version}: {applied.written} files written,"
        f" {applied.removed} removed. The replaced files are kept in {backup}."
    )
    return UPDATED


def cmd_update(args: Any, out: Any) -> int:
    """`facegen update [--wait S] [--log PATH]`, which `Start.bat --update` runs."""
    from facegen.persistence.settings import SettingsError, load_settings
    from facegen.setup.launch import Tee, open_log

    log = open_log(args.log)
    tee = Tee(out, log)
    try:
        try:
            data_dir = load_settings(defaults_path=args.config).paths.data_dir
        except SettingsError as exc:
            print(f"facegen's settings cannot be read ({exc}); using its own data folder.",
                  file=tee)  # fmt: skip
            data_dir = PROJECT_ROOT / "data"
        return run_update(tee, data_dir, wait=args.wait)
    finally:
        tee.flush()
        if log is not None:
            log.close()
