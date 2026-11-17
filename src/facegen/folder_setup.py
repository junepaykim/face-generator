"""First-run setup of FM27's folder (Phase 10a).

When FM27's user folder is neither in the local file nor found on this PC, the web UI asks for
it. This module holds the three pieces behind that:

- `check_folder` (pure over a `Disk`): is the folder the user gave FM27's folder? It must
  exist and be a folder that has `games/`, or be named `Football Manager 27` inside
  `Sports Interactive` (a new install that has not saved yet). FM27's `games` or `graphics`
  folder, a save file, or the Documents or Sports Interactive folder above it are corrected to
  FM27's folder. A refusal names what was missing.
- `write_local_path`: `[paths] fm_user_dir = '...'` in the local file (config/local.toml),
  keeping its other lines and comments, written atomically. Nothing else is ever written there.
- The Browse button: a browser never gives a page a real path, so the server opens a native
  folder dialog on this PC (`browse`): tkinter's, in a process of its own so Tk never runs in
  the server's threads. Not offered where there is no display (WSL, a Linux without one) or no
  tkinter (`dialog_available`). Phase 11's window will bring its own dialog.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Protocol

from facegen import fm_paths, platform_info, procs
from facegen.platform_info import Kind

FM_FOLDER = fm_paths.FM_SUBPATH[1]
"""`Football Manager 27`."""
SI_FOLDER = fm_paths.FM_SUBPATH[0]
"""`Sports Interactive`."""
SAVE_SUFFIX = ".fm"
INSIDE_FM = ("games", "graphics")
"""FM27's own sub-folders: picking one means the folder above."""
WHERE_IT_USUALLY_IS = {
    "windows": r"Documents\Sports Interactive\Football Manager 27",
    "wsl": r"Documents\Sports Interactive\Football Manager 27",
    "linux": "Documents/Sports Interactive/Football Manager 27",
}
DIALOG_SECONDS = 600.0
"""How long the folder dialog may stay open before it is closed unanswered."""


class Disk(Protocol):
    def is_dir(self, path: PurePath) -> bool: ...

    def is_file(self, path: PurePath) -> bool: ...


class RealDisk:
    def is_dir(self, path: PurePath) -> bool:
        try:
            return Path(path).is_dir()
        except OSError:
            return False

    def is_file(self, path: PurePath) -> bool:
        try:
            return Path(path).is_file()
        except OSError:
            return False


@dataclass(frozen=True, slots=True)
class FolderCheck:
    """The answer to "is this FM27's folder?"."""

    folder: PurePath | None
    """FM27's folder, corrected from what was given; None if refused."""
    message: str
    """Why it was refused, or what was corrected; empty when taken as given."""

    @property
    def ok(self) -> bool:
        return self.folder is not None


def _refused(message: str) -> FolderCheck:
    return FolderCheck(None, message)


def _path_class(kind: Kind) -> type[PurePath]:
    return PureWindowsPath if kind == "windows" else PurePosixPath


def _clean(text: str) -> str:
    """The path as typed or pasted: spaces and quotes around it gone (Windows's "Copy as path"
    adds double quotes)."""
    text = text.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _named(path: PurePath, name: str) -> bool:
    return path.name.casefold() == name.casefold()


def _is_fm_folder(path: PurePath, disk: Disk) -> bool:
    return disk.is_dir(path / "games") or (
        _named(path, FM_FOLDER) and _named(path.parent, SI_FOLDER) and disk.is_dir(path)
    )


def check_folder(text: str, kind: Kind, disk: Disk) -> FolderCheck:
    """Whether `text` names FM27's user folder on this system, corrected where it is plainly
    meant (see the module's docstring). Pure: the disk is asked through `disk`."""
    given = _clean(text)
    if not given:
        return _refused(
            "Enter FM27's folder: the one that holds its games and graphics folders, usually"
            f" {WHERE_IT_USUALLY_IS[kind]}."
        )
    try:
        native = fm_paths.native_path(given, kind)
    except fm_paths.PathStyleError as exc:
        return _refused(str(exc)[0].upper() + str(exc)[1:] + ".")
    path = _path_class(kind)(native)
    if not path.is_absolute():
        start = "a drive, as C:\\" if kind != "linux" else "/"
        return _refused(f"{given} is not a full path: it starts with {start}…")
    if disk.is_file(path):
        if path.suffix.lower() != SAVE_SUFFIX:
            return _refused(
                f"{path} is a file, not a folder. Pick FM27's folder, or one of its saves."
            )
        games = next((p for p in path.parents if _named(p, "games")), None)
        if games is None:
            return _refused(f"{path} is a save, but not in FM27's games folder.")
        found = games.parent
        if not _is_fm_folder(found, disk):
            return _refused(f"{path} is a save, but {found} is not FM27's folder.")
        return FolderCheck(found, f"That is a save; FM27's folder is {found}.")
    if not disk.is_dir(path):
        return _refused(f"{path} does not exist on this PC, or is not a folder.")
    if _is_fm_folder(path, disk):
        return FolderCheck(path, "")
    for inside in INSIDE_FM:
        if _named(path, inside) and _is_fm_folder(path.parent, disk):
            return FolderCheck(
                path.parent, f"That is FM27's {inside} folder; FM27's folder is {path.parent}."
            )
    for below in (path / FM_FOLDER, path / SI_FOLDER / FM_FOLDER):
        if _is_fm_folder(below, disk):
            return FolderCheck(below, f"FM27's folder is inside that one: {below}.")
    return _refused(
        f"{path} has no games folder, and it is not the folder named {FM_FOLDER} inside"
        f" {SI_FOLDER}. FM27's folder holds its games and graphics folders; usually it is"
        f" {WHERE_IT_USUALLY_IS[kind]}."
    )


LOCAL_HEADER = (
    "# This PC's own folders. Not in git (.gitignore): it holds the Windows user name.\n"
    "# Laid over [paths] in config/default.toml; only [paths] keys are allowed here.\n"
    "# facegen's first-run setup and its Settings page write fm_user_dir here.\n"
)
_TABLE = re.compile(r"^\s*\[\s*([^\]]*?)\s*\]\s*(?:#.*)?$")


class LocalFileError(ValueError):
    """The local file cannot be read or changed safely; it is left as it was."""


def toml_string(value: str) -> str:
    """`value` as a TOML string: in single quotes, which keep backslashes as they are, unless
    it holds a single quote or a control character; then a basic string with escapes."""
    if "'" not in value and not any(ord(c) < 32 or ord(c) == 127 for c in value):
        return f"'{value}'"
    return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid TOML basic string


def with_path_key(text: str, key: str, value: str) -> str:
    """The local file's text with `[paths] <key>` set to `value`; every other line kept.

    Pure. The key's line is replaced where it is, else added at the end of the `[paths]` table,
    which is added at the end of the file if there is none. The result is read back to check
    that it holds exactly that value: LocalFileError otherwise, or if `text` itself is not
    valid TOML (a file the user wrote by hand is never overwritten with a guess).
    """
    if text.strip():
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise LocalFileError(f"it is not valid TOML ({exc}); fix or delete it first") from exc
    line = f"{key} = {toml_string(value)}"
    lines = text.splitlines()
    key_line = re.compile(rf"^\s*{re.escape(key)}\s*=")
    table = None
    last_in_paths = None
    replaced = False
    for i, current in enumerate(lines):
        header = _TABLE.match(current)
        if header is not None:
            table = header.group(1)
            if table == "paths":
                last_in_paths = i
            continue
        if table == "paths":
            if key_line.match(current):
                lines[i] = line
                replaced = True
                break
            if current.strip() and not current.lstrip().startswith("#"):
                last_in_paths = i
    if not replaced:
        if last_in_paths is not None:
            lines.insert(last_in_paths + 1, line)
        else:
            if lines and lines[-1].strip():
                lines.append("")
            lines += ["[paths]", line]
    result = "\n".join(lines) + "\n"
    try:
        read = tomllib.loads(result)
    except tomllib.TOMLDecodeError as exc:
        raise LocalFileError(f"the change would not be valid TOML ({exc})") from exc
    if read.get("paths", {}).get(key) != value:
        raise LocalFileError(f"[paths] {key} could not be set there by facegen; edit it by hand")
    return result


def write_local_path(path: Path, key: str, value: str) -> None:
    """Set `[paths] <key>` in the local file at `path` (made if missing), atomically: a new
    file in the same folder, then replaced in one step, so a crash leaves the old one whole."""
    try:
        text = path.read_text(encoding="utf-8") if path.exists() else LOCAL_HEADER
    except OSError as exc:
        raise LocalFileError(f"it cannot be read: {exc}") from exc
    result = with_path_key(text, key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(result)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def value_to_write(folder: PurePath, kind: Kind) -> str:
    """How the folder is written in the local file: on WSL a folder on a Windows drive in
    Windows's form, so the same file works when facegen runs on Windows itself."""
    text = str(folder)
    if kind == "wsl" and (windows := fm_paths.windows_path_of_wsl(text)) is not None:
        return windows
    return text


def dialog_available(
    kind: Kind | None = None,
    env: Mapping[str, str] | None = None,
    has_tk: bool | None = None,
) -> bool:
    """Whether Browse can open a folder dialog here: tkinter is there and so is a display.
    Never on WSL, whose Linux side has no display of its own."""
    kind = kind or platform_info.current()
    env = os.environ if env is None else env
    if has_tk is None:
        has_tk = find_spec("tkinter") is not None
    if not has_tk or kind == "wsl":
        return False
    return kind == "windows" or bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


_DIALOG = r"""
import json, sys, tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
chosen = filedialog.askdirectory(
    parent=root, title="FM27's folder (it holds games and graphics)",
    initialdir=sys.argv[1] or None, mustexist=True,
)
root.destroy()
print(json.dumps({"path": chosen or None}))
"""


def browse(initial: str | None = None, *, timeout: float = DIALOG_SECONDS) -> str | None:
    """Open the folder dialog on this PC and wait for it: the folder chosen, or None if it was
    cancelled or could not be shown. The server calls this from a worker thread."""
    try:
        done = procs.run_quiet(
            [sys.executable, "-c", _DIALOG, initial or ""],
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    for line in reversed(done.stdout.splitlines()):
        try:
            answer = json.loads(line)
        except json.JSONDecodeError:
            continue
        chosen = answer.get("path") if isinstance(answer, dict) else None
        return str(chosen) if chosen else None
    return None
