"""Where FM27's folders are on this PC, when config/local.toml does not say.

Two folders are looked for:

- the **user folder**, `<Documents>/Sports Interactive/Football Manager 27`, which holds the
  saves (`games/`) and the graphics. Documents is the folder Windows names as the user's
  Known Folder "Documents" (which follows OneDrive and a Documents folder moved elsewhere),
  then `%OneDrive%\\Documents`, then `%USERPROFILE%\\Documents`. From WSL the same folders
  under `/mnt/<drive>/`; if none holds FM27's folder, every Windows user's under
  `/mnt/c/Users/` is tried. The first with a `games` folder wins; if several hold saves, the
  one with the newest save.
- the **install folder**, read only for the list of photos FM27 ships (photos.py): Steam's
  libraries (every one in `libraryfolders.vdf`, not only C:), then Epic's install manifests,
  then the Xbox app's `XboxGames` folders (Game Pass). Not finding it is not fatal: no one
  then counts as without a photo.

The candidates and the choice between them are pure: they take the platform's kind, what
Windows says about the user's folders (`WindowsInfo`) and a `Disk` to look at, so the tests
can describe any PC. `detect` asks this one, through `system_info` and `RealDisk`.

Windows-style paths in config/local.toml are accepted on WSL (`native_path`).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Protocol

from facegen.platform import platform_info
from facegen.platform.platform_info import Kind

FM_SUBPATH = ("Sports Interactive", "Football Manager 27")
"""FM27's user folder inside Documents."""
STEAM_APP_ID = "4530370"
"""Football Manager 27 on Steam (its appmanifest on this PC, docs/research.md)."""
INSTALL_NAME = "Football Manager 27"
"""The install folder's name in Steam's `steamapps/common`."""
INSTALL_MARKER = "fm_Data"
"""A folder that every FM27 install has beside fm.exe (Unity's data folder)."""
WSL_MOUNT_ROOT = "/mnt"

_FM27_TITLE = re.compile(r"\bfootball manager (?:20)?27\b", re.IGNORECASE)
_NOT_THE_GAME = re.compile(r"\b(?:editor|touch|console|demo|beta)\b", re.IGNORECASE)
_WINDOWS_ACCOUNTS = frozenset({"all users", "default", "default user", "public", "defaultapppool"})
_SAVE_BACKUP_PREFIX = "last save overwrite backup"


_DRIVE_PATH = re.compile(r"^([A-Za-z]):(?:[\\/](.*))?$")
_WSL_DRIVE_PATH = re.compile(r"^/mnt/([A-Za-z])(?:/(.*))?$")


class PathStyleError(ValueError):
    """A path written for another system: a WSL path on Windows, a Windows path on Linux."""


def is_windows_drive_path(text: str) -> bool:
    """`C:\\Users\\me` or `C:/Users/me`."""
    return _DRIVE_PATH.match(text.strip()) is not None


def wsl_path(windows_path: str, mount_root: str = WSL_MOUNT_ROOT) -> str | None:
    """`C:\\Users\\me` as WSL sees it, `/mnt/c/Users/me`; None for anything but a drive path
    (a network path such as `\\\\server\\share` has no place under /mnt)."""
    m = _DRIVE_PATH.match(windows_path.strip())
    if m is None:
        return None
    drive, rest = m.groups()
    parts = [p for p in re.split(r"[\\/]+", rest or "") if p]
    return "/".join([mount_root.rstrip("/"), drive.lower(), *parts])


def windows_path_of_wsl(posix_path: str) -> str | None:
    """`/mnt/c/Users/me` as Windows writes it, `C:\\Users\\me`; None if it is not on a drive."""
    m = _WSL_DRIVE_PATH.match(posix_path.strip())
    if m is None:
        return None
    drive, rest = m.groups()
    return f"{drive.upper()}:\\" + "\\".join(p for p in (rest or "").split("/") if p)


def native_path(text: str, kind: Kind, mount_root: str = WSL_MOUNT_ROOT) -> str:
    """A path from config/local.toml (or the defaults) as this system needs it.

    On WSL a Windows path (`C:\\Users\\me\\...`) becomes `/mnt/c/Users/me/...`. On Windows a
    WSL path (`/mnt/c/...`) is refused, and on Linux a Windows path, each with the form that
    would work. Anything else, relative paths included, is returned as it is.
    """
    text = text.strip()
    if kind == "wsl" and is_windows_drive_path(text):
        converted = wsl_path(text, mount_root)
        assert converted is not None
        return converted
    if kind == "windows" and (windows := windows_path_of_wsl(text)) is not None:
        raise PathStyleError(
            f"'{text}' is a path as WSL sees it; on Windows write it as '{windows}'"
            " (in single quotes in a .toml file, so the backslashes stay as they are)"
        )
    if kind == "linux" and is_windows_drive_path(text):
        raise PathStyleError(
            f"'{text}' is a Windows path, and this is Linux: write the folder as this system"
            " sees it"
        )
    return text


@dataclass(frozen=True, slots=True)
class WindowsInfo:
    """What Windows says about the user's folders. Paths are as Windows writes them."""

    documents: str | None = None
    """The Known Folder "Documents" (SHGetKnownFolderPath), wherever it was moved."""
    env: Mapping[str, str] = field(default_factory=dict)
    """USERPROFILE, OneDrive, OneDriveConsumer, OneDriveCommercial, ProgramFiles,
    ProgramFiles(x86) and ProgramData, as far as they are set."""
    steam_path: str | None = None
    """Steam's folder from the registry (HKCU\\Software\\Valve\\Steam, SteamPath)."""
    drives: tuple[str, ...] = ()
    """Drive roots such as `C:\\`, for the Xbox app's XboxGames folders."""


def _env(info: WindowsInfo, name: str) -> str | None:
    for key, value in info.env.items():
        if key.lower() == name.lower() and value.strip():
            return value.strip()
    return None


POWERSHELL_QUERY = (
    "$ErrorActionPreference='SilentlyContinue';"
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
    "'documents='+[Environment]::GetFolderPath('MyDocuments');"
    "foreach($n in 'USERPROFILE','OneDrive','OneDriveConsumer','OneDriveCommercial',"
    "'ProgramFiles','ProgramFiles(x86)','ProgramData'){"
    "'env:'+$n+'='+[Environment]::GetEnvironmentVariable($n)};"
    "'steam='+(Get-ItemProperty 'HKCU:\\Software\\Valve\\Steam').SteamPath"
)
"""Asked of Windows's PowerShell from WSL: one `key=value` line each, in UTF-8 whatever the
console's code page, so a user name in Korean arrives intact."""


def parse_powershell_info(text: str, drives: Sequence[str] = ()) -> WindowsInfo:
    """`POWERSHELL_QUERY`'s output as a WindowsInfo; empty values are left out."""
    documents = steam = None
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")
        key, sep, value = line.partition("=")
        if not sep or not value.strip():
            continue
        value = value.strip()
        if key == "documents":
            documents = value
        elif key == "steam":
            steam = value
        elif key.startswith("env:"):
            env[key.removeprefix("env:")] = value
    return WindowsInfo(documents, env, steam, tuple(drives))


class Disk(Protocol):
    """The questions the search asks of the disk; `RealDisk` answers them for this PC."""

    def is_dir(self, path: PurePath) -> bool: ...

    def read_text(self, path: PurePath) -> str | None:
        """The file's text, or None if it cannot be read."""
        ...

    def list_dir(self, path: PurePath) -> list[str]:
        """The names in a folder, or [] if it cannot be listed."""
        ...

    def newest_save(self, games: PurePath) -> float | None:
        """The modification time of the newest save in a games folder; None if none."""
        ...


class RealDisk:
    def is_dir(self, path: PurePath) -> bool:
        try:
            return Path(path).is_dir()
        except OSError:
            return False

    def read_text(self, path: PurePath) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def list_dir(self, path: PurePath) -> list[str]:
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []

    def newest_save(self, games: PurePath) -> float | None:
        newest = None
        try:
            with os.scandir(games) as entries:
                for e in entries:
                    name = e.name.lower()
                    if not name.endswith(".fm") or name.startswith(_SAVE_BACKUP_PREFIX):
                        continue
                    try:
                        if e.is_file():
                            mtime = e.stat().st_mtime
                            newest = mtime if newest is None else max(newest, mtime)
                    except OSError:
                        continue
        except OSError:
            return None
        return newest


def _native(kind: Kind, windows_path: str | None) -> PurePath | None:
    """A path Windows gave, as this system reaches it (None if it cannot)."""
    if not windows_path:
        return None
    if kind == "windows":
        return PureWindowsPath(windows_path)
    if kind == "wsl":
        converted = wsl_path(windows_path)
        return PurePosixPath(converted) if converted else None
    return None


def _unique(paths: Iterable[PurePath | None]) -> list[PurePath]:
    """Without repeats, ignoring case (Windows's drives ignore it, from WSL too)."""
    seen: set[str] = set()
    out: list[PurePath] = []
    for p in paths:
        if p is None:
            continue
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def documents_folders(info: WindowsInfo) -> list[str]:
    """The folders that may be the user's Documents, best first, as Windows writes them."""
    found = [info.documents]
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial", "USERPROFILE"):
        root = _env(info, name)
        if root:
            found.append(str(PureWindowsPath(root, "Documents")))
    seen: set[str] = set()
    out = []
    for f in found:
        if f and f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def user_dir_candidates(kind: Kind, info: WindowsInfo) -> list[PurePath]:
    """FM27's user folder in each Documents folder Windows names, best first."""
    if kind == "linux":
        return []
    return _unique(
        (d.joinpath(*FM_SUBPATH) if d else None)
        for d in (_native(kind, f) for f in documents_folders(info))
    )


def wsl_scan_candidates(disk: Disk, mount_root: str = WSL_MOUNT_ROOT) -> list[PurePath]:
    """From WSL, when Windows could not be asked or named no folder with FM27's: every
    Windows user's Documents under C:, at home and in OneDrive (`OneDrive`, `OneDrive - <org>`).
    """
    users = PurePosixPath(mount_root, "c", "Users")
    out: list[PurePath] = []
    for name in disk.list_dir(users):
        if name.lower() in _WINDOWS_ACCOUNTS:
            continue
        home = users / name
        if not disk.is_dir(home):
            continue
        docs = [home / "Documents"]
        docs += [
            home / d / "Documents"
            for d in disk.list_dir(home)
            if d == "OneDrive" or d.startswith("OneDrive - ")
        ]
        out += [d.joinpath(*FM_SUBPATH) for d in docs]
    return _unique(out)


def choose_user_dir(candidates: Sequence[PurePath], disk: Disk) -> PurePath | None:
    """The first candidate with a games folder; if several hold saves, the one with the
    newest save (the first of them if two are as new)."""
    with_games: list[tuple[PurePath, float | None]] = []
    for c in candidates:
        if disk.is_dir(c / "games"):
            with_games.append((c, disk.newest_save(c / "games")))
    with_saves = [(c, t) for c, t in with_games if t is not None]
    if with_saves:
        best = max(t for _, t in with_saves)
        return next(c for c, t in with_saves if t == best)
    return with_games[0][0] if with_games else None


def find_user_dir(kind: Kind, info: WindowsInfo, disk: Disk) -> PurePath | None:
    """FM27's user folder on this PC, or None if it is in none of the usual places."""
    found = choose_user_dir(user_dir_candidates(kind, info), disk)
    if found is None and kind == "wsl":
        found = choose_user_dir(wsl_scan_candidates(disk), disk)
    return found


_VDF_PATH = re.compile(r'"path"\s+"((?:[^"\\]|\\.)*)"', re.IGNORECASE)
_ACF_INSTALLDIR = re.compile(r'"installdir"\s+"((?:[^"\\]|\\.)*)"', re.IGNORECASE)


def _vdf_unescape(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text)


def steam_library_paths(vdf_text: str) -> list[str]:
    """Every library folder that Steam's `steamapps/libraryfolders.vdf` lists, in order."""
    return [_vdf_unescape(m) for m in _VDF_PATH.findall(vdf_text)]


def steam_install_dir_name(acf_text: str) -> str | None:
    """The `installdir` of a Steam app manifest (`appmanifest_<id>.acf`)."""
    m = _ACF_INSTALLDIR.search(acf_text)
    return _vdf_unescape(m.group(1)) if m else None


def is_fm27_title(name: str) -> bool:
    """Whether a store's title names FM27 itself, not its editor or another edition."""
    return bool(_FM27_TITLE.search(name)) and not _NOT_THE_GAME.search(name)


def steam_roots(info: WindowsInfo) -> list[str]:
    """Where Steam itself may be, as Windows writes it: the registry's answer first."""
    roots = [info.steam_path]
    for name in ("ProgramFiles(x86)", "ProgramFiles"):
        base = _env(info, name)
        if base:
            roots.append(str(PureWindowsPath(base, "Steam")))
    roots += [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam"]
    roots = [r.replace("/", "\\") for r in roots if r]
    # The registry keeps Steam's folder in lower case ("c:/program files (x86)/steam"): the
    # same folder from the environment, if there is one, is spelled as Explorer shows it.
    spelled: dict[str, str] = {}
    for r in [*roots[1:], *roots[:1]]:
        spelled.setdefault(r.lower(), r)
    return [spelled[r.lower()] for r in roots]


def steam_candidates(kind: Kind, info: WindowsInfo, disk: Disk) -> list[PurePath]:
    """FM27 in each Steam library: the one whose manifest names the app first, then the
    default folder name in each library."""
    libraries: list[PurePath] = []
    for root in _unique(_native(kind, r) for r in steam_roots(info)):
        libraries.append(root)
        vdf = disk.read_text(root / "steamapps" / "libraryfolders.vdf")
        if vdf:
            libraries += [p for p in (_native(kind, lib) for lib in steam_library_paths(vdf)) if p]
    libraries = _unique(libraries)
    listed: list[PurePath] = []
    guessed: list[PurePath] = []
    for lib in libraries:
        apps = lib / "steamapps"
        acf = disk.read_text(apps / f"appmanifest_{STEAM_APP_ID}.acf")
        name = steam_install_dir_name(acf) if acf else None
        if name:
            listed.append(apps / "common" / name)
        guessed.append(apps / "common" / INSTALL_NAME)
    return _unique([*listed, *guessed])


def epic_candidates(kind: Kind, info: WindowsInfo, disk: Disk) -> list[PurePath]:
    """FM27's folder from the Epic Games Launcher's install manifests (`*.item`, JSON with
    DisplayName and InstallLocation). Not seen with FM27 yet (docs/research.md)."""
    program_data = _native(kind, _env(info, "ProgramData") or r"C:\ProgramData")
    if program_data is None:
        return []
    folder = program_data / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
    out: list[PurePath | None] = []
    for name in disk.list_dir(folder):
        if not name.lower().endswith(".item"):
            continue
        text = disk.read_text(folder / name)
        try:
            item = json.loads(text) if text else None
        except ValueError:
            continue
        if not isinstance(item, dict):
            continue
        title = item.get("DisplayName")
        location = item.get("InstallLocation")
        if isinstance(title, str) and isinstance(location, str) and is_fm27_title(title):
            out.append(_native(kind, location))
    return _unique(out)


def xbox_candidates(kind: Kind, info: WindowsInfo, disk: Disk) -> list[PurePath]:
    """FM27's folder as the Xbox app installs PC Game Pass games: `<drive>:\\XboxGames\\<title>`,
    with the game in `Content` or, as reported for FM26, directly in it. Not seen with FM27 yet
    (docs/research.md)."""
    out: list[PurePath | None] = []
    for drive in info.drives:
        root = _native(kind, drive)
        if root is None:
            continue
        games = root / "XboxGames"
        for name in disk.list_dir(games):
            if is_fm27_title(name):
                out += [games / name / "Content", games / name]
    return _unique(out)


def install_dir_candidates(kind: Kind, info: WindowsInfo, disk: Disk) -> list[PurePath]:
    """Every place FM27 may be installed, Steam's first."""
    if kind == "linux":
        return []
    return _unique(
        [
            *steam_candidates(kind, info, disk),
            *epic_candidates(kind, info, disk),
            *xbox_candidates(kind, info, disk),
        ]
    )


def find_install_dir(kind: Kind, info: WindowsInfo, disk: Disk) -> PurePath | None:
    """The first candidate that holds an FM27 install, or None."""
    for c in install_dir_candidates(kind, info, disk):
        if disk.is_dir(c / INSTALL_MARKER):
            return c
    return None


DOCUMENTS_FOLDER_ID = "FDD39AD0-238F-46AF-ADB4-6C85480369C7"
"""FOLDERID_Documents."""


def _known_documents() -> str | None:
    """Windows's Known Folder "Documents" (SHGetKnownFolderPath), on Windows only."""
    return known_folder(DOCUMENTS_FOLDER_ID)


def known_folder(folder_id_text: str) -> str | None:
    """A Windows Known Folder by its FOLDERID (SHGetKnownFolderPath), wherever it was moved
    (OneDrive included); on Windows only."""
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    folder_id = GUID.from_buffer_copy(uuid.UUID(folder_id_text).bytes_le)
    shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
    ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
    shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(GUID),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    shell32.SHGetKnownFolderPath.restype = ctypes.HRESULT
    path = ctypes.c_wchar_p()
    try:
        shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(path))
    except OSError:  # a failing HRESULT
        return None
    try:
        return path.value
    finally:
        ole32.CoTaskMemFree(path)


def _steam_path_from_registry() -> str | None:
    import winreg  # type: ignore[import-not-found,unused-ignore]

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:  # type: ignore[attr-defined,unused-ignore]
            value, _ = winreg.QueryValueEx(key, "SteamPath")  # type: ignore[attr-defined,unused-ignore]
    except OSError:
        return None
    return value if isinstance(value, str) and value else None


def _info_from_windows() -> WindowsInfo:
    names = (
        "USERPROFILE",
        "OneDrive",
        "OneDriveConsumer",
        "OneDriveCommercial",
        "ProgramFiles",
        "ProgramFiles(x86)",
        "ProgramData",
    )
    env = {n: os.environ[n] for n in names if os.environ.get(n)}
    try:
        documents = _known_documents()
    except (OSError, AttributeError):
        documents = None
    drives = tuple(getattr(os, "listdrives", list)())
    return WindowsInfo(documents, env, _steam_path_from_registry(), drives)


def _info_from_wsl(disk: Disk) -> WindowsInfo:
    from facegen.platform.procs import run_quiet, windows_tool

    drives = tuple(
        f"{d.upper()}:\\"
        for d in disk.list_dir(PurePosixPath(WSL_MOUNT_ROOT))
        if len(d) == 1 and d.isalpha()
    )
    tool = windows_tool(r"WindowsPowerShell\v1.0\powershell.exe", "wsl", os.environ)
    if tool is None:
        return WindowsInfo(drives=drives)
    cwd = PurePosixPath(WSL_MOUNT_ROOT, "c")  # a Windows folder: no "UNC paths" warning
    try:
        out = run_quiet(
            [tool, "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_QUERY],
            timeout=20,
            capture_output=True,
            check=False,
            cwd=str(cwd) if disk.is_dir(cwd) else None,
        )
    except (OSError, ValueError):
        return WindowsInfo(drives=drives)
    if out.returncode != 0:
        return WindowsInfo(drives=drives)
    return parse_powershell_info(out.stdout.decode("utf-8", errors="replace"), drives)


def system_info(kind: Kind, disk: Disk) -> WindowsInfo:
    """What Windows says about this PC's folders; nothing on plain Linux."""
    if kind == "windows":
        return _info_from_windows()
    if kind == "wsl":
        return _info_from_wsl(disk)
    return WindowsInfo()


RETRY_SECONDS = 60.0
"""A folder that was not found is looked for again after this long (FM27 may have been
installed meanwhile); one that was found is kept for the process's life."""

_cache_lock = threading.Lock()
_cache: dict[str, tuple[Path | None, float]] = {}
_info_cache: dict[str, WindowsInfo] = {}


def _detect(
    key: str,
    find: Callable[[Kind, WindowsInfo, Disk], PurePath | None],
    clock: Callable[[], float],
) -> Path | None:
    with _cache_lock:
        hit = _cache.get(key)
        now = clock()
        if hit is not None and (hit[0] is not None or now - hit[1] < RETRY_SECONDS):
            return hit[0]
        kind = platform_info.current()
        disk = RealDisk()
        if kind not in _info_cache:
            _info_cache[kind] = system_info(kind, disk)
        found = find(kind, _info_cache[kind], disk)
        path = Path(str(found)) if found is not None else None
        _cache[key] = (path, now)
        return path


def detect_user_dir(clock: Callable[[], float] = time.monotonic) -> Path | None:
    """This PC's FM27 user folder, or None. Windows is asked once per process; the disk is
    looked at again only while the folder is missing (`RETRY_SECONDS`)."""
    return _detect("user", find_user_dir, clock)


def detect_install_dir(clock: Callable[[], float] = time.monotonic) -> Path | None:
    """This PC's FM27 install folder, or None; cached as `detect_user_dir`."""
    return _detect("install", find_install_dir, clock)


def forget() -> None:
    """Look again at the next detection (after the user set a folder, and in tests)."""
    with _cache_lock:
        _cache.clear()
        _info_cache.clear()
