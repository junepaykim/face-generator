"""kitshot-app.exe: the venv's pythonw.exe launcher with Kitshot's name and icon.

A Windows venv's pythonw.exe is a small launcher that starts the base interpreter.
Task Manager names the app's process group after that launcher's version information
and shows its icon, so a plain venv shows "Python". The launcher finds the interpreter
through `pyvenv.cfg`, not through its own file name, and passes its own path on as
`sys.executable`. A copy named kitshot-app.exe therefore starts the app and its child
processes under Kitshot's name and icon. Versions before Phase 17o named the copy
facegen-app.exe (LEGACY_APP_EXE): it is left in place, as a taskbar pin may start it, and
`launch` points our shortcuts that start it at the new copy.

`ensure` makes the copy in the venv's Scripts folder on Windows. The Win32 resource
functions replace its icons and version information and keep its other resources, such
as the manifest. The copy loses the launcher's Authenticode signature, which would no
longer match its contents. A copy that cannot start Python is deleted, and callers keep
using pythonw.exe. A stamp beside the copy records its launcher, icon and version
information (the name and version), so a change to any of them makes it again.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import struct
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("kitshot")

APP_EXE = "kitshot-app.exe"
LEGACY_APP_EXE = "facegen-app.exe"
"""The copy's name before Phase 17o (module docstring)."""
LAUNCHER = "pythonw.exe"
STAMP = "kitshot-app.stamp"
FORMAT = "2"
"""Part of the stamp: changing it remakes every copy."""
NAME = "Kitshot"
"""The product name, and the FileDescription that Task Manager shows."""
COPYRIGHT = "Python launcher: Copyright (c) Python Software Foundation"
LANGUAGE = 0x0409
CODEPAGE = 1200
RT_ICON, RT_GROUP_ICON, RT_VERSION = 3, 14, 16
START_SECONDS = 60.0
CHECK = "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 3)"
"""The copy must start Python inside this venv."""


class AppExeError(RuntimeError):
    """The copy could not be made."""


# --- the file's contents -------------------------------------------------------------------


def numeric_version(text: str) -> tuple[int, int, int, int]:
    """Four numbers for the version resource: "0.1.2" is (0, 1, 2, 0); a suffix such as
    ".dev1" is dropped."""
    m = re.match(r"\d+(?:\.\d+){0,3}", text)
    numbers = [min(int(part), 0xFFFF) for part in m[0].split(".")] if m else []
    a, b, c, d = numbers + [0] * (4 - len(numbers))
    return a, b, c, d


def _block(
    key: str, value: bytes = b"", *, wtype: int, text: bool = False, children: Sequence[bytes] = ()
) -> bytes:
    """One VS_VERSIONINFO node: header, key, value and children, each on a 32-bit bound."""
    data = bytearray(6) + (key + "\0").encode("utf-16-le")
    if value or children:
        data += bytes(-len(data) % 4)
    data += value
    for child in children:
        data += bytes(-len(data) % 4)
        data += child
    value_length = len(value) // 2 if text else len(value)
    struct.pack_into("<HHH", data, 0, len(data), value_length, wtype)
    return bytes(data)


def version_info(version: str) -> bytes:
    """The RT_VERSION resource: Kitshot's name, description and version."""
    a, b, c, d = numeric_version(version)
    fixed = struct.pack(
        "<13I",
        0xFEEF04BD,  # signature
        0x00010000,  # structure version
        (a << 16) | b,  # file version
        (c << 16) | d,
        (a << 16) | b,  # product version
        (c << 16) | d,
        0x3F,  # flags mask
        0,  # flags
        0x00040004,  # VOS_NT_WINDOWS32
        0x1,  # VFT_APP
        0,  # subtype
        0,  # date
        0,
    )
    strings = {
        "CompanyName": NAME,
        "FileDescription": NAME,
        "FileVersion": version,
        "InternalName": APP_EXE.removesuffix(".exe"),
        "LegalCopyright": COPYRIGHT,
        "OriginalFilename": APP_EXE,
        "ProductName": NAME,
        "ProductVersion": version,
    }
    table = _block(
        f"{LANGUAGE:04X}{CODEPAGE:04X}",
        wtype=1,
        children=[
            _block(key, (value + "\0").encode("utf-16-le"), wtype=1, text=True)
            for key, value in strings.items()
        ],
    )
    translation = _block("Translation", struct.pack("<HH", LANGUAGE, CODEPAGE), wtype=0)
    return _block(
        "VS_VERSION_INFO",
        fixed,
        wtype=0,
        children=[
            _block("StringFileInfo", wtype=1, children=[table]),
            _block("VarFileInfo", wtype=1, children=[translation]),
        ],
    )


def icon_resources(ico: bytes) -> tuple[list[bytes], bytes]:
    """An .ico file as resources: each picture (RT_ICON, numbered from 1) and the directory
    that lists them (RT_GROUP_ICON)."""
    try:
        reserved, kind, count = struct.unpack_from("<HHH", ico, 0)
        if reserved != 0 or kind != 1 or count == 0:
            raise AppExeError("the icon file is not an icon")
        pictures: list[bytes] = []
        group = bytearray(struct.pack("<HHH", 0, 1, count))
        for i in range(count):
            width, height, colours, _, planes, bits, size, offset = struct.unpack_from(
                "<BBBBHHII", ico, 6 + 16 * i
            )
            picture = ico[offset : offset + size]
            if len(picture) != size:
                raise AppExeError("the icon file is cut short")
            pictures.append(picture)
            group += struct.pack(
                "<BBBBHHIH", width, height, colours, 0, planes or 1, bits or 32, size, i + 1
            )
    except struct.error as exc:
        raise AppExeError("the icon file is cut short") from exc
    return pictures, bytes(group)


def split_signature(data: bytes) -> tuple[bytes, bytes]:
    """A PE file without its Authenticode signature or anything after its sections, and
    what followed its sections besides the signature (appended again afterwards)."""
    try:
        if data[:2] != b"MZ":
            raise AppExeError("the launcher is not a Windows program")
        (pe,) = struct.unpack_from("<I", data, 0x3C)
        if data[pe : pe + 4] != b"PE\0\0":
            raise AppExeError("the launcher is not a Windows program")
        (sections,) = struct.unpack_from("<H", data, pe + 6)
        (optional_size,) = struct.unpack_from("<H", data, pe + 20)
        optional = pe + 24
        (magic,) = struct.unpack_from("<H", data, optional)
        if magic not in (0x10B, 0x20B):
            raise AppExeError("the launcher is not a Windows program")
        directories = optional + (96 if magic == 0x10B else 112)
        (directory_count,) = struct.unpack_from("<I", data, directories - 4)
        table = optional + optional_size
        end = table + 40 * sections
        for i in range(sections):
            raw_size, raw_at = struct.unpack_from("<II", data, table + 40 * i + 16)
            if raw_size:
                end = max(end, raw_at + raw_size)
        security = directories + 8 * 4 if directory_count > 4 else None
        cert_at, cert_size = struct.unpack_from("<II", data, security) if security else (0, 0)
    except struct.error as exc:
        raise AppExeError("the launcher is cut short") from exc
    if end > len(data):
        raise AppExeError("the launcher is cut short")
    image = bytearray(data[:end])
    tail = data[end:]
    if security is not None and cert_size:
        image[security : security + 8] = bytes(8)
        if cert_at >= end:
            start = cert_at - end
            tail = tail[:start] + tail[start + cert_size :]
    return bytes(image), tail if tail.strip(b"\0") else b""


def stamp_key(launcher: bytes, ico: bytes, version: str) -> str:
    """What the copy is made from: the launcher, the icon and the version information
    (which holds the name, so renaming the product makes the copy again)."""
    digest = hashlib.sha256()
    for part in (FORMAT.encode(), launcher, ico, version_info(version)):
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()


# --- Windows --------------------------------------------------------------------------------


def _kernel32() -> Any:
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    k.LoadLibraryExW.restype = wintypes.HMODULE
    k.LoadLibraryExW.argtypes = (wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD)
    k.FreeLibrary.argtypes = (wintypes.HMODULE,)
    k.EnumResourceNamesW.argtypes = (
        wintypes.HMODULE,
        wintypes.LPVOID,
        ctypes.c_void_p,
        ctypes.c_ssize_t,
    )
    k.EnumResourceLanguagesW.argtypes = (
        wintypes.HMODULE,
        wintypes.LPVOID,
        wintypes.LPVOID,
        ctypes.c_void_p,
        ctypes.c_ssize_t,
    )
    k.BeginUpdateResourceW.restype = wintypes.HANDLE
    k.BeginUpdateResourceW.argtypes = (wintypes.LPCWSTR, wintypes.BOOL)
    k.UpdateResourceW.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.WORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    k.EndUpdateResourceW.argtypes = (wintypes.HANDLE, wintypes.BOOL)
    return k


def _error(what: str) -> AppExeError:
    import ctypes

    return AppExeError(f"{what}: {ctypes.WinError(ctypes.get_last_error())}")  # type: ignore[attr-defined]


def _resources(k: Any, path: Path, kinds: Sequence[int]) -> list[tuple[int, int | str, int]]:
    """(type, name, language) of each resource of these types in the file."""
    import ctypes
    from ctypes import wintypes

    names: list[int | str] = []
    languages: list[int] = []

    @ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
        wintypes.BOOL, wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t
    )
    def on_name(_module: object, _kind: object, name: int | None, _param: object) -> bool:
        value = name or 0
        names.append(value if value >> 16 == 0 else ctypes.wstring_at(value))
        return True

    @ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
        wintypes.BOOL,
        wintypes.HMODULE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.WORD,
        ctypes.c_ssize_t,
    )
    def on_language(_module: object, _kind: object, _name: object, lang: int, _p: object) -> bool:
        languages.append(lang)
        return True

    as_data = 0x2 | 0x20  # LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE
    module = k.LoadLibraryExW(str(path), None, as_data)
    if not module:
        raise _error("the launcher's resources could not be read")
    found: list[tuple[int, int | str, int]] = []
    try:
        for kind in kinds:
            names.clear()
            k.EnumResourceNamesW(module, kind, on_name, 0)  # FALSE when there are none
            for name in names:
                languages.clear()
                k.EnumResourceLanguagesW(module, kind, name, on_language, 0)
                found += [(kind, name, lang) for lang in languages]
    finally:
        k.FreeLibrary(module)
    return found


def replace_resources(path: Path, ico: bytes, version: str) -> None:
    """Replace the icons and version information of the program at `path` (Windows)."""
    pictures, group = icon_resources(ico)
    changes = [(RT_ICON, n, picture) for n, picture in enumerate(pictures, 1)]
    changes += [(RT_GROUP_ICON, 1, group), (RT_VERSION, 1, version_info(version))]
    k = _kernel32()
    old = _resources(k, path, (RT_ICON, RT_GROUP_ICON, RT_VERSION))
    handle = k.BeginUpdateResourceW(str(path), False)
    if not handle:
        raise _error("the launcher's copy could not be opened")
    done = False
    try:
        for kind, name, lang in old:
            if not k.UpdateResourceW(handle, kind, name, lang, None, 0):
                raise _error("a resource could not be removed")
        for kind, number, data in changes:
            if not k.UpdateResourceW(handle, kind, number, LANGUAGE, data, len(data)):
                raise _error("a resource could not be written")
        done = True
    finally:
        if not k.EndUpdateResourceW(handle, not done) and done:
            raise _error("the resources could not be saved")


def starts_python(exe: Path) -> bool:
    """Whether `exe` starts Python inside this venv."""
    import subprocess

    from kitshot.platform import procs

    try:
        done = procs.run_quiet([str(exe), "-c", CHECK], timeout=START_SECONDS, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("%s could not be started: %s", exe.name, exc)
        return False
    if done.returncode != 0:
        logger.warning("%s did not start Python (exit code %s)", exe.name, done.returncode)
    return done.returncode == 0


def make(
    target: Path,
    launcher: bytes,
    ico: bytes,
    version: str,
    *,
    starts: Callable[[Path], bool] = starts_python,
) -> None:
    """Write the copy to `target`, beside the launcher, and check that it starts Python.
    Raises AppExeError or OSError and leaves nothing behind."""
    image, tail = split_signature(launcher)
    target.write_bytes(image)
    try:
        replace_resources(target, ico, version)
        if tail:
            with target.open("ab") as f:
                f.write(tail)
        if not starts(target):
            raise AppExeError(f"{target.name} did not start Python")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def in_use(path: Path) -> bool:
    """Whether a running program has `path` open: Windows refuses to open it for writing."""
    try:
        with path.open("r+b"):
            return False
    except PermissionError:
        return True
    except OSError:
        return False


def _read_stamp(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def ensure(
    scripts: Path,
    icon: Path,
    version: str,
    *,
    os_name: str = os.name,
    maker: Callable[[Path, bytes, bytes, str], None] = make,
) -> Path | None:
    """kitshot-app.exe in the venv's Scripts folder, made again when its launcher, icon or
    version information changed. None off Windows, outside a venv, or when it could not be
    made (the reason is logged): the caller then uses pythonw.exe. A copy the app runs from
    cannot be replaced; it is kept until a later start."""
    if os_name != "nt":
        return None
    launcher, target, stamp = scripts / LAUNCHER, scripts / APP_EXE, scripts / STAMP
    if not launcher.is_file() or not (scripts.parent / "pyvenv.cfg").is_file():
        return None
    existing = target if target.is_file() else None
    try:
        data, ico = launcher.read_bytes(), icon.read_bytes()
    except OSError as exc:
        logger.warning("%s could not be checked: %s", APP_EXE, exc)
        return existing
    key = stamp_key(data, ico, version)
    if existing is not None and _read_stamp(stamp) == key:
        return existing
    if existing is not None and in_use(existing):
        logger.info("%s is in use, so it is made again at a later start", APP_EXE)
        return existing
    part = scripts / f".{APP_EXE.removesuffix('.exe')}-{os.getpid()}.exe"
    try:
        maker(part, data, ico, version)
        try:
            os.replace(part, target)
        except PermissionError as exc:
            logger.info("%s is in use, so it is made again later: %s", APP_EXE, exc)
            return existing
        stamp.write_text(key + "\n", encoding="ascii")
        logger.info("%s made: Task Manager shows %s's name and icon", APP_EXE, NAME)
    except (AppExeError, OSError) as exc:
        logger.warning("%s could not be made, so pythonw.exe is used: %s", APP_EXE, exc)
    finally:
        part.unlink(missing_ok=True)
    return target if target.is_file() else None
