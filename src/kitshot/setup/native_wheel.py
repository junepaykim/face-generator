"""Select, verify and install the prebuilt Rust extension without a local compiler.

Wheels come from `native-v<version>`, matching native/pyproject.toml. Only SHA-256 sums
listed in native/wheels.sha256 are accepted; downloaded mismatches are deleted.
Selection is pure; download and install adapters are injectable.

Installation uses `uv pip install`. Keep the wheel with `uv sync --inexact`; plain
`uv sync` removes it when absent from the lock's dependencies for this PC. A failed
install leaves the Python fallbacks available.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import sys
import tempfile
import tomllib
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from kitshot import locations
from kitshot.platform import procs

PROJECT_ROOT = locations.PROJECT_ROOT
NATIVE_DIR = locations.NATIVE_DIR
SUMS_PATH = NATIVE_DIR / "wheels.sha256"
RELEASES = "https://github.com/junepaykim/kitshot/releases/download"
TAG_PREFIX = "native-v"
MAX_BYTES = 64 * 1024 * 1024
"""A wheel is a few megabytes; anything far larger is not ours."""
DOWNLOAD_TIMEOUT = 120.0
INSTALL_TIMEOUT = 300.0


class WheelError(Exception):
    """The wheel could not be found, fetched, checked or installed."""


@dataclass(frozen=True, slots=True)
class Wheel:
    filename: str
    version: str
    python: str
    """Python tag, e.g. "cp312"."""
    abi: str
    """ABI tag, e.g. "abi3"."""
    platforms: tuple[str, ...]
    """Platform tags, e.g. ("win_amd64",)."""


_WHEEL = re.compile(
    r"^(?P<name>[A-Za-z0-9_.]+)-(?P<version>[^-]+)(?:-\d[^-]*)?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)


def parse_wheel_name(filename: str) -> Wheel | None:
    """The tags in a kitshot-native wheel's file name; None for another file or package."""
    m = _WHEEL.match(filename)
    if m is None or m["name"].lower().replace(".", "_") != "kitshot_native":
        return None
    return Wheel(
        filename=filename,
        version=m["version"],
        python=m["python"],
        abi=m["abi"],
        platforms=tuple(m["platform"].split(".")),
    )


def parse_sums(text: str) -> dict[str, str]:
    """`sha256sum` lines ("<64 hex>  <file name>", or with "*" for binary mode) to file name
    -> lower-case hex digest. Blank lines and "#" comments are skipped; anything else raises,
    so a damaged list is noticed rather than read as "no wheel"."""
    sums: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([0-9A-Fa-f]{64}) [ *](\S+)", line)
        if m is None:
            raise WheelError(f"{SUMS_PATH.name} line {n}: not '<sha256>  <file name>'")
        sums[m[2]] = m[1].lower()
    return sums


def native_version(pyproject_text: str) -> str:
    """The version in native/pyproject.toml."""
    try:
        return str(tomllib.loads(pyproject_text)["project"]["version"])
    except (tomllib.TOMLDecodeError, KeyError) as exc:
        raise WheelError(f"native/pyproject.toml has no project version: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Host:
    """What decides which wheel a PC can take."""

    system: str
    """sys.platform: "win32", "linux", ..."""
    machine: str
    """platform.machine(): "AMD64" on Windows, "x86_64" on Linux."""
    implementation: str
    """sys.implementation.name: "cpython"."""
    python: tuple[int, int]
    glibc: tuple[int, int] | None = None
    """The C library's version on a glibc Linux; None elsewhere (musl, Windows)."""


def this_host() -> Host:
    glibc: tuple[int, int] | None = None
    if sys.platform.startswith("linux"):
        lib, ver = platform.libc_ver()
        if lib == "glibc" and (m := re.match(r"(\d+)\.(\d+)", ver)):
            glibc = (int(m[1]), int(m[2]))
    return Host(
        system=sys.platform,
        machine=platform.machine(),
        implementation=sys.implementation.name,
        python=(sys.version_info.major, sys.version_info.minor),
        glibc=glibc,
    )


_ARCH = {"amd64": "x86_64", "x86_64": "x86_64", "x64": "x86_64", "arm64": "aarch64",
         "aarch64": "aarch64"}  # fmt: skip
_OLD_MANYLINUX = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}


def _platform_fits(host: Host, tag: str) -> bool:
    arch = _ARCH.get(host.machine.lower())
    if tag == "win_amd64":
        return host.system == "win32" and arch == "x86_64"
    if not host.system.startswith("linux") or host.glibc is None or arch is None:
        return False
    m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", tag)
    if m is not None:
        return host.glibc >= (int(m[1]), int(m[2])) and m[3] == arch
    m = re.fullmatch(r"(manylinux1|manylinux2010|manylinux2014)_(\w+)", tag)
    if m is not None:
        return host.glibc >= _OLD_MANYLINUX[m[1]] and m[2] == arch
    return False


def fits(host: Host, wheel: Wheel) -> bool:
    """Whether this PC's Python can load the wheel: CPython, the Python tag (an abi3 wheel
    for 3.12 serves 3.12 and later) and one of its platform tags."""
    if host.implementation != "cpython":
        return False
    m = re.fullmatch(r"cp(\d)(\d+)", wheel.python)
    if m is None:
        return False
    built_for = (int(m[1]), int(m[2]))
    if wheel.abi == "abi3":
        if host.python < built_for or host.python[0] != built_for[0]:
            return False
    elif wheel.abi != wheel.python or host.python != built_for:
        return False
    return any(_platform_fits(host, tag) for tag in wheel.platforms)


def choose(host: Host, version: str, sums: Mapping[str, str]) -> Wheel | None:
    """The listed wheel of `version` that this PC can take, or None."""
    for filename in sums:
        wheel = parse_wheel_name(filename)
        if wheel is not None and wheel.version == version and fits(host, wheel):
            return wheel
    return None


def release_url(version: str, filename: str, releases: str = RELEASES) -> str:
    return f"{releases}/{TAG_PREFIX}{version}/{filename}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, *, timeout: float = DOWNLOAD_TIMEOUT) -> None:
    """Fetch `url` into `dest`. Sends a plain request with a generic client name, nothing
    about this PC or its user."""
    request = urllib.request.Request(url, headers={"User-Agent": "kitshot"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as f:
            total = 0
            while block := response.read(1 << 16):
                total += len(block)
                if total > MAX_BYTES:
                    raise WheelError(f"{url}: larger than {MAX_BYTES // (1 << 20)} MB")
                f.write(block)
    except OSError as exc:  # urllib's errors are OSErrors: no network, 404, a timeout
        raise WheelError(f"could not download {url}: {exc}") from exc


def find_uv(env: Mapping[str, str] = os.environ) -> str | None:
    """uv: the one running us (`uv run` sets UV), then PATH."""
    found = env.get("UV")
    if found and Path(found).is_file():
        return found
    return shutil.which("uv")


def uv_install(wheel: Path, *, python: str = sys.executable) -> None:
    uv = find_uv()
    if uv is None:
        raise WheelError("uv is not on PATH, so the wheel cannot be installed")
    done = procs.run_quiet(
        [uv, "pip", "install", "--python", python, "--no-deps", "--reinstall", str(wheel)],
        timeout=INSTALL_TIMEOUT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if done.returncode != 0:
        lines = (done.stderr or done.stdout or "").strip().splitlines()
        raise WheelError(f"uv pip install failed: {lines[-1] if lines else done.returncode}")


def imports_cleanly(*, python: str = sys.executable) -> bool:
    """Whether a fresh Python loads the installed extension at the wanted version (this
    process may already have looked and found nothing)."""
    done = procs.run_quiet(
        [python, "-c", "from kitshot.platform import native; raise SystemExit(native.ext is None)"],
        timeout=60.0,
        capture_output=True,
    )
    return done.returncode == 0


@dataclass(frozen=True, slots=True)
class Plan:
    """What `kitshot native` would install on this PC."""

    version: str
    wheel: Wheel | None
    sha256: str | None
    url: str | None

    def describe(self) -> str:
        if self.wheel is None:
            return (
                f"No prebuilt kitshot-native {self.version} for this PC is listed in"
                f" native/{SUMS_PATH.name}."
            )
        return f"Prebuilt wheel {self.wheel.filename} from the {TAG_PREFIX}{self.version} release."


def plan(host: Host | None = None, *, native_dir: Path | None = None) -> Plan:
    """The wheel for this PC (or `host`): native/pyproject.toml's version, listed in
    native/wheels.sha256 (`native_dir` instead of native/, for tests)."""
    folder = native_dir or NATIVE_DIR
    version = native_version((folder / "pyproject.toml").read_text(encoding="utf-8"))
    sums = folder / SUMS_PATH.name
    listed = parse_sums(sums.read_text(encoding="utf-8")) if sums.is_file() else {}
    wheel = choose(host or this_host(), version, listed)
    if wheel is None:
        return Plan(version, None, None, None)
    return Plan(version, wheel, listed[wheel.filename], release_url(version, wheel.filename))


def install(
    chosen: Plan,
    *,
    source: Path | None = None,
    fetch: Callable[[str, Path], None] = download,
    installer: Callable[[Path], None] = uv_install,
) -> str:
    """Fetch the planned wheel (or take it from the folder `source`), check its SHA-256
    against the tracked list and install it. Returns the wheel's file name; raises
    WheelError, after removing a downloaded file that failed its check."""
    if chosen.wheel is None or chosen.sha256 is None or chosen.url is None:
        raise WheelError(chosen.describe())
    name = chosen.wheel.filename
    with tempfile.TemporaryDirectory(prefix="kitshot-wheel-") as tmp:
        if source is not None:
            path = source / name
            if not path.is_file():
                raise WheelError(f"{path} does not exist")
        else:
            path = Path(tmp) / name
            fetch(chosen.url, path)
        found = sha256_of(path)
        if found != chosen.sha256:
            if source is None:
                path.unlink(missing_ok=True)
            raise WheelError(
                f"{name}: SHA-256 {found} is not the {chosen.sha256} listed in"
                f" native/{SUMS_PATH.name}"
            )
        installer(path)
    return name
