"""The optional Rust extension, `facegen_native` (built from native/ by maturin).

It holds faster versions of two CPU-bound pieces: the figure scan (figure.py) and the
published-picture encoder (publish.py). Each has a Python version that gives the same
result, so the tool runs without the extension; `uv sync --no-group native` leaves it out.
Callers check `ext is not None` and fall back to Python otherwise.

Where it comes from: developers build it from native/ (`uv sync`, with Rust); elsewhere,
Windows above all, it is the prebuilt wheel that `facegen native --install` fetches
(native_wheel.py). Only the version this code was written for is used: a wheel left from
an older facegen is not even loaded (on Windows a loaded extension could not be replaced),
and the Python versions run until the matching one is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from types import ModuleType

VERSION = "0.1.0"
"""The facegen-native version this code calls: native/pyproject.toml's (a test checks)."""
DIST = "facegen-native"


def installed_version() -> str | None:
    """The installed facegen-native's version, from its metadata; None if not installed."""
    try:
        return metadata.version(DIST)
    except metadata.PackageNotFoundError:
        return None


def installed_from() -> str | None:
    """Where the installed extension came from: "wheel" for a prebuilt wheel, "build" for one
    built from a source folder (native/), None if not known (PEP 610's direct_url.json)."""
    try:
        text = metadata.distribution(DIST).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    if '"dir_info"' in text:
        return "build"
    if '"archive_info"' in text:
        return "wheel"
    return None


def _load() -> tuple[ModuleType | None, str]:
    found = installed_version()
    if found is not None and found != VERSION:
        return None, f"facegen-native {found} is installed; this facegen needs {VERSION}"
    try:
        import facegen_native
    except ImportError:  # not built or not installed: the Python versions run
        return None, "not installed"
    return facegen_native, ""


ext: ModuleType | None
reason: str
"""Why `ext` is None; empty when it is in use."""
ext, reason = _load()


@dataclass(frozen=True, slots=True)
class Status:
    in_use: bool
    version: str | None
    """The installed version, or None."""
    source: str | None
    """"wheel", "build" or None (not known)."""
    reason: str

    def describe(self) -> str:
        if self.in_use:
            how = {"wheel": ", prebuilt wheel", "build": ", built from native/"}.get(
                self.source or "", ""
            )
            return f"in use (facegen-native {self.version or VERSION}{how})"
        return f"not in use: the Python versions run ({self.reason})"


def status() -> Status:
    """Whether the Rust extension is in use, for `facegen status` and the dashboard."""
    return Status(
        in_use=ext is not None,
        version=installed_version(),
        source=installed_from(),
        reason=reason,
    )
