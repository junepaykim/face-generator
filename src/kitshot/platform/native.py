"""Load the optional Rust figure scan and PNG encoder, or use Python fallbacks.

Callers test `ext is not None`. Only a matching installed version is loaded: Windows locks
a loaded extension, preventing replacement of an outdated wheel. Developers build with
maturin; other installations obtain the wheel through `kitshot native --install`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from types import ModuleType

VERSION = "0.2.0"
"""The kitshot-native version this code calls: native/pyproject.toml's (a test checks)."""
DIST = "kitshot-native"


def installed_version() -> str | None:
    """The installed kitshot-native's version, from its metadata; None if not installed."""
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
        return None, f"kitshot-native {found} is installed; this Kitshot needs {VERSION}"
    try:
        import kitshot_native
    except ImportError:  # not built or not installed: the Python versions run
        return None, "not installed"
    return kitshot_native, ""


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
            return f"in use (kitshot-native {self.version or VERSION}{how})"
        return f"not in use: the Python versions run ({self.reason})"


def status() -> Status:
    """Whether the Rust extension is in use, for `kitshot status` and the dashboard."""
    return Status(
        in_use=ext is not None,
        version=installed_version(),
        source=installed_from(),
        reason=reason,
    )
