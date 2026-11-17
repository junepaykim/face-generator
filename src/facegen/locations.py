"""Locations in the repository-based installation, independent of the caller's depth.

Keep this module at the package root and dependent only on the standard library: desktop
startup needs these paths even when third-party packages cannot be imported. Package data
continues to use importlib.resources rather than this filesystem layout.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
NATIVE_DIR = PROJECT_ROOT / "native"
