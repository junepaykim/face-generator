"""Repository and shared child-script locations, independent of test tier."""

from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
ROOT = TESTS.parent
