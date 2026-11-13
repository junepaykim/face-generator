"""The optional Rust extension, `facegen_native` (built from native/ by maturin).

It holds faster versions of two CPU-bound pieces: the figure scan (figure.py) and the
published-picture encoder (publish.py). Each has a Python version that gives the same
result, so the tool runs without the extension; `uv sync --no-group native` leaves it out.
Callers check `ext is not None` and fall back to Python otherwise.
"""

from __future__ import annotations

from types import ModuleType

ext: ModuleType | None
try:
    import facegen_native as ext
except ImportError:  # not built: the Python versions run
    ext = None
