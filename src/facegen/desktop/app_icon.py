"""Kitshot's icon for the tray, the window and Windows shortcuts (Phase 17c).

The pictures are the user's SVG (web/static/kitshot.svg) drawn once at every size Windows asks
for and bundled in facegen/data/icon/ (scripts/build_icon.py); nothing is drawn here. The tray
icon adds a state dot over the bottom-right bracket: green while ready or drawing, amber while
paused and red after a fault (app_controller.dot_for). The .ico goes in the data folder, named
after the pictures, so a changed icon is a new file that shortcuts, the window and
facegen-app.exe pick up (app_exe.py remakes the copy when the .ico's bytes change).
"""

from __future__ import annotations

import hashlib
import io
import os
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from facegen.desktop.app_controller import Dot

DOT_COLOURS: dict[str, tuple[int, int, int, int]] = {
    "running": (46, 160, 67, 255),
    "paused": (210, 153, 34, 255),
    "fault": (209, 36, 47, 255),
}
RING = (255, 255, 255, 255)
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
"""The bundled sizes, every one Windows asks an .ico for."""
ICO_PREFIX = "kitshot-"
_SCALE = 4
"""The dot is drawn this many times larger, then reduced: smooth edges."""


def bundled_png(size: int) -> bytes:
    """The bundled picture `size` pixels square (one of ICO_SIZES)."""
    if size not in ICO_SIZES:
        raise ValueError(f"no bundled icon of {size} pixels")
    return resources.files("facegen").joinpath("data", "icon", f"kitshot-{size}.png").read_bytes()


@cache
def _decoded(size: int) -> Image.Image:
    from PIL import Image

    with Image.open(io.BytesIO(bundled_png(size))) as img:
        return img.convert("RGBA")


def icon_image(dot: Dot | None = None, size: int = 64) -> Image.Image:
    """The icon as an RGBA picture `size` pixels square, with the state dot if given. A size
    that is not bundled is reduced from the next larger one."""
    from PIL import Image, ImageDraw

    if size in ICO_SIZES:
        img = _decoded(size).copy()
    else:
        larger = min((s for s in ICO_SIZES if s >= size), default=max(ICO_SIZES))
        img = _decoded(larger).resize((size, size), Image.Resampling.LANCZOS)
    if dot is None:
        return img
    big = size * _SCALE
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    u = big / 64  # laid out on a 64-unit square: a ringed dot over the bottom-right bracket

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        return (round(x0 * u), round(y0 * u), round(x1 * u), round(y1 * u))

    d.ellipse(box(37, 37, 63, 63), fill=RING)
    d.ellipse(box(40, 40, 60, 60), fill=DOT_COLOURS[dot])
    img.alpha_composite(layer.resize((size, size), Image.Resampling.LANCZOS))
    return img


@cache
def icon_key() -> str:
    """A short hash of the bundled pictures: part of the .ico's name."""
    digest = hashlib.sha256()
    for size in ICO_SIZES:
        digest.update(hashlib.sha256(bundled_png(size)).digest())
    return digest.hexdigest()[:10]


def ico_name() -> str:
    return f"{ICO_PREFIX}{icon_key()}.ico"


def write_ico(folder: Path, name: str | None = None) -> Path:
    """The icon as an .ico in `folder` with every size Windows asks for (no dot), named
    `name` or after the pictures (ico_name); written only when it is missing, atomically.
    Returns its path."""
    path = folder / (name or ico_name())
    if path.is_file():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    largest, *rest = [icon_image(None, s) for s in sorted(ICO_SIZES, reverse=True)]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        largest.save(tmp, format="ICO", sizes=[(s, s) for s in ICO_SIZES], append_images=rest)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
