"""Draw the portrait-outline icon for the tray and Windows shortcuts.

Icons are generated locally; the .ico goes in the data folder. The tray dot is green while
ready or drawing, amber while paused and red after a fault (app_controller.dot_for).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from facegen.desktop.app_controller import Dot

BACKGROUND = (31, 58, 95, 255)
FIGURE = (244, 239, 230, 255)
DOT_COLOURS: dict[str, tuple[int, int, int, int]] = {
    "running": (46, 160, 67, 255),
    "paused": (210, 153, 34, 255),
    "fault": (209, 36, 47, 255),
}
RING = (255, 255, 255, 255)
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
ICO_NAME = "facegen.ico"
_SCALE = 4
"""Drawn this many times larger, then reduced: smooth edges without anti-aliased drawing."""


def draw_icon(dot: Dot | None = None, size: int = 64) -> Image.Image:
    """The icon as an RGBA picture `size` pixels square, with the state dot if given."""
    from PIL import Image, ImageDraw

    big = size * _SCALE
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = big / 64  # the shapes below are laid out on a 64-unit square

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        return (round(x0 * u), round(y0 * u), round(x1 * u), round(y1 * u))

    d.rounded_rectangle(box(2, 2, 62, 62), radius=round(13 * u), fill=BACKGROUND)
    # Shoulders: the top of a wide ellipse; what reaches below the square is cut off.
    d.ellipse(box(12, 40, 52, 76), fill=FIGURE)
    # The head, with a gap above the shoulders so the outline reads at 16 pixels.
    d.ellipse(box(21, 11, 43, 35), fill=BACKGROUND)
    d.ellipse(box(23, 13, 41, 33), fill=FIGURE)
    img = _clip_to_square(img, u)
    if dot is not None:
        colour = DOT_COLOURS[dot]
        d = ImageDraw.Draw(img)
        d.ellipse(box(37, 37, 63, 63), fill=RING)
        d.ellipse(box(40, 40, 60, 60), fill=colour)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _clip_to_square(img: Image.Image, u: float) -> Image.Image:
    """Keep only what is inside the rounded square (the shoulders reach below it)."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (round(2 * u), round(2 * u), round(62 * u), round(62 * u)), radius=round(13 * u), fill=255
    )
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def write_ico(folder: Path) -> Path:
    """facegen.ico in `folder`, every size Windows asks for (no dot); written only when it is
    missing, atomically. Returns its path."""
    path = folder / ICO_NAME
    if path.is_file():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    big = draw_icon(None, max(ICO_SIZES))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    big.save(tmp, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    os.replace(tmp, path)
    return path
