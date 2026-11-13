"""Finding the figure against the plain studio backdrop. Pure.

`qc.head_top` and `picture.face_box` both look at a small copy of the picture (the heuristic
of bake-off run r4, docs/research.md section 5). The backdrop colour is the per-channel
median of the two top corners; a pixel whose summed RGB difference from it is over `diff`
is foreground.

When the extension is built (native.py), Rust makes the small copy and scans it; otherwise
Pillow makes it and Python scans it. Both give the same answers (tests/test_native.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from facegen import native

if TYPE_CHECKING:
    from PIL.Image import Image

type Rgb = tuple[int, int, int]


class Grid(Protocol):
    """A small RGB picture to scan. `PyGrid` here, or `facegen_native.Grid`."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def tobytes(self) -> bytes: ...
    def backdrop(self, corner: int) -> Rgb: ...
    def first_row(self, backdrop: Rgb, diff: int, need: int) -> int | None: ...
    def column_weights(self, backdrop: Rgb, diff: int, y0: int, y1: int) -> list[int]: ...


class PyGrid:
    """The Python version of the scan."""

    def __init__(self, pixels: bytes, width: int, height: int) -> None:
        if len(pixels) != width * height * 3:
            raise ValueError(f"{len(pixels)} bytes is not a {width}x{height} RGB picture")
        self.pixels = pixels
        self.width = width
        self.height = height

    def tobytes(self) -> bytes:
        return self.pixels

    def _pixel(self, x: int, y: int) -> Rgb:
        i = (y * self.width + x) * 3
        p = self.pixels
        return p[i], p[i + 1], p[i + 2]

    def _foreground(self, x: int, y: int, bg: Rgb, diff: int) -> bool:
        r, g, b = self._pixel(x, y)
        return abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) > diff

    def backdrop(self, corner: int) -> Rgb:
        """The per-channel median of the two top `corner` x `corner` squares."""
        if not 0 < corner <= min(self.width, self.height):
            raise ValueError(f"corner {corner} does not fit a {self.width}x{self.height} grid")
        xs = (*range(corner), *range(self.width - corner, self.width))
        found = [self._pixel(x, y) for y in range(corner) for x in xs]
        r, g, b = (sorted(p[i] for p in found)[len(found) // 2] for i in range(3))
        return r, g, b

    def first_row(self, backdrop: Rgb, diff: int, need: int) -> int | None:
        """The first row with more than `need` foreground pixels."""
        for y in range(self.height):
            if sum(self._foreground(x, y, backdrop, diff) for x in range(self.width)) > need:
                return y
        return None

    def column_weights(self, backdrop: Rgb, diff: int, y0: int, y1: int) -> list[int]:
        """Foreground pixels in each column, counted over rows y0..y1."""
        if y1 > self.height:
            raise ValueError(f"row {y1} is past the grid's {self.height} rows")
        return [
            sum(self._foreground(x, y, backdrop, diff) for y in range(y0, y1))
            for x in range(self.width)
        ]


def grid(image: Image, width: int, height: int, *, rust: bool = True) -> Grid:
    """`image` scaled to width x height (Pillow's default filter, bicubic), ready to scan.

    With the Rust extension (unless `rust` is False) the scaling runs in Rust too, on all
    cores: a bit-exact port of Pillow 13's resize (native/src/resample.rs), so both versions
    scan the same pixels.
    """
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    if rust and native.ext is not None:
        return native.ext.Grid.scaled(rgb.tobytes(), rgb.width, rgb.height, width, height)
    return PyGrid(rgb.resize((width, height)).tobytes(), width, height)
