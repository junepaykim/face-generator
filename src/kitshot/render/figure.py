"""Finding the figure against the studio backdrop. Pure.

`quality_checks.head_top` and `picture.face_box` scan a small copy of the picture:

- The backdrop is followed down the picture, row by row, at each side (`edges`). It starts
  as the per-channel median of the top `corner` x `corner` square on that side. In each
  row, the median of that side's outer `corner` pixels replaces it if it is within `diff`
  of it: a gradient is followed a row at a time, while hair or an arm reaching the edge
  (a jump) is not taken for backdrop.
- A pixel is backdrop if its summed RGB difference from the row's left colour, its right
  colour, or the colour between them at its place across (a sideways gradient) is at most
  `diff`; otherwise it is foreground. Two tones split down the picture are both backdrop.

On a plain backdrop, every row keeps the top corners' colour.

When the extension is built (native.py), Rust makes the small copy and scans it; otherwise
Pillow makes it and Python scans it. Both give the same answers (tests/test_native.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from kitshot.platform import native

if TYPE_CHECKING:
    from PIL.Image import Image

type Rgb = tuple[int, int, int]
type Edge = tuple[Rgb, Rgb]
"""The backdrop's colour at the left and the right end of one row."""


class Grid(Protocol):
    """A small RGB picture to scan. `PyGrid` here, or `kitshot_native.Grid`."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def tobytes(self) -> bytes: ...
    def edges(self, corner: int, diff: int) -> list[Edge]: ...
    def first_row(self, edges: Sequence[Edge], diff: int, need: int) -> int | None: ...
    def column_weights(self, edges: Sequence[Edge], diff: int, y0: int, y1: int) -> list[int]: ...


def _distance(a: Rgb, b: Rgb) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _median(found: list[Rgb]) -> Rgb:
    """Per channel, `sorted(values)[len // 2]`: the upper median for an even count."""
    r, g, b = (sorted(p[i] for p in found)[len(found) // 2] for i in range(3))
    return r, g, b


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

    def _foreground(self, x: int, y: int, edge: Edge, diff: int) -> bool:
        p = self._pixel(x, y)
        left, right = edge
        if _distance(p, left) <= diff or _distance(p, right) <= diff:
            return False
        span = self.width - 1
        if span == 0:
            return True
        between = (
            (left[0] * (span - x) + right[0] * x) // span,
            (left[1] * (span - x) + right[1] * x) // span,
            (left[2] * (span - x) + right[2] * x) // span,
        )
        return _distance(p, between) > diff

    def _check(self, edges: Sequence[Edge]) -> None:
        if len(edges) != self.height:
            raise ValueError(f"{len(edges)} edges for the grid's {self.height} rows")

    def edges(self, corner: int, diff: int) -> list[Edge]:
        """The backdrop's colour at both ends of each row, followed down from the top
        corners while it changes by at most `diff` from one row to the next."""
        if not 0 < corner <= min(self.width, self.height):
            raise ValueError(f"corner {corner} does not fit a {self.width}x{self.height} grid")
        sides = (range(corner), range(self.width - corner, self.width))
        current = [_median([self._pixel(x, y) for y in range(corner) for x in xs]) for xs in sides]
        out: list[Edge] = []
        for y in range(self.height):
            for side, xs in enumerate(sides):
                row = _median([self._pixel(x, y) for x in xs])
                if _distance(row, current[side]) <= diff:
                    current[side] = row
            out.append((current[0], current[1]))
        return out

    def first_row(self, edges: Sequence[Edge], diff: int, need: int) -> int | None:
        """The first row with more than `need` foreground pixels."""
        self._check(edges)
        for y in range(self.height):
            edge = edges[y]
            if sum(self._foreground(x, y, edge, diff) for x in range(self.width)) > need:
                return y
        return None

    def column_weights(self, edges: Sequence[Edge], diff: int, y0: int, y1: int) -> list[int]:
        """Foreground pixels in each column, counted over rows y0..y1."""
        self._check(edges)
        if y1 > self.height:
            raise ValueError(f"row {y1} is past the grid's {self.height} rows")
        return [
            sum(self._foreground(x, y, edges[y], diff) for y in range(y0, y1))
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
