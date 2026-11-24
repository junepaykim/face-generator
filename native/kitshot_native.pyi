"""Type stubs for Kitshot's optional Rust extension (native/src/lib.rs)."""

from os import PathLike

type Rgb = tuple[int, int, int]
type CropBox = tuple[int, int, int, int]
type Size = tuple[int, int]

class Grid:
    """A small RGB copy of a picture, scanned for the figure (kitshot/figure.py)."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def __init__(self, pixels: bytes, width: int, height: int) -> None: ...
    @staticmethod
    def scaled(pixels: bytes, width: int, height: int, grid_width: int, grid_height: int) -> Grid:
        """A grid scaled from a full-size RGB picture exactly as Pillow's resize would."""
    def tobytes(self) -> bytes: ...
    def backdrop(self, corner: int) -> Rgb: ...
    def first_row(self, backdrop: Rgb, diff: int, need: int) -> int | None: ...
    def column_weights(self, backdrop: Rgb, diff: int, y0: int, y1: int) -> list[int]: ...

def encode_portrait(master: str | PathLike[str], crop: CropBox, size: Size) -> bytes:
    """One published picture as PNG bytes (kitshot/publish.py)."""

def encode_portraits(jobs: list[tuple[str | PathLike[str], CropBox, Size]]) -> list[bytes]:
    """Several published pictures at once, on all cores; the bytes in the same order."""
