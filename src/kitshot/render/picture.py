"""Picture geometry: the picture's shape, published sizes and the face crop. Pure.

A picture is a 3:4 portrait or, since Phase 16g, a 1:1 square (`picture.aspect`; each picture
keeps the shape it was drawn in, `images.aspect`). The model renders near 3:4 and the master is
cropped to exactly that aspect; a square renders square. FM27 gets the master scaled down to
the group's published height (`groups.<g>.published_long_edge`).

Legacy square pictures are centre-cropped on publication, without automatic redraws.

`face_box` locates a reference face using figure.py's background-difference heuristic,
following plain, gradient or two-tone backdrops down the picture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kitshot.render import figure

if TYPE_CHECKING:
    from PIL.Image import Image

ASPECT = (3, 4)
"""Width to height of a portrait, the default shape (settings_values.ASPECTS has both)."""

type Box = tuple[int, int, int, int]
"""left, top, right, bottom in pixels, as PIL's crop takes them."""


def aspect_box(width: int, height: int, aspect: tuple[int, int] = ASPECT) -> Box:
    """The largest box of the aspect inside a picture.

    A picture that is too wide loses the same amount on both sides; one that is too tall
    loses its bottom, so the head at the top stays.
    """
    aw, ah = aspect
    if width * ah > height * aw:
        w = height * aw // ah
        left = (width - w) // 2
        return (left, 0, left + w, height)
    return (0, 0, width, width * ah // aw)


def crop_to_aspect(image: Image, aspect: tuple[int, int] = ASPECT) -> Image:
    """The picture cropped to the aspect; the same picture if it already has it."""
    box = aspect_box(image.width, image.height, aspect)
    if box == (0, 0, image.width, image.height):
        return image
    return image.crop(box)


def published_size(long_edge: int, aspect: tuple[int, int] = ASPECT) -> tuple[int, int]:
    """(width, height) of a published picture whose long edge, its height, is `long_edge`."""
    aw, ah = aspect
    return round(long_edge * aw / ah), long_edge


FACE_SHARE = 0.36
"""Side of the square face crop as a share of picture height."""
_GRID = 128  # the mask is worked out on a picture this many cells wide
_CORNER = 0.04  # share of the width sampled in each top corner for the backdrop colour
_DIFF = 60  # summed RGB difference from the backdrop that counts as foreground
_ROW_SHARE = 0.02  # a row with more foreground than this starts the figure
_ABOVE = 0.03  # the crop starts this share of the height above the top of the head
_NO_FIGURE_TOP = 0.05  # where the crop starts when no figure is found


def face_box(image: Image) -> Box:
    """A square around the head, found against the studio backdrop.

    The backdrop is followed down from the two top corners, row by row (a plain, gradient
    or two-tone backdrop). The top of the head is the first row with enough pixels away
    from it; the square starts a little above that and is centred across on the foreground
    in the band below. Crude, but every picture shares the same media-day framing. With no
    figure found it falls back to the top centre. The scan is figure.py's.
    """
    width, height = image.size
    gw = _GRID
    gh = max(1, round(_GRID * height / width))
    grid = figure.grid(image, gw, gh)
    bg = grid.edges(max(1, round(_CORNER * gw)), _DIFF)
    first = grid.first_row(bg, _DIFF, max(1, int(_ROW_SHARE * gw)))

    side = min(round(FACE_SHARE * height), width, height)
    if first is None:
        top = round(_NO_FIGURE_TOP * height)
        cx = width // 2
    else:
        top = max(0, round(first * height / gh) - round(_ABOVE * height))
        y0 = round(top * gh / height)
        y1 = min(gh, y0 + max(1, round(side * gh / height)))
        weights = grid.column_weights(bg, _DIFF, y0, y1)
        total = sum(weights)
        if total:
            centre = sum(w * (x + 0.5) for x, w in enumerate(weights)) / total
            cx = round(centre * width / gw)
        else:
            cx = width // 2
    left = min(max(0, cx - side // 2), width - side)
    top = min(top, height - side)
    return (left, top, left + side, top + side)
