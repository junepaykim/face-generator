"""Picture geometry: the 3:4 crop, published sizes and the face crop. No GPU."""

from __future__ import annotations

import pytest
from fakes import BACKDROPS, figure
from PIL import Image

from facegen.picture import aspect_box, crop_to_aspect, face_box, published_size


@pytest.mark.parametrize(
    ("size", "box"),
    [
        ((896, 1152), (16, 0, 880, 1152)),  # the model's portrait render: 16 px off each side
        ((1344, 1728), (24, 0, 1320, 1728)),  # group A's detail pass
        ((1024, 1024), (128, 0, 896, 1024)),  # a square picture from before 3:4
        ((1536, 1536), (192, 0, 1344, 1536)),
        ((768, 1024), (0, 0, 768, 1024)),  # already 3:4
        ((600, 1000), (0, 0, 600, 800)),  # too tall: the bottom goes, the head stays
    ],
)
def test_aspect_box(size: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    assert aspect_box(*size) == box
    left, top, right, bottom = box
    assert (right - left) * 4 == (bottom - top) * 3


def test_crop_to_aspect_leaves_a_3_4_picture_alone() -> None:
    img = Image.new("RGB", (768, 1024))
    assert crop_to_aspect(img) is img
    assert crop_to_aspect(Image.new("RGB", (896, 1152))).size == (864, 1152)


@pytest.mark.parametrize(
    ("edge", "size"), [(1024, (768, 1024)), (768, (576, 768)), (512, (384, 512))]
)
def test_published_size_is_3_4_with_the_long_edge_as_height(
    edge: int, size: tuple[int, int]
) -> None:
    assert published_size(edge) == size


@pytest.mark.parametrize("shape", [(512, 512), (448, 576)])
def test_face_box_frames_the_head(shape: tuple[int, int]) -> None:
    w, h = shape
    img = figure(w, h, top=0.06)
    left, top, right, bottom = face_box(img)
    assert right - left == bottom - top == round(0.36 * h)
    assert 0 <= left and right <= w and 0 <= top and bottom <= h
    # The head is an ellipse centred across, starting 6% down.
    assert top <= 0.06 * h <= top + 0.1 * h
    assert abs((left + right) / 2 - w / 2) <= 0.05 * w


def test_face_box_follows_a_figure_off_centre() -> None:
    img = figure(512, 512, top=0.1)
    shifted = Image.new("RGB", (512, 512), (200, 200, 205))
    shifted.paste(img.crop((0, 0, 412, 512)), (100, 0))
    left, _, right, _ = face_box(shifted)
    assert (left + right) / 2 > 256 + 60


def test_face_box_without_a_figure_takes_the_top_centre() -> None:
    blank = Image.new("RGB", (300, 400), (200, 200, 205))
    left, top, right, bottom = face_box(blank)
    assert (right - left, bottom - top) == (144, 144)
    assert top == 20 and left == 78


@pytest.mark.parametrize("kind", BACKDROPS)
def test_face_box_on_every_backdrop(kind: str) -> None:
    """Gradients and two tones (shoot.py) are followed as backdrop, not taken for the head."""
    img = figure(448, 576, top=0.08, kind=kind)
    left, top, right, _ = face_box(img)
    assert top <= 0.08 * 576 <= top + 0.1 * 576
    assert abs((left + right) / 2 - 224) <= 0.05 * 448
