"""The Rust extension (native/) against the Python versions it stands in for. No GPU.

Skipped when the extension is not in use (`uv sync --no-group native`), unless
FACEGEN_REQUIRE_NATIVE is set: then that fails, as when the wheels workflow tests a
prebuilt wheel (.github/workflows/wheels.yml).
"""

from __future__ import annotations

import io
import os
import random
from pathlib import Path

import pytest
from fakes import BACKDROPS, figure
from PIL import Image, ImageChops, ImageDraw

from facegen.pipeline.publish import published_bytes, published_pictures
from facegen.platform import native
from facegen.qc import head_top
from facegen.render.figure import PyGrid, grid
from facegen.render.picture import face_box

if native.ext is None:
    if os.environ.get("FACEGEN_REQUIRE_NATIVE"):
        pytest.fail(f"the Rust extension is required here: {native.reason}", pytrace=False)
    pytest.skip(f"the Rust extension is not in use: {native.reason}", allow_module_level=True)
ext = native.ext


def noise(width: int, height: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    return Image.frombytes("RGB", (width, height), rng.randbytes(width * height * 3))


def scene(seed: int) -> Image.Image:
    """A figure on the backdrop with a few random shapes, so every scan has work to do."""
    rng = random.Random(seed)
    img = figure(448, 576, top=rng.uniform(0.0, 0.4))
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        x, y = rng.randrange(448), rng.randrange(576)
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        draw.rectangle((x, y, x + rng.randrange(80), y + rng.randrange(80)), fill=colour)
    return img


PICTURES = [
    figure(512, 512, top=0.06),
    figure(56, 72),
    *(figure(224, 288, top=0.08, kind=kind) for kind in BACKDROPS[1:]),
    *map(scene, range(8)),
    noise(64, 48, 1),
]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ((1536, 1536), (128, 128)),  # head_top on a square master
        ((1344, 1728), (128, 165)),  # face_box on a detail pass
        ((896, 1152), (128, 128)),  # head_top on a render: squashed
        ((100, 1000), (90, 100)),  # tall: Pillow 13 runs the vertical pass first
        ((40, 300), (40, 30)),  # height only
        ((300, 40), (30, 40)),  # width only
        ((20, 15), (128, 96)),  # upscaling
        ((7, 5), (1, 1)),
        ((1, 1), (5, 3)),
    ],
)
def test_the_rust_resize_matches_pillow(source: tuple[int, int], target: tuple[int, int]) -> None:
    image = noise(*source, seed=source[0] * 7 + target[1])
    rust = ext.Grid.scaled(image.tobytes(), *source, *target)
    assert rust.tobytes() == image.resize(target).tobytes()


def test_the_rust_resize_matches_pillow_on_random_sizes() -> None:
    rng = random.Random(27)
    for _ in range(150):
        source = (rng.randint(1, 160), rng.randint(1, 160))
        target = (rng.randint(1, 160), rng.randint(1, 160))
        image = noise(*source, seed=rng.randrange(1 << 30))
        rust = ext.Grid.scaled(image.tobytes(), *source, *target)
        assert rust.tobytes() == image.resize(target).tobytes(), (source, target)


def test_scaled_refuses_bad_input() -> None:
    with pytest.raises(ValueError):
        ext.Grid.scaled(b"\x00" * 10, 2, 2, 1, 1)
    with pytest.raises(ValueError):
        ext.Grid.scaled(b"\x00" * 12, 2, 2, 0, 1)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_both_grids_scan_the_same_pixels(mode: str) -> None:
    image = scene(3).convert(mode)
    rust, python = grid(image, 128, 96), grid(image, 128, 96, rust=False)
    assert rust.tobytes() == python.tobytes()


@pytest.mark.parametrize("image", PICTURES)
def test_the_rust_scan_matches_python(image: Image.Image) -> None:
    rust = grid(image, 128, 96)
    python = grid(image, 128, 96, rust=False)
    assert isinstance(rust, ext.Grid) and isinstance(python, PyGrid)
    assert (rust.width, rust.height) == (python.width, python.height) == (128, 96)
    assert rust.tobytes() == python.tobytes()
    for corner in (1, 5, 8, 48, 96):
        for diff in (0, 60, 200):
            edges = rust.edges(corner, diff)
            assert edges == python.edges(corner, diff)
            for need in (0, 2, 60):
                assert rust.first_row(edges, diff, need) == python.first_row(edges, diff, need)
            for y0, y1 in ((0, 96), (10, 40), (30, 30), (50, 20)):
                assert rust.column_weights(edges, diff, y0, y1) == python.column_weights(
                    edges, diff, y0, y1
                )


def test_both_scans_refuse_the_same_bad_input() -> None:
    for make in (ext.Grid, PyGrid):
        with pytest.raises(ValueError):
            make(b"\x00" * 10, 2, 2)
        g = make(b"\x00" * 12, 2, 2)
        for corner in (0, 3):
            with pytest.raises(ValueError):
                g.edges(corner, 60)
        edges = g.edges(1, 60)
        with pytest.raises(ValueError):
            g.column_weights(edges, 60, 0, 3)
        with pytest.raises(ValueError):
            g.column_weights(edges[:1], 60, 0, 2)  # one edge for two rows
        with pytest.raises(ValueError):
            g.first_row(edges[:1], 60, 0)


@pytest.mark.parametrize("image", PICTURES)
def test_head_top_and_face_box_do_not_depend_on_the_extension(
    image: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_rust = head_top(image), face_box(image)
    monkeypatch.setattr(native, "ext", None)
    assert (head_top(image), face_box(image)) == with_rust


def master(tmp_path: Path, name: str, image: Image.Image) -> Path:
    path = tmp_path / name
    image.save(path)
    return path


def gradient(width: int, height: int) -> Image.Image:
    """Smooth colour ramps and a figure: what a resize is tested on."""
    img = Image.merge(
        "RGB",
        (
            Image.linear_gradient("L").resize((width, height)),
            Image.linear_gradient("L").rotate(90).resize((width, height)),
            Image.new("L", (width, height), 120),
        ),
    )
    img.paste(figure(width // 2, height // 2), (width // 4, height // 4))
    return img


@pytest.mark.parametrize(
    ("size", "long_edge"),
    [((1536, 1536), 1024), ((1344, 1728), 768), ((864, 1152), 512), ((768, 1024), 1024)],
)
def test_the_rust_encoder_matches_pillow(
    tmp_path: Path, size: tuple[int, int], long_edge: int
) -> None:
    path = master(tmp_path, "m.png", gradient(*size))
    rust = Image.open(io.BytesIO(published_bytes(path, long_edge)))
    pillow = Image.open(io.BytesIO(published_bytes(path, long_edge, rust=False)))
    assert rust.format == "PNG" and rust.mode == pillow.mode == "RGB"
    assert rust.size == pillow.size == (long_edge * 3 // 4, long_edge)
    # Fixed-point rounding: almost every value within one level (see published_pictures).
    diff = ImageChops.difference(rust, pillow)
    counts = [sum(band.histogram()[2:]) for band in diff.split()]
    assert sum(counts) <= 0.001 * rust.width * rust.height * 3


def test_a_master_already_3_4_at_its_size_is_not_resampled(tmp_path: Path) -> None:
    img = noise(96, 128, 7)
    path = master(tmp_path, "m.png", img)
    rust = Image.open(io.BytesIO(published_bytes(path, 128)))
    assert rust.tobytes() == img.tobytes()


def test_a_batch_gives_the_same_bytes_as_one_at_a_time(tmp_path: Path) -> None:
    items = [
        (master(tmp_path, f"m{i}.png", gradient(300 + 40 * i, 400)), edge)
        for i, edge in enumerate((128, 96, 200, 128, 64, 128, 96, 200, 128, 64, 128))
    ]  # more than one chunk of 8
    batch = list(published_pictures(items))
    assert batch == [published_bytes(m, e) for m, e in items]
    assert [Image.open(io.BytesIO(b)).height for b in batch] == [e for _, e in items]


def test_a_missing_master_raises_oserror_either_way(tmp_path: Path) -> None:
    for rust in (True, False):
        with pytest.raises(OSError):
            published_bytes(tmp_path / "missing.png", 128, rust=rust)
