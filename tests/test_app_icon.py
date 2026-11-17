"""facegen's icon, drawn with Pillow (Phase 11)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from facegen.app_icon import BACKGROUND, DOT_COLOURS, FIGURE, ICO_SIZES, draw_icon, write_ico


def close(a: tuple[int, ...], b: tuple[int, ...], tol: int = 12) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("size", [16, 32, 64])
def test_the_icon_is_a_square_rgba_picture(size: int) -> None:
    img = draw_icon(None, size)
    assert img.mode == "RGBA" and img.size == (size, size)


def test_the_icon_shows_a_head_on_the_background() -> None:
    img = draw_icon(None, 64)
    assert close(img.getpixel((32, 22)), FIGURE)  # the head
    assert close(img.getpixel((8, 32)), BACKGROUND)
    assert img.getpixel((0, 0))[3] == 0  # the rounded corner is see-through


@pytest.mark.parametrize("dot", ["running", "paused", "fault"])
def test_the_tray_icon_has_its_state_dot(dot: str) -> None:
    img = draw_icon(dot, 64)  # type: ignore[arg-type]
    assert close(img.getpixel((50, 50)), DOT_COLOURS[dot])
    assert not close(draw_icon(None, 64).getpixel((50, 50)), DOT_COLOURS[dot])


def test_the_ico_file_has_every_size_and_is_written_once(tmp_path: Path) -> None:
    path = write_ico(tmp_path / "app")
    assert path == tmp_path / "app" / "facegen.ico"
    with Image.open(path) as ico:
        assert set(ico.info["sizes"]) == {(s, s) for s in ICO_SIZES}
    stamp = path.stat().st_mtime_ns
    assert write_ico(tmp_path / "app") == path and path.stat().st_mtime_ns == stamp
    assert not list((tmp_path / "app").glob(".*.part"))
