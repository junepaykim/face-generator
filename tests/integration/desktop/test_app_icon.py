"""Kitshot's icon (Phase 17c): the bundled pictures of the user's SVG, the tray's state dot on
top, and the .ico named after the pictures so a changed icon is a new file."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from support.paths import ROOT

from facegen.desktop import app_icon
from facegen.desktop.app_icon import DOT_COLOURS, ICO_SIZES, icon_image, write_ico

NAVY = (0x1B, 0x30, 0x4D)
CREAM = (0xF4, 0xF0, 0xE6)
SKY = (0x38, 0xBD, 0xF8)


def close(a: tuple[int, ...], b: tuple[int, ...], tol: int = 12) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


@pytest.fixture
def fresh_icon() -> Iterator[None]:
    """The module's caches emptied before and after, for a test that changes the pictures."""
    app_icon._decoded.cache_clear()
    app_icon.icon_key.cache_clear()
    yield
    app_icon._decoded.cache_clear()
    app_icon.icon_key.cache_clear()


def test_every_size_windows_asks_for_is_bundled() -> None:
    for size in ICO_SIZES:
        with Image.open(io.BytesIO(app_icon.bundled_png(size))) as img:
            assert img.format == "PNG" and img.size == (size, size)
    with pytest.raises(ValueError, match="no bundled icon"):
        app_icon.bundled_png(17)
    # Made from the SVG the pages show (scripts/build_icon.py).
    assert (ROOT / "src/facegen/web/static/kitshot.svg").read_text("utf-8").startswith("<svg")


def test_the_bundled_picture_is_the_users_icon() -> None:
    img = icon_image(None, 256)
    assert img.mode == "RGBA" and img.size == (256, 256)
    assert img.getpixel((0, 0))[3] == 0  # the rounded corner is see-through
    assert close(img.getpixel((128, 105))[:3], CREAM)  # the head
    assert close(img.getpixel((190, 36))[:3], SKY)  # the top-right bracket
    assert close(img.getpixel((128, 128 + 100))[:3], NAVY, tol=24)  # the squircle below


@pytest.mark.parametrize("size", [16, 32, 64])
def test_a_bundled_size_is_used_as_it_is(size: int) -> None:
    with Image.open(io.BytesIO(app_icon.bundled_png(size))) as png:
        bundled = png.convert("RGBA")
    assert ImageChops.difference(icon_image(None, size), bundled).getbbox() is None


def test_another_size_is_reduced_from_a_larger_one() -> None:
    img = icon_image(None, 100)
    assert img.size == (100, 100) and img.getpixel((0, 0))[3] == 0


def test_the_picture_handed_out_is_a_copy() -> None:
    img = icon_image(None, 64)
    img.putpixel((32, 32), (0, 0, 0, 255))
    assert icon_image(None, 64).getpixel((32, 32)) != (0, 0, 0, 255)


@pytest.mark.parametrize("dot", ["running", "paused", "fault"])
def test_the_tray_icon_has_its_state_dot(dot: str) -> None:
    img = icon_image(dot, 64)  # type: ignore[arg-type]
    assert close(img.getpixel((50, 50)), DOT_COLOURS[dot])
    assert close(img.getpixel((50, 38)), (255, 255, 255, 255), tol=40)  # its white ring
    assert not close(icon_image(None, 64).getpixel((50, 50)), DOT_COLOURS[dot])
    # The sky-blue accent at the top right is not covered.
    assert (
        ImageChops.difference(
            img.crop((32, 0, 64, 32)), icon_image(None, 64).crop((32, 0, 64, 32))
        ).getbbox()
        is None
    )


def test_the_dot_reads_at_sixteen_pixels() -> None:
    small = icon_image("fault", 64).resize((16, 16), Image.Resampling.LANCZOS)
    red = DOT_COLOURS["fault"]
    assert close(small.getpixel((12, 12))[:3], red[:3], tol=40)


def test_the_ico_file_has_every_size_and_is_written_once(tmp_path: Path) -> None:
    path = write_ico(tmp_path / "app")
    assert path == tmp_path / "app" / f"kitshot-{app_icon.icon_key()}.ico"
    with Image.open(path) as ico:
        assert set(ico.info["sizes"]) == {(s, s) for s in ICO_SIZES}
        for size in (16, 32, 256):  # each size is the bundled picture, not a reduced 256
            ico.size = (size, size)
            ico.load()
            frame = ico.convert("RGBA")
            assert ImageChops.difference(frame, icon_image(None, size)).getbbox() is None
    stamp = path.stat().st_mtime_ns
    assert write_ico(tmp_path / "app") == path and path.stat().st_mtime_ns == stamp
    assert not list((tmp_path / "app").glob(".*.part"))


def test_the_installer_names_its_own_ico(tmp_path: Path) -> None:
    path = write_ico(tmp_path, "kitshot.ico")
    assert path == tmp_path / "kitshot.ico" and path.is_file()


def test_a_changed_icon_is_a_new_ico_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_icon: None
) -> None:
    first = write_ico(tmp_path)
    original = app_icon.bundled_png

    def recoloured(size: int) -> bytes:
        with Image.open(io.BytesIO(original(size))) as img:
            out = io.BytesIO()
            Image.eval(img.convert("RGBA"), lambda v: 255 - v).save(out, format="PNG")
            return out.getvalue()

    monkeypatch.setattr(app_icon, "bundled_png", recoloured)
    app_icon._decoded.cache_clear()
    app_icon.icon_key.cache_clear()
    second = write_ico(tmp_path)
    assert second != first and second.is_file() and first.is_file()
    assert second.read_bytes() != first.read_bytes()
