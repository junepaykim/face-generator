"""Make Kitshot's bundled icon pictures, src/kitshot/data/icon/kitshot-<size>.png, from the
user's SVG, src/kitshot/web/static/kitshot.svg (Phase 17c).

The desktop app loads these PNGs (desktop/app_icon.py) for the tray, the window, the .ico
and kitshot-app.exe; the web pages use the SVG itself. Each size is drawn from the SVG at
that size by librsvg (the renderer behind rsvg-convert), called here through ctypes with
cairo, so no Python package is needed. On Ubuntu/WSL: the librsvg2-2 and libcairo2 packages.

    uv run --no-sync python scripts/build_icon.py [--svg PATH] [--out FOLDER]

Run it again after the SVG changes, then the tests (tests/integration/desktop).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "src" / "kitshot" / "web" / "static" / "kitshot.svg"
OUT = ROOT / "src" / "kitshot" / "data" / "icon"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
"""Every size Windows asks an .ico for (desktop/app_icon.py ICO_SIZES)."""
CAIRO_FORMAT_ARGB32 = 0


class Rectangle(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in ("x", "y", "width", "height")]


class GError(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("code", ctypes.c_int), ("message", ctypes.c_char_p)]


def _library(name: str, soname: str) -> ctypes.CDLL:
    found = ctypes.util.find_library(name) or soname
    try:
        return ctypes.CDLL(found)
    except OSError as exc:
        raise SystemExit(
            f"{soname} was not found ({exc}); install librsvg2-2 and libcairo2"
        ) from exc


def render(svg: bytes, size: int, out: Path) -> None:
    """The SVG drawn `size` pixels square into the PNG `out` (straight alpha)."""
    rsvg = _library("rsvg-2", "librsvg-2.so.2")
    cairo = _library("cairo", "libcairo.so.2")
    gobject = _library("gobject-2.0", "libgobject-2.0.so.0")
    error = ctypes.POINTER(GError)()
    rsvg.rsvg_handle_new_from_data.restype = ctypes.c_void_p
    rsvg.rsvg_handle_new_from_data.argtypes = (
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(GError)),
    )
    rsvg.rsvg_handle_render_document.restype = ctypes.c_int
    rsvg.rsvg_handle_render_document.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(Rectangle),
        ctypes.POINTER(ctypes.POINTER(GError)),
    )
    cairo.cairo_image_surface_create.restype = ctypes.c_void_p
    cairo.cairo_image_surface_create.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = (ctypes.c_void_p,)
    cairo.cairo_surface_write_to_png.restype = ctypes.c_int
    cairo.cairo_surface_write_to_png.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    cairo.cairo_destroy.argtypes = (ctypes.c_void_p,)
    cairo.cairo_surface_destroy.argtypes = (ctypes.c_void_p,)
    gobject.g_object_unref.argtypes = (ctypes.c_void_p,)

    handle = rsvg.rsvg_handle_new_from_data(svg, len(svg), ctypes.byref(error))
    if not handle:
        raise SystemExit(f"librsvg could not read the SVG: {error.contents.message!r}")
    surface = cairo.cairo_image_surface_create(CAIRO_FORMAT_ARGB32, size, size)
    context = cairo.cairo_create(surface)
    try:
        viewport = Rectangle(0, 0, size, size)
        if not rsvg.rsvg_handle_render_document(
            handle, context, ctypes.byref(viewport), ctypes.byref(error)
        ):
            raise SystemExit(f"librsvg could not draw the SVG: {error.contents.message!r}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if cairo.cairo_surface_write_to_png(surface, str(out).encode()) != 0:
            raise SystemExit(f"{out} could not be written")
    finally:
        cairo.cairo_destroy(context)
        cairo.cairo_surface_destroy(surface)
        gobject.g_object_unref(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--svg", type=Path, default=SVG, help=f"default: {SVG.relative_to(ROOT)}")
    parser.add_argument("--out", type=Path, default=OUT, help=f"default: {OUT.relative_to(ROOT)}")
    args = parser.parse_args(argv)
    svg = args.svg.read_bytes()
    for size in SIZES:
        render(svg, size, args.out / f"kitshot-{size}.png")
    print(f"{len(SIZES)} sizes written in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
