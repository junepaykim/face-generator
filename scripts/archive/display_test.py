"""Phase 0.2 spike: a test picture pack for checking how FM27 shows player pictures.

Writes labelled test cards and a classic facepack config.xml into
<FM27 user dir>/graphics/facegen/test/ (AGENTS.md hard rule 2: nowhere else). Each card
has coloured edge bands (red top, blue bottom, green left, orange right) so cropping is
visible, lines at one and two thirds of the height, and a figure with its head in the top
third, as the spec asks for.

    uv run python scripts/archive/display_test.py install 29141735:square 29193014:portrait
    uv run python scripts/archive/display_test.py list
    uv run python scripts/archive/display_test.py remove 29141735
    uv run python scripts/archive/display_test.py uninstall
    uv run python scripts/archive/display_test.py install 1:square --dir /tmp/preview

Shapes: square (1024x1024), portrait (768x1024, 3:4), tall (640x960, 2:3),
small (250x250, the classic facepack size).
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import zlib
from pathlib import Path

from facegen.settings import load_settings

SHAPES = {
    "square": (1024, 1024, "SQUARE"),
    "portrait": (768, 1024, "PORTRAIT 3:4"),
    "tall": (640, 960, "TALL 2:3"),
    "small": (250, 250, "SQUARE"),
}

BACKGROUND = (222, 226, 232)
TOP, BOTTOM, LEFT, RIGHT = (220, 50, 50), (50, 90, 220), (40, 170, 80), (240, 160, 30)
GUIDE = (120, 124, 132)
BODY = (28, 44, 92)
SKIN = (240, 208, 184)
HAIR = (70, 50, 40)
WHITE = (255, 255, 255)

_FONT = {
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    ":": (".....", "..#..", "..#..", ".....", "..#..", "..#..", "....."),
    " ": (".....",) * 7,
}


class Canvas:
    def __init__(self, width: int, height: int, color: tuple[int, int, int]) -> None:
        self.w, self.h = width, height
        self.rows = [bytearray(bytes(color) * width) for _ in range(height)]

    def span(self, y: int, x0: int, x1: int, color: tuple[int, int, int]) -> None:
        if 0 <= y < self.h:
            x0, x1 = max(0, x0), min(self.w, x1)
            if x1 > x0:
                self.rows[y][x0 * 3 : x1 * 3] = bytes(color) * (x1 - x0)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(self.h, y1)):
            self.span(y, x0, x1, color)

    def circle(self, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - r, cy + r + 1):
            dx = int((r * r - (y - cy) ** 2) ** 0.5)
            self.span(y, cx - dx, cx + dx + 1, color)

    def text(self, label: str, cx: int, y: int, scale: int, color: tuple[int, int, int]) -> None:
        x = cx - (len(label) * 6 - 1) * scale // 2
        for ch in label:
            for gy, line in enumerate(_FONT[ch]):
                for gx, bit in enumerate(line):
                    if bit == "#":
                        self.rect(
                            x + gx * scale,
                            y + gy * scale,
                            x + (gx + 1) * scale,
                            y + (gy + 1) * scale,
                            color,
                        )
            x += 6 * scale

    def png(self) -> bytes:
        def chunk(kind: bytes, data: bytes) -> bytes:
            body = kind + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        raw = b"".join(b"\x00" + bytes(row) for row in self.rows)
        header = struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )


def test_card(shape: str) -> bytes:
    w, h, name = SHAPES[shape]
    c = Canvas(w, h, BACKGROUND)
    # Figure: shoulders and torso from just below the top third, head inside it.
    head_r = int(min(w, h) * 0.13)
    head_cy = int(h * 0.20)
    top = head_cy + head_r - head_r // 3
    for y in range(top, h):
        t = min(1.0, (y - top) / (h * 0.12))  # shoulders round out, then the torso widens
        half = int(w * (0.12 + 0.24 * t**0.5 + 0.06 * (y - top) / h))
        c.span(y, w // 2 - half, w // 2 + half, BODY)
    c.rect(w // 2 - head_r // 3, head_cy, w // 2 + head_r // 3, top + head_r // 3, SKIN)  # neck
    c.circle(w // 2, head_cy, head_r, SKIN)
    c.rect(w // 2 - head_r, head_cy - head_r, w // 2 + head_r, head_cy - head_r // 3, HAIR)
    eye = max(2, head_r // 9)
    for dx in (-head_r // 3, head_r // 3):
        c.circle(w // 2 + dx, head_cy + head_r // 8, eye, BODY)
    # Thirds guides and edge bands.
    line = max(1, h // 300)
    for y in (h // 3, 2 * h // 3):
        c.rect(0, y - line, w, y + line, GUIDE)
    band = max(4, round(min(w, h) * 0.03))
    c.rect(0, 0, w, band, TOP)
    c.rect(0, h - band, w, h, BOTTOM)
    c.rect(0, 0, band, h, LEFT)
    c.rect(w - band, 0, w, h, RIGHT)
    # Label: shape and size.
    lines = (name, f"{w}X{h}")
    scale = max(1, min(int(w * 0.7) // (max(map(len, lines)) * 6), int(h * 0.07) // 7))
    y = int(h * 0.74)
    for text in lines:
        c.text(text, w // 2, y, scale, WHITE)
        y += 10 * scale
    return c.png()


def config_xml(ids: list[int]) -> str:
    """The classic facepack mapping, to be confirmed for FM27 by this test."""
    records = "\n".join(
        f'\t\t<record from="{i}" to="graphics/pictures/person/{i}/portrait"/>' for i in ids
    )
    return (
        "<record>\n"
        '\t<boolean id="preload" value="false"/>\n'
        '\t<boolean id="amap" value="false"/>\n'
        '\t<list id="maps">\n'
        f"{records}\n"
        "\t</list>\n"
        "</record>\n"
    )


def target_dir(override: str | None) -> Path:
    fm = load_settings().paths.fm_user_dir.resolve()
    allowed = fm / "graphics" / "facegen"
    folder = Path(override).resolve() if override else allowed / "test"
    if folder.is_relative_to(fm) and not folder.is_relative_to(allowed):
        sys.exit(f"refusing to write outside {allowed}")
    return folder


def write_config(folder: Path) -> list[int]:
    ids = sorted(int(p.stem) for p in folder.glob("*.png") if p.stem.isdigit())
    partial = folder / "config.xml.part"
    partial.write_text(config_xml(ids), encoding="utf-8")
    partial.replace(folder / "config.xml")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", help="write here instead of graphics/facegen/test")
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("install", help="add or replace pictures: ID:SHAPE ...")
    install.add_argument("entries", nargs="+", metavar="ID:SHAPE")
    remove = sub.add_parser("remove", help="remove pictures and their mappings")
    remove.add_argument("ids", nargs="+", type=int)
    sub.add_parser("list", help="show the mapped IDs")
    sub.add_parser("uninstall", help="delete the whole test pack")
    args = parser.parse_args()
    folder = target_dir(args.dir)

    if args.command == "install":
        folder.mkdir(parents=True, exist_ok=True)
        for entry in args.entries:
            uid, _, shape = entry.partition(":")
            shape = shape or "square"
            if not uid.isdigit() or shape not in SHAPES:
                sys.exit(f"bad entry {entry!r}: use ID:SHAPE with SHAPE in {', '.join(SHAPES)}")
            (folder / f"{uid}.png").write_bytes(test_card(shape))
            print(f"wrote {uid}.png ({shape})")
        print("mapped:", write_config(folder))
    elif args.command == "remove":
        for uid in args.ids:
            (folder / f"{uid}.png").unlink(missing_ok=True)
        print("mapped:", write_config(folder))
    elif args.command == "list":
        print(folder)
        ids = sorted(int(f.stem) for f in folder.glob("*.png") if f.stem.isdigit())
        print("mapped:", ids or "nothing")
    elif args.command == "uninstall":
        if folder.is_dir():
            shutil.rmtree(folder)
        print("removed", folder)


if __name__ == "__main__":
    main()
