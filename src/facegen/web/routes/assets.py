"""Assets routes and their app-local helpers."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from facegen.web.dependencies import AppDependencies

THUMB_SIZES = (96, 192, 384)
_PICTURE_SUFFIXES = (".png", ".webp")


def _within(path: Path, folder: Path) -> bool:
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


def make_thumb(source: Path, size: int, cache_dir: Path) -> Path:
    """A cached WebP thumbnail of a picture, `size` pixels on its long edge (3:4 pictures are
    `size` high). Masters never change once written."""
    st = source.stat()
    key = hashlib.sha1(f"{source.resolve()}|{st.st_mtime_ns}|{size}".encode()).hexdigest()[:20]
    dest = cache_dir / f"{key}-{size}.webp"
    if dest.is_file():
        return dest
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        img = im.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    img.save(tmp, format="WEBP", quality=85)
    os.replace(tmp, dest)
    return dest


def create_router(deps: AppDependencies) -> APIRouter:
    router = APIRouter()
    data_dir = deps.data_dir
    masters_dir = deps.masters_dir
    thumbs_dir = deps.thumbs_dir
    Conn = deps.connection

    def send_picture(path: Path, size: int | None) -> Response:
        if not _within(path, masters_dir) or path.suffix not in _PICTURE_SUFFIXES:
            raise HTTPException(404)
        if not path.is_file():
            raise HTTPException(404, "picture file missing")
        if size is not None:
            if size not in THUMB_SIZES:
                raise HTTPException(400, f"size must be one of {THUMB_SIZES}")
            path = make_thumb(path, size, thumbs_dir)
        media_type = "image/webp" if path.suffix == ".webp" else "image/png"
        return FileResponse(
            path, media_type=media_type, headers={"Cache-Control": "private, max-age=86400"}
        )

    @router.get("/pictures/{image_id}")
    def picture(
        image_id: int, size: int | None = None, conn: sqlite3.Connection = Conn
    ) -> Response:
        row = conn.execute("SELECT master_path FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None or not row["master_path"]:
            raise HTTPException(404)
        return send_picture(data_dir / row["master_path"], size)

    @router.get("/files")
    def files(path: str, size: int | None = None) -> Response:
        """A candidate picture, by its path relative to data/ as stored in check results."""
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts or not path.startswith("masters/"):
            raise HTTPException(404)
        return send_picture(data_dir / rel, size)

    return router
