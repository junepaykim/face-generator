"""Assets routes and their app-local helpers."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from facegen.web.dependencies import AppDependencies

THUMB_SIZES = (96, 192, 384)
_PICTURE_SUFFIXES = (".png", ".webp")
_KEEP = "private, max-age=86400"
"""A file addressed by its own path never changes (candidates, references)."""
_REVALIDATE = "private, no-cache"
"""A picture addressed by its image ID can change file (a candidate picked by hand), so the
browser asks again each time; an unchanged one costs a 304."""


def _within(path: Path, folder: Path) -> bool:
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


def _etag(path: Path) -> str:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    return '"' + hashlib.sha1(key.encode()).hexdigest()[:20] + '"'


def _matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match:
        return False
    tags = {t.strip().removeprefix("W/") for t in if_none_match.split(",")}
    return etag in tags or "*" in tags


def make_thumb(source: Path, size: int, cache_dir: Path) -> Path:
    """A cached WebP thumbnail of a picture, `size` pixels on its long edge (3:4 pictures are
    `size` high). Masters never change once written; a picked candidate is another file."""
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
    tmp = dest.with_name(f".{dest.name}.{uuid4().hex}.part")
    try:
        img.save(tmp, format="WEBP", quality=85)
        try:
            tmp.replace(dest)
        except PermissionError:
            # On Windows another request can already be serving the completed thumbnail.
            if not dest.is_file():
                raise
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def create_router(deps: AppDependencies) -> APIRouter:
    router = APIRouter()
    data_dir = deps.data_dir
    masters_dir = deps.masters_dir
    thumbs_dir = deps.thumbs_dir
    Conn = deps.connection

    def send_picture(path: Path, size: int | None, if_none_match: str | None = None) -> Response:
        """The picture or its thumbnail. With `if_none_match` (a page asking by image ID) the
        browser revalidates by ETag instead of keeping its copy for a day."""
        if not _within(path, masters_dir) or path.suffix not in _PICTURE_SUFFIXES:
            raise HTTPException(404)
        if not path.is_file():
            raise HTTPException(404, "picture file missing")
        if size is not None:
            if size not in THUMB_SIZES:
                raise HTTPException(400, f"size must be one of {THUMB_SIZES}")
            path = make_thumb(path, size, thumbs_dir)
        media_type = "image/webp" if path.suffix == ".webp" else "image/png"
        if if_none_match is None:
            return FileResponse(path, media_type=media_type, headers={"Cache-Control": _KEEP})
        headers = {"Cache-Control": _REVALIDATE, "ETag": _etag(path)}
        if _matches(if_none_match, headers["ETag"]):
            return Response(status_code=304, headers=headers)
        return FileResponse(path, media_type=media_type, headers=headers)

    @router.get("/pictures/{image_id}")
    def picture(
        image_id: int,
        request: Request,
        size: int | None = None,
        conn: sqlite3.Connection = Conn,
    ) -> Response:
        row = conn.execute("SELECT master_path FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None or not row["master_path"]:
            raise HTTPException(404)
        asked = request.headers.get("if-none-match", "")
        return send_picture(data_dir / row["master_path"], size, asked)

    @router.get("/files")
    def files(path: str, size: int | None = None) -> Response:
        """A candidate picture, by its path relative to data/ as stored in check results."""
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts or not path.startswith("masters/"):
            raise HTTPException(404)
        return send_picture(data_dir / rel, size)

    return router
