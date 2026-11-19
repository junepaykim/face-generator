"""Image history and atomic master/candidate writes; no GPU dependencies."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from facegen.core.job_records import Career, Job
from facegen.core.prompt import Prompt
from facegen.persistence.jobs import now
from facegen.render.picture import crop_to_aspect

if TYPE_CHECKING:
    from PIL.Image import Image


class CandidateFile(Protocol):
    attempt: int
    index: int
    seed: int
    image: Image


def master_dir(data_dir: Path, career_key: str, unique_id: int) -> Path:
    return data_dir / "masters" / career_key / str(unique_id)


class ImageStore:
    def __init__(self, conn: sqlite3.Connection, data_dir: Path, career: Career) -> None:
        self.conn = conn
        self.data_dir = data_dir
        self.career = career

    def next_version(self, unique_id: int) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM images WHERE career_id = ? AND unique_id = ?",
            (self.career.id, unique_id),
        ).fetchone()
        return int(row[0])

    def rel(self, path: Path) -> str:
        return path.relative_to(self.data_dir).as_posix()

    def insert_image(
        self,
        job: Job,
        version: int,
        sheet_version: int,
        wording: str,
        prompt: Prompt,
        seed: int,
        master: Path,
        render: dict[str, Any],
        check_result: dict[str, Any],
        *,
        current: bool,
        model: str,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO images (career_id, unique_id, version, reason, group_name, club_id,"
            " age_band, traits_version, model, prompt, negative_prompt, seed, master_path,"
            " published_path, check_result, is_current, created_at, render)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
            (
                self.career.id,
                job.unique_id,
                version,
                job.kind,
                job.group,
                job.target_club_id,
                wording,
                sheet_version,
                model,
                prompt.positive,
                prompt.negative,
                seed,
                self.rel(master),
                json.dumps(check_result),
                int(current),
                now(),
                json.dumps(render),
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def save_candidate(self, folder: Path, cand: CandidateFile) -> str:
        """The candidate as it would be published: cropped to 3:4, as WebP."""
        path = folder / f"a{cand.attempt}c{cand.index}-{cand.seed}.webp"
        self.save_webp(path, crop_to_aspect(cand.image))
        return self.rel(path)

    def save_webp(self, path: Path, image: Image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        image.convert("RGB").save(tmp, format="WEBP", quality=90)
        os.replace(tmp, path)

    def save_master(
        self,
        path: Path,
        image: Image,
        prompt: Prompt,
        seed: int,
        model: str,
        render_settings: dict[str, Any],
    ) -> None:
        meta = {
            "model": model,
            "prompt": prompt.positive,
            "negative": prompt.negative,
            "seed": seed,
            **render_settings,
        }
        save_png(path, image, meta)


def save_png(path: Path, image: Image, meta: dict[str, Any]) -> None:
    """A master picture as RGB PNG with facegen's metadata, written atomically."""
    from PIL.PngImagePlugin import PngInfo

    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    info.add_text("facegen", json.dumps(meta))
    tmp = path.with_name(f".{path.name}.part")
    image.convert("RGB").save(tmp, format="PNG", pnginfo=info)
    os.replace(tmp, path)
