"""Image history and atomic master/candidate writes; no GPU dependencies."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from kitshot.core.job_records import Career, Job
from kitshot.core.prompt import Prompt
from kitshot.core.settings_values import IMAGE_MODELS, Settings
from kitshot.persistence.jobs import now
from kitshot.render.picture import ASPECT, crop_to_aspect

if TYPE_CHECKING:
    from PIL.Image import Image


class CandidateFile(Protocol):
    attempt: int
    index: int
    seed: int
    image: Image


def master_dir(data_dir: Path, career_key: str, unique_id: int) -> Path:
    return data_dir / "masters" / career_key / str(unique_id)


def mark_pictures_changed(conn: sqlite3.Connection, career_id: int) -> None:
    """Record a change to existing pictures inside its write transaction."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS picture_revisions ("
        "career_id INTEGER PRIMARY KEY REFERENCES careers(id) ON DELETE CASCADE,"
        " revision INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO picture_revisions VALUES (?, 1) ON CONFLICT(career_id)"
        " DO UPDATE SET revision = revision + 1",
        (career_id,),
    )


def picture_revision(conn: sqlite3.Connection, career_id: int) -> int:
    """A cheap live-page check, including databases created before manual picks."""
    if not conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'picture_revisions'"
    ).fetchone():
        return 0
    row = conn.execute(
        "SELECT revision FROM picture_revisions WHERE career_id = ?", (career_id,)
    ).fetchone()
    return row[0] if row else 0


def reference_problem(
    settings: Settings,
    current: Mapping[str, Any] | sqlite3.Row | None,
    *,
    renderer_model: str | None = None,
    adapter: bool = True,
) -> str | None:
    """Why the current master cannot guide a redraw, or None if it can be used."""
    model = IMAGE_MODELS[settings.style.model]
    if not model.reference or not adapter:
        return f"{model.name} takes no reference picture"
    if current is None or not current["master_path"]:
        return "no current picture"
    if current["gender"] != settings.style.gender:
        return f"the current picture is of a {current['gender']}"
    if current["model"] != (renderer_model or model.repo):
        return f"the current picture was drawn by {current['model']}"
    if not (settings.paths.data_dir / current["master_path"]).is_file():
        return f"picture file missing: {current['master_path']}"
    return None


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

    def current_reference(self, unique_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT id, version, master_path, model, gender"
            " FROM images WHERE career_id = ? AND unique_id = ? AND is_current = 1",
            (self.career.id, unique_id),
        ).fetchone()

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
        gender: str,
        aspect: str = "3:4",
    ) -> int:
        """`gender` is who the picture shows: "woman" or "man" (`style.gender`); `aspect` its
        shape, "3:4" or "1:1" (`picture.aspect` when it was drawn)."""
        cur = self.conn.execute(
            "INSERT INTO images (career_id, unique_id, version, reason, group_name, club_id,"
            " age_band, traits_version, model, prompt, negative_prompt, seed, master_path,"
            " published_path, check_result, is_current, created_at, render, gender, aspect)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
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
                gender,
                aspect,
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def save_candidate(
        self, folder: Path, cand: CandidateFile, aspect: tuple[int, int] = ASPECT
    ) -> str:
        """The candidate as it would be published: cropped to its shape (3:4 or 1:1), as WebP."""
        path = folder / f"a{cand.attempt}c{cand.index}-{cand.seed}.webp"
        self.save_webp(path, crop_to_aspect(cand.image, aspect))
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
    """A master picture as RGB PNG with Kitshot's metadata, written atomically."""
    from PIL.PngImagePlugin import PngInfo

    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    info.add_text("facegen", json.dumps(meta))
    tmp = path.with_name(f".{path.name}.part")
    image.convert("RGB").save(tmp, format="PNG", pnginfo=info)
    os.replace(tmp, path)
