# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Phase 16g on the GPU: 1:1 pictures. For each model, a Detailed (group A) job drawn square as
the worker runs it passes the check (the head-position check included), peaks under 8.0 GiB
reserved, keeps its shape in the master, the record and the published picture, and a "new
picture" crops its reference face from the square. Opt-in:

    KITSHOT_TEST_GPU=1 uv run --no-sync pytest tests/opt_in/test_gpu_square.py -s

Needs the GPU, every model's files and the EVA02 tagger in the Hugging Face cache (with
h94/IP-Adapter for the SDXL models' re-draws), and FM27 closed. Nothing is downloaded. One
made-up player of my club per model, drawn as a woman, in a temporary data folder and FM27 user
folder. One model is loaded at a time.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("KITSHOT_TEST_GPU") != "1", reason="set KITSHOT_TEST_GPU=1 to use the GPU"
    ),
]

MODELS = ("animagine", "realvis", "zimage_turbo")
UID = 1001  # a player of my club: group A
MASTER_SIZE = {True: (1536, 1536), False: (1024, 1024)}
"""The square master: group A's detail pass at 1536x1536, or the 1024x1024 candidate if the
detail pass failed the check."""


@pytest.fixture(scope="module")
def tagger() -> Iterator[Any]:
    from kitshot.render.tagger import Eva02Tagger

    tagger = Eva02Tagger()
    yield tagger
    tagger.close()


@pytest.fixture(scope="module", params=MODELS)
def loaded(request: pytest.FixtureRequest) -> Iterator[tuple[str, Any]]:
    from kitshot.platform.memory import release_free_memory
    from kitshot.runtime.dependencies import default_renderer

    renderer = default_renderer(request.param)
    yield request.param, renderer
    print(f"\n{request.param} memory:", renderer.memory())
    renderer.close()
    release_free_memory()


def _setup(conn: sqlite3.Connection, settings: Any, tmp_path: Path, model: str) -> Any:
    from factories import player, snapshot

    from kitshot.core.settings_values import PictureSettings, StyleSettings
    from kitshot.persistence.settings import Paths
    from kitshot.pipeline.sync import run_sync

    s = replace(
        settings,
        paths=Paths(
            fm_user_dir=tmp_path / "fm",
            data_dir=tmp_path / "data",
            fm_install_dir=tmp_path / "fm-install",
        ),
        style=StyleSettings(model=model, gender="woman"),
        picture=PictureSettings(aspect="1:1"),
    )
    run_sync(conn, snapshot([player(UID, name="Test Player", age=27)]), s, save_file="career.fm")
    return s


def _worker(conn: sqlite3.Connection, s: Any, renderer: Any, tagger: Any) -> Any:
    from kitshot.datasource.kits import load_kits
    from kitshot.pipeline.worker import Worker, load_career

    career = load_career(conn)
    assert career is not None
    return Worker(
        conn,
        s,
        load_kits(),
        career,
        renderer_factory=lambda key: renderer,
        tagger_factory=lambda: tagger,
        fm_check=lambda: False,
        log=print,
    )


def _run_one(w: Any) -> sqlite3.Row:
    from kitshot.pipeline.worker import JobFilter

    summary = w.run(JobFilter(groups=frozenset({"a"}), my_club_only=True))
    assert not summary.failed, [o.message for o in summary.failed]
    assert [o.job.unique_id for o in summary.done] == [UID]
    image = w.conn.execute(
        "SELECT * FROM images WHERE unique_id = ? AND is_current = 1", (UID,)
    ).fetchone()
    assert image is not None
    return image


def _check_square(image: sqlite3.Row, data: Path, fm: Path, career_key: str) -> dict[str, Any]:
    from PIL import Image

    assert image["aspect"] == "1:1"
    render = json.loads(image["render"])
    assert (render["aspect"], render["width"], render["height"]) == ("1:1", 1024, 1024)
    check = json.loads(image["check_result"])
    assert check["passed"] is True and check["result"]["failures"] == []
    head = check["result"].get("head_top")
    print(f"  head top {head}, detail pass {'used' if render['detail']['used'] else 'rejected'}")
    detailed = render["detail"]["used"]
    assert render["detail"]["target"] == [1536, 1536]
    with Image.open(data / image["master_path"]) as master:
        assert master.size == MASTER_SIZE[detailed]
    for cand in check["candidates"]:
        with Image.open(data / cand["path"]) as c:
            assert c.size == (1024, 1024)
    with Image.open(fm / "graphics" / "kitshot" / career_key / f"{UID}.png") as published:
        assert published.size == (1024, 1024)  # group A's long edge, square
    return render


def test_a_square_group_a_job_and_its_redraw(
    loaded: tuple[str, Any],
    tagger: Any,
    conn: sqlite3.Connection,
    settings: Any,
    tmp_path: Path,
) -> None:
    import torch

    from kitshot.core.settings_values import IMAGE_MODELS
    from kitshot.pipeline import manual

    model, renderer = loaded
    s = _setup(conn, settings, tmp_path, model)
    w = _worker(conn, s, renderer, tagger)
    torch.cuda.reset_peak_memory_stats()

    first = _run_one(w)
    render = _check_square(first, s.paths.data_dir, s.paths.fm_user_dir, w.career.key)
    assert render["reference"] is None

    # "New picture": the first, a square, as the reference where the model takes one.
    manual.new_picture(conn, w.career.id, UID, s)
    again = _run_one(w)
    render = _check_square(again, s.paths.data_dir, s.paths.fm_user_dir, w.career.key)
    info = render["reference"]
    if IMAGE_MODELS[model].reference:
        assert info["used"] is True and info["image_id"] == first["id"]
        from PIL import Image

        with Image.open(s.paths.data_dir / info["crop"]) as crop:
            side = crop.width
            assert crop.width == crop.height and 300 <= side <= 600, crop.size
    else:
        assert info["used"] is False

    reserved = torch.cuda.max_memory_reserved() / 2**30
    allocated = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"{model}, 1:1: two group A jobs, peak reserved {reserved:.2f} GiB, allocated {allocated:.2f}"
    )
    assert reserved < 8.0
