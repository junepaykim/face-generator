# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Phase 15's exit test on the GPU: for every model and gender, a group A job as the worker runs
it passes the check, peaks under 8.0 GiB reserved, records the model, prompt, seed and settings,
and publishes. Opt-in: FACEGEN_TEST_GPU=1 uv run --no-sync pytest tests/opt_in/test_gpu_jobs.py

Needs the GPU, every model's files and the EVA02 tagger in the Hugging Face cache (with
h94/IP-Adapter for the SDXL models' re-draws), and FM27 closed. The real worker, prompts, checks
and publishing draw one made-up player of my club per model and gender, in a temporary data
folder and FM27 user folder: a first picture, then "new picture", which takes the first as a
reference on a model that takes one. One model is loaded at a time. Run with -s to see each
job's log and memory.
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
        os.environ.get("FACEGEN_TEST_GPU") != "1", reason="set FACEGEN_TEST_GPU=1 to use the GPU"
    ),
]

MODELS = ("animagine", "realvis", "zimage_turbo")
GENDERS = ("woman", "man")
UID = 1001  # a player of my club: group A
MASTER_SIZE = {True: (1296, 1728), False: (864, 1152)}
"""The 3:4 master: group A's detail pass at 1344x1728, or the 896x1152 candidate if the
detail pass failed the check."""


@pytest.fixture(scope="module")
def tagger() -> Iterator[Any]:
    from facegen.render.tagger import Eva02Tagger

    tagger = Eva02Tagger()
    yield tagger
    tagger.close()


@pytest.fixture(scope="module", params=MODELS)
def loaded(request: pytest.FixtureRequest) -> Iterator[tuple[str, Any]]:
    """One model at a time, as the worker builds it from `style.model`; closed before the
    next is loaded, so each job's peak is its own model's."""
    from facegen.platform.memory import release_free_memory
    from facegen.runtime.dependencies import default_renderer

    renderer = default_renderer(request.param)
    yield request.param, renderer
    print(f"\n{request.param} memory:", renderer.memory())
    renderer.close()
    release_free_memory()


def _env(conn: sqlite3.Connection, settings: Any, tmp_path: Path, model: str, gender: str) -> Any:
    from factories import player, snapshot

    from facegen.core.settings_values import StyleSettings
    from facegen.persistence.settings import Paths
    from facegen.pipeline.sync import run_sync

    s = replace(
        settings,
        paths=Paths(
            fm_user_dir=tmp_path / "fm",
            data_dir=tmp_path / "data",
            fm_install_dir=tmp_path / "fm-install",
        ),
        style=StyleSettings(model=model, gender=gender),
    )
    run_sync(conn, snapshot([player(UID, name="Test Player", age=27)]), s, save_file="career.fm")
    return s


def _worker(conn: sqlite3.Connection, s: Any, model: str, renderer: Any, tagger: Any) -> Any:
    from facegen.datasource.kits import load_kits
    from facegen.pipeline.worker import Worker, load_career

    def built(key: str) -> Any:
        assert key == model  # the worker asks for the model the settings name
        return renderer

    career = load_career(conn)
    assert career is not None
    return Worker(
        conn,
        s,
        load_kits(),
        career,
        renderer_factory=built,
        tagger_factory=lambda: tagger,
        fm_check=lambda: False,
        log=print,
    )


def _encoders(renderer: Any) -> list[Any]:
    """The models that wait in system memory between trips."""
    if hasattr(renderer, "encoder"):  # Z-Image-Turbo: its INT8 text encoder
        return [renderer.encoder.text_encoder]
    pipe = renderer.pipe
    return [
        m
        for m in (pipe.text_encoder, pipe.text_encoder_2, getattr(pipe, "image_encoder", None))
        if m is not None
    ]


def _denoiser(renderer: Any) -> Any:
    pipe = renderer.pipe
    return pipe.transformer if hasattr(renderer, "encoder") else pipe.unet


def _run_one(w: Any, gender: str) -> tuple[Any, sqlite3.Row]:
    """Runs the queued job of UID; returns its preview and the picture it made current."""
    from facegen.pipeline.worker import JobFilter

    mine = JobFilter(groups=frozenset({"a"}), my_club_only=True)
    (preview,) = [p for p in w.preview(mine) if p.job.unique_id == UID]
    summary = w.run(mine)
    assert not summary.failed, [o.message for o in summary.failed]
    assert [o.job.unique_id for o in summary.done] == [UID]
    assert summary.publish is not None
    image = w.conn.execute(
        "SELECT * FROM images WHERE unique_id = ? AND is_current = 1", (UID,)
    ).fetchone()
    assert image is not None and image["gender"] == gender
    return preview, image


def _check_record(
    model: str, image: sqlite3.Row, preview: Any, renderer: Any, data: Path
) -> dict[str, Any]:
    """The picture records the model, prompt, seed and settings, in the database and in
    the master's own metadata; it passed the check."""
    from PIL import Image

    from facegen.core.settings_values import IMAGE_MODELS

    assert image["model"] == renderer.model
    assert (image["prompt"], image["negative_prompt"]) == (
        preview.prompt.positive,
        preview.prompt.negative,
    )
    render = json.loads(image["render"])
    settings = renderer.settings()
    assert {k: render[k] for k in settings} == settings
    detailed = render["detail"]["used"]
    print(f"  detail pass {'used' if detailed else 'rejected'}: {render['detail']}")
    check = json.loads(image["check_result"])
    assert check["passed"] is True and check["result"]["failures"] == []
    assert check["result"]["rating"] == "general"
    assert image["seed"] == check["chosen"]["seed"]
    with Image.open(data / image["master_path"]) as master:
        meta = json.loads(master.text["facegen"])
        assert master.size == MASTER_SIZE[detailed]
    assert (meta["model"], meta["prompt"], meta["negative"], meta["seed"]) == (
        image["model"],
        image["prompt"],
        image["negative_prompt"],
        image["seed"],
    )
    if not IMAGE_MODELS[model].negative:  # Z-Image-Turbo draws without one: none stored
        assert image["negative_prompt"] == ""
    else:
        assert image["negative_prompt"]
    return render


def _check_prompt(model: str, gender: str, positive: str) -> None:
    """The prompt branch the settings ask for (core/style.py)."""
    from facegen.core.settings_values import IMAGE_MODELS

    if gender == "woman":
        assert "1girl" in positive  # the anime women's prompt, on every model for now
    elif IMAGE_MODELS[model].photo:
        assert positive.startswith("Professional studio headshot photograph of ")
        assert "He is the only person in the picture." in positive
    else:
        assert positive.startswith("1boy, solo, male focus")


def _check_published(fm: Path, career_key: str) -> None:
    from PIL import Image

    pack = fm / "graphics" / "facegen" / career_key
    with Image.open(pack / f"{UID}.png") as published:
        assert published.size == (768, 1024)  # group A's published long edge
    xml = (pack / "config.xml").read_text()
    assert f'<record from="{UID}" to="graphics/pictures/person/{UID}/portrait"/>' in xml


@pytest.mark.parametrize("gender", GENDERS)
def test_a_group_a_job_and_its_redraw(
    loaded: tuple[str, Any],
    tagger: Any,
    gender: str,
    conn: sqlite3.Connection,
    settings: Any,
    tmp_path: Path,
) -> None:
    import torch

    from facegen.core.settings_values import IMAGE_MODELS
    from facegen.pipeline import manual

    model, renderer = loaded
    s = _env(conn, settings, tmp_path, model, gender)
    w = _worker(conn, s, model, renderer, tagger)
    torch.cuda.reset_peak_memory_stats()

    # The first picture: never a reference.
    preview, first = _run_one(w, gender)
    _check_prompt(model, gender, first["prompt"])
    render = _check_record(model, first, preview, renderer, s.paths.data_dir)
    assert render["reference"] is None
    _check_published(s.paths.fm_user_dir, w.career.key)

    # "New picture": the same sheet, with the first as a reference where the model takes one.
    manual.new_picture(conn, w.career.id, UID, s)
    preview, again = _run_one(w, gender)
    assert again["version"] == 2 and again["traits_version"] == first["traits_version"]
    render = _check_record(model, again, preview, renderer, s.paths.data_dir)
    info = render["reference"]
    if IMAGE_MODELS[model].reference:
        assert info["used"] is True and info["image_id"] == first["id"]
        assert info["scale"] == s.reference.scale
        assert (s.paths.data_dir / info["crop"]).is_file()
    else:
        assert info == {
            "used": False,
            "why": "Z-Image-Turbo takes no reference picture",
            "scale": s.reference.scale,
        }
    _check_published(s.paths.fm_user_dir, w.career.key)

    reserved = torch.cuda.max_memory_reserved() / 2**30
    allocated = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"{model}, {gender}: two group A jobs, peak reserved {reserved:.2f} GiB,"
        f" allocated {allocated:.2f}"
    )
    assert reserved < 8.0
    for part in (*_encoders(renderer), tagger.model):
        assert next(part.parameters()).device.type == "cpu"
    assert _denoiser(renderer).device.type == "cuda"
