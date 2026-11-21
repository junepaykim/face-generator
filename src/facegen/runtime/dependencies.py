"""Default runtime dependencies; save-reader and GPU imports happen only on use."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from facegen.datasource.base import DataSource
    from facegen.render.renderer import Renderer, Tagger


def default_source() -> DataSource:
    from facegen.datasource.fmsave_source import FmsaveDataSource

    return FmsaveDataSource()


def default_renderer(model: str = "animagine") -> Renderer:
    """The renderer for a `style.model` key, loaded now (seconds, and the GPU)."""
    if model == "animagine":
        from facegen.render.renderer import AnimagineRenderer

        return AnimagineRenderer()
    if model == "realvis":
        from facegen.render.renderer import REALVIS, SdxlRenderer

        return SdxlRenderer(REALVIS)
    if model == "zimage_turbo":
        from facegen.render.zimage import ZImageRenderer

        return ZImageRenderer()
    raise ValueError(f"no image model called {model!r}")


def default_tagger() -> Tagger:
    from facegen.render.tagger import Eva02Tagger

    return Eva02Tagger()


def default_fm_check() -> bool | None:
    from facegen.pipeline.worker import fm_running

    return fm_running()


@dataclass(frozen=True, slots=True)
class Gpu:
    """What `run` and `worker` use for the GPU and the FM27 check; the tests pass fakes."""

    renderer: Callable[[str], Renderer] = default_renderer
    """Builds the renderer for the selected model (`style.model`)."""
    tagger: Callable[[], Tagger] = default_tagger
    fm_check: Callable[[], bool | None] = default_fm_check
    program: tuple[str, ...] | None = None
    """How `serve` starts facegen for its worker process, before `[--config X] worker`;
    None: this Python with `-m facegen.cli`. The tests give a script that passes fakes."""
    models_ready: Callable[[str], bool] | None = None
    """Whether the files of a model (a `style.model` key) are on this PC; None: asked of the
    Hugging Face cache when the real renderer is used (models.ready_now), and not at all with
    the tests' fakes."""

    def models_check(self) -> Callable[[str], bool] | None:
        if self.models_ready is not None:
            return self.models_ready
        from facegen.setup import models

        return models.ready_now if self.renderer is default_renderer else None
