"""Default runtime dependencies; save-reader and GPU imports happen only on use."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from facegen.datasource.base import DataSource
    from facegen.renderer import Renderer, Tagger


def default_source() -> DataSource:
    from facegen.datasource.fmsave_source import FmsaveDataSource

    return FmsaveDataSource()


def default_renderer() -> Renderer:
    from facegen.renderer import AnimagineRenderer

    return AnimagineRenderer()


def default_tagger() -> Tagger:
    from facegen.tagger import Eva02Tagger

    return Eva02Tagger()


def default_fm_check() -> bool | None:
    from facegen.worker import fm_running

    return fm_running()


@dataclass(frozen=True, slots=True)
class Gpu:
    """What `run` and `worker` use for the GPU and the FM27 check; the tests pass fakes."""

    renderer: Callable[[], Renderer] = default_renderer
    tagger: Callable[[], Tagger] = default_tagger
    fm_check: Callable[[], bool | None] = default_fm_check
    program: tuple[str, ...] | None = None
    """How `serve` starts facegen for its worker process, before `[--config X] worker`;
    None: this Python with `-m facegen.cli`. The tests give a script that passes fakes."""
    models_ready: Callable[[], bool] | None = None
    """Whether the model files are on this PC; None: asked of the Hugging Face cache when the
    real renderer is used (models.ready_now), and not at all with the tests' fakes."""

    def models_check(self) -> Callable[[], bool] | None:
        if self.models_ready is not None:
            return self.models_ready
        from facegen import models

        return models.ready_now if self.renderer is default_renderer else None
