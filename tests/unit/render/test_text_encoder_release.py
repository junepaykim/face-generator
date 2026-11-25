"""Text encoders are let go of between trips (user, 2026-11-24): while pictures are drawn only
the prompt embeddings stay in system memory. The renderers' trip logic with stand-ins for
torch and the models; no GPU, no weights. That the encoders read for a trip are the very ones
kept before is checked on the GPU (tests/opt_in) and in docs/research.md."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from kitshot.core.prompt import Prompt
from kitshot.render import renderer as sdxl
from kitshot.render.prompt_cache import EmbeddingCache
from kitshot.render.renderer import TEXT_CACHE_SIZE, SdxlRenderer
from kitshot.render.zimage import ZImageRenderer


class FakeTorch:
    """`no_grad` and `cuda.empty_cache`, all the trips ask of torch."""

    def __init__(self) -> None:
        self.emptied = 0
        self.cuda = SimpleNamespace(empty_cache=self._empty)

    def _empty(self) -> None:
        self.emptied += 1

    @contextmanager
    def no_grad(self) -> Iterator[None]:
        yield


class Model:
    """A model that records where it is moved, in a shared list of events."""

    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def to(self, device: str) -> Model:
        self.events.append(f"{self.name} to {device}")
        return self


def _sdxl(events: list[str]) -> SdxlRenderer:
    r = object.__new__(SdxlRenderer)
    r._torch = FakeTorch()
    r.pipe = SimpleNamespace(text_encoder=None, text_encoder_2=None)
    r.release_text_encoders = True
    r.offload_text_encoders = True
    r._texts = EmbeddingCache(TEXT_CACHE_SIZE)

    def load() -> tuple[Model, Model]:
        events.append("load")
        return Model("clip-l", events), Model("clip-g", events)

    r._load_text_encoders = load  # type: ignore[method-assign]
    return r


def test_sdxl_encoders_are_loaded_for_a_trip_and_let_go_of_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    r = _sdxl(events)
    seen: list[weakref.ref[Model]] = []

    def encode(pipe: Any, text: str) -> str:
        assert pipe.text_encoder is not None and pipe.text_encoder_2 is not None
        seen.extend((weakref.ref(pipe.text_encoder), weakref.ref(pipe.text_encoder_2)))
        return f"emb({text})"

    monkeypatch.setattr(sdxl, "_encode_text", encode)
    prompts = [Prompt("p1", "neg"), Prompt("p2", "neg")]
    assert r.prepare(prompts) == 4  # "", p1, neg, p2 in one trip
    assert events == ["load"]
    assert r.pipe.text_encoder is None and r.pipe.text_encoder_2 is None
    gc.collect()
    assert seen and all(ref() is None for ref in seen)  # nothing keeps them
    assert r._torch.emptied == 1

    assert r.prepare(prompts) == 0  # cached: no trip, nothing read from disk
    assert events == ["load"] and r.encoded(prompts[0])
    assert r.prepare([Prompt("p3", "neg")]) == 1
    assert events == ["load", "load"] and r.pipe.text_encoder is None


def test_sdxl_encoders_are_let_go_of_when_the_trip_fails() -> None:
    r = _sdxl([])
    with pytest.raises(RuntimeError, match="out of memory"), r._encoders_on_gpu():
        assert r.pipe.text_encoder is not None
        raise RuntimeError("CUDA out of memory")
    assert r.pipe.text_encoder is None and r.pipe.text_encoder_2 is None


def test_sdxl_kept_encoders_stay_when_release_is_off() -> None:
    """The previous behaviour, for comparisons: offloading off keeps them on the card."""
    events: list[str] = []
    r = _sdxl(events)
    r.release_text_encoders = False
    r.offload_text_encoders = False
    kept = Model("clip-l", events), Model("clip-g", events)
    r.pipe.text_encoder, r.pipe.text_encoder_2 = kept
    with r._encoders_on_gpu():
        pass
    assert (r.pipe.text_encoder, r.pipe.text_encoder_2) == kept and events == []


def _zimage(events: list[str]) -> ZImageRenderer:
    r = object.__new__(ZImageRenderer)
    r._torch = FakeTorch()
    r.encoder = SimpleNamespace(text_encoder=None)
    r.release_text_encoder = True
    r.offload_text_encoder = True
    r.transformer_off_for_encoding = True
    r._texts = EmbeddingCache(TEXT_CACHE_SIZE)

    class Transformer(Model):
        def to(self, device: str) -> Model:
            # Back on the card only once the encoder is gone: never both there at once.
            if device == "cuda":
                assert r.encoder.text_encoder is None, "the encoder is still held"
            return super().to(device)

    r.pipe = SimpleNamespace(transformer=Transformer("transformer", events))

    def load() -> Model:
        events.append("load")
        return Model("qwen3", events)

    r._load_text_encoder = load  # type: ignore[method-assign]
    return r


def test_zimage_encoder_is_read_for_a_trip_and_gone_before_the_transformer_returns() -> None:
    events: list[str] = []
    r = _zimage(events)
    seen: list[weakref.ref[Model]] = []

    def encode(text: str) -> str:
        encoder = r.encoder.text_encoder
        assert encoder is not None
        seen.append(weakref.ref(encoder))
        return f"emb({text})"

    r._encode = encode  # type: ignore[method-assign]
    assert r.prepare([Prompt("p1", ""), Prompt("p2", "")]) == 2
    assert events == ["transformer to cpu", "load", "qwen3 to cuda", "transformer to cuda"]
    assert r.encoder.text_encoder is None
    gc.collect()
    assert len(seen) == 2 and all(ref() is None for ref in seen)

    assert r.prepare([Prompt("p1", "")]) == 0  # cached: nothing moves or loads
    assert len(events) == 4


def test_zimage_encoder_is_let_go_of_when_the_trip_fails() -> None:
    events: list[str] = []
    r = _zimage(events)
    with pytest.raises(ValueError, match="512"), r._encoder_on_gpu():
        raise ValueError("the prompt is 600 tokens; Z-Image-Turbo reads 512")
    assert r.encoder.text_encoder is None
    assert events[-1] == "transformer to cuda"
