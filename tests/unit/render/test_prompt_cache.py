"""The prompt embedding cache and the two-stage encoding (facegen.prompt_cache). Pure."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from facegen.render.prompt_cache import EmbeddingCache, prepare


class Encoders:
    """Stands in for the text encoders: counts trips to the GPU and what each one encoded."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.trips: list[list[str]] = []
        self.on_gpu = False
        self.fail_on = fail_on

    @contextmanager
    def session(self) -> Iterator[None]:
        self.trips.append([])
        self.on_gpu = True
        try:
            yield
        finally:
            self.on_gpu = False

    def encode(self, text: str) -> str:
        assert self.on_gpu, "encoded outside a trip"
        if text == self.fail_on:
            raise RuntimeError("out of memory")
        self.trips[-1].append(text)
        return f"emb({text})"

    def prepare(self, cache: EmbeddingCache[str], texts: list[str]) -> int:
        return prepare(cache, texts, self.encode, self.session)


def test_the_least_recently_used_text_goes_first() -> None:
    cache: EmbeddingCache[int] = EmbeddingCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # now "b" is the least recently used
    cache.put("c", 3)
    assert "b" not in cache and "a" in cache and "c" in cache
    assert len(cache) == 2
    with pytest.raises(KeyError):
        cache.get("b")


def test_putting_a_cached_text_replaces_it_without_growing() -> None:
    cache: EmbeddingCache[int] = EmbeddingCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)  # "a" becomes the most recent
    cache.put("c", 3)
    assert cache.get("a") == 10 and "b" not in cache


def test_missing_lists_each_uncached_text_once_in_order() -> None:
    cache: EmbeddingCache[int] = EmbeddingCache(8)
    cache.put("neg", 0)
    assert cache.missing(["p2", "neg", "p1", "p2", "", "p1"]) == ["p2", "p1", ""]
    assert cache.missing(["neg"]) == []


def test_capacity_must_hold_something() -> None:
    with pytest.raises(ValueError):
        EmbeddingCache(0)


def test_a_batch_is_encoded_in_one_trip_and_shared_texts_once() -> None:
    cache: EmbeddingCache[str] = EmbeddingCache(16)
    enc = Encoders()
    # Three jobs' prompts: the negative prompt is the same for two of them.
    texts = ["", "pos1", "neg", "pos2", "neg", "pos3", "neg young"]
    assert enc.prepare(cache, texts) == 6
    assert enc.trips == [["", "pos1", "neg", "pos2", "pos3", "neg young"]]
    assert cache.get("pos2") == "emb(pos2)"


def test_nothing_moves_when_every_text_is_cached() -> None:
    cache: EmbeddingCache[str] = EmbeddingCache(16)
    enc = Encoders()
    enc.prepare(cache, ["", "pos1", "neg"])
    assert enc.prepare(cache, ["pos1", "neg", ""]) == 0
    assert len(enc.trips) == 1


def test_a_later_prompt_costs_one_trip_for_its_new_texts_only() -> None:
    """The single-job path: a prompt nobody prepared is encoded on its own when drawn."""
    cache: EmbeddingCache[str] = EmbeddingCache(16)
    enc = Encoders()
    enc.prepare(cache, ["", "pos1", "neg"])
    assert enc.prepare(cache, ["", "pos9", "neg"]) == 1
    assert enc.trips == [["", "pos1", "neg"], ["pos9"]]


def test_texts_needed_again_are_not_pushed_out_by_the_new_ones() -> None:
    cache: EmbeddingCache[str] = EmbeddingCache(3)
    enc = Encoders()
    enc.prepare(cache, ["neg", "old1", "old2"])  # "neg" is now the least recently used
    enc.prepare(cache, ["neg", "new"])  # needs "neg" again: "old1" goes instead
    assert "neg" in cache and "new" in cache and "old1" not in cache
    assert enc.trips[-1] == ["new"]


def test_more_texts_than_the_cache_holds_is_refused_before_any_trip() -> None:
    cache: EmbeddingCache[str] = EmbeddingCache(2)
    enc = Encoders()
    with pytest.raises(ValueError):
        enc.prepare(cache, ["a", "b", "c", "a"])
    assert enc.trips == [] and len(cache) == 0
    assert enc.prepare(cache, ["a", "b", "a"]) == 2  # duplicates count once


def test_a_failed_encode_still_ends_the_trip_and_keeps_what_was_done() -> None:
    cache: EmbeddingCache[str] = EmbeddingCache(8)
    enc = Encoders(fail_on="bad")
    with pytest.raises(RuntimeError):
        enc.prepare(cache, ["ok", "bad", "later"])
    assert not enc.on_gpu  # the encoders went back to system memory
    assert "ok" in cache and "bad" not in cache and "later" not in cache
