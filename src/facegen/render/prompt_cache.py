"""Pure LRU storage and batched preparation of prompt embeddings.

Preparing upcoming prompts together amortizes text-encoder transfers to the GPU. The
renderer prepares any uncached prompt on first use. Values are opaque here: CPU tensors
in the renderer, plain values in tests.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager


class EmbeddingCache[V]:
    """The text encoders' output per prompt text; the least recently used goes first."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self._entries: OrderedDict[str, V] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, text: object) -> bool:
        return text in self._entries

    def get(self, text: str) -> V:
        """The text's value, which becomes the most recently used. KeyError if not cached."""
        value = self._entries[text]
        self._entries.move_to_end(text)
        return value

    def put(self, text: str, value: V) -> None:
        """Store a value as the most recently used, dropping the least recently used."""
        self._entries[text] = value
        self._entries.move_to_end(text)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def missing(self, texts: Iterable[str]) -> list[str]:
        """The texts not cached, each once, in their first order."""
        return [t for t in dict.fromkeys(texts) if t not in self._entries]

    def clear(self) -> None:
        self._entries.clear()


def prepare[V](
    cache: EmbeddingCache[V],
    texts: Iterable[str],
    encode: Callable[[str], V],
    session: Callable[[], AbstractContextManager[object]],
) -> int:
    """Encode every text not cached yet, all in one session; returns how many were encoded.

    `session` brings the encoders onto the GPU and back; it is not opened when every text is
    cached already. The texts already cached become the most recently used first, so the
    new ones cannot push them out. More distinct texts than the cache holds is an error:
    the first ones would be dropped before they were used.
    """
    needed = list(dict.fromkeys(texts))
    if len(needed) > cache.capacity:
        raise ValueError(f"{len(needed)} prompt texts, but the cache holds {cache.capacity}")
    todo = []
    for text in needed:
        if text in cache:
            cache.get(text)
        else:
            todo.append(text)
    if todo:
        with session():
            for text in todo:
                cache.put(text, encode(text))
    return len(todo)
