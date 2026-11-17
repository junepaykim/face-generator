"""Telling newgens apart and recognising a career across saves. Pure.

Newgens: the game gives every person it creates a unique ID above the last ID in its
database, so the save's IDs above 2,000,000,000 split into a dense database block and a
dense block of created people, with a large gap between them (docs/research.md).

Careers: a save file can be renamed, and two careers can share a manager profile, so a
career is recognised by the people the game created for it. Their names and birth dates
are random per career, and most of them persist from one save to the next.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import date

ID_FLOOR = 2_000_000_000
MIN_GAP = 100_000
MIN_BLOCK = 50
FINGERPRINT_SIZE = 64
MATCH_THRESHOLD = 0.5


def detect_newgen_boundary(unique_ids: Iterable[int]) -> int | None:
    """The lowest ID of the block of created people, or None when there is no clear gap.

    Looks at IDs from ID_FLOOR up, finds the largest gap between neighbours, and accepts it
    when it is at least MIN_GAP wide with at least MIN_BLOCK IDs above it.
    """
    ids = sorted({u for u in unique_ids if u >= ID_FLOOR})
    if len(ids) < 2:
        return None
    gap, index = max((ids[i + 1] - ids[i], i) for i in range(len(ids) - 1))
    if gap < MIN_GAP or len(ids) - (index + 1) < MIN_BLOCK:
        return None
    return ids[index + 1]


def is_newgen(
    unique_id: int,
    boundary: int | None,
    *,
    previously_newgen: bool = False,
    first_seen_mid_career: bool = False,
) -> bool:
    """Whether a player is a newgen.

    A newgen is a player whose ID is at or above the career's boundary, or a player who first
    appeared after the career's first sync (the game created them, even if it reused an old
    ID). Once a newgen, always a newgen.
    """
    if previously_newgen or first_seen_mid_career:
        return True
    return boundary is not None and unique_id >= boundary


@dataclass(frozen=True, slots=True)
class Person:
    unique_id: int
    name: str
    birth_date: date | None


def person_hash(p: Person) -> str:
    text = f"{p.unique_id}|{p.name}|{p.birth_date.isoformat() if p.birth_date else ''}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fingerprint(people: Iterable[Person], boundary: int | None, fallback: str) -> tuple[str, ...]:
    """A sample identifying the career: hashes of its FINGERPRINT_SIZE lowest created people.

    With no boundary, or no one above it, the fallback text (the save file name) is used.
    """
    created = sorted(
        (p for p in people if boundary is not None and p.unique_id >= boundary),
        key=lambda p: p.unique_id,
    )[:FINGERPRINT_SIZE]
    if not created:
        return ("file:" + hashlib.sha256(fallback.encode()).hexdigest()[:16],)
    return tuple(person_hash(p) for p in created)


def match_score(stored: Sequence[str], current: Collection[str]) -> float:
    """The share of a stored fingerprint still present in a save's hashes."""
    if not stored:
        return 0.0
    return sum(1 for h in stored if h in current) / len(stored)


def best_career[K](
    candidates: Iterable[tuple[K, Sequence[str]]], current: Collection[str]
) -> K | None:
    """The candidate whose fingerprint best matches, if any reaches MATCH_THRESHOLD."""
    best: tuple[float, K] | None = None
    for key, stored in candidates:
        score = match_score(stored, current)
        if score >= MATCH_THRESHOLD and (best is None or score > best[0]):
            best = (score, key)
    return None if best is None else best[1]


def career_key(fp: Sequence[str]) -> str:
    """A short, stable folder name for a career, from its first fingerprint."""
    return "c" + hashlib.sha256("".join(fp).encode()).hexdigest()[:10]
