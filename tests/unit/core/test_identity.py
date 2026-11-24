from __future__ import annotations

from datetime import date

from kitshot.core.identity import (
    FINGERPRINT_SIZE,
    Person,
    best_career,
    career_key,
    detect_newgen_boundary,
    fingerprint,
    is_newgen,
    match_score,
    person_hash,
)

# Shaped like the Phase 0.1 save: a dense database block, one stray database ID, a gap of
# about a million, then the people the game created.
DATABASE = list(range(2_000_005_719, 2_000_594_452, 997))
STRAY = [2_001_040_995]
CREATED = list(range(2_002_064_399, 2_002_072_705, 7))
LOW = list(range(352, 90_000_000, 1_000_003))


def test_boundary_is_the_first_created_id() -> None:
    assert detect_newgen_boundary(LOW + DATABASE + STRAY + CREATED) == 2_002_064_399


def test_boundary_ignores_order_and_repeats() -> None:
    ids = list(reversed(CREATED)) + DATABASE + CREATED[:10] + LOW
    assert detect_newgen_boundary(ids) == 2_002_064_399


def test_no_boundary_without_a_wide_gap() -> None:
    dense = list(range(2_000_000_000, 2_000_100_000, 50))
    assert detect_newgen_boundary(dense) is None


def test_no_boundary_when_too_few_ids_sit_above_the_gap() -> None:
    assert detect_newgen_boundary(DATABASE + STRAY) is None  # one stray is not a block


def test_no_boundary_from_ids_below_the_floor() -> None:
    assert detect_newgen_boundary(LOW) is None
    assert detect_newgen_boundary([]) is None


def test_is_newgen() -> None:
    b = 2_002_064_399
    assert is_newgen(b, b)
    assert is_newgen(b + 5, b)
    assert not is_newgen(b - 1, b)
    assert not is_newgen(2_002_100_000, None)
    # Sticky, and anyone who appears mid-career was created by the game.
    assert is_newgen(12345, b, previously_newgen=True)
    assert is_newgen(12345, b, first_seen_mid_career=True)


def people(prefix: str, ids: list[int]) -> list[Person]:
    return [Person(i, f"{prefix} {i}", date(2008, 1, 1)) for i in ids]


def test_fingerprint_samples_the_lowest_created_people() -> None:
    created = people("x", CREATED)
    fp = fingerprint(people("db", DATABASE) + created, 2_002_064_399, "save.fm")
    assert len(fp) == FINGERPRINT_SIZE
    assert fp == tuple(person_hash(p) for p in created[:FINGERPRINT_SIZE])


def test_fingerprint_is_deterministic_and_order_free() -> None:
    created = people("x", CREATED)
    a = fingerprint(created, 2_002_064_399, "save.fm")
    b = fingerprint(list(reversed(created)), 2_002_064_399, "save.fm")
    assert a == b


def test_fingerprint_falls_back_to_the_file_name() -> None:
    fp = fingerprint(people("db", DATABASE), None, "save.fm")
    assert len(fp) == 1 and fp[0].startswith("file:")
    assert fp == fingerprint([], 2_002_064_399, "save.fm")
    assert fp != fingerprint([], None, "other.fm")


def test_person_hash_changes_with_name_or_birth_date() -> None:
    p = Person(1, "Ana", date(2008, 1, 1))
    assert person_hash(p) != person_hash(Person(1, "Ana", date(2008, 1, 2)))
    assert person_hash(p) != person_hash(Person(1, "Ann", date(2008, 1, 1)))
    assert person_hash(Person(1, "Ana", None))


def test_match_score() -> None:
    assert match_score(["a", "b", "c", "d"], {"a", "b", "z"}) == 0.5
    assert match_score([], {"a"}) == 0.0


def test_same_career_is_recognised_after_people_retire() -> None:
    created = people("x", CREATED)
    stored = fingerprint(created, 2_002_064_399, "a.fm")
    later = {person_hash(p) for p in created[40:]}  # the 40 oldest created people are gone
    assert match_score(stored, later) == (FINGERPRINT_SIZE - 40) / FINGERPRINT_SIZE


def test_best_career_picks_the_best_match_above_the_threshold() -> None:
    current = {"a", "b", "c"}
    candidates = [(1, ["a", "x", "y", "z"]), (2, ["a", "b", "c", "x"]), (3, ["a", "b", "x", "y"])]
    assert best_career(candidates, current) == 2
    assert best_career([(1, ["a", "x", "y", "z"])], current) is None
    assert best_career([], current) is None


def test_other_career_does_not_match() -> None:
    mine = fingerprint(people("mine", CREATED), 2_002_064_399, "a.fm")
    theirs = {person_hash(p) for p in people("theirs", CREATED)}
    assert best_career([(1, mine)], theirs) is None


def test_career_key_is_short_and_stable() -> None:
    fp = ("abc", "def")
    assert career_key(fp) == career_key(list(fp))
    assert career_key(fp).startswith("c") and len(career_key(fp)) == 11
    assert career_key(fp) != career_key(("abc",))
