from __future__ import annotations

import sqlite3
from collections import Counter

import pytest

from facegen import traits
from facegen.traits import (
    BANNABLE,
    FIELDS,
    POSE,
    BodyThresholds,
    CharacterSheet,
    build_class,
    height_class,
    load_sheet,
    pick_pose,
    picture_seed,
    sample_sheet,
    sheet_for,
    skin_words,
)


def test_same_inputs_give_the_same_sheet() -> None:
    a = sample_sheet("career-1", 29141735, 1, height_cm=177, strength=12)
    b = sample_sheet("career-1", 29141735, 1, height_cm=177, strength=12)
    assert a == b


def test_career_player_and_version_each_change_the_sheet() -> None:
    base = sample_sheet("career-1", 29141735, 1)
    others = [
        sample_sheet("career-2", 29141735, 1),
        sample_sheet("career-1", 29141736, 1),
        sample_sheet("career-1", 29141735, 2),
    ]
    for other in others:
        assert other.base_seed != base.base_seed
    # Over many players the sampled fields differ too.
    sheets = {sample_sheet("career-1", uid, 1).to_json() for uid in range(200)}
    assert len(sheets) > 150


def test_sheet_values_come_from_the_vocabulary() -> None:
    for uid in range(300):
        s = sample_sheet("k", uid, 1)
        for name, table in FIELDS.items():
            assert getattr(s, name) in table
        assert 0 <= s.base_seed < 2**32


def test_weights_shape_the_draw() -> None:
    """Accessory "none" has weight 4 of 8, so about half the players have no accessory."""
    counts = Counter(sample_sheet("k", uid, 1).accessory for uid in range(4000))
    assert 0.45 < counts["none"] / 4000 < 0.55
    assert set(counts) == set(traits.ACCESSORY)


def test_known_sheet_is_stable_across_releases() -> None:
    """Changing the hashing or the vocabulary order would re-roll every stored player."""
    s = sample_sheet("ca0a7bbf751", 29141735, 1, height_cm=177, strength=12)
    assert s == CharacterSheet(
        hair_colour="black",
        hair_style="long",
        fringe="swept",
        eye_colour="violet",
        eye_shape="gentle",
        skin="fair",
        accessory="clip",
        mark="freckles",
        expression="confident",
        height="average",
        build="average",
        base_seed=107270051,
    )
    assert pick_pose(s.base_seed, 1) == "behind"
    assert picture_seed(s.base_seed, 2, 1, 2) == 107270051 + 100 + 10 + 2


@pytest.mark.parametrize(
    ("cm", "cls"),
    [(None, "average"), (161, "short"), (173, "short"), (174, "average"), (190, "average"),
     (191, "tall"), (206, "tall")],
)  # fmt: skip
def test_height_class_marks_the_tenth_at_each_end(cm: int | None, cls: str) -> None:
    assert height_class(cm) == cls


@pytest.mark.parametrize(
    ("strength", "cls"),
    [(None, "average"), (1, "slim"), (4, "slim"), (5, "average"), (13, "average"),
     (14, "muscular"), (20, "muscular")],
)  # fmt: skip
def test_build_class_marks_the_tenth_at_each_end(strength: int | None, cls: str) -> None:
    assert build_class(strength) == cls


def test_body_thresholds_are_settings() -> None:
    t = BodyThresholds(
        short_below_cm=170, tall_from_cm=195, slim_strength_at_most=2, muscular_strength_from=18
    )
    assert [height_class(cm, t) for cm in (169, 170, 194, 195)] == [
        "short", "average", "average", "tall"
    ]  # fmt: skip
    assert [build_class(x, t) for x in (2, 3, 17, 18)] == ["slim", "average", "average", "muscular"]
    s = sample_sheet("k", 1, 1, height_cm=169, strength=18, body=t)
    assert (s.height, s.build) == ("short", "muscular")


def test_body_classes_have_words_only_at_the_ends() -> None:
    assert traits.HEIGHT["short"].words == "petite"
    assert traits.HEIGHT["tall"].words == "tall female"
    assert traits.BUILD["slim"].words == "skinny"
    assert traits.BUILD["muscular"].words == "muscular female"
    assert traits.HEIGHT["average"].words == traits.BUILD["average"].words == ""


def test_skin_softening_follows_the_scale() -> None:
    # Spec section 3: at step 2 tan becomes light and dark becomes slightly tanned.
    assert skin_words("tan", 2) == ""
    assert skin_words("dark", 2) == "slightly tanned skin"
    assert skin_words("tan", 1) == "slightly tanned skin"
    assert skin_words("dark", 1) == "tan"
    # Never lighter than light; fair and light unchanged.
    assert skin_words("dark", 9) == ""
    assert skin_words("fair", 2) == "pale skin"
    assert skin_words("light", 2) == ""
    assert skin_words("dark", 0) == "dark skin, dark-skinned female"


def test_picture_seeds_are_distinct_and_offset_from_the_base() -> None:
    base = 4_000_000_000
    seeds = {picture_seed(base, v, a, c) for v in (1, 2, 3) for a in range(3) for c in range(8)}
    assert len(seeds) == 3 * 3 * 8
    assert picture_seed(base, 1, 0, 0) == base
    assert picture_seed(base, 2, 0, 0) == (base + 100) % 2**32
    assert all(0 <= s < 2**32 for s in seeds)
    with pytest.raises(ValueError):
        picture_seed(base, 1, 0, 10)


def test_pose_is_from_the_library_and_varies_by_picture() -> None:
    poses = {pick_pose(12345, v) for v in range(1, 50)}
    assert poses <= set(POSE) and len(poses) > 3
    assert pick_pose(12345, 1) == pick_pose(12345, 1)


def test_bannable_traits_are_accessories_marks_and_body_words() -> None:
    assert set(BANNABLE) == {
        "hairband", "ribbon", "clip", "glasses", "freckles", "mole",
        "short", "tall", "slim", "muscular",
    }  # fmt: skip


def test_hair_and_eye_colours_have_tagger_tags() -> None:
    for table in (traits.HAIR_COLOUR, traits.EYE_COLOUR):
        for name, option in table.items():
            assert option.tags, name


def test_sheet_json_round_trip() -> None:
    s = sample_sheet("k", 7, 1, height_cm=195, strength=16)
    back = CharacterSheet.from_json(s.to_json(), s.base_seed)
    assert back == s
    assert "base_seed" not in s.to_json()


def test_sheet_for_samples_once_and_stores(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO careers (id, key, display_name, fingerprint, created_at, updated_at)"
        " VALUES (1, 'k', 'K', '[]', '', '')"
    )
    v1, s1 = sheet_for(conn, 1, "k", 42, height_cm=180, strength=10)
    v2, s2 = sheet_for(conn, 1, "k", 42, height_cm=150, strength=1)  # facts changed: kept
    assert (v1, v2) == (1, 1) and s1 == s2
    assert load_sheet(conn, 1, 42) == (1, s1)
    v3, s3 = sheet_for(conn, 1, "k", 42, height_cm=180, strength=10, new_face=True)
    assert v3 == 2 and s3.base_seed != s1.base_seed
    assert load_sheet(conn, 1, 42) == (2, s3)
    assert conn.execute("SELECT COUNT(*) FROM traits").fetchone()[0] == 2
