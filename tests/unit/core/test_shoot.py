"""A player's shoot: kit look, extra, pose and backdrop (shoot.py). No GPU.

The content-rule tests here guard AGENTS.md hard rule 3 for players' clothes and poses.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import pytest

from kitshot.core.age import WORDINGS
from kitshot.core.man_traits import HAIR_COLOUR as MAN_HAIR_COLOUR
from kitshot.core.prompt import CONTENT_NEGATIVE, CONTENT_POSITIVE, QUALITY, build_prompt
from kitshot.core.shoot import (
    ANGLES,
    ARMS,
    BACKDROP_COLOURS,
    BACKDROP_KINDS,
    DEFAULT_BACKDROP,
    EXTRAS,
    HAIR_FAMILIES,
    JACKET_ARMS,
    KEEPER_ARMS,
    LOOKS,
    STYLES,
    Backdrop,
    KitLook,
    Pose,
    clothes_families,
    favourite_style,
    pick_backdrop,
    season_of,
    shoot_for,
)
from kitshot.core.traits import HAIR_COLOUR, CharacterSheet
from kitshot.datasource.kits import COLOURS, Kit

COLCHESTER = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
KEEPER = Kit(shirt=("yellow",), goalkeeper=True)
RED = Kit(shirt=("red",), shorts="white")
SHEET = CharacterSheet(
    hair_colour="black",
    hair_style="long",
    fringe="blunt",
    eye_colour="brown",
    eye_shape="gentle",
    skin="light",
    accessory="none",
    mark="none",
    expression="serious",
    height="average",
    build="average",
    base_seed=1,
)
WINTER = date(2026, 12, 5)
SUMMER = date(2027, 7, 10)
SEEDS = range(800)


def shoots(kit: Kit = COLCHESTER, hair: str = "black", **kw: object) -> list:
    return [shoot_for(kit, hair, seed, 1, **kw) for seed in SEEDS]  # type: ignore[arg-type]


# --- content rule ------------------------------------------------------------------------------

# Winter wear is left out on purpose (user, 2026-11-12), and nothing may show skin.
FORBIDDEN = (
    "coat", "beanie", "hat", "cap", "scarf", "hood", "glove ", "sleeveless", "crop", "bare",
    "midriff", "navel", "cleavage", "swimsuit", "bikini", "underwear", "tank top", "unzipped",
    "shirt lift", "no shirt", "topless", "lying", "sitting", "from above",
)  # fmt: skip


def all_words() -> list[str]:
    kits = (COLCHESTER, KEEPER, RED, Kit(shirt=("red", "white"), pattern="sleeves", trim="black"))
    words = [KitLook(k, look, extra).words() for k in kits for look in LOOKS for extra in EXTRAS]
    words += [o.words for o in (*ARMS.values(), *ANGLES.values(), *BACKDROP_KINDS.values())]
    words += [c.words for c in BACKDROP_COLOURS.values()]
    return words


@pytest.mark.parametrize("words", all_words())
def test_no_word_weakens_the_content_rule(words: str) -> None:
    for term in FORBIDDEN:
        assert term not in f"{words} ", (term, words)
    for term in CONTENT_NEGATIVE.split(", "):
        assert term not in words.split(", "), (term, words)


@pytest.mark.parametrize("look", sorted(LOOKS))
def test_every_look_is_a_full_football_top(look: str) -> None:
    for kit in (COLCHESTER, KEEPER):
        words = KitLook(kit, look).words()
        assert "shirt" in words or "jacket" in words
        assert "sleeves" in words or "jacket" in words  # sleeved, or a long-sleeved jacket
        assert ("gloves" in words) == kit.goalkeeper


@pytest.mark.parametrize("wording", sorted(WORDINGS))
def test_player_prompts_keep_the_content_terms(wording: str) -> None:
    for seed in range(40):
        for kit in (COLCHESTER, KEEPER):
            shoot = shoot_for(kit, "black", seed, 1, age=31, game_date=WINTER)
            p = build_prompt(SHEET, wording, shoot.wear, shoot.pose, backdrop=shoot.backdrop)
            assert p.positive.startswith("1girl, solo, safe, ")
            assert CONTENT_POSITIVE in p.positive and p.positive.endswith(QUALITY)
            assert p.negative.startswith(CONTENT_NEGATIVE)
            assert "simple background" in p.positive


# --- clothes -----------------------------------------------------------------------------------


def test_the_classic_looks_are_the_kit_words_of_before() -> None:
    assert KitLook(COLCHESTER, "short sleeves").words() == COLCHESTER.words()
    assert KitLook(KEEPER, "long sleeves").words() == KEEPER.words()


def test_look_words() -> None:
    assert KitLook(COLCHESTER, "layered sleeves", "armband").words() == (
        "blue and white vertical-striped shirt, layered sleeves, blue shorts, armband"
    )
    assert KitLook(COLCHESTER, "collared shirt").words() == (
        "blue and white vertical-striped shirt, collared shirt, short sleeves, blue shorts"
    )
    assert KitLook(KEEPER, "collared shirt").words() == (
        "goalkeeper, yellow shirt, collared shirt, long sleeves, black gloves"
    )
    # Training and warm-up tops: the jacket in the kit's colours, no shorts, no extra.
    assert KitLook(COLCHESTER, "track jacket").words() == (
        "blue and white track jacket, long sleeves"
    )
    assert KitLook(COLCHESTER, "open jacket").words() == (
        "blue and white track jacket, open jacket, blue and white vertical-striped shirt"
    )
    assert KitLook(RED, "high collar").words() == "red track jacket, high collar, zipper"


def test_rank_tags_follow_the_garment() -> None:
    assert KitLook(COLCHESTER, "short sleeves", "wristband").rank_tags() == {
        "shirt colour 1": frozenset({"blue shirt"}),
        "shirt colour 2": frozenset({"white shirt"}),
        "look": frozenset({"short sleeves"}),
        "extra": frozenset({"wristband"}),
    }
    jacket = KitLook(COLCHESTER, "track jacket").rank_tags()
    assert jacket["jacket colour 1"] == frozenset({"blue jacket"})
    assert jacket["look"] == frozenset({"track jacket"})
    assert "shirt colour 1" in KitLook(COLCHESTER, "open jacket").rank_tags()


def test_conditional_poses() -> None:
    for shoot in shoots() + shoots(KEEPER):
        if shoot.pose.arms in JACKET_ARMS:
            assert LOOKS[shoot.wear.look].jacket
        if shoot.pose.arms in KEEPER_ARMS:
            assert shoot.wear.kit.goalkeeper
    keepers = Counter(s.pose.arms for s in shoots(KEEPER))
    assert keepers["gloves"] > 50
    players = Counter(s.pose.arms for s in shoots())
    assert "gloves" not in players and players["pockets"] > 0


def test_extras_go_with_the_match_kit_and_armbands_to_seniors() -> None:
    for shoot in shoots(age=33):
        if LOOKS[shoot.wear.look].jacket:
            assert shoot.wear.extra == "none"
    assert all(s.wear.extra != "wristband" for s in shoots(KEEPER))
    young = Counter(s.wear.extra for s in shoots(age=21))
    senior = Counter(s.wear.extra for s in shoots(age=31))
    assert senior["armband"] > 2 * young["armband"] > 0
    assert young["none"] > 0.75 * len(SEEDS)  # extras stay rare


def test_season_shifts_the_looks() -> None:
    def share(game_date: date | None, looks: set[str]) -> float:
        drawn = shoots(game_date=game_date)
        return sum(s.wear.look in looks for s in drawn) / len(drawn)

    warm = {"long sleeves", "layered sleeves", "track jacket"}
    assert share(WINTER, warm) > share(None, warm) + 0.1
    assert share(SUMMER, {"short sleeves"}) > share(None, {"short sleeves"}) + 0.1
    assert [season_of(date(2026, m, 1)) for m in (1, 2, 3, 6, 8, 9, 11, 12)] == [
        "winter", "winter", "spring", "summer", "summer", "autumn", "winter", "winter",
    ]  # fmt: skip
    assert season_of(None) == "unknown"


def test_a_favourite_style_is_fixed_per_player_and_shows() -> None:
    styles = Counter(favourite_style(seed) for seed in SEEDS)
    assert set(styles) == set(STYLES) and styles["none"] == max(styles.values())
    jacket_fans = [seed for seed in SEEDS if favourite_style(seed) == "jacket"][:20]
    others = [seed for seed in SEEDS if favourite_style(seed) == "none"][:20]

    def jacket_share(seeds: list[int]) -> float:
        drawn = [shoot_for(COLCHESTER, "black", seed, v) for seed in seeds for v in range(1, 30)]
        return sum(s.wear.look in ("track jacket", "open jacket") for s in drawn) / len(drawn)

    assert jacket_share(jacket_fans) > 2 * jacket_share(others)
    # The same player keeps their style in every picture; the picture's look still changes.
    looks = {shoot_for(COLCHESTER, "black", 7, v).wear.look for v in range(1, 40)}
    assert {shoot_for(COLCHESTER, "black", 7, v).style for v in range(1, 40)} == {
        favourite_style(7)
    }
    assert len(looks) > 2


def test_same_inputs_same_shoot() -> None:
    a = shoot_for(COLCHESTER, "pink", 99, 3, age=30, game_date=WINTER)
    assert a == shoot_for(COLCHESTER, "pink", 99, 3, age=30, game_date=WINTER)
    assert len({shoot_for(COLCHESTER, "pink", 99, v) for v in range(1, 30)}) > 20


# --- pose ----------------------------------------------------------------------------------------


def test_front_is_the_most_common_angle_and_from_below_the_rarest() -> None:
    angles = Counter(s.pose.angle for s in shoots())
    assert angles.most_common(1)[0][0] == "front"
    assert angles["below"] == min(angles.values()) and angles["below"] < 0.1 * len(SEEDS)
    assert Pose("crossed", "side").words() == "crossed arms, from side"
    assert Pose("hips").words() == "hands on own hips"


# --- backdrop ------------------------------------------------------------------------------------


def test_every_hair_colour_has_backdrop_families() -> None:
    """Women's and men's (Phase 15g); a colour with both names blends with the same."""
    assert set(HAIR_FAMILIES) == set(HAIR_COLOUR) | set(MAN_HAIR_COLOUR)


@pytest.mark.parametrize("hair", sorted(set(HAIR_COLOUR) | set(MAN_HAIR_COLOUR)))
@pytest.mark.parametrize(
    "kit",
    [COLCHESTER, KEEPER, RED, Kit(shirt=("amber",), trim="black", shorts="black"),
     Kit(shirt=("grey", "pink"), pattern="mixed")],
)  # fmt: skip
def test_the_backdrop_never_blends_with_the_clothes_or_hair(kit: Kit, hair: str) -> None:
    avoid = clothes_families(kit) | HAIR_FAMILIES[hair]
    kinds = Counter()
    for shoot in shoots(kit, hair):
        backdrop = shoot.backdrop
        kinds[backdrop.kind] += 1
        assert backdrop.colours and len(set(backdrop.colours)) == len(backdrop.colours)
        assert len(backdrop.colours) == (2 if backdrop.kind == "two-tone" else 1)
        for colour in backdrop.colours:
            assert not BACKDROP_COLOURS[colour].families & avoid, (colour, kit, hair)
    assert kinds["plain"] > kinds["gradient"] > kinds["two-tone"] > 0


def test_a_goalkeeper_backdrop_favours_the_club_colours() -> None:
    plain = Counter(c for s in shoots(KEEPER) for c in s.backdrop.colours)
    club = Counter(c for s in shoots(KEEPER, home=COLCHESTER) for c in s.backdrop.colours)
    assert club["blue"] > 2 * plain["blue"]
    # An outfield player in the same kit never gets it: it is the shirt's colour.
    assert all("blue" not in s.backdrop.colours for s in shoots(COLCHESTER, home=COLCHESTER))


def test_backdrop_words() -> None:
    assert DEFAULT_BACKDROP.words() == "simple background, grey background"
    assert Backdrop("gradient", ("blue",)).words() == (
        "simple background, gradient background, blue background"
    )
    assert Backdrop("two-tone", ("grey", "yellow")).words() == (
        "simple background, two-tone background, grey background, yellow background"
    )
    assert Backdrop("two-tone", ("grey", "yellow")).describe() == "two-tone grey and yellow"
    only_grey = pick_backdrop(1, 1, avoid=frozenset(COLOURS) - {"grey"})
    assert only_grey == DEFAULT_BACKDROP  # one colour left: never two-tone


def test_staff_prompts_keep_the_grey_backdrop() -> None:
    p = build_prompt(SHEET, "W07", RED, "crossed")
    assert "standing, simple background, grey background, sportswear" in p.positive
