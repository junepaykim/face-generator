"""What my club's staff wear. The content tests here guard AGENTS.md hard rule 3."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from facegen.age import WORDINGS
from facegen.kits import Kit, load_kits
from facegen.prompt import (
    CONTENT_NEGATIVE,
    CONTENT_POSITIVE,
    QUALITY,
    STAFF_NEGATIVE,
    STAFF_POSITIVE,
    YOUNG_NEGATIVE,
    build_prompt,
)
from facegen.qc import expectation
from facegen.traits import CharacterSheet
from facegen.wardrobe import (
    LOOKS,
    PROPS,
    ROLES,
    accent_colour,
    dress,
    outfit_for,
    role_for,
    seniority,
    track_jacket_words,
)

COLCHESTER = Kit(shirt=("blue", "white"), pattern="stripes", shorts="blue")
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
    base_seed=7,
)

# Words no staff look or prop may carry: they would show skin or put the playing kit back.
FORBIDDEN = (
    "short sleeves",
    "sleeveless",
    "unzipped",
    "unbuttoned",
    "open shirt",
    "crop",
    "tank top",
    "shorts",
    "skirt",
    "soccer",
    "jersey",
    "sleeves rolled up",
    *(t.strip() for t in CONTENT_NEGATIVE.split(",")),
)


@pytest.mark.parametrize(
    ("job", "role"),
    [
        # fmsave's named jobs (StaffJob), as staff_job() words them, and the human manager.
        ("coach", "coach"),
        ("chairperson", "director"),
        ("director", "director"),
        ("managing director", "director"),
        ("director of football", "director"),
        ("physio", "medical"),
        ("scout", "scout"),
        ("manager", "manager"),
        ("assistant manager", "manager"),
        ("goalkeeping coach", "coach"),
        ("chief doctor", "doctor"),
        ("head of sports science", "medical"),
        ("sports scientist", "medical"),
        ("head physio", "medical"),
        ("performance analyst", "analyst"),
        ("head of youth development", "coach"),
        ("technical director", "director"),
        ("manager (you)", "manager"),
        # Jobs fmsave does not name yet, matched by their words.
        ("chief scout", "scout"),
        ("recruitment analyst", "analyst"),
        ("loan manager", "director"),
        ("fitness coach", "coach"),
        ("club doctor", "doctor"),
        (None, "staff"),
        ("kit man", "staff"),
    ],
)
def test_role_for_each_job(job: str | None, role: str) -> None:
    assert role_for(job) == role


def test_seniority_from_title_and_age() -> None:
    assert seniority("head physio", 30) == "senior"
    assert seniority("chief doctor", None) == "senior"
    assert seniority("technical director", 33) == "senior"
    assert seniority("chairperson", 72) == "senior"
    assert seniority("coach", 50) == "senior"
    assert seniority("coach", 49) is None
    assert seniority("coach", 35) is None
    assert seniority("coach", 34) == "young"
    assert seniority(None, None) is None


def test_every_role_names_known_looks_and_props() -> None:
    for name, role in ROLES.items():
        assert set(role.looks) <= set(LOOKS), name
        assert set(role.props) <= set(PROPS), name
        assert all(w > 0 for w in (*role.looks.values(), *role.props.values())), name


def test_look_templates_are_consistent() -> None:
    for name, look in LOOKS.items():
        fields = set(re.findall(r"{(\w+)}", look.words))
        assert fields <= {"tone", "club", "accent", "jacket"}, name
        assert ("tone" in fields) == bool(look.tones), name
        assert not {"club", "accent"} <= fields, name  # one club colour per look
        assert bool(look.colour_on) == bool(fields & {"club", "accent"}), name
        assert look.lean in (None, "senior", "young"), name
        assert look.tags, name


def test_no_look_or_prop_shows_skin_or_the_playing_kit() -> None:
    texts = [p.words for p in PROPS.values()]
    for name, look in LOOKS.items():
        for tone in look.tones or (None,):
            texts.append(dress(COLCHESTER, "staff", name, tone=tone).words())
    for text in texts:
        for word in FORBIDDEN:
            assert word not in text, (word, text)


def test_same_inputs_same_outfit_and_pictures_vary() -> None:
    a = outfit_for(COLCHESTER, "coach", 30, 1234, 1)
    assert a == outfit_for(COLCHESTER, "coach", 30, 1234, 1)
    looks = {outfit_for(COLCHESTER, "coach", 30, 1234, v).look for v in range(1, 30)}
    assert len(looks) >= 4


def test_roles_keep_to_their_looks() -> None:
    for seed in range(200):
        assert outfit_for(RED, "director of football", 60, seed, 1).sporty is False
        assert outfit_for(RED, "coach", 40, seed, 1).look in ROLES["coach"].looks
        doctor = outfit_for(RED, "chief doctor", 45, seed, 1)
        assert doctor.look in ROLES["doctor"].looks
        assert doctor.prop in ROLES["doctor"].props


def test_seniority_shifts_the_looks() -> None:
    def senior_share(age: int) -> float:
        looks = Counter(outfit_for(RED, "manager", age, seed, 1).look for seed in range(600))
        return sum(n for look, n in looks.items() if LOOKS[look].lean == "senior") / 600

    assert senior_share(60) > senior_share(40) + 0.1


def test_track_jacket_in_the_home_colours() -> None:
    assert track_jacket_words(COLCHESTER) == "blue and white track jacket"
    sleeves = Kit(shirt=("red", "white"), pattern="sleeves", shorts="white", trim="black")
    assert track_jacket_words(sleeves) == "red track jacket, white sleeves, black collar"
    hoops = Kit(shirt=("navy", "sky"), pattern="hoops", shorts="navy")
    assert track_jacket_words(hoops) == "navy blue and light blue track jacket"
    jacket = dress(COLCHESTER, "coach", "track jacket", "whistle")
    assert jacket.words() == "blue and white track jacket, whistle around neck"
    assert jacket.rank_tags() == {
        "outfit": frozenset({"track jacket"}),
        "jacket colour 1": frozenset({"blue jacket"}),
        "jacket colour 2": frozenset({"white jacket"}),
        "prop": frozenset({"whistle", "whistle around neck"}),
    }


def test_necktie_and_scarf_take_a_colour_that_shows() -> None:
    assert accent_colour(RED) == "red"
    assert accent_colour(Kit(shirt=("white", "red"), pattern="sleeves", shorts="red")) == "red"
    assert accent_colour(Kit(shirt=("white",), shorts="black", trim="black")) == "black"
    assert accent_colour(Kit(shirt=("white",), shorts="black")) == "navy"
    suit = dress(Kit(shirt=("white", "claret"), pattern="sleeves", shorts="white"), "director",
                 "suit and tie", tone="grey")  # fmt: skip
    assert suit.words() == "grey suit, white collared shirt, dark red necktie"
    assert suit.rank_tags()["club colour"] == frozenset({"red necktie"})


def test_fallback_club_is_dressed() -> None:
    kits = load_kits()
    home, source = kits.kit_for(999, goalkeeper=False)
    assert source == "fallback"
    for job in ("coach", "director", "chief doctor", None):
        assert outfit_for(home, job, 40, 5, 1).words()


def test_unknown_tone_is_an_error() -> None:
    with pytest.raises(ValueError):
        dress(RED, "director", "suit and tie", tone="pink")


@pytest.mark.parametrize("wording", sorted(WORDINGS))
def test_staff_prompts_keep_the_content_rule(wording: str) -> None:
    for name, look in LOOKS.items():
        for prop in PROPS:
            outfit = dress(RED, "staff", name, prop)
            p = build_prompt(SHEET, wording, outfit, "crossed")
            assert p.positive.startswith("1girl, solo, safe, ")
            assert p.positive.endswith(QUALITY)
            assert STAFF_POSITIVE in p.positive
            assert CONTENT_POSITIVE not in p.positive and "soccer" not in p.positive
            assert p.negative.startswith(CONTENT_NEGATIVE)
            assert STAFF_NEGATIVE in p.negative
            assert (YOUNG_NEGATIVE in p.negative) == WORDINGS[wording].young
            # Sportswear is asked for in club sportswear and kept out of smart wear.
            assert ("sportswear" in p.positive) == look.sporty
            assert ("sportswear" in p.negative) == (not look.sporty)


def test_staff_prompt_order() -> None:
    outfit = dress(RED, "director", "suit and tie", "wristwatch", tone="navy blue")
    p = build_prompt(SHEET, "W07", outfit, "crossed")
    assert p.positive == (
        "1girl, solo, safe, young adult woman, black hair, long hair, straight hair, "
        "blunt bangs, brown eyes, tareme, navy blue suit, white collared shirt, red necktie, "
        "wristwatch, crossed arms, serious, closed mouth, upper body, looking at viewer, "
        "standing, simple background, grey background, long sleeves, "
        "masterpiece, high score, great score, absurdres"
    )
    assert f"{CONTENT_NEGATIVE}, {STAFF_NEGATIVE}, sportswear, multiple girls" in p.negative


def test_quality_check_ranks_on_the_outfit() -> None:
    outfit = dress(RED, "doctor", "lab coat and tie", "stethoscope")
    rank = expectation(SHEET, outfit).rank
    assert rank["outfit"] == frozenset({"lab coat"})
    assert rank["club colour"] == frozenset({"red necktie"})
    assert rank["prop"] == frozenset({"stethoscope"})
    assert "shirt colour 1" not in rank
