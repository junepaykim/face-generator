"""Who is drawn as a man (gender.py). Pure."""

from __future__ import annotations

import pytest

from facegen.gender import DEFAULT_WOMEN_SHARE, drawn_male, women_share


@pytest.mark.parametrize(
    ("job", "share"),
    [
        ("chief doctor", 0.9),
        ("physio", 0.9),
        ("head physio", 0.9),
        ("head of youth development", 0.6),
        ("loan manager", 0.6),
        ("sports scientist", 0.45),
        ("head of sports science", 0.45),
        ("recruitment analyst", 0.45),
        ("scout", 0.15),
        ("chief scout", 0.15),
        ("director of football", 0.15),
        ("technical director", 0.15),
        ("chairperson", 0.15),
        ("coach", DEFAULT_WOMEN_SHARE),
        ("assistant manager", DEFAULT_WOMEN_SHARE),
        ("manager (you)", DEFAULT_WOMEN_SHARE),
        ("performance analyst", DEFAULT_WOMEN_SHARE),
        (None, DEFAULT_WOMEN_SHARE),
    ],
)
def test_women_share_by_job(job: str | None, share: float) -> None:
    assert women_share(job) == share


def test_players_are_never_drawn_as_men() -> None:
    for uid in range(200):
        assert not drawn_male("c1", uid, staff=False, media=False, female=False)


def test_media_keep_the_gender_the_save_gives() -> None:
    for uid in range(50):
        assert drawn_male("c1", uid, staff=False, media=True, female=False)
        assert not drawn_male("c1", uid, staff=False, media=True, female=True)
        assert not drawn_male("c1", uid, staff=False, media=True, female=None)


def test_women_on_the_staff_are_drawn_as_women() -> None:
    for uid in range(200):
        assert not drawn_male("c1", uid, staff=True, media=False, female=True, job="scout")


def share_of_women(job: str | None, female: bool | None = False) -> float:
    n = 4000
    men = sum(
        drawn_male("c1", uid, staff=True, media=False, female=female, job=job) for uid in range(n)
    )
    return 1 - men / n


@pytest.mark.parametrize("job", ["physio", "loan manager", "coach", "scout", None])
def test_men_on_the_staff_are_drawn_as_women_at_the_jobs_share(job: str | None) -> None:
    assert share_of_women(job) == pytest.approx(women_share(job), abs=0.03)
    # Someone the save says nothing about is treated as a man.
    assert share_of_women(job, None) == pytest.approx(women_share(job), abs=0.03)


def test_the_draw_is_seeded_by_career_and_person() -> None:
    a = [drawn_male("c1", uid, staff=True, media=False, female=False) for uid in range(300)]
    assert a == [drawn_male("c1", uid, staff=True, media=False, female=False) for uid in range(300)]
    b = [drawn_male("c2", uid, staff=True, media=False, female=False) for uid in range(300)]
    assert a != b
