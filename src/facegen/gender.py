"""Whether a member of my club's staff, a journalist or a press officer is drawn as a man. Pure.

Players are always drawn as girls (spec section 3). The people around them are drawn as
themselves where the save says they are women. Journalists and press officers keep the
gender the save gives them; a man on my club's staff is drawn as a man or a woman, by a share
that depends on the job. Who gets which is seeded by the career and the person's unique ID,
so it holds for every picture of them and does not change with "New face".
"""

from __future__ import annotations

from facegen.traits import derive_seed

DEFAULT_WOMEN_SHARE = 0.3
"""Of the men on my club's staff, the share drawn as women where the job sets no other."""

# Job words (fmsave's labels, lower case) to the share of men in that job drawn as women.
# First match wins, so "loan manager" is a loan job and "head of youth development" a youth
# one. A job the save does not name takes DEFAULT_WOMEN_SHARE.
WOMEN_SHARE_BY_JOB = (
    ("doctor", 0.9),
    ("physio", 0.9),
    ("youth", 0.6),
    ("loan", 0.6),
    ("scien", 0.45),
    ("recruit", 0.45),
    ("scout", 0.15),
    ("director", 0.15),
    ("chair", 0.15),
)


def women_share(job: str | None) -> float:
    """The share of men in this staff job who are drawn as women."""
    if job:
        for word, share in WOMEN_SHARE_BY_JOB:
            if word in job:
                return share
    return DEFAULT_WOMEN_SHARE


def drawn_male(
    career_key: str,
    unique_id: int,
    *,
    staff: bool,
    media: bool,
    female: bool | None,
    job: str | None = None,
) -> bool:
    """Whether the person is drawn as a man.

    `female` is what the save says (None: it does not say). A player is never drawn as a man.
    A journalist or press officer is drawn as a man exactly when the save names a man. A
    woman on my club's staff is drawn as a woman; a man, or someone the save does not tell
    about, is drawn as a woman at the job's share (`women_share`), seeded per person.
    """
    if media:
        return female is False
    if not staff or female:
        return False
    return derive_seed("drawn gender", career_key, unique_id) / 2**32 >= women_share(job)
