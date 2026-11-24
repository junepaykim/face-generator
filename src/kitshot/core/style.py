"""Which prompt a picture gets from the model and gender settings (Phase 15g; user, 2026-11-20).

One model draws everyone, and men and women differ only by prompt (and check) branch:

- a woman: the women's anime prompt (prompt.build_prompt) from her character sheet and age
  wording, with my club's rules. The photo models draw it too, for now (user, 2026-11-20): a
  photo prompt for women is not decided;
- a man: his own sheet (man_traits.ManSheet) at his real age in years, as a photo prompt in
  sentences on a photo model, or Animagine's tags (man_prompt.py). My club's rules are the
  women's and do not apply. Without an age there is no prompt.

A model that draws without guidance (Z-Image-Turbo) gets no negative prompt, so none is
stored with its pictures. Standard library only; the same inputs give the same prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kitshot.core.man_prompt import anime_prompt, photo_prompt
from kitshot.core.man_traits import HAIR_STYLE, ManSheet, drawn
from kitshot.core.prompt import Prompt, Wear, build_prompt
from kitshot.core.settings_values import IMAGE_MODELS, MAN, WOMAN
from kitshot.core.shoot import Backdrop, Pose
from kitshot.core.traits import CharacterSheet

if TYPE_CHECKING:
    from kitshot.core.settings_values import MyClubSettings


def portrait_prompt(
    model: str,
    gender: str,
    *,
    sheet: CharacterSheet,
    man: ManSheet | None,
    wording: str,
    age: int | None,
    wear: Wear,
    pose: str | Pose,
    backdrop: Backdrop,
    my_club: MyClubSettings | None = None,
) -> Prompt | None:
    """The prompt for one picture, or None for a man whose age is unknown.

    `model` and `gender` are the `style.model` and `style.gender` keys; `sheet` the character
    sheet, `man` the men's sheet of the same version (needed for a man); `wording` the age
    wording key (W01 to W10); `age` the real age in years; `my_club` only for someone at my
    club, as for prompt.build_prompt.
    """
    spec = IMAGE_MODELS[model]  # KeyError on an unknown model: never guess one
    if gender == MAN:
        if man is None:
            raise ValueError("a picture of a man needs his sheet")
        if age is None:
            return None
        if spec.photo:
            prompt = photo_prompt(man, age, wear, backdrop=backdrop)
        else:
            prompt = anime_prompt(man, age, wear, pose, backdrop=backdrop)
    elif gender == WOMAN:
        prompt = build_prompt(sheet, wording, wear, pose, my_club, backdrop=backdrop)
    else:
        raise ValueError(f"no gender called {gender!r}")
    return prompt if spec.negative else Prompt(prompt.positive, "")


def shoot_hair(sheet: CharacterSheet, man: ManSheet | None, age: int | None) -> str:
    """The hair colour a player's shoot keeps the backdrop clear of (shoot.HAIR_FAMILIES):
    the women's sheet's, or a man's as drawn at `age` ("" for a shaved head)."""
    if man is None:
        return sheet.hair_colour
    if not HAIR_STYLE[man.hair_style].coloured:
        return ""
    return man.hair_colour if age is None else drawn(man, age).hair_colour
