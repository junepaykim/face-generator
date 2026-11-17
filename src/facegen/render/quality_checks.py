"""Quality and content checks (docs/spec.md section 3; docs/research.md section 5 recipe).

A picture passes only if all of these hold:

- the tagger's top rating is "general", and none of CONTENT_TAGS (midriff, navel, crop top
  and the like) is tagged (content rule; never configurable);
- a hair colour tag and an eye colour tag of the sheet are present (tag threshold 0.35);
  the eye colour also shows a face is there;
- `solo` and `1girl` are present and none of `multiple views`, `multiple girls`, `2girls`,
  `no humans`;
- the figure's head starts in the top fifth of the picture, as a proxy for face size.

Kit pattern and logos are not checked. Group A's candidates that pass
are ranked by `score`: p(general), plus the tagger's mean confidence in the traits, the kit's
colours and a player's look and extra (for staff, the outfit's look, colour and prop) the
prompt asked for, plus 1 - p(cowboy shot), which prefers the tighter framing.

The checks are pure functions of tagger output and the picture. The GPU tagger lives in
`tagger.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import TYPE_CHECKING, Any

from facegen.core.prompt import drawn_traits
from facegen.core.traits import EYE_COLOUR, HAIR_COLOUR, TRAIT_TABLES, CharacterSheet
from facegen.render import figure

if TYPE_CHECKING:
    from PIL.Image import Image

    from facegen.core.prompt import Wear
    from facegen.core.settings_values import MyClubSettings

TAG_THRESHOLD = 0.35
REQUIRED_TAGS = ("solo", "1girl")
# Reject these tags even when the top rating is "general"; ratings alone miss violations.
CONTENT_TAGS = (
    "nude", "nipples", "underwear", "lingerie", "swimsuit", "bikini", "cleavage", "midriff",
    "navel", "see-through", "crop top", "revealing clothes",
)  # fmt: skip
FORBIDDEN_TAGS = ("multiple views", "multiple girls", "2girls", "no humans")
HEAD_TOP_MAX = 0.20
"""Maximum distance from the top edge to the figure, as a fraction of picture height."""

# Kept in the stored check result, at any probability: a rough record of age and framing.
PROBE_TAGS = ("child", "aged down", "mature female", "cowboy shot", "upper body")


@dataclass(frozen=True, slots=True)
class TagResult:
    ratings: Mapping[str, float]
    tags: Mapping[str, float]
    """General tags and their probabilities; a missing tag counts as 0."""

    @property
    def rating(self) -> str:
        return max(self.ratings, key=lambda k: self.ratings[k])

    def p(self, tag: str) -> float:
        return float(self.tags.get(tag, 0.0))

    def present(self, tag: str) -> bool:
        return self.p(tag) >= TAG_THRESHOLD

    def above_threshold(self, limit: int = 60) -> dict[str, float]:
        found = sorted(((t, p) for t, p in self.tags.items() if p >= TAG_THRESHOLD), key=_by_p)
        return {t: round(p, 3) for t, p in found[:limit]}


def _by_p(item: tuple[str, float]) -> float:
    return -item[1]


@dataclass(frozen=True, slots=True)
class Expectation:
    hair: frozenset[str]
    eyes: frozenset[str]
    rank: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Ranking only: trait or shirt colour name to the tags that show it."""


def expectation(
    sheet: CharacterSheet, wear: Wear, my_club: MyClubSettings | None = None
) -> Expectation:
    """What the tagger should see: the sheet's hair and eyes, and, for ranking only, the
    traits drawn and the kit's colours and look, or the staff outfit's look, colour and prop."""
    rank: dict[str, frozenset[str]] = {}
    for name, value in drawn_traits(sheet, my_club).items():
        tags = TRAIT_TABLES[name][value].tags
        if tags:
            rank[name] = tags
    rank.update(wear.rank_tags())
    return Expectation(
        hair=HAIR_COLOUR[sheet.hair_colour].tags,
        eyes=EYE_COLOUR[sheet.eye_colour].tags,
        rank=rank,
    )


@dataclass(frozen=True, slots=True)
class CheckResult:
    passed: bool
    failures: tuple[str, ...]
    rating: str
    p_general: float
    score: float
    traits: Mapping[str, float]
    head_top: float | None

    def to_json(self, tags: TagResult | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "passed": self.passed,
            "failures": list(self.failures),
            "rating": self.rating,
            "p_general": round(self.p_general, 4),
            "score": round(self.score, 4),
            "traits": {k: round(v, 3) for k, v in self.traits.items()},
            "head_top": None if self.head_top is None else round(self.head_top, 3),
        }
        if tags is not None:
            data["ratings"] = {k: round(v, 4) for k, v in tags.ratings.items()}
            data["probes"] = {t: round(tags.p(t), 3) for t in PROBE_TAGS}
            data["tags"] = tags.above_threshold()
        return data


def score(tags: TagResult, expect: Expectation) -> tuple[float, dict[str, float]]:
    """Ranking score for a passing candidate, and the per-trait confidences behind it."""
    traits = {name: max(tags.p(t) for t in group) for name, group in expect.rank.items()}
    trait_mean = fmean(traits.values()) if traits else 0.0
    framing = 1.0 - tags.p("cowboy shot")
    return tags.ratings.get("general", 0.0) + trait_mean + framing, traits


def check(tags: TagResult, expect: Expectation, head_top: float | None) -> CheckResult:
    failures: list[str] = []
    p_general = float(tags.ratings.get("general", 0.0))
    if tags.rating != "general":
        failures.append(f"rated {tags.rating} (general {p_general:.2f})")
    flagged = [t for t in CONTENT_TAGS if tags.present(t)]
    if flagged:
        failures.append("content: " + ", ".join(flagged))
    if not any(tags.present(t) for t in expect.hair):
        failures.append("hair colour not seen")
    if not any(tags.present(t) for t in expect.eyes):
        failures.append("eye colour not seen")
    missing = [t for t in REQUIRED_TAGS if not tags.present(t)]
    extra = [t for t in FORBIDDEN_TAGS if tags.present(t)]
    if missing or extra:
        failures.append("not one girl: " + ", ".join([f"no {t}" for t in missing] + extra))
    if head_top is None:
        failures.append("no figure found against the background")
    elif head_top > HEAD_TOP_MAX:
        failures.append(f"head starts {head_top:.0%} down the picture")
    total, traits = score(tags, expect)
    return CheckResult(
        passed=not failures,
        failures=tuple(failures),
        rating=tags.rating,
        p_general=p_general,
        score=total,
        traits=traits,
        head_top=head_top,
    )


def best(results: Sequence[CheckResult]) -> int | None:
    """Index of the best passing result, or None if none passed. Ties go to the first."""
    passing = [i for i, r in enumerate(results) if r.passed]
    if not passing:
        return None
    return max(passing, key=lambda i: (results[i].score, -i))


_GRID = 128
_CORNER = 8
_DIFF = 60  # summed RGB difference from the backdrop that counts as foreground
_ROW_SHARE = 0.02  # a row with more foreground than this starts the figure


def head_top(image: Image) -> float | None:
    """Where the figure starts, as a fraction of the height, or None if no figure is found.

    The backdrop is followed down from the two top corners, row by row, so a gradient or
    two-tone backdrop is backdrop too; the first row with enough pixels away from it is the
    top of the head (the heuristic of the reference test, docs/research.md run r4). A busy
    background gives 0 here, so the check fails open and the tagger's checks still apply.
    The scan is figure.py's.
    """
    g = figure.grid(image, _GRID, _GRID)
    edges = g.edges(_CORNER, _DIFF)
    first = g.first_row(edges, _DIFF, max(1, int(_ROW_SHARE * _GRID)))
    return None if first is None else first / _GRID
