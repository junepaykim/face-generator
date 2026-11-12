"""Quality and content checks (docs/spec.md section 3; docs/research.md section 5 recipe).

A picture passes only if all of these hold:

- the tagger's top rating is "general", and none of CONTENT_TAGS (midriff, navel, crop top
  and the like) is tagged (content rule; never configurable);
- a hair colour tag and an eye colour tag of the sheet are present (tag threshold 0.35);
  the eye colour also shows a face is there;
- `solo` and `1girl` are present and none of `multiple views`, `multiple girls`, `2girls`,
  `no humans`;
- the figure's head starts in the top fifth of the picture (a stand-in for a face-size
  check until a face detector is approved; decisions.md).

Kit pattern and logos are not checked (user, 2026-11-10). Group A's candidates that pass
are ranked by `score`: p(general), plus the tagger's mean confidence in the traits and shirt
colours the prompt asked for, plus 1 - p(cowboy shot), which prefers the tighter framing.

The checks are pure functions of tagger output and the picture. The tagger itself runs on
the GPU and is loaded only by `Eva02Tagger`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import TYPE_CHECKING, Any

from facegen.prompt import drawn_traits
from facegen.traits import EYE_COLOUR, HAIR_COLOUR, TRAIT_TABLES, CharacterSheet

if TYPE_CHECKING:
    from PIL.Image import Image

    from facegen.kits import Kit
    from facegen.settings import MyClubSettings

TAG_THRESHOLD = 0.35
REQUIRED_TAGS = ("solo", "1girl")
# Content rule: a picture tagged with any of these is rejected even if its top rating is
# "general" (decisions.md, from the Z-Image test: a bare midriff rated general 0.66).
CONTENT_TAGS = (
    "nude", "nipples", "underwear", "lingerie", "swimsuit", "bikini", "cleavage", "midriff",
    "navel", "see-through", "crop top", "revealing clothes",
)  # fmt: skip
FORBIDDEN_TAGS = ("multiple views", "multiple girls", "2girls", "no humans")
HEAD_TOP_MAX = 0.20
"""Highest allowed start of the figure, as a fraction of the picture height. Every one of
the 426 Animagine bake-off pictures started at 0.10 or above it (docs/research.md)."""

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
    sheet: CharacterSheet, kit: Kit, my_club: MyClubSettings | None = None
) -> Expectation:
    rank: dict[str, frozenset[str]] = {}
    for name, value in drawn_traits(sheet, my_club).items():
        tags = TRAIT_TABLES[name][value].tags
        if tags:
            rank[name] = tags
    for i, group in enumerate(kit.shirt_tag_groups(), 1):
        rank[f"shirt colour {i}"] = group
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


# --- head position -------------------------------------------------------------------------

_GRID = 128
_CORNER = 8
_DIFF = 60  # summed RGB difference from the backdrop that counts as foreground
_ROW_SHARE = 0.02  # a row with more foreground than this starts the figure


def head_top(image: Image) -> float | None:
    """Where the figure starts, as a fraction of the height, or None if no figure is found.

    The backdrop colour is the median of the two top corners; the first row with enough
    pixels away from it is the top of the head (the heuristic of the reference test,
    docs/research.md run r4). A busy background gives 0 here, so the check fails open and
    the tagger's checks still apply.
    """
    small = image.convert("RGB").resize((_GRID, _GRID))
    px = small.load()
    assert px is not None
    corner = [
        px[x, y] for y in range(_CORNER) for x in (*range(_CORNER), *range(_GRID - _CORNER, _GRID))
    ]
    bg = tuple(sorted(c[i] for c in corner)[len(corner) // 2] for i in range(3))
    need = max(1, int(_ROW_SHARE * _GRID))
    for y in range(_GRID):
        count = 0
        for x in range(_GRID):
            r, g, b = px[x, y][:3]
            if abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) > _DIFF:
                count += 1
        if count > need:
            return y / _GRID
    return None


# --- the tagger (GPU) ------------------------------------------------------------------------

TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
_KEEP_FLOOR = 0.01  # probabilities below this are dropped from TagResult.tags


class Eva02Tagger:
    """WD EVA02-Large tagger v3 through timm, as in the bake-off. Loads onto the GPU."""

    name = TAGGER_REPO

    def __init__(self, device: str = "cuda") -> None:
        from facegen.gpu import prepare_environment

        prepare_environment()
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        self._torch = torch
        self.device = device
        model = timm.create_model(f"hf-hub:{TAGGER_REPO}", pretrained=True)
        self.model = model.eval().to(device, torch.float16)
        config = timm.data.resolve_data_config(model.pretrained_cfg, model=model)
        self.transform = timm.data.create_transform(**config)
        with open(hf_hub_download(TAGGER_REPO, "selected_tags.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"].replace("_", " ") for r in rows]
        categories = [int(r["category"]) for r in rows]
        self.rating_idx = [i for i, c in enumerate(categories) if c == 9]
        self.general_idx = [i for i, c in enumerate(categories) if c == 0]

    def __call__(self, images: Sequence[Image]) -> list[TagResult]:
        torch = self._torch
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images])
        x = batch[:, [2, 1, 0]].to(self.device, torch.float16)  # the model wants BGR
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x)).float().cpu().tolist()
        out = []
        for row in probs:
            ratings = {self.names[i]: row[i] for i in self.rating_idx}
            tags = {self.names[i]: row[i] for i in self.general_idx if row[i] >= _KEEP_FLOOR}
            out.append(TagResult(ratings, tags))
        return out
