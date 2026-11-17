"""Settings values shared by policy code, without loading or folder discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from facegen.core.traits import BodyThresholds

GROUP_NAMES = ("a", "b", "c")

DETAIL_SIZES: dict[str, tuple[int, int] | None] = {
    "1344x1728": (1344, 1728),
    "1152x1472": (1152, 1472),
    "off": None,
}
"""`groups.*.detail_size`: the detail pass's target size (width, height), or none. 1344x1728 is
1.5 times the 896x1152 render; 1152x1472 needs less graphics memory (Phase 10c). Both sides
are multiples of 64, as the model needs."""

FOUND = "found on this PC"
NOT_SET = "not set"


@dataclass(frozen=True, slots=True)
class Paths:
    fm_user_dir: Path | None
    """FM27's user folder (saves, graphics); None when it is neither set nor found."""
    data_dir: Path
    fm_install_dir: Path | None
    """FM27's install folder: read only for the list of photos the game ships. None when it
    is neither set nor found: then no one counts as without a photo."""
    origins: Mapping[str, str] = field(default_factory=dict)
    """Where each folder came from, by key: the file that set it, `FOUND` or `NOT_SET`."""

    @property
    def games_dir(self) -> Path | None:
        return self.fm_user_dir / "games" if self.fm_user_dir is not None else None

    @property
    def graphics_dir(self) -> Path | None:
        return self.fm_user_dir / "graphics" if self.fm_user_dir is not None else None

    def describe(self, key: str) -> str:
        """A folder and where it came from, for the CLI and the UI."""
        origin = self.origins.get(key)
        value = {"paths.fm_user_dir": self.fm_user_dir, "paths.fm_install_dir": self.fm_install_dir}
        path = value.get(key)
        if path is None:
            return "not set, and not found on this PC"
        if origin == FOUND:
            return f"{path} (found on this PC)"
        return f"{path} (from {origin})" if origin else str(path)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "facegen.db"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"


@dataclass(frozen=True, slots=True)
class Eligibility:
    mens_football_only: bool
    continental_competition_ids: frozenset[int]
    loaned_out_at_my_club: bool


@dataclass(frozen=True, slots=True)
class Newgen:
    fallback_min_unique_id: int


@dataclass(frozen=True, slots=True)
class GroupA:
    ca_above: int
    pa_above: int
    published_long_edge: int
    candidates: int
    detail_size: tuple[int, int] | None
    """The detail pass's target size (width, height); None: no detail pass."""


@dataclass(frozen=True, slots=True)
class GroupB:
    my_academy: bool
    league_ca_above: int
    league_pa_above: int
    any_ca_above: int
    any_pa_above: int
    published_long_edge: int
    candidates: int
    detail_size: tuple[int, int] | None
    """The detail pass's target size (width, height); None: no detail pass."""


@dataclass(frozen=True, slots=True)
class GroupC:
    league_ca_above: int
    league_pa_above: int
    published_long_edge: int
    candidates: int
    detail_size: tuple[int, int] | None
    """The detail pass's target size (width, height); None: no detail pass."""


@dataclass(frozen=True, slots=True)
class Groups:
    a: GroupA
    b: GroupB
    c: GroupC

    def published_sizes(self) -> dict[str, int]:
        """Group name to the height of its published 3:4 picture, in pixels."""
        return {
            "a": self.a.published_long_edge,
            "b": self.b.published_long_edge,
            "c": self.c.published_long_edge,
        }


@dataclass(frozen=True, slots=True)
class AgeSettings:
    """Per age table, (first real age, wording key) rows with ages rising (spec section 3)."""

    a_my_club: tuple[tuple[int, str], ...]
    a_other_club: tuple[tuple[int, str], ...]
    b: tuple[tuple[int, str], ...]
    c: tuple[tuple[int, str], ...]
    staff: tuple[tuple[int, str], ...]
    """My club's staff, whatever their group: wide bands, W02 the youngest (user, 2026-11-12)."""
    media: tuple[tuple[int, str], ...]
    """Journalists and press officers: the staff bands from W04 up (user, 2026-11-14)."""

    def table(self, name: str) -> tuple[tuple[int, str], ...]:
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class StaffSettings:
    """My club's staff get pictures like its players (user, 2026-11-12)."""

    enabled: bool
    """False: staff are in no group unless pinned; their pictures are kept."""


@dataclass(frozen=True, slots=True)
class YouthSettings:
    """My club's teams other than the first team (user, 2026-11-13)."""

    enabled: bool
    """True: players in my club's reserve, under-23 and under-18 teams are in group B, or A
    when their CA or PA reaches group A's thresholds, and A once they join the first team.
    False: they are in group A like the first team."""


@dataclass(frozen=True, slots=True)
class MediaSettings:
    """Journalists and press officers (user, 2026-11-13), drawn once at group B."""

    journalists: bool
    """Journalists from my club's country are drawn without a request."""
    press_officers: bool
    """My club's press officers are drawn without a request."""


@dataclass(frozen=True, slots=True)
class NoPhotoSettings:
    """Pre-existing players without a photo are picked like newgens (user, 2026-11-12)."""

    enabled: bool
    """False: only newgens are picked outside my club; drawn pictures are kept."""


@dataclass(frozen=True, slots=True)
class ReferenceSettings:
    """Re-draws may use the player's existing picture as a reference (spec section 5)."""

    enabled: bool
    scale: float
    other_clubs: bool


@dataclass(frozen=True, slots=True)
class MyClubSettings:
    """Prompt adjustments for players at my club (spec section 3)."""

    banned_traits: frozenset[str]
    soften_skin: int
    round_eyes: float = 0.0
    """Share of sharp-eyed players drawn with round eyes (0 to 1)."""


@dataclass(frozen=True, slots=True)
class PersonalitySettings:
    """How much personality weights a newly sampled character sheet (spec section 4)."""

    weight: int
    """Each lean multiplies its option's weight by (1 + weight); 0 turns weighting off."""


@dataclass(frozen=True, slots=True)
class Schedule:
    poll_seconds: int
    """How often `facegen serve` looks at FM27's saves folder."""
    generate_while_fm_running: bool
    auto_groups: frozenset[str]
    """Groups whose automatic jobs `facegen serve` draws; manual requests always run."""


@dataclass(frozen=True, slots=True)
class RenderSettings:
    """How the model draws, for the graphics card it has (Phase 10d)."""

    vae_tiling: bool
    """Decode in tiles and one picture at a time (diffusers' VAE tiling and slicing): less
    graphics memory, a little slower."""


@dataclass(frozen=True, slots=True)
class UI:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class Settings:
    paths: Paths
    eligibility: Eligibility
    newgen: Newgen
    groups: Groups
    staff: StaffSettings
    no_photo: NoPhotoSettings
    youth: YouthSettings
    media: MediaSettings
    age: AgeSettings
    reference: ReferenceSettings
    my_club: MyClubSettings
    body: BodyThresholds
    personality: PersonalitySettings
    schedule: Schedule
    render: RenderSettings
    ui: UI
    values: Mapping[str, Any]
    """Every effective value by dotted key, for display."""
    overridden: frozenset[str]
    """Keys whose value comes from the database rather than the defaults file."""
