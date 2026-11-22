"""Settings values shared by policy code, without loading or folder discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from facegen.core.traits import BodyThresholds

GROUP_NAMES = ("a", "b", "c")

LANGUAGE_CHOICES = ("auto", "en", "ko")
"""`display.language`: auto follows the window's language (Phase 12a)."""
THEME_CHOICES = ("auto", "light", "dark")
"""`display.theme`: auto follows Windows's light or dark app mode (Phase 12b)."""


@dataclass(frozen=True, slots=True)
class ImageModel:
    """What a `style.model` choice means for the pictures it draws."""

    name: str
    photo: bool
    """Draws photos: a man gets the photo prompt in sentences (core/man_prompt.py)."""
    reference: bool
    """Takes the current picture as a reference on a re-draw (an IP-Adapter)."""
    negative: bool
    """Uses a negative prompt. Z-Image-Turbo draws without guidance, so it has none."""


IMAGE_MODELS: dict[str, ImageModel] = {
    "animagine": ImageModel("Animagine XL 4.0", photo=False, reference=True, negative=True),
    "realvis": ImageModel("RealVisXL V5.0", photo=True, reference=True, negative=True),
    "zimage_turbo": ImageModel("Z-Image-Turbo", photo=True, reference=False, negative=False),
}
MODEL_CHOICES = tuple(IMAGE_MODELS)
"""`style.model`: the image model everyone is drawn with (user, 2026-11-20)."""
WOMAN, MAN = "woman", "man"
GENDER_CHOICES = (WOMAN, MAN)
"""`style.gender`: everyone is drawn as a woman or as a man (user, 2026-11-20)."""

ASPECTS: dict[str, tuple[int, int]] = {"3:4": (3, 4), "1:1": (1, 1)}
"""`picture.aspect`: the shape of pictures drawn from now on, width to height (Phase 16g). Each
picture keeps the shape it was drawn in (`images.aspect`)."""
ASPECT_CHOICES = tuple(ASPECTS)
RENDER_SIZES: dict[str, tuple[int, int]] = {"3:4": (896, 1152), "1:1": (1024, 1024)}
"""The size each shape is rendered at: about SDXL's native million pixels. A 3:4 render is
cropped to exactly 3:4 (864x1152); a square needs no crop."""

DETAIL_SIZES: dict[str, dict[str, tuple[int, int]] | None] = {
    "x1.5": {"3:4": (1344, 1728), "1:1": (1536, 1536)},
    "smaller": {"3:4": (1152, 1472), "1:1": (1280, 1280)},
    "off": None,
}
"""`groups.*.detail_size`: the detail pass by scale, resolved per picture shape to (width,
height), or none. x1.5 is 1.5 times the render; "smaller" needs less graphics memory. Every side
is a multiple of 64, as the models need. Before Phase 16 the values were the 3:4 sizes
("1344x1728", "1152x1472"), migrated by `persistence.db.CONVERTED_SETTINGS`."""


def detail_size(name: str, aspect: str) -> tuple[int, int] | None:
    """A detail-pass choice's target size for a picture shape; None for off."""
    sizes = DETAIL_SIZES[name]
    return None if sizes is None else sizes[aspect]


CUTOFF_MODES = ("auto", "manual")
"""`cutoffs.mode`: my league's cells as shares ("top 6%") or numbers set by hand (Phase 16)."""
TIER_CHOICES = ("a", "b", "c")
"""`cutoffs.my_club.*`: the tier (group) a part of my club is drawn in."""
STAFF_AGE_MODES = ("manual", "auto")
"""`age.staff.mode`: the staff table set look by look, or spread over an age span (Phase 16f)."""


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
class GroupSpec:
    """How one group's pictures are drawn: who is in it is `CutoffSettings`' (Phase 16)."""

    published_long_edge: int
    candidates: int
    detail: str
    """A key of DETAIL_SIZES."""

    def detail_size(self, aspect: str = "3:4") -> tuple[int, int] | None:
        """The detail pass's target size (width, height) for a picture shape; None: no pass."""
        return detail_size(self.detail, aspect)


@dataclass(frozen=True, slots=True)
class Groups:
    a: GroupSpec
    b: GroupSpec
    c: GroupSpec

    def published_sizes(self) -> dict[str, int]:
        """Group name to the height of its published picture, in pixels."""
        return {
            "a": self.a.published_long_edge,
            "b": self.b.published_long_edge,
            "c": self.c.published_long_edge,
        }


@dataclass(frozen=True, slots=True)
class MyClubTiers:
    """The tier each part of my club is drawn in (`cutoffs.my_club`)."""

    first_team: str = "a"
    youth: str = "b"
    """My club's teams other than the first team (reserves, under-23s, under-18s)."""
    loans: str = "b"
    """Players I loaned out to other clubs, while I hold the contract (user, 2026-11-21)."""
    academy_alumni: bool = True
    """Newgens who came through my academy stay in Standard (B), tracked, after they leave."""


@dataclass(frozen=True, slots=True)
class CutoffSettings:
    """Who is picked for each tier (spec section 2, Phase 16): shares of a league, numbers
    everywhere. A cell's None is "-": switched off. See core/cutoffs.py."""

    mode: str = "auto"
    """A key of CUTOFF_MODES: my league's cells as shares (auto) or numbers (manual)."""
    light_above: bool = True
    """The league above gets Light pictures by its share (`above_c_ca`)."""
    light_below: bool = False
    """The league below gets Light pictures by its share (`below_c_ca`)."""
    my_club: MyClubTiers = field(default_factory=MyClubTiers)
    share: Mapping[str, float | None] = field(default_factory=dict)
    """My league's cells and the other leagues' cells ("above_b_ca"), in percent."""
    manual: Mapping[str, int | None] = field(default_factory=dict)
    """My league's cells as numbers (at least), used in manual mode."""
    everywhere: Mapping[str, int | None] = field(default_factory=dict)
    """The global cells: at least this CA or PA, at any club."""


@dataclass(frozen=True, slots=True)
class StaffSpread:
    """`age.staff` in auto mode: looks `first` to `last` spread over the ages `start` to `end`."""

    first: str
    last: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PictureSettings:
    """The shape of pictures drawn from now on."""

    aspect: str = "1:1"
    """A key of ASPECTS. A change re-draws nobody; each picture keeps its own shape."""


@dataclass(frozen=True, slots=True)
class AgeSettings:
    """Per age table, (first real age, wording key) rows with ages rising (spec section 3)."""

    a_my_club: tuple[tuple[int, str], ...]
    a_other_club: tuple[tuple[int, str], ...]
    b: tuple[tuple[int, str], ...]
    c: tuple[tuple[int, str], ...]
    staff: tuple[tuple[int, str], ...]
    """My club's staff, whatever their group: wide bands, W02 the youngest."""
    media: tuple[tuple[int, str], ...]
    """Journalists and press officers: the staff bands from W04 up."""
    staff_spread: StaffSpread | None = None
    """In auto mode (`age.staff.mode`) the spread `staff` was made from; None in manual."""

    def table(self, name: str) -> tuple[tuple[int, str], ...]:
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class StaffSettings:
    """My club's staff get pictures like its players."""

    enabled: bool
    """False: staff are in no group unless pinned; their pictures are kept."""


@dataclass(frozen=True, slots=True)
class MediaSettings:
    """Journalists and press officers, drawn once at group B."""

    journalists: bool
    """Journalists from my club's country are drawn without a request."""
    press_officers: bool
    """My club's press officers are drawn without a request."""


@dataclass(frozen=True, slots=True)
class NoPhotoSettings:
    """Pre-existing players without a photo are picked like newgens."""

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
    """Rendering options for the graphics card."""

    vae_tiling: bool
    """Decode in tiles and one picture at a time (diffusers' VAE tiling and slicing): less
    graphics memory, a little slower."""


@dataclass(frozen=True, slots=True)
class StyleSettings:
    """How everyone is drawn: one model and one gender for players, staff and media alike.
    A change applies to pictures drawn from then on; it re-draws nobody by itself."""

    model: str
    """A key of MODEL_CHOICES."""
    gender: str
    """A key of GENDER_CHOICES."""


@dataclass(frozen=True, slots=True)
class DisplaySettings:
    """How the UI looks: its language and theme; "auto" follows Windows."""

    language: str
    theme: str


@dataclass(frozen=True, slots=True)
class NotificationSettings:
    """Which of the tray's notifications are shown; the others are always shown."""

    new_pictures: bool
    """"New pictures" once drawing stops."""


@dataclass(frozen=True, slots=True)
class UpdateSettings:
    """Whether the app asks GitHub for a newer release (Phase 13)."""

    check: bool


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
    cutoffs: CutoffSettings
    staff: StaffSettings
    no_photo: NoPhotoSettings
    media: MediaSettings
    age: AgeSettings
    reference: ReferenceSettings
    my_club: MyClubSettings
    body: BodyThresholds
    personality: PersonalitySettings
    schedule: Schedule
    render: RenderSettings
    picture: PictureSettings
    style: StyleSettings
    display: DisplaySettings
    notifications: NotificationSettings
    updates: UpdateSettings
    ui: UI
    values: Mapping[str, Any]
    """Every effective value by dotted key, for display."""
    overridden: frozenset[str]
    """Keys whose value comes from the database rather than the defaults file."""
