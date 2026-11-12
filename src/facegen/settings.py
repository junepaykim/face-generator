"""Settings: defaults from config/default.toml, overrides from the state database.

Read settings only through `load_settings`. The UI writes overrides with `set_override`.
The content rule (docs/spec.md section 3) is code, not a setting, and has no key here.
"""

from __future__ import annotations

import json
import sqlite3
import tomllib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from facegen.age import TABLES, WORDINGS
from facegen.traits import BANNABLE, BodyThresholds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = PROJECT_ROOT / "config" / "default.toml"

# Keys the UI may not override: where things live on disk.
_FIXED_PREFIXES = ("paths.",)


class SettingsError(ValueError):
    """A setting is unknown, has the wrong type, or is out of range."""


@dataclass(frozen=True, slots=True)
class Paths:
    fm_user_dir: Path
    data_dir: Path

    @property
    def games_dir(self) -> Path:
        return self.fm_user_dir / "games"

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
    detail_pass: bool


@dataclass(frozen=True, slots=True)
class GroupB:
    my_academy: bool
    league_ca_above: int
    league_pa_above: int
    any_ca_above: int
    any_pa_above: int
    published_long_edge: int
    candidates: int
    detail_pass: bool


@dataclass(frozen=True, slots=True)
class GroupC:
    league_ca_above: int
    league_pa_above: int
    published_long_edge: int
    candidates: int
    detail_pass: bool


@dataclass(frozen=True, slots=True)
class Groups:
    a: GroupA
    b: GroupB
    c: GroupC


@dataclass(frozen=True, slots=True)
class AgeSettings:
    """Per age table, (first real age, wording key) rows with ages rising (spec section 3)."""

    a_my_club: tuple[tuple[int, str], ...]
    a_other_club: tuple[tuple[int, str], ...]
    b: tuple[tuple[int, str], ...]
    c: tuple[tuple[int, str], ...]

    def table(self, name: str) -> tuple[tuple[int, str], ...]:
        return getattr(self, name)


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


@dataclass(frozen=True, slots=True)
class Schedule:
    poll_seconds: int
    generate_while_fm_running: bool


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
    age: AgeSettings
    reference: ReferenceSettings
    my_club: MyClubSettings
    body: BodyThresholds
    schedule: Schedule
    ui: UI
    values: Mapping[str, Any]
    """Every effective value by dotted key, for display."""
    overridden: frozenset[str]
    """Keys whose value comes from the database rather than the defaults file."""


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Turn nested tables into dotted keys: {"groups": {"a": {"x": 1}}} -> {"groups.a.x": 1}."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def load_defaults(path: Path | None = None) -> dict[str, Any]:
    """The defaults file as dotted keys."""
    with (path or DEFAULTS_PATH).open("rb") as f:
        return flatten(tomllib.load(f))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_override(key: str, value: Any, defaults: Mapping[str, Any]) -> Any:
    """Return value if it may override key, else raise SettingsError.

    The value must have the same type as the default. A list holds the same kind of item
    as its default: text if the default holds text, otherwise whole numbers.
    """
    if key not in defaults:
        raise SettingsError(f"unknown setting {key!r}")
    if key.startswith(_FIXED_PREFIXES):
        raise SettingsError(f"setting {key!r} can only be changed in the defaults file")
    default = defaults[key]
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif _is_int(default):
        ok = _is_int(value)
    elif isinstance(default, float):
        ok = _is_int(value) or isinstance(value, float)
    elif isinstance(default, str):
        ok = isinstance(value, str)
    elif isinstance(default, list):
        item_ok = (
            (lambda v: isinstance(v, str)) if default and isinstance(default[0], str) else _is_int
        )
        ok = isinstance(value, list) and all(item_ok(v) for v in value)
    else:
        ok = False
    if not ok:
        raise SettingsError(
            f"setting {key!r} needs a {type(default).__name__}, got {type(value).__name__}"
        )
    return value


def _range(key: str, value: int, low: int, high: int) -> int:
    if not low <= value <= high:
        raise SettingsError(f"setting {key!r} must be between {low} and {high}, got {value}")
    return value


def build_settings(values: Mapping[str, Any], overridden: frozenset[str] = frozenset()) -> Settings:
    """Build and range-check a Settings from dotted keys."""
    v = values

    def threshold(key: str) -> int:
        return _range(key, v[key], 0, 200)

    def edge(key: str) -> int:
        return _range(key, v[key], 64, 4096)

    def candidates(key: str) -> int:
        return _range(key, v[key], 1, 8)

    data_dir = Path(v["paths.data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    def age_table(name: str) -> tuple[tuple[int, str], ...]:
        prefix = f"age.{name}."
        rows = sorted((k.removeprefix(prefix), v[k]) for k in v if k.startswith(prefix))
        if not rows:
            raise SettingsError(f"age.{name} needs at least one wording")
        for wording, first_age in rows:
            if wording not in WORDINGS:
                raise SettingsError(f"unknown age wording {prefix}{wording}")
            if not _is_int(first_age):
                raise SettingsError(f"setting {prefix}{wording!r} needs an int")
            _range(prefix + wording, first_age, 0, 60)
        firsts = [first_age for _, first_age in rows]
        if firsts != sorted(set(firsts)):
            raise SettingsError(
                f"age.{name}: first ages must rise strictly from the youngest wording to the oldest"
            )
        return tuple((first_age, wording) for wording, first_age in rows)

    age = AgeSettings(**{name: age_table(name) for name in TABLES})

    scale = v["reference.scale"]
    if not 0 < scale <= 1:
        raise SettingsError(f"setting 'reference.scale' must be above 0 and at most 1, got {scale}")
    banned = v["my_club.banned_traits"]
    if any(not isinstance(x, str) or not x.strip() or x != x.strip().lower() for x in banned):
        raise SettingsError("my_club.banned_traits holds lower-case trait names, e.g. 'freckles'")
    # Only sheet traits: anything else would put free text into the negative prompt.
    unknown = sorted(set(banned) - BANNABLE.keys())
    if unknown:
        raise SettingsError(
            f"my_club.banned_traits: unknown traits {unknown}; choose from {sorted(BANNABLE)}"
        )

    body = BodyThresholds(
        short_below_cm=_range("body.short_below_cm", v["body.short_below_cm"], 100, 250),
        tall_from_cm=_range("body.tall_from_cm", v["body.tall_from_cm"], 100, 250),
        slim_strength_at_most=_range(
            "body.slim_strength_at_most", v["body.slim_strength_at_most"], 0, 20
        ),
        muscular_strength_from=_range(
            "body.muscular_strength_from", v["body.muscular_strength_from"], 1, 21
        ),
    )
    if body.short_below_cm > body.tall_from_cm:
        raise SettingsError("body.short_below_cm must not be above body.tall_from_cm")
    if body.slim_strength_at_most >= body.muscular_strength_from:
        raise SettingsError("body.slim_strength_at_most must be below body.muscular_strength_from")

    return Settings(
        paths=Paths(fm_user_dir=Path(v["paths.fm_user_dir"]), data_dir=data_dir),
        eligibility=Eligibility(
            mens_football_only=v["eligibility.mens_football_only"],
            continental_competition_ids=frozenset(v["eligibility.continental_competition_ids"]),
            loaned_out_at_my_club=v["eligibility.loaned_out_at_my_club"],
        ),
        newgen=Newgen(fallback_min_unique_id=v["newgen.fallback_min_unique_id"]),
        groups=Groups(
            a=GroupA(
                ca_above=threshold("groups.a.ca_above"),
                pa_above=threshold("groups.a.pa_above"),
                published_long_edge=edge("groups.a.published_long_edge"),
                candidates=candidates("groups.a.candidates"),
                detail_pass=v["groups.a.detail_pass"],
            ),
            b=GroupB(
                my_academy=v["groups.b.my_academy"],
                league_ca_above=threshold("groups.b.league_ca_above"),
                league_pa_above=threshold("groups.b.league_pa_above"),
                any_ca_above=threshold("groups.b.any_ca_above"),
                any_pa_above=threshold("groups.b.any_pa_above"),
                published_long_edge=edge("groups.b.published_long_edge"),
                candidates=candidates("groups.b.candidates"),
                detail_pass=v["groups.b.detail_pass"],
            ),
            c=GroupC(
                league_ca_above=threshold("groups.c.league_ca_above"),
                league_pa_above=threshold("groups.c.league_pa_above"),
                published_long_edge=edge("groups.c.published_long_edge"),
                candidates=candidates("groups.c.candidates"),
                detail_pass=v["groups.c.detail_pass"],
            ),
        ),
        age=age,
        reference=ReferenceSettings(
            enabled=v["reference.enabled"],
            scale=float(scale),
            other_clubs=v["reference.other_clubs"],
        ),
        my_club=MyClubSettings(
            banned_traits=frozenset(banned),
            soften_skin=_range("my_club.soften_skin", v["my_club.soften_skin"], 0, 2),
        ),
        body=body,
        schedule=Schedule(
            poll_seconds=_range("schedule.poll_seconds", v["schedule.poll_seconds"], 5, 86400),
            generate_while_fm_running=v["schedule.generate_while_fm_running"],
        ),
        ui=UI(host=v["ui.host"], port=_range("ui.port", v["ui.port"], 1, 65535)),
        values=dict(values),
        overridden=overridden,
    )


def read_overrides(conn: sqlite3.Connection) -> dict[str, Any]:
    """Raw overrides from the settings table, without checks."""
    return {
        key: json.loads(value) for key, value in conn.execute("SELECT key, value FROM settings")
    }


def load_settings(
    conn: sqlite3.Connection | None = None, defaults_path: Path | None = None
) -> Settings:
    """Defaults plus any database overrides.

    A stored override that no longer fits (its key was removed from the defaults, or its type
    changed) is skipped with a warning rather than stopping the tool.
    """
    defaults = load_defaults(defaults_path)
    merged = dict(defaults)
    overridden: set[str] = set()
    if conn is not None:
        for key, value in read_overrides(conn).items():
            try:
                merged[key] = check_override(key, value, defaults)
            except SettingsError as exc:
                warnings.warn(f"ignoring stored setting: {exc}", stacklevel=2)
                continue
            overridden.add(key)
    return build_settings(merged, frozenset(overridden))


def set_override(
    conn: sqlite3.Connection, key: str, value: Any, defaults_path: Path | None = None
) -> Settings:
    """Check and store one override; return the settings it produces.

    The whole result is range-checked before anything is written, so an override that
    would leave the settings invalid is refused.
    """
    defaults = load_defaults(defaults_path)
    check_override(key, value, defaults)
    current = load_settings(conn, defaults_path)
    merged = dict(current.values)
    merged[key] = value
    result = build_settings(merged, current.overridden | {key})
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, json.dumps(value), now),
    )
    return result


def reset_overrides(conn: sqlite3.Connection, keys: list[str] | None = None) -> None:
    """Remove overrides, all of them or only the given keys."""
    if keys is None:
        conn.execute("DELETE FROM settings")
    else:
        conn.executemany("DELETE FROM settings WHERE key = ?", [(k,) for k in keys])
