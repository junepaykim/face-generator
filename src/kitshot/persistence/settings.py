"""Settings: defaults from config/default.toml, overrides from the state database.

Read settings only through `load_settings`. The UI saves through `settings_history`.
The content rule (docs/spec.md section 3) is code, not a setting, and has no key here.
"""

from __future__ import annotations

import json
import sqlite3
import tomllib
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kitshot.core.age import NOT_USED, TABLES, WORDINGS, spread_table
from kitshot.core.cutoffs import LEAGUE_CELLS, OFF, OTHER_CELLS
from kitshot.core.settings_values import (
    AGE_MODES,
    ASPECT_CHOICES,
    CANDIDATE_RANGE,
    CUTOFF_MODES,
    DETAIL_SIZES,
    EDGE_RANGE,
    FOUND,
    GENDER_CHOICES,
    GROUP_NAMES,
    LANGUAGE_CHOICES,
    MODEL_CHOICES,
    NOT_SET,
    SOFTEN_SKIN_STEPS,
    THEME_CHOICES,
    TIER_CHOICES,
    UI,
    AgeSettings,
    AgeSpread,
    CutoffSettings,
    DisplaySettings,
    Eligibility,
    Groups,
    GroupSpec,
    MediaSettings,
    MyClubSettings,
    MyClubTiers,
    Newgen,
    NoPhotoSettings,
    NotificationSettings,
    Paths,
    PersonalitySettings,
    PictureSettings,
    ReferenceSettings,
    RenderSettings,
    Schedule,
    Settings,
    StaffSettings,
    StyleSettings,
    UpdateSettings,
)
from kitshot.core.traits import BANNABLE, BodyThresholds
from kitshot.locations import CONFIG_DIR, PROJECT_ROOT
from kitshot.persistence.settings_store import write_overrides
from kitshot.platform import fm_paths, platform_info

DEFAULTS_PATH = CONFIG_DIR / "default.toml"
LOCAL_PATH = CONFIG_DIR / "local.toml"
"""This PC's own folders, laid over the defaults' `[paths]`. Optional and not in git, so the
repository holds no Windows user name. The first-run setup writes FM27's folder here."""
LOCAL_NAME = "local.toml"

# Keys the UI may not override: where things live on disk, and where the web UI listens (a
# browser must not be able to open the tool to the network; AGENTS.md hard rule 6).
_FIXED_PREFIXES = ("paths.", "ui.")

SKIP_WORDING = NOT_USED
"""An age table's value for a wording it does not use (Phase 12, user 2026-11-17)."""
SPREAD_KEYS = frozenset({"mode", "from", "to", "start", "end"})
"""Each age table's keys for its auto mode, beside the looks (staff Phase 16f, all Phase 17l)."""
CELL_PREFIXES = ("cutoffs.share.", "cutoffs.manual.", "cutoffs.everywhere.")
"""Cutoff cells: a share or a number, or OFF ("-") to switch the cell off (Phase 16)."""

DETECTED_KEYS = ("paths.fm_user_dir", "paths.fm_install_dir")
"""Folders that are looked for on this PC (fm_paths.py) when no file sets them."""


class SettingsError(ValueError):
    """A setting is unknown, has the wrong type, or is out of range."""


def folder_not_set_message(kind: platform_info.Kind | None = None) -> str:
    """What to tell the user when FM27's user folder is neither set nor found."""
    kind = kind or platform_info.current()
    example = (
        "/home/you/Football Manager 27"
        if kind == "linux"
        else r"C:\Users\you\Documents\Sports Interactive\Football Manager 27"
    )
    return (
        "FM27's folder is not set and was not found on this PC (it is the folder that holds"
        " FM27's games and graphics folders). Put it in config/local.toml as"
        f" [paths] fm_user_dir = '{example}', in single quotes, then start Kitshot again."
    )


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


def detect_folder(key: str) -> Path | None:
    """FM27's user or install folder on this PC (fm_paths.py), by setting key; the tests
    replace it."""
    if key == "paths.fm_user_dir":
        return fm_paths.detect_user_dir()
    return fm_paths.detect_install_dir()


def local_path(defaults_path: Path | None = None) -> Path:
    """The local file laid over a defaults file: config/local.toml for the standard defaults,
    and for a defaults file given explicitly (`--config`, the tests) the local.toml beside it."""
    return LOCAL_PATH if defaults_path is None else defaults_path.with_name(LOCAL_NAME)


def local_label(defaults_path: Path | None = None) -> str:
    """How the local file is named to the user: "config/local.toml", or its full path."""
    return f"config/{LOCAL_PATH.name}" if defaults_path is None else str(local_path(defaults_path))


def load_defaults(path: Path | None = None) -> dict[str, Any]:
    """The defaults file as dotted keys.

    The local file (`local_path`: config/local.toml, or the local.toml beside a defaults file
    given explicitly), if present, is laid over it. FM27's folders that neither sets (an empty
    value) are looked for on this PC (fm_paths.py); one not found stays empty, "not set".
    """
    return _load_defaults(path)[0]


def _load_defaults(path: Path | None) -> tuple[dict[str, Any], dict[str, str]]:
    """The defaults as dotted keys, and where each folder in `[paths]` came from."""
    source = path or DEFAULTS_PATH
    with source.open("rb") as f:
        values = flatten(tomllib.load(f))
    named = f"config/{source.name}" if path is None else str(source)
    origins = {k: named for k in values if k.startswith("paths.") and values[k]}
    local = load_local(values, local_path(path))
    values.update(local)
    origins.update({k: local_label(path) for k, v in local.items() if v})
    kind = platform_info.current()
    for key in [k for k in values if k.startswith("paths.")]:
        if not isinstance(values[key], str):
            raise SettingsError(f"{key} needs a str, got {type(values[key]).__name__}")
        try:
            values[key] = fm_paths.native_path(values[key], kind)
        except fm_paths.PathStyleError as exc:
            raise SettingsError(f"{origins.get(key, source.name)}, {key}: {exc}") from exc
    for key in DETECTED_KEYS:
        if key in values and not values[key]:
            folder = detect_folder(key)
            values[key] = str(folder) if folder is not None else ""
            origins[key] = FOUND if folder is not None else NOT_SET
    return values, origins


TOML_PATH_HINT = (
    "Write a Windows path in single quotes, which keep backslashes as they are:"
    " fm_user_dir = 'C:\\Users\\you\\Documents\\Sports Interactive\\Football Manager 27'"
)


def load_local(defaults: Mapping[str, Any], path: Path = LOCAL_PATH) -> dict[str, Any]:
    """config/local.toml as dotted keys, or nothing if there is none.

    It may set only `[paths]` keys that the defaults have, each as text.
    """
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        try:
            local = flatten(tomllib.load(f))
        except tomllib.TOMLDecodeError as exc:
            raise SettingsError(f"{path.name} cannot be read: {exc}. {TOML_PATH_HINT}") from exc
    for key, value in local.items():
        if not key.startswith("paths.") or key not in defaults:
            raise SettingsError(f"{path.name} may only set the defaults' [paths] keys, not {key!r}")
        if not isinstance(value, str):
            raise SettingsError(f"{path.name}: {key!r} needs a str, got {type(value).__name__}")
    return local


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
    if is_age_wording(key):
        ok = _is_int(value) or value == SKIP_WORDING
    elif is_cut_cell(key):
        share = key.startswith("cutoffs.share.")
        ok = _is_int(value) or (share and isinstance(value, float)) or value == OFF
    elif isinstance(default, bool):
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


def build_settings(
    values: Mapping[str, Any],
    overridden: frozenset[str] = frozenset(),
    origins: Mapping[str, str] | None = None,
) -> Settings:
    """Build and range-check a Settings from dotted keys. `origins` says where each folder in
    `[paths]` came from (`_load_defaults`)."""
    v = values

    def edge(key: str) -> int:
        return _range(key, v[key], *EDGE_RANGE)

    def candidates(key: str) -> int:
        return _range(key, v[key], *CANDIDATE_RANGE)

    def detail(key: str) -> str:
        return one_of(key, tuple(DETAIL_SIZES))

    def group(name: str) -> GroupSpec:
        return GroupSpec(
            published_long_edge=edge(f"groups.{name}.published_long_edge"),
            candidates=candidates(f"groups.{name}.candidates"),
            detail=detail(f"groups.{name}.detail_size"),
        )

    def share(key: str) -> float | None:
        value = v[key]
        if value == OFF:
            return None
        if not (_is_int(value) or isinstance(value, float)):
            raise SettingsError(f"setting {key!r} needs a share or {OFF!r}")
        tenths = float(value) * 10
        if not 1 <= tenths <= 1000 or abs(tenths - round(tenths)) > 1e-6:
            raise SettingsError(
                f"setting {key!r} must be a share from 0.1 to 100 with one decimal, got {value}"
            )
        return round(tenths) / 10

    def number(key: str) -> int | None:
        value = v[key]
        if value == OFF:
            return None
        if not _is_int(value):
            raise SettingsError(f"setting {key!r} needs a whole number or {OFF!r}")
        return _range(key, value, 1, 200)

    def one_of(key: str, choices: tuple[str, ...]) -> str:
        if v[key] not in choices:
            raise SettingsError(
                f"setting {key!r} must be one of {', '.join(choices)}, got {v[key]!r}"
            )
        return v[key]

    data_dir = Path(v["paths.data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    def age_table(name: str) -> tuple[tuple[int, str], ...]:
        prefix = f"age.{name}."
        rows = sorted(
            (k.removeprefix(prefix), v[k])
            for k in v
            if k.startswith(prefix)
            and v[k] != SKIP_WORDING
            and k.removeprefix(prefix) not in SPREAD_KEYS
        )
        if not rows:
            raise SettingsError(f"age.{name} needs at least one wording")
        for wording, first_age in rows:
            if wording not in WORDINGS:
                raise SettingsError(f"unknown age wording {prefix}{wording}")
            if not _is_int(first_age):
                raise SettingsError(f"setting {prefix}{wording!r} needs an int or {SKIP_WORDING!r}")
            _range(prefix + wording, first_age, 0, 100)
        firsts = [first_age for _, first_age in rows]
        if firsts != sorted(set(firsts)):
            raise SettingsError(
                f"age.{name}: first ages must rise strictly from the youngest wording to the oldest"
            )
        return tuple((first_age, wording) for wording, first_age in rows)

    # Every table keeps its looks (Custom) and, in auto mode, is the spread instead (Phase 17l).
    tables = {name: age_table(name) for name in TABLES}
    spreads: dict[str, AgeSpread] = {}
    looks = tuple(WORDINGS)
    for name in TABLES:
        prefix = f"age.{name}."
        if one_of(prefix + "mode", AGE_MODES) != "auto":
            continue
        spread = AgeSpread(
            first=one_of(prefix + "from", looks),
            last=one_of(prefix + "to", looks),
            start=_range(prefix + "start", v[prefix + "start"], 0, 100),
            end=_range(prefix + "end", v[prefix + "end"], 0, 100),
        )
        try:
            tables[name] = spread_table(spread.first, spread.last, spread.start, spread.end)
        except ValueError as exc:
            raise SettingsError(f"age.{name}: {exc}") from exc
        spreads[name] = spread
    age = AgeSettings(**tables, spreads=spreads)

    scale = v["reference.scale"]
    if not 0 < scale <= 1:
        raise SettingsError(f"setting 'reference.scale' must be above 0 and at most 1, got {scale}")
    round_eyes = v["my_club.round_eyes"]
    if not 0 <= round_eyes <= 1:
        raise SettingsError(
            f"setting 'my_club.round_eyes' must be between 0 and 1, got {round_eyes}"
        )
    banned = v["my_club.banned_traits"]
    if any(not isinstance(x, str) or not x.strip() or x != x.strip().lower() for x in banned):
        raise SettingsError("my_club.banned_traits holds lower-case trait names, e.g. 'freckles'")
    # Only sheet traits: anything else would put free text into the negative prompt.
    unknown = sorted(set(banned) - BANNABLE.keys())
    if unknown:
        raise SettingsError(
            f"my_club.banned_traits: unknown traits {unknown}; choose from {sorted(BANNABLE)}"
        )

    auto_groups = v["schedule.auto_groups"]
    if any(g not in GROUP_NAMES for g in auto_groups):
        raise SettingsError(f"schedule.auto_groups lists groups from {list(GROUP_NAMES)}")
    auto_groups = frozenset(auto_groups)

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
        paths=Paths(
            fm_user_dir=Path(v["paths.fm_user_dir"]) if v["paths.fm_user_dir"] else None,
            data_dir=data_dir,
            fm_install_dir=Path(v["paths.fm_install_dir"]) if v["paths.fm_install_dir"] else None,
            origins=dict(origins or {}),
        ),
        eligibility=Eligibility(
            mens_football_only=v["eligibility.mens_football_only"],
            continental_competition_ids=frozenset(v["eligibility.continental_competition_ids"]),
            loaned_out_at_my_club=v["eligibility.loaned_out_at_my_club"],
        ),
        newgen=Newgen(fallback_min_unique_id=v["newgen.fallback_min_unique_id"]),
        groups=Groups(a=group("a"), b=group("b"), c=group("c")),
        cutoffs=CutoffSettings(
            mode=one_of("cutoffs.mode", CUTOFF_MODES),
            light_above=v["cutoffs.light_above"],
            light_below=v["cutoffs.light_below"],
            my_club=MyClubTiers(
                first_team=one_of("cutoffs.my_club.first_team", TIER_CHOICES),
                youth=one_of("cutoffs.my_club.youth", TIER_CHOICES),
                loans=one_of("cutoffs.my_club.loans", TIER_CHOICES),
                academy_alumni=v["cutoffs.my_club.academy_alumni"],
            ),
            share={c: share(f"cutoffs.share.{c}") for c in (*LEAGUE_CELLS, *OTHER_CELLS)},
            manual={c: number(f"cutoffs.manual.{c}") for c in LEAGUE_CELLS},
            everywhere={c: number(f"cutoffs.everywhere.{c}") for c in LEAGUE_CELLS},
        ),
        staff=StaffSettings(enabled=v["staff.enabled"]),
        no_photo=NoPhotoSettings(enabled=v["no_photo.enabled"]),
        media=MediaSettings(
            journalists=v["media.journalists"], press_officers=v["media.press_officers"]
        ),
        age=age,
        reference=ReferenceSettings(
            enabled=v["reference.enabled"],
            scale=float(scale),
            other_clubs=v["reference.other_clubs"],
        ),
        my_club=MyClubSettings(
            banned_traits=frozenset(banned),
            soften_skin=_range(
                "my_club.soften_skin",
                v["my_club.soften_skin"],
                SOFTEN_SKIN_STEPS[0],
                SOFTEN_SKIN_STEPS[-1],
            ),
            round_eyes=float(round_eyes),
        ),
        body=body,
        personality=PersonalitySettings(
            weight=_range("personality.weight", v["personality.weight"], 0, 10)
        ),
        schedule=Schedule(
            poll_seconds=_range("schedule.poll_seconds", v["schedule.poll_seconds"], 5, 86400),
            generate_while_fm_running=v["schedule.generate_while_fm_running"],
            auto_groups=auto_groups,
        ),
        render=RenderSettings(vae_tiling=v["render.vae_tiling"]),
        picture=PictureSettings(aspect=one_of("picture.aspect", ASPECT_CHOICES)),
        style=StyleSettings(
            model=one_of("style.model", MODEL_CHOICES),
            gender=one_of("style.gender", GENDER_CHOICES),
        ),
        display=DisplaySettings(
            language=one_of("display.language", LANGUAGE_CHOICES),
            theme=one_of("display.theme", THEME_CHOICES),
        ),
        notifications=NotificationSettings(new_pictures=v["notifications.new_pictures"]),
        updates=UpdateSettings(check=v["updates.check"]),
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
    defaults, origins = _load_defaults(defaults_path)
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
    return build_settings(merged, frozenset(overridden), origins)


def load_saved_settings(defaults_path: Path | None = None) -> Settings:
    """The settings with the overrides saved so far, for commands that otherwise run without
    the database (`kitshot models`, `kitshot launch`). It is never created here: before
    Kitshot's first start there is none, and the defaults alone apply."""
    from kitshot.persistence import db

    settings = load_settings(defaults_path=defaults_path)
    if not settings.paths.db_path.is_file():
        return settings
    conn = db.connect(settings.paths.db_path)
    try:
        return load_settings(conn, defaults_path)
    finally:
        conn.close()


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
    result = build_settings(merged, current.overridden | {key}, current.paths.origins)
    now = datetime.now(UTC).isoformat()
    write_overrides(conn, {key: value}, updated_at=now)
    return result


def is_age_wording(key: str) -> bool:
    """An age table's first age for a wording ("age.b.W04"): a whole number, or SKIP_WORDING."""
    return key.startswith("age.") and key.rpartition(".")[2] in WORDINGS


def is_cut_cell(key: str) -> bool:
    """A cutoff cell ("cutoffs.share.a_ca"): a share or a number, or OFF to switch it off."""
    return key.startswith(CELL_PREFIXES)


def is_fixed(key: str) -> bool:
    """Whether a key can only be changed in the defaults file, not from the UI."""
    return key.startswith(_FIXED_PREFIXES)


def merge_overrides(
    conn: sqlite3.Connection, changes: Mapping[str, Any], defaults_path: Path | None = None
) -> Settings:
    """The settings that storing `changes` would give, without storing anything.

    Every value is checked, then the whole result is range-checked, so changes that only
    make sense together (say, both body height cut-offs) are judged together. A value equal
    to its default clears that override.
    """
    defaults = load_defaults(defaults_path)
    for key, value in changes.items():
        check_override(key, value, defaults)
    current = load_settings(conn, defaults_path)
    merged = {**current.values, **changes}
    overridden = set(current.overridden) | set(changes)
    overridden -= {
        k for k, v in changes.items() if v == defaults[k] and type(v) is type(defaults[k])
    }
    return build_settings(merged, frozenset(overridden), current.paths.origins)


def set_overrides(
    conn: sqlite3.Connection, changes: Mapping[str, Any], defaults_path: Path | None = None
) -> Settings:
    """Check and store several overrides at once (the settings page); return the result.

    Nothing is written unless every change is valid. A value equal to its default removes
    the override, so "reset" is just entering the default again.
    """
    result = merge_overrides(conn, changes, defaults_path)
    now = datetime.now(UTC).isoformat()
    write_overrides(
        conn,
        {key: value if key in result.overridden else None for key, value in changes.items()},
        updated_at=now,
    )
    return result


def reset_overrides(conn: sqlite3.Connection, keys: list[str] | None = None) -> None:
    """Remove overrides, all of them or only the given keys."""
    if keys is None:
        conn.execute("DELETE FROM settings")
    else:
        conn.executemany("DELETE FROM settings WHERE key = ?", [(k,) for k in keys])
