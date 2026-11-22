from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from facegen.persistence import settings as settings_module
from facegen.persistence.db import transaction
from facegen.persistence.settings import (
    DEFAULTS_PATH,
    FOUND,
    NOT_SET,
    PROJECT_ROOT,
    SettingsError,
    build_settings,
    check_override,
    flatten,
    load_defaults,
    load_local,
    load_settings,
    read_overrides,
    reset_overrides,
    set_override,
    set_overrides,
)


def test_defaults_load_into_typed_settings() -> None:
    s = load_settings()
    assert s.cutoffs.everywhere["a_ca"] == 165
    assert s.cutoffs.my_club.academy_alumni is True
    assert s.groups.c.published_long_edge == 512
    assert s.age.b == ((15, "W02"), (19, "W04"), (25, "W07"), (29, "W09"))
    assert (s.reference.enabled, s.reference.scale, s.reference.other_clubs) == (True, 0.1, False)
    assert s.my_club.banned_traits == frozenset({"freckles", "muscular"})
    assert (s.body.short_below_cm, s.body.tall_from_cm) == (174, 191)
    assert (s.body.slim_strength_at_most, s.body.muscular_strength_from) == (4, 14)
    assert s.my_club.soften_skin == 2
    assert s.personality.weight == 2
    assert s.my_club.round_eyes == 0.2
    assert s.age.a_my_club[0] == (15, "W01") and s.age.a_my_club[-1] == (34, "W10")
    assert s.eligibility.mens_football_only is True
    assert s.eligibility.continental_competition_ids == frozenset()
    assert s.overridden == frozenset()


def test_relative_data_dir_resolves_against_the_project() -> None:
    s = load_settings()
    assert s.paths.data_dir == PROJECT_ROOT / "data"
    assert s.paths.db_path == PROJECT_ROOT / "data" / "facegen.db"
    games = s.paths.games_dir
    assert games is None or games.name == "games"


# --- FM27's folders: config/local.toml, then found on this PC, then not set (Phase 8) ---------


@pytest.fixture
def pc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The standard defaults file and a local.toml of our own, on a system of our choice, with
    whatever detection finds in `found`."""
    defaults = tmp_path / "default.toml"
    defaults.write_text(DEFAULTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    local = tmp_path / "local.toml"
    found: dict[str, Path | None] = {}
    asked: list[str] = []

    def detect(key: str) -> Path | None:
        asked.append(key)
        return found.get(key)

    monkeypatch.setattr(settings_module, "DEFAULTS_PATH", defaults)
    monkeypatch.setattr(settings_module, "LOCAL_PATH", local)
    monkeypatch.setattr(settings_module, "detect_folder", detect)
    monkeypatch.setattr(settings_module.platform_info, "current", lambda: "windows")
    return {"local": local, "found": found, "asked": asked, "monkeypatch": monkeypatch}


def test_no_local_file_and_nothing_found_leaves_the_folders_not_set(pc: dict[str, Any]) -> None:
    s = load_settings()
    assert s.paths.fm_user_dir is None and s.paths.fm_install_dir is None
    assert s.paths.games_dir is None and s.paths.graphics_dir is None
    assert s.paths.origins["paths.fm_user_dir"] == NOT_SET
    assert s.paths.describe("paths.fm_user_dir") == "not set, and not found on this PC"
    assert s.values["paths.fm_user_dir"] == ""
    assert pc["asked"] == ["paths.fm_user_dir", "paths.fm_install_dir"]


def test_folders_found_on_this_pc(pc: dict[str, Any]) -> None:
    pc["found"].update(
        {"paths.fm_user_dir": Path("/found/FM27"), "paths.fm_install_dir": Path("/steam/FM27")}
    )
    s = load_settings()
    assert s.paths.fm_user_dir == Path("/found/FM27")
    assert s.paths.games_dir == Path("/found/FM27/games")
    assert s.paths.fm_install_dir == Path("/steam/FM27")
    assert s.paths.origins["paths.fm_user_dir"] == FOUND
    assert s.paths.describe("paths.fm_user_dir") == f"{Path('/found/FM27')} (found on this PC)"


def test_a_folder_in_local_toml_is_not_looked_for(pc: dict[str, Any]) -> None:
    pc["local"].write_text("[paths]\nfm_user_dir = 'D:\\FM27'\n", encoding="utf-8")
    pc["found"]["paths.fm_user_dir"] = Path("/elsewhere")
    s = load_settings()
    assert s.paths.fm_user_dir == Path("D:\\FM27")
    assert s.paths.origins["paths.fm_user_dir"] == "config/local.toml"
    assert pc["asked"] == ["paths.fm_install_dir"]


def test_a_windows_path_in_local_toml_works_from_wsl(pc: dict[str, Any]) -> None:
    pc["monkeypatch"].setattr(settings_module.platform_info, "current", lambda: "wsl")
    pc["local"].write_text(
        "[paths]\nfm_user_dir = 'C:\\Users\\me\\Documents\\Sports Interactive\\Football"
        " Manager 27'\n",
        encoding="utf-8",
    )
    s = load_settings()
    assert s.paths.fm_user_dir == Path(
        "/mnt/c/Users/me/Documents/Sports Interactive/Football Manager 27"
    )


def test_a_wsl_path_in_local_toml_on_windows_is_an_error_that_says_what_to_write(
    pc: dict[str, Any],
) -> None:
    pc["local"].write_text('[paths]\nfm_user_dir = "/mnt/c/Users/me/FM27"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match=r"config/local.toml.*write it as 'C:\\Users"):
        load_settings()


def test_a_windows_path_in_double_quotes_gets_a_hint(pc: dict[str, Any]) -> None:
    """TOML reads \\U in double quotes as an escape: the file cannot be read."""
    pc["local"].write_text('[paths]\nfm_user_dir = "C:\\Users\\me"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="single quotes"):
        load_settings()


def test_local_file_sets_this_pcs_paths(tmp_path: Path) -> None:
    defaults = load_defaults()
    assert load_local(defaults, tmp_path / "missing.toml") == {}
    local = tmp_path / "local.toml"
    local.write_text('[paths]\nfm_user_dir = "/mnt/c/Users/me/FM27"\n', encoding="utf-8")
    assert load_local(defaults, local) == {"paths.fm_user_dir": "/mnt/c/Users/me/FM27"}


@pytest.mark.parametrize(
    "text",
    [
        "[groups.a]\ncandidates = 1\n",  # not a path
        '[paths]\nsomewhere = "x"\n',  # not a key of the defaults
        "[paths]\nfm_user_dir = 3\n",  # not text
    ],
)
def test_local_file_refuses_anything_but_paths(tmp_path: Path, text: str) -> None:
    defaults = load_defaults()
    local = tmp_path / "local.toml"
    local.write_text(text, encoding="utf-8")
    with pytest.raises(SettingsError):
        load_local(defaults, local)


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": [2]}}, "e": True}) == {
        "a.b": 1,
        "a.c.d": [2],
        "e": True,
    }


def test_override_changes_one_value(conn: sqlite3.Connection) -> None:
    result = set_override(conn, "cutoffs.everywhere.a_ca", 150)
    assert result.cutoffs.everywhere["a_ca"] == 150
    again = load_settings(conn)
    assert again.cutoffs.everywhere["a_ca"] == 150
    assert again.cutoffs.everywhere["a_pa"] == 180
    assert again.overridden == frozenset({"cutoffs.everywhere.a_ca"})


def test_a_single_override_stores_an_explicit_default(conn: sqlite3.Connection) -> None:
    result = set_override(conn, "cutoffs.everywhere.a_ca", 165)
    assert result.overridden == frozenset({"cutoffs.everywhere.a_ca"})
    assert read_overrides(conn) == {"cutoffs.everywhere.a_ca": 165}
    assert load_settings(conn).overridden == result.overridden

    set_overrides(conn, {"cutoffs.everywhere.a_ca": 165})
    assert read_overrides(conn) == {}


def test_override_writes_leave_the_transaction_to_the_caller(conn: sqlite3.Connection) -> None:
    set_overrides(conn, {"cutoffs.everywhere.a_ca": 150, "cutoffs.everywhere.a_pa": 160})
    before = list(conn.execute("SELECT * FROM settings ORDER BY key"))

    with transaction(conn, rollback=True):
        set_override(conn, "staff.enabled", False)
        set_overrides(conn, {"cutoffs.everywhere.a_ca": 165, "cutoffs.everywhere.a_pa": 170})
        assert read_overrides(conn) == {"staff.enabled": False, "cutoffs.everywhere.a_pa": 170}

    assert list(conn.execute("SELECT * FROM settings ORDER BY key")) == before


def test_override_of_a_list(conn: sqlite3.Connection) -> None:
    set_override(conn, "eligibility.continental_competition_ids", [1301385, 1301388])
    assert load_settings(conn).eligibility.continental_competition_ids == {1301385, 1301388}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("groups.a.nonsense", 1),  # unknown
        ("cutoffs.everywhere.a_ca", "150"),  # string for int
        ("cutoffs.everywhere.a_ca", True),  # bool is not an int here
        ("cutoffs.everywhere.a_ca", 150.5),  # a number, not a share
        ("cutoffs.share.a_ca", "6"),  # string for a share
        ("cutoffs.share.a_ca", "off"),  # only "-" switches a cell off
        ("groups.a.detail_size", 1),  # int for text
        ("render.vae_tiling", 1),  # int for bool
        ("eligibility.continental_competition_ids", [1, "2"]),
        ("paths.data_dir", "/tmp/elsewhere"),  # paths are fixed
    ],
)
def test_bad_overrides_are_refused(conn: sqlite3.Connection, key: str, value: object) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, key, value)
    assert conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cutoffs.everywhere.a_ca", 201),
        ("cutoffs.everywhere.a_ca", 0),  # at least 1
        ("cutoffs.share.a_ca", 0.05),  # at least 0.1
        ("cutoffs.share.a_ca", 6.25),  # one decimal
        ("cutoffs.share.b_ca", 101),
        ("cutoffs.manual.b_ca", 300),
        ("cutoffs.mode", "sometimes"),
        ("cutoffs.my_club.youth", "d"),
        ("picture.aspect", "16:9"),
        ("age.staff.mode", "spread"),
        ("groups.b.candidates", 0),
        ("groups.c.published_long_edge", 10),
        ("age.b.W04", 15),  # first ages must rise from W02 (15) to W04
        ("age.b.W07", 30),  # past W09 (29)
        ("age.a_my_club.W01", -1),
        ("age.c.W02", "15"),  # must be a whole number
        ("reference.scale", 0),
        ("reference.scale", 1.5),
        ("my_club.soften_skin", 3),
        ("my_club.banned_traits", [1, 2]),  # text, not numbers
        ("my_club.banned_traits", ["Freckles"]),  # lower case
        ("ui.port", 0),
    ],
)
def test_out_of_range_overrides_are_refused(
    conn: sqlite3.Connection, key: str, value: object
) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, key, value)
    assert load_settings(conn).overridden == frozenset()


def test_stale_stored_override_is_skipped_with_a_warning(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, 'now')",
        ("groups.z.removed", json.dumps(3)),
    )
    with pytest.warns(UserWarning, match="ignoring stored setting"):
        s = load_settings(conn)
    assert s.overridden == frozenset()


def test_reset_overrides(conn: sqlite3.Connection) -> None:
    set_override(conn, "cutoffs.everywhere.a_ca", 150)
    set_override(conn, "cutoffs.everywhere.a_pa", 160)
    reset_overrides(conn, ["cutoffs.everywhere.a_ca"])
    assert load_settings(conn).overridden == frozenset({"cutoffs.everywhere.a_pa"})
    reset_overrides(conn)
    assert load_settings(conn).cutoffs.everywhere["a_pa"] == 180


def test_check_override_returns_the_value() -> None:
    defaults = load_defaults()
    assert check_override("schedule.poll_seconds", 30, defaults) == 30


def test_build_settings_with_custom_defaults_file(tmp_path: Path) -> None:
    # UTF-8 both ways: the defaults hold Korean text, which a Korean Windows's own code page
    # (cp949) cannot read as written.
    text = (PROJECT_ROOT / "config" / "default.toml").read_text(encoding="utf-8")
    custom = tmp_path / "custom.toml"
    custom.write_text(
        text.replace('data_dir = "data"', f'data_dir = "{tmp_path.as_posix()}/d"'), encoding="utf-8"
    )
    s = load_settings(defaults_path=custom)
    assert s.paths.data_dir == tmp_path / "d"
    assert build_settings(dict(s.values)).paths.data_dir == tmp_path / "d"


def test_the_content_rule_is_not_a_setting() -> None:
    keys = " ".join(load_defaults())
    for word in ("nsfw", "content", "rating", "negative", "wardrobe", "safe"):
        assert word not in keys


def test_text_list_override(conn: sqlite3.Connection) -> None:
    s = set_override(conn, "my_club.banned_traits", ["freckles", "mole"])
    assert s.my_club.banned_traits == frozenset({"freckles", "mole"})
    s = set_override(conn, "my_club.banned_traits", [])
    assert s.my_club.banned_traits == frozenset()


def test_reference_can_be_switched_off(conn: sqlite3.Connection) -> None:
    s = set_override(conn, "reference.enabled", False)
    assert s.reference.enabled is False
    assert set_override(conn, "reference.scale", 0.2).reference.scale == 0.2


def test_banned_traits_must_be_sheet_traits(conn: sqlite3.Connection) -> None:
    """Free text in the negative prompt could weaken the content rule (AGENTS.md rule 3)."""
    for bad in (["shirt"], ["freckles", "clothes"]):
        with pytest.raises(SettingsError):
            set_override(conn, "my_club.banned_traits", bad)
    s = set_override(conn, "my_club.banned_traits", ["glasses", "mole"])
    assert s.my_club.banned_traits == frozenset({"glasses", "mole"})


def test_body_thresholds_are_checked(conn: sqlite3.Connection) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, "body.short_below_cm", 200)  # above tall_from_cm
    with pytest.raises(SettingsError):
        set_override(conn, "body.slim_strength_at_most", 14)  # not below muscular
    s = set_override(conn, "body.tall_from_cm", 193)
    assert s.body.tall_from_cm == 193


def test_round_eyes_share_is_checked(conn: sqlite3.Connection) -> None:
    for bad in (-0.1, 1.5):
        with pytest.raises(SettingsError):
            set_override(conn, "my_club.round_eyes", bad)
    assert set_override(conn, "my_club.round_eyes", 0).my_club.round_eyes == 0.0
    assert set_override(conn, "my_club.round_eyes", 0.35).my_club.round_eyes == 0.35


def test_personality_weight_is_checked(conn: sqlite3.Connection) -> None:
    for bad in (-1, 11):
        with pytest.raises(SettingsError):
            set_override(conn, "personality.weight", bad)
    assert set_override(conn, "personality.weight", 0).personality.weight == 0


def test_several_overrides_are_checked_together(conn: sqlite3.Connection) -> None:
    from facegen.persistence.settings import set_overrides

    # Each alone would put short_below_cm above tall_from_cm; together they are fine.
    s = set_overrides(conn, {"body.short_below_cm": 195, "body.tall_from_cm": 200})
    assert (s.body.short_below_cm, s.body.tall_from_cm) == (195, 200)
    with pytest.raises(SettingsError):
        set_overrides(conn, {"cutoffs.everywhere.a_ca": 150, "groups.b.candidates": 0})
    # Nothing of a refused batch is stored.
    assert load_settings(conn).cutoffs.everywhere["a_ca"] == 165


def test_entering_the_default_again_clears_the_override(conn: sqlite3.Connection) -> None:
    from facegen.persistence.settings import merge_overrides, set_overrides

    set_overrides(conn, {"cutoffs.everywhere.a_ca": 150, "cutoffs.everywhere.a_pa": 160})
    assert merge_overrides(conn, {"cutoffs.everywhere.a_ca": 165}).overridden == {
        "cutoffs.everywhere.a_pa"
    }
    s = set_overrides(conn, {"cutoffs.everywhere.a_ca": 165})
    assert s.overridden == frozenset({"cutoffs.everywhere.a_pa"})
    assert load_settings(conn).overridden == frozenset({"cutoffs.everywhere.a_pa"})


def test_where_the_ui_listens_is_not_a_ui_setting(conn: sqlite3.Connection) -> None:
    """A browser must not be able to open the tool to the network."""
    from facegen.persistence.settings import is_fixed

    for key, value in (("ui.host", "0.0.0.0"), ("ui.port", 9000)):
        assert is_fixed(key)
        with pytest.raises(SettingsError, match="defaults file"):
            set_override(conn, key, value)


def test_published_sizes() -> None:
    assert load_settings().groups.published_sizes() == {"a": 1024, "b": 768, "c": 512}


def test_auto_groups(conn: sqlite3.Connection) -> None:
    # Phase 4 switched groups B and C on.
    assert load_settings().schedule.auto_groups == frozenset({"a", "b", "c"})
    assert set_override(conn, "schedule.auto_groups", ["a"]).schedule.auto_groups == {"a"}
    assert set_override(conn, "schedule.auto_groups", []).schedule.auto_groups == frozenset()
    with pytest.raises(SettingsError):
        set_override(conn, "schedule.auto_groups", ["d"])


# --- Phase 10: the detail pass's size, VAE tiling, the local file beside --config -------------


def test_the_detail_pass_size_and_vae_tiling_defaults() -> None:
    s = load_settings()
    assert s.groups.a.detail_size() == (1344, 1728)
    assert s.groups.a.detail_size("1:1") == (1536, 1536)
    assert s.groups.b.detail_size() is None and s.groups.c.detail_size("1:1") is None
    assert s.render.vae_tiling is False
    assert "groups.a.detail_pass" not in s.values


@pytest.mark.parametrize(
    ("text", "size", "square"),
    [("x1.5", (1344, 1728), (1536, 1536)), ("smaller", (1152, 1472), (1280, 1280)),
     ("off", None, None)],
)  # fmt: skip
def test_each_detail_size(
    conn: sqlite3.Connection, text: str, size: object, square: object
) -> None:
    """Phase 16g: named by scale and resolved per picture shape."""
    for group in ("a", "b", "c"):
        spec = getattr(set_override(conn, f"groups.{group}.detail_size", text).groups, group)
        assert spec.detail_size("3:4") == size
        assert spec.detail_size("1:1") == square


@pytest.mark.parametrize("text", ["1536x1536", "1344x1728", "X1.5", "", "on"])
def test_a_detail_size_outside_the_choices_is_refused(conn: sqlite3.Connection, text: str) -> None:
    with pytest.raises(SettingsError, match="must be one of x1.5, smaller, off"):
        set_override(conn, "groups.a.detail_size", text)
    assert conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0


def test_the_cutoff_cells_take_a_share_a_number_or_off(conn: sqlite3.Connection) -> None:
    """Phase 16: shares 0.1-100 with one decimal, numbers 1-200, "-" off."""
    s = set_overrides(
        conn,
        {
            "cutoffs.share.a_ca": 12.5,
            "cutoffs.share.b_ca": 30,  # a whole share is stored as given
            "cutoffs.share.c_ca": "-",
            "cutoffs.share.c_pa": 10.0,
            "cutoffs.manual.a_ca": 99,
            "cutoffs.everywhere.a_pa": "-",
            "cutoffs.mode": "manual",
            "cutoffs.my_club.loans": "c",
            "picture.aspect": "1:1",
        },
    )
    assert s.cutoffs.share["a_ca"] == 12.5 and s.cutoffs.share["b_ca"] == 30.0
    assert s.cutoffs.share["c_ca"] is None and s.cutoffs.share["c_pa"] == 10.0
    assert s.cutoffs.manual["a_ca"] == 99 and s.cutoffs.everywhere["a_pa"] is None
    assert s.cutoffs.mode == "manual" and s.cutoffs.my_club.loans == "c"
    assert s.picture.aspect == "1:1"
    assert load_settings(conn).cutoffs == s.cutoffs


def test_an_explicit_defaults_file_takes_the_local_file_beside_it(tmp_path: Path) -> None:
    from facegen.persistence.settings import local_label, local_path

    config = tmp_path / "test.toml"
    config.write_text(DEFAULTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    assert local_path(config) == tmp_path / "local.toml"
    assert local_path() == settings_module.LOCAL_PATH
    assert load_settings(defaults_path=config).paths.fm_user_dir is None
    (tmp_path / "local.toml").write_text(
        f"[paths]\nfm_user_dir = '{tmp_path / 'FM27'}'\n", encoding="utf-8"
    )
    s = load_settings(defaults_path=config)
    assert s.paths.fm_user_dir == tmp_path / "FM27"
    assert (
        s.paths.origins["paths.fm_user_dir"] == local_label(config) == str(tmp_path / "local.toml")
    )


# --- Phase 12: every age table offers all ten wordings; "-" leaves one out -------------------


def test_a_left_out_wording_is_skipped_and_can_be_switched_on(conn: sqlite3.Connection) -> None:
    before = load_settings(conn).age.b
    assert [w for _, w in before] == ["W02", "W04", "W07", "W09"]
    after = set_override(conn, "age.b.W03", 17).age.b  # between W02 (15) and W04 (19)
    assert after[:3] == ((15, "W02"), (17, "W03"), (19, "W04"))
    assert set_override(conn, "age.b.W03", "-").age.b == before  # left out again


def test_my_clubs_table_has_w06_back(conn: sqlite3.Connection) -> None:
    """User, 2026-11-17: W05 from 24 and W06 at 26 at my club; W06 was unused there."""
    table = {w: first for first, w in load_settings(conn).age.a_my_club}
    assert (table["W04"], table["W05"], table["W06"], table["W07"]) == (22, 24, 26, 27)


@pytest.mark.parametrize("value", ["15", "skip", 15.0, None, True])
def test_a_wording_takes_a_whole_number_or_a_dash_only(
    conn: sqlite3.Connection, value: object
) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, "age.c.W02", value)


def test_a_table_needs_one_wording(conn: sqlite3.Connection) -> None:
    with pytest.raises(SettingsError, match="at least one wording"):
        set_overrides(conn, {"age.c.W03": "-", "age.c.W06": "-", "age.c.W08": "-"})


# --- Phase 15: one model and one gender for everyone (user, 2026-11-20) -------------------------


def test_the_style_defaults_draw_anime_women() -> None:
    """Animagine and women until the user chooses otherwise, so nothing changes on update."""
    s = load_settings()
    assert (s.style.model, s.style.gender) == ("animagine", "woman")
    assert {"style.model", "style.gender"} <= set(s.values)


@pytest.mark.parametrize("model", ["animagine", "realvis", "zimage_turbo"])
@pytest.mark.parametrize("gender", ["woman", "man"])
def test_each_model_and_gender_can_be_chosen(
    conn: sqlite3.Connection, model: str, gender: str
) -> None:
    s = set_overrides(conn, {"style.model": model, "style.gender": gender})
    assert (s.style.model, s.style.gender) == (model, gender)
    assert load_settings(conn).style == s.style  # stored, read back the same
    # Only what differs from the defaults is stored as an override.
    changed = {"style.model"} if model != "animagine" else set()
    changed |= {"style.gender"} if gender != "woman" else set()
    assert s.overridden == frozenset(changed)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("style.model", "flux"),
        ("style.model", "RealVis"),  # keys are exact
        ("style.model", "RealVisXL V5.0"),  # the label is not the key
        ("style.model", ""),
        ("style.model", 1),
        ("style.model", None),
        ("style.gender", "other"),
        ("style.gender", "Man"),
        ("style.gender", "men"),
        ("style.gender", ["woman", "man"]),  # one gender for everyone
    ],
)
def test_a_style_outside_the_choices_is_refused(
    conn: sqlite3.Connection, key: str, value: object
) -> None:
    with pytest.raises(SettingsError):
        set_override(conn, key, value)
    assert conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0


def test_a_style_change_is_one_transaction(conn: sqlite3.Connection) -> None:
    """Model and gender are saved together or not at all: a bad gender keeps the model."""
    with pytest.raises(SettingsError):
        set_overrides(conn, {"style.model": "realvis", "style.gender": "other"})
    assert load_settings(conn).style.model == "animagine"
    assert conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0
