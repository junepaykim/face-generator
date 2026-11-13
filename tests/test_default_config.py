"""Guards config/default.toml: it must parse and stay consistent with docs/spec.md."""

import tomllib
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.toml"


def load() -> dict:
    with CONFIG.open("rb") as f:
        return tomllib.load(f)


def test_three_groups_with_valid_thresholds():
    groups = load()["groups"]
    assert set(groups) == {"a", "b", "c"}
    for name, group in groups.items():
        thresholds = {k: v for k, v in group.items() if k.endswith("_above")}
        assert thresholds, f"group {name} has no thresholds"
        for key, value in thresholds.items():
            assert isinstance(value, int) and 1 <= value <= 200, f"{name}.{key}"


def test_default_bars_descend_from_a_to_c():
    g = load()["groups"]
    ca = [
        g["a"]["ca_above"],
        g["b"]["any_ca_above"],
        g["b"]["league_ca_above"],
        g["c"]["league_ca_above"],
    ]
    pa = [
        g["a"]["pa_above"],
        g["b"]["any_pa_above"],
        g["b"]["league_pa_above"],
        g["c"]["league_pa_above"],
    ]
    assert ca == sorted(ca, reverse=True) and len(set(ca)) == len(ca)
    assert pa == sorted(pa, reverse=True) and len(set(pa)) == len(pa)


def test_academy_newgens_fall_back_to_group_b():
    g = load()["groups"]
    assert "my_academy" not in g["a"]
    assert g["b"]["my_academy"] is True


def test_age_tables_match_spec():
    """Spec section 3: first real age per wording, per table (user, 2026-11-11)."""
    age = load()["age"]
    assert age == {
        "a_my_club": {"W01": 15, "W02": 17, "W03": 19, "W04": 22, "W05": 25, "W07": 27,
                      "W08": 29, "W09": 31, "W10": 34},
        "a_other_club": {"W02": 15, "W03": 19, "W04": 22, "W07": 25, "W09": 29, "W10": 34},
        "b": {"W02": 15, "W04": 19, "W07": 25, "W09": 29},
        "c": {"W03": 15, "W06": 22, "W08": 28},
        # My club's staff (user, 2026-11-12): wide bands, W02 the youngest.
        "staff": {"W02": 18, "W03": 25, "W04": 32, "W05": 39, "W06": 46, "W07": 53,
                  "W08": 61, "W09": 68, "W10": 75},
    }  # fmt: skip


def test_paths_name_no_windows_user():
    """This PC's real folder goes in config/local.toml, which is not in git (user, 2026-11-12)."""
    fm_user_dir = load()["paths"]["fm_user_dir"]
    assert fm_user_dir.startswith("/mnt/c/Users/<Windows user>/")


def test_my_clubs_staff_are_drawn_by_default():
    assert load()["staff"] == {"enabled": True}


def test_pre_existing_players_without_a_photo_are_picked_by_default():
    """User, 2026-11-12: include them, by default."""
    assert load()["no_photo"] == {"enabled": True}


def test_the_install_folder_names_no_windows_user():
    assert "/Users/" not in load()["paths"]["fm_install_dir"]
