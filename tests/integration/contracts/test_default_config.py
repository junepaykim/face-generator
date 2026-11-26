# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Guards config/default.toml: it must parse and stay consistent with docs/spec.md."""

import tomllib

from support.paths import ROOT

CONFIG = ROOT / "config" / "default.toml"


def load() -> dict:
    with CONFIG.open("rb") as f:
        return tomllib.load(f)


def test_three_groups_draw_and_the_cutoffs_pick():
    """Phase 16 (user, 2026-11-20): a group keeps how it is drawn (height, candidates, detail
    pass); who is in it is [cutoffs]."""
    groups = load()["groups"]
    assert set(groups) == {"a", "b", "c"}
    for name, group in groups.items():
        assert set(group) == {"published_long_edge", "candidates", "detail_size"}, name
    cutoffs = load()["cutoffs"]
    cells = ("a_ca", "a_pa", "b_ca", "b_pa", "c_ca", "c_pa")
    others = ("above_b_ca", "above_c_ca", "below_c_ca", "continental_b_ca")
    assert set(cutoffs["share"]) == {*cells, *others}
    assert set(cutoffs["manual"]) == set(cells) == set(cutoffs["everywhere"])
    for key, value in cutoffs["share"].items():
        assert value == "-" or (isinstance(value, float) and 0.1 <= value <= 100), key
    for kind in ("manual", "everywhere"):
        for key, value in cutoffs[kind].items():
            assert value == "-" or (isinstance(value, int) and 1 <= value <= 200), (kind, key)


def test_the_users_cutoff_table():
    """The user's table (2026-11-20, "at least" 2026-11-21, new numbers 2026-11-22): Detailed my
    league CA or PA top 2.5%, everywhere CA 165 or PA 180; Standard my league CA or PA top 12%,
    the league above CA top 15%, a continental competition CA top 30%, everywhere CA 145 or PA
    160; Light my league CA top 80%, the league above CA top 50% (on) and the league below CA top
    50% (off), each its own switch; no global cut."""
    c = load()["cutoffs"]
    assert c["mode"] == "auto"
    assert (c["light_above"], c["light_below"]) == (True, False)
    assert "adjacent_light" not in c
    assert c["share"] == {
        "a_ca": 2.5, "a_pa": 2.5, "b_ca": 12.0, "b_pa": 12.0, "c_ca": 80.0, "c_pa": "-",
        "above_b_ca": 15.0, "above_c_ca": 50.0, "below_c_ca": 50.0, "continental_b_ca": 30.0,
    }  # fmt: skip
    assert c["everywhere"] == {
        "a_ca": 165, "a_pa": 180, "b_ca": 145, "b_pa": 160, "c_ca": "-", "c_pa": "-",
    }  # fmt: skip
    assert c["my_club"] == {
        "first_team": "a", "youth": "b", "loans": "b", "academy_alumni": True,
    }  # fmt: skip


def test_default_cuts_descend_from_a_to_c():
    c = load()["cutoffs"]
    everywhere = [c["everywhere"][k] for k in ("a_ca", "b_ca")]
    assert everywhere == sorted(everywhere, reverse=True) and len(set(everywhere)) == 2
    everywhere = [c["everywhere"][k] for k in ("a_pa", "b_pa")]
    assert everywhere == sorted(everywhere, reverse=True) and len(set(everywhere)) == 2
    shares = [c["share"][k] for k in ("a_ca", "b_ca", "c_ca")]
    assert shares == sorted(shares) and len(set(shares)) == 3  # a wider share picks more
    manual = [c["manual"][k] for k in ("b_ca", "c_ca")]
    assert manual == sorted(manual, reverse=True)
    manual = [c["manual"][k] for k in ("b_pa", "c_pa")]
    assert manual == sorted(manual, reverse=True)


def test_academy_newgens_fall_back_to_group_b():
    assert load()["cutoffs"]["my_club"]["academy_alumni"] is True
    assert "my_academy" not in load()["groups"]["b"]


def test_age_tables_match_spec():
    """Spec section 3: first real age per wording, per table (user, 2026-11-11). Every table
    offers all ten wordings, "-" for one it leaves out; W06 back at my club at 26, W05 from 24
    (user, 2026-11-17). My club's staff spread looks over an age span instead (Phase 16f): W02 to
    W10 over 28 to 64, on by default (user, 2026-11-22); the manual table stays below it. Every
    table can be spread (Phase 17l, D9): the players' tables stay Custom ("manual") with their
    looks, the media table is Auto over the measured ages, 30 to 60 (user, 2026-11-22)."""
    looks = {f"W{n:02d}" for n in range(1, 11)}
    tables = load()["age"]
    age = {
        table: {w: first for w, first in rows.items() if w in looks and first != "-"}
        for table, rows in tables.items()
    }
    assert all(len([w for w in rows if w in looks]) == 10 for rows in tables.values())
    spreads = {
        name: {k: v for k, v in rows.items() if k not in looks} for name, rows in tables.items()
    }
    assert spreads == {
        "a_my_club": {"mode": "manual", "from": "W01", "to": "W10", "start": 16, "end": 34},
        "a_other_club": {"mode": "manual", "from": "W02", "to": "W10", "start": 18, "end": 34},
        "b": {"mode": "manual", "from": "W02", "to": "W09", "start": 18, "end": 29},
        "c": {"mode": "manual", "from": "W03", "to": "W08", "start": 21, "end": 28},
        "staff": {"mode": "auto", "from": "W02", "to": "W10", "start": 28, "end": 64},
        "media": {"mode": "auto", "from": "W04", "to": "W10", "start": 30, "end": 60},
    }  # fmt: skip
    assert age == {
        "a_my_club": {"W01": 15, "W02": 17, "W03": 19, "W04": 22, "W05": 24, "W06": 26,
                      "W07": 27, "W08": 29, "W09": 31, "W10": 34},
        "a_other_club": {"W02": 15, "W03": 19, "W04": 22, "W07": 25, "W09": 29, "W10": 34},
        "b": {"W02": 15, "W04": 19, "W07": 25, "W09": 29},
        "c": {"W03": 15, "W06": 22, "W08": 28},
        # My club's staff (user, 2026-11-12): wide bands, W02 the youngest.
        "staff": {"W02": 18, "W03": 25, "W04": 32, "W05": 39, "W06": 46, "W07": 53,
                  "W08": 61, "W09": 68, "W10": 75},
        # Journalists and press officers (user, 2026-11-14): the staff bands from W04 up.
        "media": {
            "W04": 18, "W05": 39, "W06": 46, "W07": 53, "W08": 61, "W09": 68, "W10": 75,
        },
    }  # fmt: skip


def test_youth_and_media_are_drawn_by_default():
    """User, 2026-11-13: my club's youth teams at group B, journalists and press officers.
    Since Phase 16 the youth teams' group is `cutoffs.my_club.youth`."""
    d = load()
    assert "youth" not in d
    assert d["cutoffs"]["my_club"]["youth"] == "b"
    assert d["media"] == {"journalists": True, "press_officers": True}


def test_pictures_are_1_by_1_by_default():
    """Phase 16g: 1:1 is a choice beside 3:4 (user, 2026-11-11), and the default since
    2026-11-22 (user); a database that already holds pictures keeps 3:4 (migration 15)."""
    assert load()["picture"] == {"aspect": "1:1"}


def test_photos_of_men_are_drawn_by_default():
    """RealVisXL and men since 0.1.8 (user, 2026-11-25); a database that already holds
    pictures keeps Animagine and women (migration 18)."""
    assert load()["style"] == {"model": "realvis", "gender": "man"}


def test_paths_name_no_windows_user():
    """This PC's real folder goes in config/local.toml, which is not in git (user, 2026-11-12),
    or is found on the PC (Phase 8): the defaults name no folder of FM27's."""
    paths = load()["paths"]
    assert paths["fm_user_dir"] == "" and paths["fm_install_dir"] == ""


def test_my_clubs_staff_are_drawn_by_default():
    assert load()["staff"] == {"enabled": True, "include_human_manager": False}


def test_pre_existing_players_without_a_photo_are_picked_by_default():
    """User, 2026-11-12: include them, by default."""
    assert load()["no_photo"] == {"enabled": True}


def test_the_install_folder_names_no_windows_user():
    assert "/Users/" not in load()["paths"]["fm_install_dir"]
