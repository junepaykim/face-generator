"""League-share cutoffs (Phase 16): "top N%", ties, empty leagues, the numbers in force and the
Auto <-> Manual conversion."""

from __future__ import annotations

from dataclasses import replace

import pytest

from facegen.core.cutoffs import (
    LEAGUE_CELLS,
    NO_CUT,
    Cut,
    Histogram,
    ScopeStats,
    convert_mode,
    cut_for_share,
    effective_cutoffs,
    share_at_or_above,
    share_text,
    split_cell,
)
from facegen.core.settings_values import CutoffSettings, MyClubTiers

DEFAULT = CutoffSettings(
    mode="auto",
    light_above=False,
    light_below=False,
    my_club=MyClubTiers(),
    share={
        "a_ca": 6.0,
        "a_pa": 3.0,
        "b_ca": 40.0,
        "b_pa": 20.0,
        "c_ca": 80.0,
        "c_pa": None,
        "above_b_ca": 15.0,
        "above_c_ca": 50.0,
        "below_c_ca": 50.0,
        "continental_b_ca": 30.0,
    },
    manual={"a_ca": None, "a_pa": None, "b_ca": 131, "b_pa": 151, "c_ca": 116, "c_pa": 141},
    everywhere={"a_ca": 155, "a_pa": 170, "b_ca": 140, "b_pa": 155, "c_ca": None, "c_pa": None},
)


def hist(*values: int | None) -> Histogram:
    return Histogram.of(values)


def stats(ca: Histogram, pa: Histogram | None = None, **kw: Histogram) -> ScopeStats:
    return ScopeStats(competitions=(1,), newgen_ca=ca, newgen_pa=pa or ca, **kw)


# --- "top N%" --------------------------------------------------------------------------------


def test_the_cut_is_the_value_at_rank_ceil_share_times_count() -> None:
    h = hist(*range(1, 101))  # 100 candidates, CA 1 to 100
    assert cut_for_share(h, 6.0) == 95  # rank 6
    assert cut_for_share(h, 0.5) == 100  # rank ceil(0.5) = 1
    assert cut_for_share(h, 100.0) == 1
    assert cut_for_share(h, 6.1) == 94  # rank ceil(6.1) = 7
    assert cut_for_share(hist(*range(1, 51)), 6.0) == 48  # 6% of 50 is exactly rank 3


def test_ties_are_included() -> None:
    h = hist(90, 80, 80, 80, 70, 60, 50, 40, 30, 20)
    assert cut_for_share(h, 20.0) == 80  # rank 2 is 80: all three 80s qualify
    assert h.at_or_above(80) == 4


def test_an_empty_league_picks_nobody() -> None:
    assert cut_for_share(Histogram(), 50.0) is None
    assert share_at_or_above(Histogram(), 80) is None


def test_values_are_clamped_and_unknown_pa_left_out() -> None:
    h = Histogram.of([-5, 250, None, 100])
    assert h.total == 3
    assert h.counts[0] == 1 and h.counts[200] == 1
    assert Histogram.from_list([1, 2]).counts[:3] == (1, 2, 0)
    assert len(Histogram.from_list(range(300)).counts) == 201
    assert (hist(1) + hist(1, 2)).counts[1] == 2


@pytest.mark.parametrize("n", [7, 50, 100, 450, 805, 999])
@pytest.mark.parametrize("share", [0.5, 3.0, 6.0, 15.0, 40.0, 80.0, 100.0])
def test_manual_from_auto_and_back_picks_the_same_people(n: int, share: float) -> None:
    # Many ties, as CA has: values 40-120 repeating.
    h = hist(*(40 + (i * 37) % 81 for i in range(n)))
    cut = cut_for_share(h, share)
    assert cut is not None
    back = share_at_or_above(h, cut)
    assert back is not None
    assert cut_for_share(h, back) == cut
    assert back == round(back, 1) and 0.1 <= back <= 100


def test_share_of_a_number_nobody_reaches_is_zero() -> None:
    assert share_at_or_above(hist(50, 60), 150) == 0.0


def test_a_number_nobody_holds_becomes_the_share_that_picks_the_same_people() -> None:
    h = hist(90, 80, 60)
    assert h.at_or_above(85) == 1  # "at least 85" picks the 90 alone
    share = share_at_or_above(h, 85)
    assert share == 33.3  # 33.4% would be rank 2 and pick the 80 too
    cut = cut_for_share(h, share)
    assert cut == 90 and h.at_or_above(cut) == 1


@pytest.mark.parametrize("n", [3, 7, 50, 450])
@pytest.mark.parametrize("number", [41, 55, 63, 86, 99, 118])
def test_every_number_becomes_a_share_that_picks_the_same_people(n: int, number: int) -> None:
    h = hist(*(40 + (i * 37) % 81 for i in range(n)))
    share = share_at_or_above(h, number)
    assert share is not None
    cut = cut_for_share(h, share)
    assert (h.at_or_above(cut) if cut is not None else 0) == h.at_or_above(number)


def test_a_number_nobody_holds_converts_to_the_share_that_keeps_its_people() -> None:
    league = stats(hist(90, 80, 60))
    manual = replace(
        DEFAULT, mode="manual", manual={**DEFAULT.manual, "a_ca": 85}, share={"a_ca": 50.0}
    )
    changes = convert_mode(manual, "auto", {"league": league})
    assert changes["cutoffs.share.a_ca"] == 33.3  # picks the 90 alone, as 85 did


def test_switching_keeps_equivalent_cutoffs_across_a_gap_in_ability() -> None:
    league = {"league": stats(hist(90, 80, 80, 60))}
    settings = replace(
        DEFAULT, manual={**DEFAULT.manual, "a_ca": 85}, share={**DEFAULT.share, "a_ca": 20.0}
    )
    assert "cutoffs.manual.a_ca" not in convert_mode(settings, "manual", league)
    back = convert_mode(replace(settings, mode="manual"), "auto", league)
    assert "cutoffs.share.a_ca" not in back


def test_cell_names_and_share_text() -> None:
    assert split_cell("b_ca") == ("b", "ca")
    assert split_cell("above_c_ca") == ("c", "ca")
    assert share_text(6.0) == "6" and share_text(12.5) == "12.5"


# --- the numbers in force -----------------------------------------------------------------------


def test_effective_cutoffs_from_shares_numbers_and_switches() -> None:
    league = stats(hist(*range(1, 101)), hist(*range(101, 201)))
    above = stats(hist(*range(1, 21)))
    cuts = effective_cutoffs(DEFAULT, {"league": league, "above": above})
    assert cuts.league_cut("a_ca") == Cut(95, 6.0)
    assert cuts.league_cut("a_pa") == Cut(198, 3.0)
    assert cuts.league_cut("c_pa") == NO_CUT  # "-"
    assert cuts.other_cut("above_b_ca") == Cut(18, 15.0)
    assert cuts.other_cut("above_c_ca") == NO_CUT  # Light for the league above is off
    assert cuts.other_cut("below_c_ca") == NO_CUT  # no league below found
    assert cuts.other_cut("continental_b_ca") == NO_CUT
    assert cuts.everywhere_cut("a_ca") == Cut(155)
    assert cuts.everywhere_cut("c_ca") == NO_CUT
    both = {"above": above, "below": above}
    on = effective_cutoffs(replace(DEFAULT, light_above=True), both)
    assert on.other_cut("above_c_ca") == Cut(11, 50.0)
    assert on.other_cut("below_c_ca") == NO_CUT  # each league has its own switch
    below = effective_cutoffs(replace(DEFAULT, light_below=True), both)
    assert below.other_cut("below_c_ca") == Cut(11, 50.0)
    assert below.other_cut("above_c_ca") == NO_CUT
    manual = effective_cutoffs(replace(DEFAULT, mode="manual"), {"league": league})
    assert manual.league_cut("b_ca") == Cut(131)
    assert manual.league_cut("a_ca") == NO_CUT
    assert (manual.first_team, manual.youth, manual.loans) == ("a", "b", "b")


def test_photo_less_candidates_count_only_while_the_setting_is_on() -> None:
    league = ScopeStats((1,), hist(50), hist(50), hist(90, 80, 70), hist(90, 80, 70))
    assert league.candidates("ca", no_photo=True).total == 4
    on = effective_cutoffs(DEFAULT, {"league": league}, no_photo=True)
    off = effective_cutoffs(DEFAULT, {"league": league}, no_photo=False)
    assert on.league_cut("b_ca").value == 80  # rank ceil(1.6) = 2 of 90, 80, 70, 50
    assert off.league_cut("b_ca").value == 50  # newgens only


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_without_stats_only_the_global_cuts_work(mode: str) -> None:
    cuts = effective_cutoffs(replace(DEFAULT, mode=mode), None)
    assert all(cuts.league_cut(c) == NO_CUT for c in LEAGUE_CELLS)
    assert cuts.everywhere_cut("b_pa") == Cut(155)


# --- Auto <-> Manual ---------------------------------------------------------------------------


def test_switching_to_manual_converts_each_share_to_its_number() -> None:
    league = stats(hist(*range(1, 101)))
    changes = convert_mode(DEFAULT, "manual", {"league": league})
    assert changes["cutoffs.mode"] == "manual"
    assert changes["cutoffs.manual.a_ca"] == 95
    assert changes["cutoffs.manual.b_ca"] == 61
    assert "cutoffs.manual.c_pa" not in changes  # Manual 141 already picks nobody, as Auto off did
    assert not any(k.startswith("cutoffs.share.") for k in changes)


def test_auto_manual_auto_keeps_the_shares_and_the_people() -> None:
    league = stats(hist(*(40 + (i * 37) % 81 for i in range(805))))
    to_manual = convert_mode(DEFAULT, "manual", {"league": league})
    manual = replace(
        DEFAULT,
        mode="manual",
        manual={
            c: (None if (v := to_manual.get(f"cutoffs.manual.{c}")) == "-" else v)
            for c in LEAGUE_CELLS
        },
    )
    back = convert_mode(manual, "auto", {"league": league})
    assert back == {"cutoffs.mode": "auto"}  # every share already gives its number
    a = effective_cutoffs(DEFAULT, {"league": league})
    m = effective_cutoffs(manual, {"league": league})
    assert all(a.league_cut(c).value == m.league_cut(c).value for c in LEAGUE_CELLS)


def test_a_number_set_by_hand_becomes_the_share_that_gives_it() -> None:
    league = stats(hist(*range(1, 101)))
    manual = replace(DEFAULT, mode="manual", manual={**DEFAULT.manual, "b_ca": 91})
    back = convert_mode(manual, "auto", {"league": league})
    assert back["cutoffs.share.b_ca"] == 10.0
    assert back["cutoffs.share.a_ca"] == "-"  # the number was off


def test_without_a_synced_save_only_the_mode_changes() -> None:
    assert convert_mode(DEFAULT, "manual", {}) == {"cutoffs.mode": "manual"}
    assert convert_mode(DEFAULT, "auto", {"league": stats(hist(1))}) == {"cutoffs.mode": "auto"}
