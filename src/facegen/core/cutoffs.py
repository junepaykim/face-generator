"""League-share cutoffs: the CA and PA numbers each picture tier uses (Phase 16). Pure.

Detailed, Standard and Light (groups A, B and C) are picked by a player's share of a league
rather than by fixed numbers, so the picks follow a career through promotion and relegation
(spec section 2). The shares are measured over a league's candidates, the people the rules can
pick: newgens and pre-existing players without a photo at its clubs. Each scope's candidates are
kept as CA and PA histograms (one count per value, 0 to 200), so a settings change needs no new
save.

"Top N%": the candidates sorted by CA (or PA), highest first; the cut is the value of the
candidate at rank ceil(N% x count), and everyone at or above that value qualifies, ties
included. Every number means "at least" (user, 2026-11-21). An empty league picks nobody.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from facegen.core.settings_values import CutoffSettings

TOP = 200
"""The highest CA or PA."""
OFF = "-"
"""A cell switched off: its rule picks nobody."""

SCOPES = ("league", "above", "below", "continental")
"""Where a share is measured: my league, the league above, the league below (each a level of
my nation's ladder, core/leagues.py) and the continental competitions my club plays in."""

TIERS = ("a", "b", "c")
LEAGUE_CELLS = ("a_ca", "a_pa", "b_ca", "b_pa", "c_ca", "c_pa")
"""My league's cells: a share in Auto, a number in Manual. Also the global cells' names."""
OTHER_CELLS = ("above_b_ca", "above_c_ca", "below_c_ca", "continental_b_ca")
"""Other leagues' cells: always a share."""
CELL_SCOPE = {"above_b_ca": "above", "above_c_ca": "above", "below_c_ca": "below"} | {
    "continental_b_ca": "continental"
}
"""The scope each other league's cell is measured over."""
LIGHT_SWITCHES = {"above_c_ca": "light_above", "below_c_ca": "light_below"}
"""Cells that work only while their switch (`cutoffs.light_above`, `cutoffs.light_below`) is on."""


def split_cell(cell: str) -> tuple[str, str]:
    """("b", "ca") from "b_ca" or "above_b_ca"."""
    parts = cell.split("_")
    return parts[-2], parts[-1]


@dataclass(frozen=True, slots=True)
class Histogram:
    """How many candidates have each value from 0 to TOP."""

    counts: tuple[int, ...] = (0,) * (TOP + 1)

    @classmethod
    def of(cls, values: Iterable[int | None]) -> Histogram:
        """Count the values, clamped to 0 to TOP; None (an unknown PA) is left out."""
        counts = [0] * (TOP + 1)
        for v in values:
            if v is not None:
                counts[min(TOP, max(0, int(v)))] += 1
        return cls(tuple(counts))

    @classmethod
    def from_list(cls, counts: Iterable[int]) -> Histogram:
        """From a stored list; a short or long one is padded or cut to TOP + 1 values."""
        values = [int(c) for c in counts][: TOP + 1]
        return cls(tuple(values + [0] * (TOP + 1 - len(values))))

    @property
    def total(self) -> int:
        return sum(self.counts)

    def __add__(self, other: Histogram) -> Histogram:
        return Histogram(tuple(a + b for a, b in zip(self.counts, other.counts, strict=True)))

    def at_or_above(self, value: int) -> int:
        """How many have this value or more."""
        return sum(self.counts[max(0, value) :]) if value <= TOP else 0


def cut_for_share(hist: Histogram, share: float) -> int | None:
    """The value of the candidate at rank ceil(share% x count), highest first; None for an
    empty league. Everyone at or above it qualifies, ties included."""
    n = hist.total
    if n == 0 or share <= 0:
        return None
    # The share has one decimal: work in whole tenths so 6.0% of 50 is exactly rank 3.
    tenths = round(share * 10)
    rank = max(1, min(n, -(-tenths * n // 1000)))
    seen = 0
    for value in range(TOP, -1, -1):
        seen += hist.counts[value]
        if seen >= rank:
            return value
    return 0  # unreachable: the counts add up to n


def share_at_or_above(hist: Histogram, value: int) -> float | None:
    """The share (one decimal) whose cut is exactly `value`, for Manual -> Auto; None for an
    empty league.

    The largest one-decimal share at most count(>= value) / n is taken, so the cut picks the
    same people; if no one-decimal share lands between the people above `value` and those at
    it (a very large league), the nearest one above is taken instead. Nobody reaching the value
    gives the smallest share, 0.1.
    """
    n = hist.total
    if n == 0:
        return None
    at = hist.at_or_above(value)
    above = hist.at_or_above(value + 1)
    if at == 0:
        return 0.1
    tenths = (1000 * at) // n  # floor of the share in tenths of a percent
    if tenths * n <= 1000 * above:  # rank would fall among those above the value
        tenths = -((-1000 * at) // n)  # ceiling instead
    return max(1, min(1000, tenths)) / 10


def share_text(share: float) -> str:
    """6.0 -> "6", 12.5 -> "12.5"."""
    return f"{share:g}"


@dataclass(frozen=True, slots=True)
class Cut:
    """One cell's number in force: everyone at or above `value` qualifies."""

    value: int | None
    """None: the rule picks nobody (switched off, an empty league or no league found)."""
    share: float | None = None
    """The share it came from, in Auto; None for a number set by hand."""

    def holds(self, value: int | None) -> bool:
        return self.value is not None and value is not None and value >= self.value

    def why(self, name: str, value: int | None) -> str | None:
        """ "CA 98 (top 6%: 98+)" when the value qualifies, else None."""
        if not self.holds(value):
            return None
        if self.share is not None:
            return f"{name} {value} (top {share_text(self.share)}%: {self.value}+)"
        return f"{name} {value} ({self.value}+)"


NO_CUT = Cut(None)


@dataclass(frozen=True, slots=True)
class ScopeStats:
    """One scope's candidates at the last sync: newgens and pre-existing players without a
    photo, kept apart so `no_photo.enabled` can change without a new save."""

    competitions: tuple[int, ...] = ()
    newgen_ca: Histogram = field(default_factory=Histogram)
    newgen_pa: Histogram = field(default_factory=Histogram)
    no_photo_ca: Histogram = field(default_factory=Histogram)
    no_photo_pa: Histogram = field(default_factory=Histogram)

    def candidates(self, attribute: str, no_photo: bool) -> Histogram:
        """The CA ("ca") or PA ("pa") histogram of the candidates."""
        newgen = self.newgen_ca if attribute == "ca" else self.newgen_pa
        if not no_photo:
            return newgen
        return newgen + (self.no_photo_ca if attribute == "ca" else self.no_photo_pa)


@dataclass(frozen=True, slots=True)
class Cutoffs:
    """The numbers in force for one classification (`effective_cutoffs`), and my club's tiers."""

    league: Mapping[str, Cut] = field(default_factory=dict)
    """My league's cells by name ("a_ca"); a missing cell picks nobody."""
    other: Mapping[str, Cut] = field(default_factory=dict)
    """The other leagues' cells by name ("above_b_ca")."""
    everywhere: Mapping[str, Cut] = field(default_factory=dict)
    """The global cells by name ("a_ca")."""
    first_team: str = "a"
    youth: str = "b"
    loans: str = "b"
    academy_alumni: bool = True

    def league_cut(self, cell: str) -> Cut:
        return self.league.get(cell, NO_CUT)

    def other_cut(self, cell: str) -> Cut:
        return self.other.get(cell, NO_CUT)

    def everywhere_cut(self, cell: str) -> Cut:
        return self.everywhere.get(cell, NO_CUT)


def effective_cutoffs(
    settings: CutoffSettings,
    stats: Mapping[str, ScopeStats] | None,
    *,
    no_photo: bool = True,
) -> Cutoffs:
    """The numbers in force per tier and scope.

    `stats` is the last sync's candidates by scope (missing: no league found there, so its
    share rules pick nobody). `no_photo` says whether pre-existing players without a photo are
    candidates (`no_photo.enabled`). In Manual, my league's cells are the numbers set by hand.
    """
    stats = stats or {}

    def from_share(scope: str, cell: str, share: float | None) -> Cut:
        if share is None or scope not in stats:
            return NO_CUT
        _, attribute = split_cell(cell)
        cut = cut_for_share(stats[scope].candidates(attribute, no_photo), share)
        return Cut(cut, share) if cut is not None else NO_CUT

    league: dict[str, Cut] = {}
    for cell in LEAGUE_CELLS:
        if settings.mode == "manual":
            number = settings.manual.get(cell)
            league[cell] = Cut(number) if number is not None else NO_CUT
        else:
            league[cell] = from_share("league", cell, settings.share.get(cell))
    other = {
        cell: (
            NO_CUT
            if cell in LIGHT_SWITCHES and not getattr(settings, LIGHT_SWITCHES[cell])
            else from_share(CELL_SCOPE[cell], cell, settings.share.get(cell))
        )
        for cell in OTHER_CELLS
    }
    everywhere = {
        cell: Cut(n) if (n := settings.everywhere.get(cell)) is not None else NO_CUT
        for cell in LEAGUE_CELLS
    }
    tiers = settings.my_club
    return Cutoffs(
        league=league,
        other=other,
        everywhere=everywhere,
        first_team=tiers.first_team,
        youth=tiers.youth,
        loans=tiers.loans,
        academy_alumni=tiers.academy_alumni,
    )


def convert_mode(
    settings: CutoffSettings,
    to: str,
    stats: Mapping[str, ScopeStats] | None,
    *,
    no_photo: bool = True,
) -> dict[str, float | int | str]:
    """The setting values that switching my league's cells to `to` ("auto" or "manual") sets,
    converted from the current save: each Auto share becomes the number it gives ("top 6%" ->
    "98"), each Manual number the share that gives it. A cell whose other form already gives
    the same people is kept as it is, so Auto -> Manual -> Auto keeps "6"; "-" stays "-". With
    no save synced, only the mode changes. `cutoffs.mode` is in the result.
    """
    changes: dict[str, float | int | str] = {"cutoffs.mode": to}
    league = (stats or {}).get("league")
    if league is None or settings.mode == to:
        return changes
    for cell in LEAGUE_CELLS:
        _, attribute = split_cell(cell)
        hist = league.candidates(attribute, no_photo)
        share = settings.share.get(cell)
        number = settings.manual.get(cell)
        if to == "manual":
            if share is None:
                if number is not None:
                    changes[f"cutoffs.manual.{cell}"] = OFF
                continue
            cut = cut_for_share(hist, share)
            if cut is None or (number is not None and number == cut):
                continue
            changes[f"cutoffs.manual.{cell}"] = max(1, cut)
        else:
            if number is None:
                if share is not None:
                    changes[f"cutoffs.share.{cell}"] = OFF
                continue
            if share is not None and cut_for_share(hist, share) == number:
                continue
            converted = share_at_or_above(hist, number)
            if converted is not None:
                changes[f"cutoffs.share.{cell}"] = converted
    return changes
