"""Read models for the Advanced settings page's live parts (Phase 16e, 16f): what each cutoff
cell gives in the latest save, each group's count, the leagues found, and my club's staff per
age look. Pure reads; the routes render them."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from facegen.core.age import age_wording, band_ages
from facegen.core.cutoffs import (
    CELL_SCOPE,
    LEAGUE_CELLS,
    OTHER_CELLS,
    ScopeStats,
    cut_for_share,
    share_at_or_above,
    share_text,
    split_cell,
)
from facegen.core.settings_values import Settings
from facegen.i18n import Translator
from facegen.persistence import league_stats
from facegen.persistence.league_stats import FoundLeague

NONE = "–"
"""What a live value shows before any save is synced, or where no league was found."""


@dataclass(frozen=True, slots=True)
class CutoffView:
    effective: dict[str, str]
    """Setting key to its value in the latest save, in grey beside the field: a share's cut
    ("≈ CA 84+"), a number's share ("≈ top 12%"); NONE without one."""
    counts: dict[str, int] | None
    """Everyone in each group (players, staff, media); None before any sync."""
    leagues: list[FoundLeague] = field(default_factory=list)
    synced: bool = False


def tier_counts(conn: sqlite3.Connection, career_id: int) -> dict[str, int]:
    """How many people are in each group now: players, staff and media."""
    counts = dict.fromkeys(("a", "b", "c"), 0)
    for group, n in conn.execute(
        "SELECT group_name, COUNT(*) FROM players WHERE career_id = ? AND present = 1"
        " AND group_name IS NOT NULL GROUP BY group_name",
        (career_id,),
    ):
        counts[group] = n
    return counts


def effective_texts(
    settings: Settings, stats: dict[str, ScopeStats], t: Translator
) -> dict[str, str]:
    """Every cutoff cell's live value as text, by setting key."""
    no_photo = settings.no_photo.enabled
    out: dict[str, str] = {}

    def cut_text(scope: str, cell: str, share: float | None) -> str:
        if share is None or scope not in stats:
            return NONE
        _, attribute = split_cell(cell)
        cut = cut_for_share(stats[scope].candidates(attribute, no_photo), share)
        if cut is None:
            return NONE
        return t("cutoffs.effective.value", name=attribute.upper(), value=cut)

    for cell in LEAGUE_CELLS:
        out[f"cutoffs.share.{cell}"] = cut_text("league", cell, settings.cutoffs.share.get(cell))
        number = settings.cutoffs.manual.get(cell)
        text = NONE
        if number is not None and "league" in stats:
            _, attribute = split_cell(cell)
            share = share_at_or_above(stats["league"].candidates(attribute, no_photo), number)
            if share is not None:
                text = t("cutoffs.effective.share", share=share_text(share))
        out[f"cutoffs.manual.{cell}"] = text
    for cell in OTHER_CELLS:
        out[f"cutoffs.share.{cell}"] = cut_text(
            CELL_SCOPE[cell], cell, settings.cutoffs.share.get(cell)
        )
    return out


def cutoff_view(
    conn: sqlite3.Connection, settings: Settings, career_id: int | None, t: Translator
) -> CutoffView:
    if career_id is None:
        keys = [f"cutoffs.{k}.{c}" for k in ("share", "manual") for c in LEAGUE_CELLS]
        keys += [f"cutoffs.share.{c}" for c in OTHER_CELLS]
        return CutoffView(dict.fromkeys(keys, NONE), None)
    stats = league_stats.load(conn, career_id)
    return CutoffView(
        effective_texts(settings, stats, t),
        tier_counts(conn, career_id),
        league_stats.found_leagues(conn, career_id),
        synced=True,
    )


@dataclass(frozen=True, slots=True)
class LookRow:
    look: str
    first: int | None
    """The first age; None: "and under"."""
    last: int | None
    """The last age; None: "and over"."""
    staff: int | None
    """My club's staff in it now; None before any sync."""


def staff_looks(
    conn: sqlite3.Connection, settings: Settings, career_id: int | None
) -> list[LookRow]:
    """Each look of my club's staff table with its ages and how many staff have it now."""
    table = settings.age.staff
    counts: dict[str, int] | None = None
    if career_id is not None:
        counts = dict.fromkeys((key for _, key in table), 0)
        for (age,) in conn.execute(
            "SELECT age FROM players WHERE career_id = ? AND present = 1 AND is_staff = 1"
            " AND at_my_club = 1",
            (career_id,),
        ):
            look = age_wording(age, "a", True, settings.age, staff=True)
            if look is not None:
                counts[look] = counts.get(look, 0) + 1
    return [
        LookRow(look, first, last, counts.get(look, 0) if counts is not None else None)
        for look, first, last in band_ages(table)
    ]


def live_parts(
    conn: sqlite3.Connection,
    settings: Settings,
    career_id: int | None,
    t: Translator,
    before: dict[str, int] | None = None,
) -> dict[str, Any]:
    """What the page's live parts show: the cutoff view, the counts before the change (None:
    unchanged), and the staff looks."""
    return {
        "cutoffs": cutoff_view(conn, settings, career_id, t),
        "counts_before": before,
        "staff_looks": staff_looks(conn, settings, career_id),
    }
