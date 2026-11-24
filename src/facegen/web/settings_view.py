"""Read models for the Advanced settings page's live parts (Phase 16e, 16f, 17l): what each
cutoff cell gives in the latest save, each group's count, the leagues found, and each age
table's looks with their ages and, for staff and media, how many people have each. Pure reads;
the routes render them."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from facegen.core.age import TABLES, WORDINGS, age_wording, band_ages
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

OFF_SCOPES = ("league", "above", "below")
"""The scopes whose cells are shown turned off when the latest sync did not find them: my league
(unknown, or its seasons conflict) and the leagues above and below it."""
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
    off: tuple[str, ...] = ()
    """The scopes of OFF_SCOPES the latest sync did not find: their cells (and Light switch) are
    shown turned off, as their rules pick nobody."""


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
        off=tuple(scope for scope in OFF_SCOPES if scope not in stats),
    )


COUNTED_TABLES = ("staff", "media")
"""The tables whose looks show how many people have each now (staff since Phase 16f)."""


@dataclass(frozen=True, slots=True)
class LookCell:
    """One look in one age table."""

    look: str
    used: bool
    first: int | None
    """The first age; None: "and under" (or not used)."""
    last: int | None
    """The last age; None: "and over" (or not used)."""
    count: int | None
    """People with this look now: staff and media only, None before any sync."""


@dataclass(frozen=True, slots=True)
class AgeColumn:
    """An age table as the Appearance tab shows it: every look W01-W10, in order."""

    table: str
    auto: bool
    """Its looks are the spread (Auto); greyed, with the spread's inputs above them."""
    cells: dict[str, LookCell]
    counted: bool


def _counts(
    conn: sqlite3.Connection, settings: Settings, career_id: int, table: str
) -> dict[str, int]:
    """How many people of a counted table have each look now: my club's staff, or the
    journalists and press officers who get a picture (in a group)."""
    if table == "staff":
        rows = conn.execute(
            "SELECT age, 'a' AS grp, 1 AS mine FROM players WHERE career_id = ? AND present = 1"
            " AND is_staff = 1 AND at_my_club = 1",
            (career_id,),
        )
    else:
        rows = conn.execute(
            "SELECT age, group_name AS grp, at_my_club AS mine FROM players WHERE career_id = ?"
            " AND present = 1 AND media_role IS NOT NULL AND group_name IS NOT NULL",
            (career_id,),
        )
    counts: dict[str, int] = {}
    for age, group, mine in rows:
        look = age_wording(
            age, group, bool(mine), settings.age, staff=table == "staff", media=table == "media"
        )
        if look is not None:
            counts[look] = counts.get(look, 0) + 1
    return counts


def age_columns(
    conn: sqlite3.Connection, settings: Settings, career_id: int | None
) -> dict[str, AgeColumn]:
    """Every age table by name, each look with its ages; staff and media with their counts."""
    out = {}
    for table in TABLES:
        counted = table in COUNTED_TABLES
        counts = _counts(conn, settings, career_id, table) if counted and career_id else None
        ages = {look: (first, last) for look, first, last in band_ages(settings.age.table(table))}
        cells = {}
        for look in WORDINGS:
            first, last = ages.get(look, (None, None))
            cells[look] = LookCell(
                look,
                look in ages,
                first,
                last,
                None if counts is None else counts.get(look, 0),
            )
        out[table] = AgeColumn(table, table in settings.age.spreads, cells, counted)
    return out


def range_text(cell: LookCell, t: Translator) -> str:
    """A look's ages for people: "29–33", "28 and under", "Not used"."""
    if not cell.used:
        return t("ages.not_used")
    if cell.first is None and cell.last is None:
        return t("ages.any")
    if cell.first is None:
        return t("ages.and_under", age=cell.last)
    if cell.last is None:
        return t("ages.and_over", age=cell.first)
    if cell.first == cell.last:
        return str(cell.first)
    return f"{cell.first}–{cell.last}"


def age_texts(columns: dict[str, AgeColumn], t: Translator) -> dict[str, str]:
    """The live texts of the age tables, by the `data-effective` name the page gives them: each
    look's ages ("ages.b.W04") and, for staff and media, its count ("ages.staff.W04.n")."""
    out = {}
    for table, column in columns.items():
        for look, cell in column.cells.items():
            out[f"ages.{table}.{look}"] = range_text(cell, t)
            if column.counted:
                count = cell.count
                out[f"ages.{table}.{look}.n"] = (
                    NONE if count is None else t("ages.count", count=count)
                )
    return out


def live_parts(
    conn: sqlite3.Connection,
    settings: Settings,
    career_id: int | None,
    t: Translator,
    before: dict[str, int] | None = None,
) -> dict[str, Any]:
    """What the page's live parts show: the cutoff view, the counts before the change (None:
    unchanged), and the age tables with their live texts."""
    ages = age_columns(conn, settings, career_id)
    return {
        "cutoffs": cutoff_view(conn, settings, career_id, t),
        "counts_before": before,
        "ages": ages,
        "age_texts": age_texts(ages, t),
    }
