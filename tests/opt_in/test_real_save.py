"""Read an explicitly supplied snapshot copy; never use FM27's games/ directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from facegen.core.records import Gender
from facegen.datasource.fmsave_source import FmsaveDataSource

pytestmark = pytest.mark.real_save


@pytest.mark.skipif(
    not os.environ.get("FACEGEN_TEST_SAVE"), reason="set FACEGEN_TEST_SAVE to a save copy"
)
def test_reads_a_real_save_copy() -> None:
    snap = FmsaveDataSource().load(Path(os.environ["FACEGEN_TEST_SAVE"]))
    assert snap.meta.game == "FM27"
    assert len(snap.players) > 1000
    assert snap.managed_club_id in snap.clubs
    assert {c.gender for c in snap.clubs.values()} >= {Gender.MEN}
    assert len({p.unique_id for p in snap.players}) == len(snap.players)
    with_personality = sum(p.personality is not None for p in snap.players)
    assert with_personality > 0.9 * len(snap.players)
    # Journalists, press officers and genders: read from the game database's bytes.
    assert snap.media_read, snap.meta.warnings
    assert sum(m.role == "journalist" for m in snap.media) > 100
    assert {m.female for m in snap.media} == {True, False}
    assert any(p.team_slot == 0 for p in snap.players if p.club_id == snap.managed_club_id)
