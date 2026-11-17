"""Synthetic fmsave-shaped codes and staff records, with a stable club ID map."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace as NS


def coded(name: str) -> NS:
    return NS(label=NS(name=name), raw=0)


def fm_staff(unique_id: int | None, **kw: object) -> NS:
    base: dict[str, object] = {
        "uid": 5,
        "unique_id": unique_id,
        "is_human_manager": False,
        "name": "Rhys Morris",
        "full_name": "Rhys Morris",
        "legal_name": None,
        "birth_date": date(1980, 2, 1),
        "age": 46,
        "club_uid": 10,
        "job": NS(label=NS(name="ASSISTANT_MANAGER"), raw=20),
        "contract_start": date(2025, 6, 1),
        "ability": NS(current=55, potential=70),
        "personality": None,
    }
    base.update(kw)
    return NS(**base)


UID_MAP = {10: 1010, 11: 1011, 12: 1012}
