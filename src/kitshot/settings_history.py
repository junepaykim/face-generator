"""Compatibility exports for the web app."""

from kitshot.pipeline.settings_history import (
    Outcome,
    can_redo,
    can_undo,
    entries,
    redo,
    save,
    save_defaults,
    undo,
)

__all__ = ["Outcome", "can_redo", "can_undo", "entries", "redo", "save", "save_defaults", "undo"]
