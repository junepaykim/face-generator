"""Compatibility exports for the web app."""

from facegen.persistence.settings import (
    DETAIL_SIZES,
    Settings,
    SettingsError,
    folder_not_set_message,
    is_fixed,
    load_defaults,
    load_settings,
    local_label,
    local_path,
)

__all__ = [
    "DETAIL_SIZES",
    "Settings",
    "SettingsError",
    "folder_not_set_message",
    "is_fixed",
    "load_defaults",
    "load_settings",
    "local_label",
    "local_path",
]
