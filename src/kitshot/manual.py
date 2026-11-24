"""Compatibility exports for the web app."""

from kitshot.pipeline.manual import (
    DEFAULT_GROUP,
    ManualError,
    cancel,
    generate,
    new_face,
    new_picture,
    new_pictures,
    remove_picture,
    restorable,
    restore_picture,
    retry,
    set_pin,
)

__all__ = [
    "DEFAULT_GROUP",
    "ManualError",
    "cancel",
    "generate",
    "new_face",
    "new_picture",
    "new_pictures",
    "remove_picture",
    "restorable",
    "restore_picture",
    "retry",
    "set_pin",
]
