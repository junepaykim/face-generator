"""Compatibility exports for the web app."""

from kitshot.setup.folder_setup import (
    LocalFileError,
    RealDisk,
    check_folder,
    value_to_write,
    write_local_path,
)

__all__ = ["LocalFileError", "RealDisk", "check_folder", "value_to_write", "write_local_path"]
