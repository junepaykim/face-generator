"""The application log and local timestamps used by its status feeds."""

import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_LINES = 200
LOG_FILE = Path("logs") / "kitshot.log"
"""Relative to the data folder."""

logger = logging.getLogger("kitshot")

PYTHON_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
"""A worker log line's level (contracts.LOG_LEVELS) for the log file."""


def log_to_file(data_dir: Path) -> Path:
    """Send the "kitshot" logger to data/logs/kitshot.log (1 MB, three old files kept)."""
    path = data_dir / LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(getattr(h, "baseFilename", None) == str(path.resolve()) for h in logger.handlers):
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return path


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def utc_now() -> str:
    """Now in UTC, ISO 8601 to the second: when a worker state began (WorkerStatus.since)."""
    return datetime.now(UTC).isoformat(timespec="seconds")
