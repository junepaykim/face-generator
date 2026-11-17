"""Shared connection and retry bounds for the running application."""

BUSY_TIMEOUT_MS = 60_000
FM_POLL_SECONDS = 30.0
"""Worker checks while FM27 runs, and retry delay after a service-loop error."""
