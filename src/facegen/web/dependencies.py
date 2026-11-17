"""Per-app paths and request/blocking-work connection ownership."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import Depends, HTTPException

from facegen.core.settings_values import Settings
from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline.worker import Career, load_career
from facegen.runtime.policy import BUSY_TIMEOUT_MS
from facegen.web.context import AppContext


class AppDependencies:
    """Keep one stable request dependency per app, never a shared live connection."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.data_dir = load_settings(defaults_path=ctx.defaults_path).paths.data_dir
        self.masters_dir = self.data_dir / "masters"
        self.thumbs_dir = self.data_dir / "thumbs"
        self.connection = Depends(self.get_conn)

    def connect(self) -> sqlite3.Connection:
        """The caller owns this connection and must close it in its work unit."""
        return db.connect(self.ctx.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)

    def get_conn(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def settings_now(self, conn: sqlite3.Connection) -> Settings:
        return load_settings(conn, self.ctx.defaults_path)

    def wake(self) -> None:
        if self.ctx.worker is not None:
            self.ctx.worker.wake()


def need_career(conn: sqlite3.Connection) -> Career:
    career = load_career(conn)
    if career is None:
        raise HTTPException(409, "No career yet: sync a save first.")
    return career
