"""Per-app paths and request/blocking-work connection ownership."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from fastapi import Depends, HTTPException, Request

from facegen.core.job_records import Career
from facegen.core.settings_values import Settings
from facegen.i18n import Translator, negotiate
from facegen.persistence import db
from facegen.persistence.jobs import load_career
from facegen.persistence.settings import load_settings
from facegen.runtime.policy import BUSY_TIMEOUT_MS
from facegen.setup.hardware import Hardware
from facegen.web.context import AppContext


class CardProbe:
    """The graphics card, asked once per app: a child process imports torch for it, which
    takes a few seconds, so the dashboard asks in the background (Phase 12f) and the GPU page
    waits for the same answer."""

    def __init__(self, ask: Callable[[], Hardware] | None) -> None:
        self._ask = ask
        self._lock = threading.Lock()
        self._card: Hardware | None = None
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        """This process can ask the card (the GPU worker's setup is part of it)."""
        return self._ask is not None

    def get(self, again: bool = False) -> Hardware | None:
        """The card, asking now if not yet asked (or `again`); None where it cannot be asked."""
        if self._ask is None:
            return None
        with self._lock:
            if again or self._card is None:
                self._card = self._ask()
            return self._card

    def start(self) -> None:
        """Ask in the background, once."""
        with self._lock:
            if self._ask is None or self._card is not None or self._thread is not None:
                return
            self._thread = threading.Thread(target=self.get, name="facegen-card", daemon=True)
        self._thread.start()

    def known(self) -> Hardware | None:
        """The answer if there is one yet; never waits."""
        return self._card

    def state(self) -> tuple[Any, ...]:
        """For the live fingerprint: not asked, asking, or the answer."""
        card = self._card
        if card is not None:
            return ("known", card.name, card.total_bytes, card.reason)
        return ("asking",) if self._thread is not None else ("not asked",)


class AppDependencies:
    """Keep one stable request dependency per app, never a shared live connection."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.data_dir = load_settings(defaults_path=ctx.defaults_path).paths.data_dir
        self.masters_dir = self.data_dir / "masters"
        self.thumbs_dir = self.data_dir / "thumbs"
        self.connection = Depends(self.get_conn)
        self.card = CardProbe(ctx.hardware)
        self.closing = threading.Event()
        """Set when the server stops (web.app.close_streams): live event streams end."""

    def connect(self) -> sqlite3.Connection:
        """The caller owns this connection and must close it in its work unit."""
        return db.connect(self.ctx.db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)

    def get_conn(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def live_extra(self) -> tuple[Any, ...]:
        """State outside the services and the database that live pages show: the card's
        answer and whether the first-run guide was closed (web.live, Phase 12f)."""
        from facegen.web import guide  # the guide's module needs nothing from here

        return (self.card.state(), guide.closed(self.data_dir))

    def update_offer(self) -> dict[str, Any] | None:
        """The update check's state for the pages (Phase 13), and how this facegen is updated:
        "app" (the dashboard's button), "git" (a checkout) or "manual" (Start.bat --update).
        None where nothing looks for releases."""
        if self.ctx.updates is None:
            return None
        status = self.ctx.updates.status()
        if self.ctx.install_update is not None:
            how = "app"
        else:
            how = "git" if status.checkout else "manual"
        checked = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(status.checked_at))
            if status.checked_at
            else None
        )
        return {"status": status, "how": how, "checked": checked}

    def settings_now(self, conn: sqlite3.Connection) -> Settings:
        return load_settings(conn, self.ctx.defaults_path)

    def translator(self, request: Request, conn: sqlite3.Connection | None = None) -> Translator:
        """The request's language; opens and closes a connection of its own if given none."""
        if conn is not None:
            return translator_for(request, self.settings_now(conn))
        own = self.connect()
        try:
            return translator_for(request, self.settings_now(own))
        finally:
            own.close()

    def wake(self) -> None:
        if self.ctx.worker is not None:
            self.ctx.worker.wake()


def translator_for(request: Request, settings: Settings) -> Translator:
    """The request's language: the setting, or for "auto" the window's (Accept-Language)."""
    return Translator(negotiate(settings.display.language, request.headers.get("accept-language")))


def need_career(conn: sqlite3.Connection, t: Translator | None = None) -> Career:
    career = load_career(conn)
    if career is None:
        raise HTTPException(409, t("msg.no_career") if t else "No career yet: sync a save first.")
    return career
