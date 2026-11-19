"""The update offer on the pages (Phase 13): the dashboard's card, Settings' About rows, the
switch, and Check now / Later / Update now. The releases list is made up; nothing is fetched."""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from fastapi.testclient import TestClient
from support.web import BASE, KOREAN_WINDOWS

from facegen.persistence import db
from facegen.persistence.settings import load_settings
from facegen.pipeline import settings_history
from facegen.runtime.update_service import UpdateService
from facegen.setup import update
from facegen.web.app import AppContext, create_app


class Releases:
    def __init__(self, *tags: str) -> None:
        self.tags = list(tags)
        self.error: str | None = None

    def __call__(self, url: str) -> Any:
        if self.error:
            raise update.UpdateError(self.error)
        return [
            {"tag_name": t, "draft": False, "prerelease": False,
             "html_url": f"https://github.com/junepaykim/face-generator/releases/tag/{t}"}
            for t in self.tags
        ]  # fmt: skip


class Env:
    def __init__(self, config_file: Path, tmp_path: Path, *, checkout: bool = False) -> None:
        self.config = config_file
        settings = load_settings(defaults_path=config_file)
        self.db_path = settings.paths.db_path
        db.connect(self.db_path).close()
        root = tmp_path / "app"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(
            '[project]\nname = "facegen"\nversion = "0.9.0"\n', encoding="utf-8"
        )
        if checkout:
            (root / ".git").mkdir()
        self.releases = Releases("v1.0.0")
        self.installs: list[str] = []

        def enabled() -> bool:
            conn = db.connect(self.db_path)
            try:
                return load_settings(conn, config_file).updates.check
            finally:
                conn.close()

        self.updates = UpdateService(
            settings.paths.data_dir, enabled, root=root, fetch=self.releases
        )

    def client(self, *, can_install: bool = False) -> TestClient:
        ctx = AppContext(self.db_path, self.config, updates=self.updates)
        if can_install:
            ctx.install_update = lambda: self.installs.append("Start.bat --update")
        client = TestClient(create_app(ctx), base_url=BASE, follow_redirects=False)
        token = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/settings").text)
        assert token is not None
        client.headers["X-CSRF-Token"] = token[1]
        return client


@pytest.fixture
def env(config_file: Path, tmp_path: Path) -> Iterator[Env]:
    yield Env(config_file, tmp_path)


def card(client: TestClient, **kw: Any) -> str | None:
    page = unescape(client.get("/", **kw).text)
    m = re.search(r'<section class="card update" id="update">(.*?)</section>', page, re.DOTALL)
    return m[1] if m else None


def test_no_card_until_a_newer_release_is_known(env: Env) -> None:
    with env.client() as client:
        assert card(client) is None
        env.releases.tags = ["v0.9.0"]
        env.updates.check_now()
        assert card(client) is None


def test_the_dashboard_offers_a_newer_release(env: Env) -> None:
    env.updates.check_now()
    with env.client(can_install=True) as client:
        shown = card(client)
        assert shown is not None
        assert "facegen 1.0.0 is out" in shown and "You have 0.9.0." in shown
        assert 'href="https://github.com/junepaykim/face-generator/releases/tag/v1.0.0"' in shown
        assert 'action="/update/install"' in shown and "Update now" in shown
        korean = card(client, headers=KOREAN_WINDOWS)
        assert korean is not None and "지금 업데이트" in korean and "나중에" in korean


def test_without_the_desktop_app_it_says_how_to_update(config_file: Path, tmp_path: Path) -> None:
    env = Env(config_file, tmp_path)
    env.updates.check_now()
    with env.client() as client:
        shown = card(client)
        assert shown is not None and "/update/install" not in shown
        assert "Start.bat --update" in shown
    checkout = Env(config_file, tmp_path / "git", checkout=True)
    checkout.updates.check_now()
    with checkout.client() as client:
        shown = card(client)
        assert shown is not None and "git pull" in shown


def test_later_puts_the_offer_off(env: Env) -> None:
    env.updates.check_now()
    with env.client(can_install=True) as client:
        assert 'name="version" value="1.0.0"' in card(client)
        form = {"next": "/", "version": "1.0.0"}
        response = client.post("/update/later", data=form)
        assert response.status_code == 303 and response.headers["location"] == "/"
        assert card(client) is None
        env.releases.tags.append("v1.1.0")
        env.updates.check_now()
        client.post("/update/later", data=form)  # an older dashboard tab
        assert card(client) is not None


def test_update_now_starts_the_update_and_says_so(env: Env) -> None:
    env.updates.check_now()
    with env.client(can_install=True) as client:
        response = client.post("/update/install")
        assert response.status_code == 200
        assert "Updating facegen" in response.text and "1.0.0" in response.text
        assert env.installs == ["Start.bat --update"]


def test_update_now_needs_the_desktop_app_and_a_newer_release(env: Env) -> None:
    with env.client() as client:
        response = client.post("/update/install")
        assert response.status_code == 303 and "cannot update itself" in unquote(
            response.headers["location"]
        )
    with env.client(can_install=True) as client:
        response = client.post("/update/install")  # nothing newer known yet
        assert response.status_code == 303 and env.installs == []
        assert "0.9.0 is the newest version" in unquote(response.headers["location"])


def test_check_now_asks_and_says_what_it_found(env: Env) -> None:
    with env.client() as client:
        response = client.post(
            "/update/check", data={"next": "/settings/advanced?lang=en&err=previous#about"}
        )
        assert response.status_code == 303
        parts = urlsplit(response.headers["location"])
        assert parts.path == "/settings/advanced" and parts.fragment == "about"
        assert parse_qs(parts.query) == {"lang": ["en"], "msg": ["facegen 1.0.0 is available."]}
        env.releases.error = "could not reach GitHub: timed out"
        parts = urlsplit(client.post("/update/check").headers["location"])
        assert parts.fragment == "about"
        error = parse_qs(parts.query)["err"][0]
        assert "could not ask GitHub" in error and "timed out" not in error
        # The error itself is under Technical details in About this run.
        page = unescape(client.get("/settings/advanced").text)
        assert "could not reach GitHub: timed out" in page


def test_check_now_without_a_checker_says_so(config_file: Path) -> None:
    settings = load_settings(defaults_path=config_file)
    app = create_app(AppContext(settings.paths.db_path, config_file))
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        token = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/").text)
        assert token is not None
        response = client.post("/update/check", headers={"X-CSRF-Token": token[1]})
        assert "does not look for updates" in unquote(response.headers["location"])
        page = unescape(client.get("/settings/advanced").text)
        assert "Check for updates now" not in page and 'id="update"' not in client.get("/").text


def test_settings_show_the_version_and_the_switch(env: Env) -> None:
    with env.client() as client:
        page = unescape(client.get("/settings/advanced").text)
        assert re.search(r"<th>Version</th><td>0\.9\.0</td>", page)
        assert "not checked yet" in page and "Check for updates now" in page
        assert 'name="updates.check"' in page and "Look for a newer facegen once a day" in page
        env.updates.check_now()
        page = unescape(client.get("/settings/advanced").text)
        assert "facegen 1.0.0 is available" in page and "checked " in page


def test_the_switch_turns_the_check_off(env: Env) -> None:
    conn = db.connect(env.db_path)
    try:
        settings_history.save(conn, {"updates.check": False}, defaults_path=env.config)
    finally:
        conn.close()
    assert not env.updates.enabled()
    with env.client() as client:
        page = unescape(client.get("/settings/advanced").text)
        assert "not looked for (switched off below)" in page
        # Check now still asks: the user asked.
        location = unquote(client.post("/update/check").headers["location"])
        assert "facegen 1.0.0 is available" in location


def test_posts_need_the_form_token(env: Env) -> None:
    env.updates.check_now()
    with env.client(can_install=True) as client:
        del client.headers["X-CSRF-Token"]
        for path in ("/update/install", "/update/later", "/update/check"):
            assert client.post(path).status_code == 403
        assert env.installs == []
