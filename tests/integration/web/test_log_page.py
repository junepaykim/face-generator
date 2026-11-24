"""The Log page (Phase 17i): Copy report and Open log folder in its header, the report itself,
the worker's log oldest first with its levels and the toggle, the save watcher as a badge and
a short line in both languages, and the Reload Skin note on the dashboard."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from factories import NEWGEN, RIVAL, player
from fastapi.testclient import TestClient
from support.desktop import FakeWorker, status
from support.web import BASE, KOREAN_WINDOWS, client_for, fake_worker, synced

from kitshot.persistence.settings import load_settings
from kitshot.runtime.contracts import LogLine, PollerStatus
from kitshot.web.app import AppContext, create_app

TODAY = datetime.now().strftime("%Y-%m-%d")
LINES = (
    LogLine("18:55:01", "Loading the image model..."),
    LogLine("18:55:09", "  attempt 1 failed the check: blurry", "warning"),
    LogLine("18:55:10", r"Publishing failed: cannot write C:\Users\Sam Lee\x", "error"),
    LogLine("18:55:11", "Published 1 picture(s)."),
)


@dataclass
class StubPoller:
    s: PollerStatus

    def status(self) -> PollerStatus:
        return self.s


@dataclass
class Opener:
    opened: list[Path] = field(default_factory=list)
    fail: bool = False

    def __call__(self, path: Path) -> None:
        if self.fail:
            raise OSError("no shell")
        self.opened.append(path)


@pytest.fixture
def db_path(config_file: Path) -> Path:
    return synced(config_file, [player(1001, name="Mine"), player(NEWGEN + 2, club_id=RIVAL)])


def client(config_file: Path, db_path: Path, **kw: Any) -> TestClient:
    kw.setdefault("worker", FakeWorker(status("running", log=LINES)))
    app = create_app(AppContext(db_path, config_file, **kw))
    c = TestClient(app, base_url=BASE, follow_redirects=False)
    token = re.search(r'name="csrf_token" value="([^"]+)"', c.get("/settings").text)
    assert token is not None
    c.headers["X-CSRF-Token"] = token[1]
    return c


@pytest.fixture
def opener() -> Opener:
    return Opener()


@pytest.fixture
def log_client(config_file: Path, db_path: Path, opener: Opener) -> Iterator[TestClient]:
    c = client(config_file, db_path, open_folder=opener)
    yield c
    c.close()


def head(page: str) -> str:
    return page[page.index('<header class="page-head">') : page.index("</header>")]


def test_the_log_reads_oldest_first_with_levels_and_a_toggle(log_client: TestClient) -> None:
    page = log_client.get("/log").text
    assert '<main class="fill">' in page  # the log fills the window
    pre = re.search(
        r'<pre class="log" data-log data-problems="2"[^>]*>(.*?)</pre>', page, re.DOTALL
    )
    assert pre is not None
    lines = re.findall(r'<span class="ll ll-(\w+)">([^<]*)\n</span>', pre[1])
    assert lines == [
        ("info", "18:55:01  Loading the image model..."),
        ("warning", "18:55:09    attempt 1 failed the check: blurry"),
        ("error", r"18:55:10  Publishing failed: cannot write C:\Users\Sam Lee\x"),
        ("info", "18:55:11  Published 1 picture(s)."),
    ]
    card = page[page.index('id="log-worker"') :]
    assert 'data-log-filter="all" aria-pressed="true">All</button>' in card
    assert (
        'data-log-filter="problems" aria-pressed="false">Errors and warnings'
        ' <span class="badge badge-count tone-bad">2</span></button>'
    ) in card
    assert '<p class="muted small" data-log-none hidden>No errors or warnings.</p>' in card
    # The live part is the same, with its sections by id so unchanged ones are kept.
    part = log_client.get("/log/part").text
    assert 'id="log-sync"' in part and 'id="log-worker"' in part and "ll-warning" in part


def test_no_problem_lines_no_count(config_file: Path, db_path: Path) -> None:
    quiet = FakeWorker(status("idle", log=(LogLine("09:00:00", "Worker started."),)))
    with client(config_file, db_path, worker=quiet) as c:
        page = c.get("/log").text
    assert 'data-problems="0"' in page and "badge-count tone-bad" not in page


def test_copy_report_and_open_log_folder_in_the_header(log_client: TestClient) -> None:
    h = head(log_client.get("/log").text)
    assert '<button type="button" data-copy-report="/log/report" hidden>Copy report</button>' in h
    form = re.search(r'<form method="post" action="/folders/open" data-open-folder>(.*?)</form>', h)
    assert form is not None
    assert 'name="csrf_token"' in form[1] and 'name="name" value="logs"' in form[1]
    assert "<button>Open log folder</button>" in form[1]
    assert "data-report-note" in h
    ko = head(log_client.get("/log", headers=KOREAN_WINDOWS).text)
    assert ">보고서 복사</button>" in ko and "<button>기록 폴더 열기</button>" in ko


def test_without_explorer_there_is_no_open_folder(config_file: Path, db_path: Path) -> None:
    """WSL's `kitshot serve` and Linux: no Explorer, so no button, and the route says so."""
    with client(config_file, db_path) as c:
        assert "/folders/open" not in head(c.get("/log").text)
        r = c.post("/folders/open", data={"name": "logs"}, headers={"Accept": "application/json"})
        assert r.json() == {"ok": False, "error": "Folders open in Explorer on Windows only."}


def test_open_folder_opens_known_folders_only(
    log_client: TestClient, opener: Opener, config_file: Path
) -> None:
    settings = load_settings(defaults_path=config_file)
    logs = settings.paths.data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    json = {"Accept": "application/json"}
    assert log_client.post("/folders/open", data={"name": "logs"}, headers=json).json() == {
        "ok": True,
        "error": None,
    }
    # A plain form goes back where it came from.
    r = log_client.post("/folders/open", data={"name": "data", "next": "/log"})
    assert r.status_code == 303 and r.headers["location"] == "/log"
    assert opener.opened == [logs, settings.paths.data_dir]
    # Never a path from the request: an unknown name, or a path, is refused.
    for name in ("", "x", "..", str(logs), "logs/.."):
        assert log_client.post("/folders/open", data={"name": name}).status_code == 404
    # FM27's own folders (Phase 17k) by name too; the pack folder may not be there yet.
    fm = settings.paths.fm_user_dir
    assert fm is not None
    fm.mkdir(parents=True, exist_ok=True)
    assert log_client.post("/folders/open", data={"name": "fm"}, headers=json).json()["ok"]
    assert opener.opened[-1] == fm
    r = log_client.post("/folders/open", data={"name": "pack"}, headers=json)
    assert r.json() == {"ok": False, "error": "That folder is not there yet."}
    assert opener.opened[-1] == fm


def test_open_folder_needs_the_token_and_says_when_it_fails(
    config_file: Path, db_path: Path, opener: Opener
) -> None:
    with client(config_file, db_path, open_folder=opener) as c:
        load_settings(defaults_path=config_file).paths.data_dir.mkdir(parents=True, exist_ok=True)
        r = c.post("/folders/open", data={"name": "data"}, headers={"X-CSRF-Token": "wrong"})
        assert r.status_code == 403 and opener.opened == []
        opener.fail = True
        r = c.post("/folders/open", data={"name": "data"}, headers={"Accept": "application/json"})
        assert r.json() == {"ok": False, "error": "The folder could not be opened."}
        r = c.post("/folders/open", data={"name": "data"}, headers=KOREAN_WINDOWS)
        assert r.status_code == 303 and "err=" in r.headers["location"]


def test_the_report_is_plain_text_from_the_running_app(log_client: TestClient) -> None:
    r = log_client.get("/log/report")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert r.headers["cache-control"] == "no-store"
    text = r.text
    assert text.startswith("Kitshot report, ")
    for expected in (
        "Graphics card: not asked in this process",
        "Image model: Animagine XL 4.0 (fp16, INT8 UNet blocks); files not checked in this process",
        "Save watcher: off",
        "Last sync: career.fm at ",
        "Worker log, oldest first (4 line(s)):",
        r"18:55:10 ERROR   Publishing failed: cannot write %USERPROFILE%\x",
    ):
        assert expected in text, expected
    assert "Sam Lee" not in text


def watcher(page: str) -> str:
    found = re.search(r'<p class="watcher">(.*?)</p>', page, re.DOTALL)
    assert found is not None
    return re.sub(r"\s+", " ", found[1]).strip()


def test_the_save_watcher_as_a_badge_and_a_short_line(config_file: Path, db_path: Path) -> None:
    poller = StubPoller(
        PollerStatus(
            "synced",
            "newest save Sam Lee - 레드우드.fm (saved ...) is synced",
            "18:55:10",
            "synced",
            {"name": "Sam Lee - 레드우드.fm", "saved": f"{TODAY} 18:19"},
        )
    )
    with client(config_file, db_path, poller=poller) as c:
        en = watcher(c.get("/log").text)
        ko = watcher(c.get("/log", headers=KOREAN_WINDOWS).text)
        poller.s = PollerStatus(
            "synced", "", "18:55:10", "synced", {"name": "a.fm", "saved": "2026-01-02 18:19"}
        )
        older = watcher(c.get("/log").text)
        poller.s = PollerStatus(
            "error",
            "database is locked; trying again shortly",
            "18:56:00",
            "sync_failed",
            {"error": "database is locked"},
        )
        failed = c.get("/log").text
        poller.s = PollerStatus("settling", "an English line", None, "new_code", {})
        unknown = watcher(c.get("/log").text)
    assert en == (
        '<span class="watcher-label">Save watcher</span> <span class="badge badge-status tone-ok"'
        ' data-watcher="synced">Synced</span> <span>Sam Lee - 레드우드.fm (18:19)</span>'
        ' <span class="muted">· checked 18:55:10</span>'
    )
    assert ">동기화됨</span> <span>Sam Lee - 레드우드.fm (18:19)</span>" in ko
    assert "· 확인 18:55:10" in ko
    assert "<span>a.fm (2026-01-02 18:19)</span>" in older  # not today: the date too
    line = watcher(failed)
    assert 'tone-bad" data-watcher="error">Error</span>' in line
    assert "<span>The sync failed; trying again shortly.</span>" in line
    assert "database is locked" not in line  # the raw error is folded away
    assert re.search(
        r'<details class="problem-detail" id="watcher-detail"><summary>Technical details</summary>'
        r"<pre>database is locked</pre></details>",
        failed,
    )
    assert "<span>an English line</span>" in unknown  # an unknown code: the English line


def test_without_a_watcher(log_client: TestClient) -> None:
    assert "not watching" in watcher(log_client.get("/log").text)


def test_the_dashboard_says_reload_skin_once_after_new_pictures(
    config_file: Path, db_path: Path
) -> None:
    worker = fake_worker(config_file, db_path)
    note = "New pictures show in FM27 after Settings → Reload Skin."
    with client_for(config_file, db_path, worker) as c:
        assert note not in c.get("/").text  # nothing made yet
        worker.step()  # draws and publishes one picture
        page = c.get("/").text
        recent = page[page.index('id="recent"') :]
        assert recent.count(note) == 1
        assert f'<p class="muted small recent-note" data-note>{note}</p>' in recent
        assert (
            "새 그림은 FM27에서 설정 → 스킨 다시 불러오기를 하면 보입니다."
            in c.get("/", headers=KOREAN_WINDOWS).text
        )
    # Another run that has made nothing yet shows the pictures without the note.
    with client_for(config_file, db_path, fake_worker(config_file, db_path)) as c:
        page = c.get("/").text
        assert 'id="recent"' in page and note not in page
