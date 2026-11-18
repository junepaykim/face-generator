"""facegen's releases (Phase 13): the check, the release payload, and the update in place.

Nothing here reaches the network: the root conftest refuses it, and each test gives its own
releases list and archive. The folders updated are made-up trees under tmp_path.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from support.paths import ROOT

from facegen.setup import update
from facegen.setup.update import Release, UpdateError, Updater


def release(tag: str, **kw: Any) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "html_url": f"https://github.com/junepaykim/face-generator/releases/tag/{tag}",
        **kw,
    }


# --- versions and the releases list ---------------------------------------------------------


def test_versions_are_three_whole_numbers() -> None:
    assert update.parse_version("v1.2.3") == (1, 2, 3)
    assert update.parse_version("0.10.0") == (0, 10, 0)
    for text in ("v1.2", "1.2.3rc1", "native-v0.1.0", "", None, "v1.2.3.4"):
        assert update.parse_version(text) is None


def test_the_newest_app_release_is_chosen() -> None:
    items = [
        release("native-v9.0.0"),  # the Rust wheels' releases are not the app's
        release("v0.2.0"),
        release("v0.10.0"),  # numbers, not text: 0.10 is newer than 0.9 and 0.2
        release("v0.9.0"),
        release("v1.0.0", draft=True),
        release("v0.11.0", prerelease=True),
        release("v0.12"),
        "not a release",
    ]
    chosen = update.choose_release(items)
    assert chosen == Release(
        "0.10.0", "v0.10.0", "https://github.com/junepaykim/face-generator/releases/tag/v0.10.0"
    )
    assert update.choose_release([release("native-v0.1.0")]) is None


def test_a_page_elsewhere_is_replaced_by_the_release_page() -> None:
    chosen = update.choose_release([release("v1.0.0", html_url="https://example.com/x")])
    assert chosen is not None
    assert chosen.page == "https://github.com/junepaykim/face-generator/releases/tag/v1.0.0"


def test_newer_compares_versions() -> None:
    new = Release("0.10.0", "v0.10.0", "p")
    assert update.is_newer(new, "0.9.3")
    assert not update.is_newer(new, "0.10.0")
    assert not update.is_newer(new, "1.0.0")
    assert not update.is_newer(None, "0.1.0")
    assert update.is_newer(new, "not a version")


def test_the_check_is_one_request_for_the_releases_list() -> None:
    asked: list[str] = []

    def fetch(url: str) -> Any:
        asked.append(url)
        return [release("v0.3.0")]

    found = update.latest_release(fetch)
    assert found is not None and found.version == "0.3.0"
    assert asked == ["https://api.github.com/repos/junepaykim/face-generator/releases?per_page=30"]
    with pytest.raises(UpdateError, match="not a list"):
        update.latest_release(lambda url: {"message": "rate limited"})


def test_tests_cannot_reach_github() -> None:
    with pytest.raises(UpdateError, match="tests do not reach the network"):
        update.latest_release()


def test_the_request_says_only_the_programs_name() -> None:
    request = update._request(update.RELEASES_API)
    headers = {k.lower(): v for k, v in request.header_items()}
    assert headers == {"user-agent": "facegen", "accept": "application/vnd.github+json"}


def test_the_record_of_a_check_survives_a_restart(tmp_path: Path) -> None:
    assert update.load_record(tmp_path) is None
    record = update.CheckRecord(1000.0, Release("1.0.0", "v1.0.0", "p"), None, "0.9.0")
    update.save_record(tmp_path, record)
    assert update.load_record(tmp_path) == record
    update.check_path(tmp_path).write_text("{broken", encoding="utf-8")
    assert update.load_record(tmp_path) is None


# --- the payload ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "shipped"),
    [
        ("src/facegen/cli.py", True),
        ("native/src/lib.rs", True),
        ("native/wheels.sha256", True),
        ("config/default.toml", True),
        ("pyproject.toml", True),
        ("uv.lock", True),
        ("Start.bat", True),
        (".python-version", True),
        ("README.md", True),
        ("LICENSE", True),
        ("config/local.toml", False),
        ("data/facegen.db", False),
        (".venv/Scripts/python.exe", False),
        (".bin/uv.exe", False),
        (".git/HEAD", False),
        ("tests/conftest.py", False),
        ("scripts/build_kit_catalog.py", False),
        ("docs/images/hero.jpg", False),
        ("installer/facegen.iss", False),
        (".github/workflows/ci.yml", False),
        ("AGENTS.md", False),
        ("src/facegen/__pycache__/cli.cpython-312.pyc", False),
        ("native/target/release/x.dll", False),
    ],
)
def test_the_payload_is_the_release_without_developer_files(path: str, shipped: bool) -> None:
    assert update.in_payload(PurePosixPath(path)) is shipped


def test_the_repositorys_payload_has_what_a_start_needs() -> None:
    """From git's own list of this repository's files (skipped without git)."""
    try:
        listed = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=30
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available here")
    shipped = {p for p in listed if update.in_payload(PurePosixPath(p))}
    for needed in (
        "Start.bat",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "LICENSE",
        "README.md",
        "config/default.toml",
        "config/kits.toml",
        "native/pyproject.toml",
        "native/wheels.sha256",
        "src/facegen/__init__.py",
        "src/facegen/i18n/ko.toml",
        "src/facegen/data/kit_catalog.json",
        "src/facegen/web/templates/base.html",
    ):
        assert needed in shipped
    assert not any(p.startswith(("tests/", "docs/", "scripts/", "installer/")) for p in shipped)


def source_zip(
    path: Path, files: dict[str, str | bytes], top: str = "face-generator-1.0.0"
) -> Path:
    """A GitHub-style source archive: every file under one top folder."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(f"{top}/{name}" if top else name, data)
    return path


SOURCE = {
    "pyproject.toml": '[project]\nname = "facegen"\nversion = "1.0.0"\n',
    "Start.bat": "@echo off\r\n",
    "uv.lock": "lock 1.0.0\n",
    "src/facegen/__init__.py": "",
    "src/facegen/new_module.py": "NEW = True\n",
    "config/default.toml": "[ui]\n",
    "tests/test_x.py": "def test(): pass\n",
    "docs/images/a.jpg": b"\xff\xd8",
    "README.md": "facegen 1.0.0\n",
}


def test_an_archive_unpacks_to_its_payload(tmp_path: Path) -> None:
    archive = source_zip(tmp_path / "s.zip", SOURCE)
    files = update.unpack_payload(archive, tmp_path / "out")
    assert [str(f) for f in files] == [
        "README.md",
        "Start.bat",
        "config/default.toml",
        "pyproject.toml",
        "src/facegen/__init__.py",
        "src/facegen/new_module.py",
        "uv.lock",
    ]
    assert (tmp_path / "out" / "Start.bat").read_bytes() == b"@echo off\r\n"
    assert not (tmp_path / "out" / "tests").exists()


@pytest.mark.parametrize(
    ("files", "top", "why"),
    [
        ({"a/../../evil.py": "x"}, "f", "unsafe path"),
        ({"/abs.py": "x"}, "", "unsafe path"),
        ({"C:/x.py": "x"}, "", "unsafe path"),
        ({**SOURCE, "data/facegen.db": "x"}, "f", "this PC's own"),
        ({**SOURCE, "config/local.toml": "x"}, "f", "this PC's own"),
        ({"src/x.py": "x"}, "f", "not facegen's source"),
    ],
)
def test_an_archive_that_is_not_facegens_source_is_refused(
    tmp_path: Path, files: dict[str, str], top: str, why: str
) -> None:
    with pytest.raises(UpdateError, match=why):
        update.unpack_payload(source_zip(tmp_path / "s.zip", files, top), tmp_path / "out")


def test_an_archive_with_two_top_folders_or_a_link_is_refused(tmp_path: Path) -> None:
    two = tmp_path / "two.zip"
    with zipfile.ZipFile(two, "w") as zf:
        zf.writestr("a/pyproject.toml", "x")
        zf.writestr("b/Start.bat", "x")
    with pytest.raises(UpdateError, match="one folder"):
        update.unpack_payload(two, tmp_path / "out")
    link = tmp_path / "link.zip"
    with zipfile.ZipFile(link, "w") as zf:
        for name, data in SOURCE.items():
            zf.writestr(f"top/{name}", data)
        info = zipfile.ZipInfo("top/src/facegen/link.py")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(UpdateError, match="link"):
        update.unpack_payload(link, tmp_path / "out2")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(UpdateError, match="cannot be read"):
        update.unpack_payload(bad, tmp_path / "out3")


# --- putting a payload over a folder -------------------------------------------------------


def installed(root: Path) -> Path:
    """An installed facegen 0.9.0 with this PC's own files beside the release's."""
    files = {
        "pyproject.toml": '[project]\nname = "facegen"\nversion = "0.9.0"\n',
        "Start.bat": "@echo off\r\nrem old\r\n",
        "uv.lock": "lock 0.9.0\n",
        "src/facegen/__init__.py": "",
        "src/facegen/old_module.py": "OLD = True\n",
        "src/facegen/gone/only.py": "x\n",
        "src/facegen/__pycache__/old_module.cpython-312.pyc": "pyc",
        "config/default.toml": "[ui]\n",
        "config/local.toml": "[paths]\nfm_user_dir = 'C:\\\\FM'\n",
        "config/kits.toml": "# my own edits\n",
        "data/facegen.db": "the database",
        ".venv/Scripts/python.exe": "python",
        ".bin/uv.exe": "uv",
        "README.md": "facegen 0.9.0\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def staged(tmp_path: Path) -> tuple[Path, list[PurePosixPath]]:
    archive = source_zip(tmp_path / "s.zip", SOURCE)
    return tmp_path / "staged", update.unpack_payload(archive, tmp_path / "staged")


def test_a_payload_replaces_the_releases_files_and_keeps_this_pcs(tmp_path: Path) -> None:
    root = installed(tmp_path / "app")
    stage, files = staged(tmp_path)
    applied = update.apply_payload(root, stage, files, tmp_path / "backup")
    text = {p: (root / p).read_text(encoding="utf-8") for p in ("pyproject.toml", "uv.lock")}
    assert 'version = "1.0.0"' in text["pyproject.toml"] and text["uv.lock"] == "lock 1.0.0\n"
    assert (root / "src/facegen/new_module.py").is_file()
    # src/ files the new release lacks are gone, and so is the folder left empty.
    assert not (root / "src/facegen/old_module.py").exists()
    assert not (root / "src/facegen/gone").exists()
    # This PC's own: untouched.
    assert (root / "config/local.toml").read_text(encoding="utf-8").startswith("[paths]")
    assert (root / "data/facegen.db").read_text(encoding="utf-8") == "the database"
    assert (root / ".venv/Scripts/python.exe").is_file() and (root / ".bin/uv.exe").is_file()
    # A file the release does not ship (here config/kits.toml) is left as it is.
    assert (root / "config/kits.toml").read_text(encoding="utf-8") == "# my own edits\n"
    # Everything replaced or removed is in the backup.
    backup = tmp_path / "backup"
    assert (backup / "src/facegen/old_module.py").read_text(encoding="utf-8") == "OLD = True\n"
    assert 'version = "0.9.0"' in (backup / "pyproject.toml").read_text(encoding="utf-8")
    assert applied.removed == 2 and applied.unchanged == 2  # __init__.py, default.toml


def test_a_failure_while_replacing_puts_everything_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = installed(tmp_path / "app")
    before = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    stage, files = staged(tmp_path)
    real_replace = os.replace

    def fail_on_uv_lock(src: Any, dst: Any) -> None:
        if str(dst).endswith("uv.lock"):  # after Start.bat, README.md and others were written
            raise PermissionError("[WinError 32] being used by another process")
        real_replace(src, dst)

    monkeypatch.setattr(update.os, "replace", fail_on_uv_lock)
    with pytest.raises(UpdateError, match="nothing changed"):
        update.apply_payload(root, stage, files, tmp_path / "backup")
    after = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and not p.name.endswith(".update-part")
    }
    assert after == before


# --- `facegen update` ----------------------------------------------------------------------


class Archive:
    """The download: a source archive of `version` written where asked."""

    def __init__(self, tmp_path: Path, version: str = "1.0.0", **extra: str) -> None:
        files = {**SOURCE, **extra}
        files["pyproject.toml"] = f'[project]\nname = "facegen"\nversion = "{version}"\n'
        self.zip = source_zip(tmp_path / f"release-{version}.zip", files)
        self.urls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        self.urls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.zip, dest)


def run(
    tmp_path: Path, root: Path, *, latest: str = "v1.0.0", archive: Any = None, **kw: Any
) -> tuple[int, str]:
    out = io.StringIO()
    steps = Updater(
        root=root,
        fetch=lambda url: [release(latest)],
        fetch_archive=archive or Archive(tmp_path),
        running=kw.pop("running", lambda data: False),
        sleep=kw.pop("sleep", lambda s: None),
        now=lambda: 5000.0,
        **kw,
    )
    code = update.run_update(out, tmp_path / "data", steps=steps)
    return code, out.getvalue()


def test_an_update_puts_the_newest_release_over_the_folder(tmp_path: Path) -> None:
    root = installed(tmp_path / "app")
    archive = Archive(tmp_path)
    code, said = run(tmp_path, root, archive=archive)
    assert code == update.UPDATED, said
    assert "Updated facegen 0.9.0 to 1.0.0" in said
    assert archive.urls == [
        "https://github.com/junepaykim/face-generator/archive/refs/tags/v1.0.0.zip"
    ]
    assert update.installed_version(root) == "1.0.0"
    assert (root / "data/facegen.db").is_file()  # the app's own data folder is not root/data
    # The work folder keeps only the backup; the check is recorded as up to date.
    work = tmp_path / "data" / "update"
    assert sorted(p.name for p in work.iterdir()) == ["backup-0.9.0"]
    record = update.load_record(tmp_path / "data")
    assert record is not None and record.latest is not None and record.latest.version == "1.0.0"


def test_the_newest_version_is_left_alone(tmp_path: Path) -> None:
    root = installed(tmp_path / "app")
    archive = Archive(tmp_path)
    code, said = run(tmp_path, root, latest="v0.9.0", archive=archive)
    assert code == update.UPDATED and "0.9.0 is the newest version" in said
    assert archive.urls == []


def test_a_git_checkout_is_not_updated_in_place(tmp_path: Path) -> None:
    root = installed(tmp_path / "app")
    (root / ".git").mkdir()
    code, said = run(tmp_path, root)
    assert code == update.CHECKOUT and "git pull" in said
    assert update.installed_version(root) == "0.9.0"


def test_the_update_waits_for_facegen_to_quit(tmp_path: Path) -> None:
    root = installed(tmp_path / "app")
    answers = iter([True, True, True, False])
    code, said = run(tmp_path, root, running=lambda data: next(answers))
    assert code == update.UPDATED and "Waiting for facegen to quit" in said

    clock = iter(range(0, 1000, 60))
    code, said = run(
        tmp_path, installed(tmp_path / "app2"), running=lambda data: True, clock=lambda: next(clock)
    )
    assert code == update.STILL_RUNNING and "Quit it from the tray icon" in said


def test_a_failed_update_leaves_the_folder_as_it_was(tmp_path: Path) -> None:
    root = installed(tmp_path / "app")
    # The archive says another version than its tag.
    code, said = run(tmp_path, root, archive=Archive(tmp_path, version="2.0.0"))
    assert code == update.FAILED and "says it is version 2.0.0" in said
    assert update.installed_version(root) == "0.9.0"
    assert (root / "src/facegen/old_module.py").is_file()

    def offline(url: str) -> Any:
        raise UpdateError("could not reach GitHub: [Errno -3] name resolution")

    out = io.StringIO()
    code = update.run_update(
        out, tmp_path / "data", steps=Updater(root=root, fetch=offline, running=lambda d: False)
    )
    assert code == update.FAILED and "could not reach GitHub" in out.getvalue()


def test_facegen_update_on_this_checkout_says_to_use_git(tmp_path: Path) -> None:
    """The real command on this repository: a checkout, so it changes nothing."""
    from facegen.cli import main

    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = io.StringIO()
    code = main(["update", "--wait", "0", "--log", str(tmp_path / "start.log")], out=out)
    assert code == update.CHECKOUT and "git pull" in out.getvalue()
    assert "git pull" in (tmp_path / "start.log").read_text(encoding="utf-8")


# --- the installer --------------------------------------------------------------------------


def load_stage() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("installer_stage", ROOT / "installer/stage.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_installer_is_staged_from_the_same_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stage = load_stage()
    archive = source_zip(tmp_path / "s.zip", SOURCE, top="facegen")
    uv_zip = tmp_path / "uv.zip"
    with zipfile.ZipFile(uv_zip, "w") as zf:
        zf.writestr("uv.exe", b"MZ fake uv")
    sums = tmp_path / "uv.sha256"
    sums.write_text(f"{stage.sha256_of(uv_zip)}  {stage.UV_ZIP}\n", encoding="utf-8")
    stage.UV_SUMS = sums
    out = tmp_path / "stage"
    assert stage.main(["--archive", str(archive), "--uv-zip", str(uv_zip), "--out", str(out),
                       "--tag", "v1.0.0"]) == 0  # fmt: skip
    assert capsys.readouterr().out.strip().endswith("version=1.0.0")
    assert (out / ".bin" / "uv.exe").read_bytes() == b"MZ fake uv"
    assert (out / "facegen.ico").is_file()
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == sorted(
        [*(str(f) for f in update.unpack_payload(archive, tmp_path / "again")),
         ".bin/uv.exe", "facegen.ico"]
    )  # fmt: skip
    with pytest.raises(SystemExit, match="must be v1.0.0"):
        stage.main(["--archive", str(archive), "--out", str(out), "--tag", "v1.0.1"])
    sums.write_text(f"{'0' * 64}  {stage.UV_ZIP}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256"):
        stage.main(["--archive", str(archive), "--uv-zip", str(uv_zip), "--out", str(out)])


def test_the_pinned_uv_is_the_one_start_bat_installs() -> None:
    stage = load_stage()
    assert len(stage.expected_sum(stage.UV_ZIP)) == 64
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    start = (ROOT / "Start.bat").read_bytes().decode("ascii")
    assert 'UV_VERSION: "0.10.0"' in workflow and "astral.sh/uv/0.10.0/install.ps1" in start


def test_the_installer_script_agrees_with_the_app() -> None:
    from facegen.cli import build_parser
    from facegen.desktop import shell, shortcuts

    raw = (ROOT / "installer/facegen.iss").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 with BOM, for the Korean texts
    text = raw.decode("utf-8-sig")
    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={localappdata}\facegen" in text
    assert f"AppMutex={shell.APP_MUTEX}" in text
    # The uninstaller's command exists, and the shortcut name is the one the app refreshes.
    assert "-m facegen.cli remove-pack --yes" in text
    build_parser().parse_args(["remove-pack", "--yes"])
    assert shortcuts.LINK_NAME == "facegen.lnk" and r'Name: "{userprograms}\facegen";' in text
    # Uninstalling keeps data unless asked, and never deletes the data folder by itself.
    assert r'Name: "{app}\data"' not in text.split("[UninstallDelete]")[1].split("[Code]")[0]


def test_start_bat_updates_from_a_block_it_has_read_in_full() -> None:
    text = (ROOT / "Start.bat").read_bytes().decode("ascii")
    assert text.count("\r\n") == text.count("\n")
    block = text[text.index(":update\r\n") :]
    block = block[block.index("(\r\n") : block.index("\r\n)\r\n")]
    lines = [line.strip() for line in block.splitlines()[1:]]
    assert lines == [
        '"%PY%" -m facegen.cli update --log "%LOG%"',
        "if errorlevel 1 pause",
        "endlocal",
        'call "%~f0"',
        "exit /b",
    ]
    # --update skips asking a running facegen for its window: the update waits for it to quit.
    assert text.index("if defined UPDATE goto update") < text.index("launch --open-running")


def test_the_check_record_is_plain_json(tmp_path: Path) -> None:
    update.save_record(tmp_path, update.CheckRecord(1.5, None, "offline"))
    assert json.loads(update.check_path(tmp_path).read_text(encoding="utf-8")) == {
        "checked_at": 1.5,
        "latest": None,
        "error": "offline",
        "dismissed": None,
    }
