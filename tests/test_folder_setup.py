"""First-run setup of FM27's folder (Phase 10a): the check, the local file, the dialog switch."""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

import pytest

from facegen import folder_setup, platform_info
from facegen.folder_setup import (
    LocalFileError,
    RealDisk,
    check_folder,
    dialog_available,
    value_to_write,
    with_path_key,
    write_local_path,
)
from facegen.settings import load_local

DISK = RealDisk()
NATIVE = platform_info.current()


@pytest.fixture
def fm(tmp_path: Path) -> Path:
    """Documents/Sports Interactive/Football Manager 27 with a save in games/."""
    folder = tmp_path / "Documents" / "Sports Interactive" / "Football Manager 27"
    (folder / "games").mkdir(parents=True)
    (folder / "graphics").mkdir()
    (folder / "games" / "Career.fm").write_bytes(b"save")
    return folder


# --- the check ------------------------------------------------------------------------------


def test_fm27s_folder_is_taken_as_it_is(fm: Path) -> None:
    check = check_folder(str(fm), NATIVE, DISK)
    assert check.ok and check.folder == fm and check.message == ""


def test_quotes_and_spaces_around_a_pasted_path_are_dropped(fm: Path) -> None:
    assert check_folder(f'  "{fm}"  ', NATIVE, DISK).folder == fm


@pytest.mark.parametrize("inside", ["games", "graphics", "games/Career.fm"])
def test_a_folder_or_save_inside_is_corrected_to_the_folder_above(fm: Path, inside: str) -> None:
    check = check_folder(str(fm / inside), NATIVE, DISK)
    assert check.folder == fm
    assert str(fm) in check.message


@pytest.mark.parametrize("above", ["Documents", "Documents/Sports Interactive"])
def test_the_documents_or_sports_interactive_folder_is_corrected_downwards(
    fm: Path, above: str
) -> None:
    root = fm.parents[2]
    check = check_folder(str(root / above), NATIVE, DISK)
    assert check.folder == fm and "inside" in check.message


def test_a_new_install_with_no_saves_yet_is_accepted_by_its_name(tmp_path: Path) -> None:
    folder = tmp_path / "Sports Interactive" / "Football Manager 27"
    folder.mkdir(parents=True)
    assert check_folder(str(folder), NATIVE, DISK).folder == folder


def test_a_wrong_folder_is_refused_with_what_was_missing(tmp_path: Path) -> None:
    other = tmp_path / "Football Manager 26"
    other.mkdir()
    check = check_folder(str(other), NATIVE, DISK)
    assert not check.ok
    assert "has no games folder" in check.message
    assert "Football Manager 27 inside Sports Interactive" in check.message


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("", "Enter FM27's folder"),
        ("   ", "Enter FM27's folder"),
        ("relative/folder", "not a full path"),
        ("/no/such/folder", "does not exist"),
        ("C:\\Users\\me", "is a Windows path, and this is Linux"),
    ],
)
def test_refusals(text: str, why: str) -> None:
    check = check_folder(text, "linux", DISK)
    assert not check.ok and why in check.message


def test_a_file_that_is_not_a_save_is_refused(fm: Path) -> None:
    (fm / "notes.txt").write_text("x", encoding="utf-8")
    assert "is a file, not a folder" in check_folder(str(fm / "notes.txt"), NATIVE, DISK).message


def test_a_save_outside_fm27s_games_folder_is_refused(tmp_path: Path) -> None:
    (tmp_path / "copy.fm").write_bytes(b"save")
    assert (
        "not in FM27's games folder"
        in check_folder(str(tmp_path / "copy.fm"), NATIVE, DISK).message
    )


class WindowsDisk:
    """A Windows PC's folders, by their lower-cased paths."""

    def __init__(self, folders: set[str], files: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        self.folders = {f.lower() for f in folders}
        self.files = {f.lower() for f in files}

    def is_dir(self, path: PurePath) -> bool:
        return str(path).lower() in self.folders

    def is_file(self, path: PurePath) -> bool:
        return str(path).lower() in self.files


FM_WIN = r"C:\Users\me\OneDrive\Documents\Sports Interactive\Football Manager 27"


def test_on_windows_names_are_matched_whatever_their_case() -> None:
    disk = WindowsDisk({FM_WIN, FM_WIN + r"\games"}, {FM_WIN + r"\games\career.fm"})
    check = check_folder(FM_WIN.upper() + r"\GAMES\CAREER.FM", "windows", disk)
    assert check.ok and str(check.folder).lower() == FM_WIN.lower()
    assert isinstance(check.folder, PureWindowsPath)
    assert check_folder(FM_WIN.replace("\\", "/"), "windows", disk).ok


def test_on_windows_a_wsl_path_is_refused_with_the_form_to_write() -> None:
    check = check_folder("/mnt/c/Users/me/FM27", "windows", WindowsDisk(set()))
    assert not check.ok and r"C:\Users\me\FM27" in check.message


def test_on_wsl_a_windows_path_is_read_as_wsl_sees_it(tmp_path: Path) -> None:
    folder = "/mnt/c/users/me/documents/sports interactive/football manager 27"
    disk = WindowsDisk({folder, folder + "/games"})
    check = check_folder(
        r"C:\Users\me\Documents\Sports Interactive\Football Manager 27", "wsl", disk
    )
    assert check.ok
    assert str(check.folder) == "/mnt/c/Users/me/Documents/Sports Interactive/Football Manager 27"


# --- the local file -------------------------------------------------------------------------


def _paths(text: str) -> dict[str, object]:
    return tomllib.loads(text)["paths"]


def test_a_new_local_file_gets_the_header_and_the_folder() -> None:
    text = with_path_key(folder_setup.LOCAL_HEADER, "fm_user_dir", FM_WIN)
    assert text.startswith("# This PC's own folders")
    assert f"fm_user_dir = '{FM_WIN}'" in text  # single quotes: backslashes stay as they are
    assert _paths(text) == {"fm_user_dir": FM_WIN}


def test_the_key_is_replaced_where_it_is_and_everything_else_is_kept() -> None:
    before = (
        "# my notes\n"
        "[paths]\n"
        "# the old folder\n"
        'fm_user_dir = "/mnt/c/old"   \n'
        "fm_install_dir = 'D:\\SteamLibrary\\steamapps\\common\\Football Manager 27'\n"
        "# trailing comment\n"
    )
    after = with_path_key(before, "fm_user_dir", "/mnt/d/FM27")
    lines = after.splitlines()
    assert lines[0] == "# my notes" and lines[2] == "# the old folder"
    assert lines[3] == "fm_user_dir = '/mnt/d/FM27'"
    assert lines[-1] == "# trailing comment"
    assert _paths(after) == {
        "fm_user_dir": "/mnt/d/FM27",
        "fm_install_dir": "D:\\SteamLibrary\\steamapps\\common\\Football Manager 27",
    }


def test_the_key_is_added_to_an_existing_paths_table() -> None:
    before = "[paths]\nfm_install_dir = '/games/FM27'\n\n# end\n"
    after = with_path_key(before, "fm_user_dir", "/home/me/FM27")
    assert after.splitlines()[:3] == [
        "[paths]",
        "fm_install_dir = '/games/FM27'",
        "fm_user_dir = '/home/me/FM27'",
    ]
    assert after.endswith("# end\n")


def test_a_path_with_a_single_quote_is_written_as_a_basic_string() -> None:
    folder = r"C:\Users\O'Neil\Documents\Sports Interactive\Football Manager 27"
    text = with_path_key("", "fm_user_dir", folder)
    assert _paths(text) == {"fm_user_dir": folder}
    assert "'O'Neil" not in text


def test_a_local_file_that_is_not_toml_is_not_overwritten() -> None:
    with pytest.raises(LocalFileError, match="not valid TOML"):
        with_path_key('[paths]\nfm_user_dir = "C:\\Users\\me"\n', "fm_user_dir", "/x")


def test_write_local_path_writes_a_file_settings_can_read(tmp_path: Path) -> None:
    local = tmp_path / "config" / "local.toml"
    write_local_path(local, "fm_user_dir", "/home/me/FM27")
    write_local_path(local, "fm_user_dir", "/home/me/FM27 moved")
    defaults = {"paths.fm_user_dir": "", "paths.fm_install_dir": ""}
    assert load_local(defaults, local) == {"paths.fm_user_dir": "/home/me/FM27 moved"}
    assert [p.name for p in local.parent.iterdir()] == ["local.toml"]  # no temporary left


def test_a_failed_write_leaves_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.toml"
    write_local_path(local, "fm_user_dir", "/old")

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(folder_setup.os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_local_path(local, "fm_user_dir", "/new")
    assert _paths(local.read_text(encoding="utf-8")) == {"fm_user_dir": "/old"}
    assert [p.name for p in tmp_path.iterdir()] == ["local.toml"]


def test_on_wsl_a_windows_drive_folder_is_written_in_windows_form() -> None:
    folder = PurePosixPath("/mnt/c/Users/me/Documents/Sports Interactive/Football Manager 27")
    assert value_to_write(folder, "wsl") == (
        r"C:\Users\me\Documents\Sports Interactive\Football Manager 27"
    )
    assert value_to_write(PurePosixPath("/home/me/FM27"), "wsl") == "/home/me/FM27"
    assert value_to_write(PurePosixPath("/home/me/FM27"), "linux") == "/home/me/FM27"
    assert value_to_write(PureWindowsPath(FM_WIN), "windows") == FM_WIN


# --- Browse -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "env", "has_tk", "available"),
    [
        ("windows", {}, True, True),
        ("windows", {}, False, False),
        ("wsl", {"DISPLAY": ":0"}, True, False),  # WSL: hidden, whatever WSLg offers
        ("linux", {"DISPLAY": ":0"}, True, True),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, True, True),
        ("linux", {}, True, False),
    ],
)
def test_browse_is_offered_only_with_a_display_and_tkinter(
    kind: str, env: dict[str, str], has_tk: bool, available: bool
) -> None:
    assert dialog_available(kind, env, has_tk) is available  # type: ignore[arg-type]


def test_browse_reads_the_dialogs_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    answers = iter(['noise\n{"path": "C:/Users/me/FM27"}\n', '{"path": null}\n', "garbage"])
    calls: list[list[str]] = []

    def run(args: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, next(answers), "")

    monkeypatch.setattr(folder_setup.procs, "run_quiet", run)
    assert folder_setup.browse("C:/start") == "C:/Users/me/FM27"
    assert calls[0][-1] == "C:/start"
    assert folder_setup.browse() is None  # cancelled
    assert folder_setup.browse() is None  # no answer


def test_browse_that_times_out_is_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def run(args: list[str], **kw: object) -> None:
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(folder_setup.procs, "run_quiet", run)
    assert folder_setup.browse() is None
