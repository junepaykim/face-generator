"""Finding FM27's folders on a PC (fm_paths.py), with the PC described by a fake disk."""

from __future__ import annotations

import json
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

import pytest

from facegen import fm_paths
from facegen.fm_paths import (
    PathStyleError,
    WindowsInfo,
    choose_user_dir,
    documents_folders,
    find_install_dir,
    find_user_dir,
    install_dir_candidates,
    is_fm27_title,
    native_path,
    parse_powershell_info,
    steam_install_dir_name,
    steam_library_paths,
    steam_roots,
    user_dir_candidates,
    windows_path_of_wsl,
    wsl_path,
)

FM = ("Sports Interactive", "Football Manager 27")
WIN_FM = r"\Sports Interactive\Football Manager 27"


class FakeDisk:
    """Files by path (text, or a save's modification time) and empty folders. Like Windows's
    drives, it ignores case in paths but keeps names as they were written."""

    def __init__(self, flavour: type[PurePath]) -> None:
        self.flavour = flavour
        self.sep = "\\" if flavour is PureWindowsPath else "/"
        self.files: dict[str, tuple[str, str, float]] = {}  # key: (path, text, mtime)
        self.dirs: dict[str, str] = {}  # key: path

    def _path(self, path: PurePath | str) -> str:
        return str(self.flavour(path))

    def add_file(self, path: str, text: str = "", mtime: float = 0.0) -> FakeDisk:
        p = self._path(path)
        self.files[p.lower()] = (p, text, mtime)
        return self

    def add_dir(self, path: str) -> FakeDisk:
        p = self._path(path)
        self.dirs[p.lower()] = p
        return self

    def add_save(self, fm_dir: str, name: str = "Career.fm", mtime: float = 1.0) -> FakeDisk:
        return self.add_file(self._path(self.flavour(fm_dir, "games", name)), mtime=mtime)

    def _below(self, path: PurePath) -> list[str]:
        """What lies below a folder, relative to it, with the names' own case."""
        prefix = self._path(path).rstrip(self.sep).lower() + self.sep
        every = [p for p, _, _ in self.files.values()] + list(self.dirs.values())
        return [p[len(prefix) :] for p in every if p.lower().startswith(prefix)]

    def is_dir(self, path: PurePath) -> bool:
        return self._path(path).lower() in self.dirs or bool(self._below(path))

    def read_text(self, path: PurePath) -> str | None:
        hit = self.files.get(self._path(path).lower())
        return hit[1] if hit else None

    def list_dir(self, path: PurePath) -> list[str]:
        return sorted({rest.split(self.sep)[0] for rest in self._below(path)})

    def newest_save(self, games: PurePath) -> float | None:
        prefix = self._path(games).lower() + self.sep
        times = [
            t
            for key, (_, _, t) in self.files.items()
            if key.startswith(prefix) and key.endswith(".fm") and self.sep not in key[len(prefix) :]
        ]
        return max(times) if times else None


def win(*parts: str) -> PureWindowsPath:
    return PureWindowsPath(*parts)


# --- paths between Windows and WSL ------------------------------------------------------------


@pytest.mark.parametrize(
    "windows,expected",
    [
        (r"C:\Users\me\Documents", "/mnt/c/Users/me/Documents"),
        ("D:/SteamLibrary", "/mnt/d/SteamLibrary"),
        ("c:\\", "/mnt/c"),
        ("E:", "/mnt/e"),
        (r"C:\Users\김\문서\\", "/mnt/c/Users/김/문서"),
        (r"\\server\share\x", None),
        ("/mnt/c/Users", None),
        ("relative\\path", None),
    ],
)
def test_wsl_path(windows: str, expected: str | None) -> None:
    assert wsl_path(windows) == expected


def test_windows_path_of_wsl() -> None:
    assert windows_path_of_wsl("/mnt/c/Users/me/x") == r"C:\Users\me\x"
    assert windows_path_of_wsl("/mnt/d") == "D:\\"
    assert windows_path_of_wsl("/home/me") is None


def test_a_windows_path_in_local_toml_works_from_wsl() -> None:
    assert native_path(r"C:\Users\me\FM27", "wsl") == "/mnt/c/Users/me/FM27"
    assert native_path("/mnt/c/Users/me/FM27", "wsl") == "/mnt/c/Users/me/FM27"
    assert native_path("data", "wsl") == "data"


def test_a_wsl_path_on_windows_is_refused_with_the_windows_form() -> None:
    with pytest.raises(PathStyleError, match=r"write it as 'C:\\Users\\me\\FM27'"):
        native_path("/mnt/c/Users/me/FM27", "windows")
    assert native_path(r"C:\Users\me\FM27", "windows") == r"C:\Users\me\FM27"
    assert native_path("data", "windows") == "data"


def test_a_windows_path_on_linux_is_refused() -> None:
    with pytest.raises(PathStyleError, match="this is Linux"):
        native_path(r"C:\Users\me", "linux")
    assert native_path("/home/me/fm", "linux") == "/home/me/fm"


# --- what Windows says ------------------------------------------------------------------------


def test_powershell_output_is_read_into_windows_info() -> None:
    text = (
        "\ufeffdocuments=C:\\Users\\김\\OneDrive\\문서\r\n"
        "env:USERPROFILE=C:\\Users\\김\r\n"
        "env:OneDrive=C:\\Users\\김\\OneDrive\r\n"
        "env:OneDriveConsumer=\r\n"
        "env:ProgramFiles(x86)=C:\\Program Files (x86)\r\n"
        "steam=c:/program files (x86)/steam\r\n"
        "noise without a sign\r\n"
    )
    info = parse_powershell_info(text, ["C:\\", "D:\\"])
    assert info.documents == r"C:\Users\김\OneDrive\문서"
    assert info.env == {
        "USERPROFILE": r"C:\Users\김",
        "OneDrive": r"C:\Users\김\OneDrive",
        "ProgramFiles(x86)": r"C:\Program Files (x86)",
    }
    assert info.steam_path == "c:/program files (x86)/steam"
    assert info.drives == ("C:\\", "D:\\")


def test_documents_folders_in_order_without_repeats() -> None:
    info = WindowsInfo(
        documents=r"C:\Users\me\Documents",
        env={"USERPROFILE": r"C:\Users\me", "OneDrive": r"C:\Users\me\OneDrive"},
    )
    assert documents_folders(info) == [
        r"C:\Users\me\Documents",
        r"C:\Users\me\OneDrive\Documents",
    ]


# --- the user folder --------------------------------------------------------------------------


def test_plain_documents_on_windows() -> None:
    info = WindowsInfo(documents=r"C:\Users\me\Documents", env={"USERPROFILE": r"C:\Users\me"})
    disk = FakeDisk(PureWindowsPath).add_save(r"C:\Users\me\Documents" + WIN_FM)
    assert find_user_dir("windows", info, disk) == win(r"C:\Users\me\Documents" + WIN_FM)


def test_onedrive_documents_on_windows() -> None:
    """Known Folder Move: Documents lives in OneDrive, under a localised name."""
    info = WindowsInfo(
        documents=r"C:\Users\me\OneDrive\문서",
        env={"USERPROFILE": r"C:\Users\me", "OneDrive": r"C:\Users\me\OneDrive"},
    )
    disk = FakeDisk(PureWindowsPath).add_save(r"C:\Users\me\OneDrive\문서" + WIN_FM)
    assert find_user_dir("windows", info, disk) == win(r"C:\Users\me\OneDrive\문서" + WIN_FM)


def test_a_moved_documents_folder_on_windows() -> None:
    info = WindowsInfo(documents=r"D:\Docs", env={"USERPROFILE": r"C:\Users\me"})
    disk = FakeDisk(PureWindowsPath).add_save(r"D:\Docs" + WIN_FM)
    assert user_dir_candidates("windows", info)[0] == win(r"D:\Docs" + WIN_FM)
    assert find_user_dir("windows", info, disk) == win(r"D:\Docs" + WIN_FM)


def test_onedrive_from_the_environment_when_the_known_folder_is_unknown() -> None:
    info = WindowsInfo(env={"USERPROFILE": r"C:\Users\me", "OneDrive": r"C:\Users\me\OneDrive"})
    disk = FakeDisk(PureWindowsPath).add_save(r"C:\Users\me\OneDrive\Documents" + WIN_FM)
    assert find_user_dir("windows", info, disk) == win(r"C:\Users\me\OneDrive\Documents" + WIN_FM)


def test_the_folder_with_the_newest_save_wins() -> None:
    info = WindowsInfo(env={"USERPROFILE": r"C:\Users\me", "OneDrive": r"C:\Users\me\OneDrive"})
    disk = (
        FakeDisk(PureWindowsPath)
        .add_save(r"C:\Users\me\OneDrive\Documents" + WIN_FM, mtime=100)
        .add_save(r"C:\Users\me\Documents" + WIN_FM, mtime=200)
    )
    assert find_user_dir("windows", info, disk) == win(r"C:\Users\me\Documents" + WIN_FM)


def test_a_folder_with_saves_beats_an_earlier_one_with_an_empty_games_folder() -> None:
    info = WindowsInfo(env={"USERPROFILE": r"C:\Users\me", "OneDrive": r"C:\Users\me\OneDrive"})
    disk = (
        FakeDisk(PureWindowsPath)
        .add_dir(r"C:\Users\me\OneDrive\Documents" + WIN_FM + r"\games")
        .add_save(r"C:\Users\me\Documents" + WIN_FM)
    )
    assert find_user_dir("windows", info, disk) == win(r"C:\Users\me\Documents" + WIN_FM)


def test_without_saves_the_first_folder_with_games_wins() -> None:
    info = WindowsInfo(env={"USERPROFILE": r"C:\Users\me", "OneDrive": r"C:\Users\me\OneDrive"})
    disk = (
        FakeDisk(PureWindowsPath)
        .add_dir(r"C:\Users\me\OneDrive\Documents" + WIN_FM + r"\games")
        .add_dir(r"C:\Users\me\Documents" + WIN_FM + r"\games")
    )
    assert find_user_dir("windows", info, disk) == win(r"C:\Users\me\OneDrive\Documents" + WIN_FM)


def test_a_folder_without_games_is_not_fm27s() -> None:
    info = WindowsInfo(documents=r"C:\Users\me\Documents")
    disk = FakeDisk(PureWindowsPath).add_dir(r"C:\Users\me\Documents" + WIN_FM + r"\graphics")
    assert find_user_dir("windows", info, disk) is None


def test_nothing_found() -> None:
    info = WindowsInfo(documents=r"C:\Users\me\Documents", env={"USERPROFILE": r"C:\Users\me"})
    assert find_user_dir("windows", info, FakeDisk(PureWindowsPath)) is None
    assert find_user_dir("wsl", WindowsInfo(), FakeDisk(PurePosixPath)) is None
    assert find_user_dir("linux", info, FakeDisk(PurePosixPath)) is None


def test_wsl_uses_what_windows_says_under_mnt() -> None:
    info = WindowsInfo(documents=r"C:\Users\me\Documents", env={"USERPROFILE": r"C:\Users\me"})
    disk = FakeDisk(PurePosixPath).add_save("/mnt/c/Users/me/Documents/" + "/".join(FM))
    assert find_user_dir("wsl", info, disk) == PurePosixPath("/mnt/c/Users/me/Documents", *FM)


def test_wsl_scans_every_windows_user_when_windows_cannot_be_asked() -> None:
    """Two Windows users, each with FM27: the newer save decides."""
    disk = (
        FakeDisk(PurePosixPath)
        .add_dir("/mnt/c/Users/Public/Documents")
        .add_save("/mnt/c/Users/ann/Documents/" + "/".join(FM), mtime=100)
        .add_save("/mnt/c/Users/bob/OneDrive - Club/Documents/" + "/".join(FM), mtime=300)
    )
    assert find_user_dir("wsl", WindowsInfo(), disk) == PurePosixPath(
        "/mnt/c/Users/bob/OneDrive - Club/Documents", *FM
    )


def test_wsl_prefers_the_windows_users_own_folder_to_another_users_newer_save() -> None:
    info = WindowsInfo(env={"USERPROFILE": r"C:\Users\ann"})
    disk = (
        FakeDisk(PurePosixPath)
        .add_save("/mnt/c/Users/ann/Documents/" + "/".join(FM), mtime=100)
        .add_save("/mnt/c/Users/bob/Documents/" + "/".join(FM), mtime=300)
    )
    assert find_user_dir("wsl", info, disk) == PurePosixPath("/mnt/c/Users/ann/Documents", *FM)


def test_choose_takes_the_first_of_two_equally_new_saves() -> None:
    disk = FakeDisk(PurePosixPath).add_save("/a", mtime=5).add_save("/b", mtime=5)
    assert choose_user_dir([PurePosixPath("/b"), PurePosixPath("/a")], disk) == PurePosixPath("/b")


# --- the install folder -----------------------------------------------------------------------

LIBRARY_VDF = r"""
"libraryfolders"
{
	"0"
	{
		"path"		"C:\\Program Files (x86)\\Steam"
		"apps"
		{
			"228980"		"333950174"
		}
	}
	"1"
	{
		"path"		"D:\\SteamLibrary"
		"apps"
		{
			"4530370"		"8732056054"
		}
	}
}
"""

FM27_ACF = r"""
"AppState"
{
	"appid"		"4530370"
	"name"		"Football Manager 27"
	"installdir"		"Football Manager 27"
}
"""


def test_steam_library_parsing() -> None:
    assert steam_library_paths(LIBRARY_VDF) == [r"C:\Program Files (x86)\Steam", r"D:\SteamLibrary"]
    assert steam_install_dir_name(FM27_ACF) == "Football Manager 27"
    assert steam_install_dir_name('"AppState" { "appid" "1" }') is None


def steam_pc(flavour: type[PurePath], root: str, library: str) -> FakeDisk:
    """Steam on C:, FM27 in a second library on D:."""
    sep = "\\" if flavour is PureWindowsPath else "/"
    disk = FakeDisk(flavour)
    disk.add_file(sep.join([root, "steamapps", "libraryfolders.vdf"]), LIBRARY_VDF)
    disk.add_file(sep.join([library, "steamapps", "appmanifest_4530370.acf"]), FM27_ACF)
    disk.add_dir(sep.join([library, "steamapps", "common", "Football Manager 27", "fm_Data"]))
    return disk


def test_fm27_in_a_second_steam_library_on_windows() -> None:
    info = WindowsInfo(env={"ProgramFiles(x86)": r"C:\Program Files (x86)"})
    disk = steam_pc(PureWindowsPath, r"C:\Program Files (x86)\Steam", r"D:\SteamLibrary")
    assert find_install_dir("windows", info, disk) == win(
        r"D:\SteamLibrary\steamapps\common\Football Manager 27"
    )


def test_fm27_in_a_second_steam_library_from_wsl() -> None:
    disk = steam_pc(PurePosixPath, "/mnt/c/Program Files (x86)/Steam", "/mnt/d/SteamLibrary")
    assert find_install_dir("wsl", WindowsInfo(), disk) == PurePosixPath(
        "/mnt/d/SteamLibrary/steamapps/common/Football Manager 27"
    )


def test_steam_from_the_registry_comes_first() -> None:
    info = WindowsInfo(steam_path="e:/games/steam")
    disk = FakeDisk(PureWindowsPath)
    disk.add_dir(r"E:\games\steam\steamapps\common\Football Manager 27\fm_Data")
    assert install_dir_candidates("windows", info, disk)[0] == win(
        r"E:\games\steam\steamapps\common\Football Manager 27"
    )
    assert find_install_dir("windows", info, disk) == win(
        r"E:\games\steam\steamapps\common\Football Manager 27"
    )


def test_steams_registry_path_is_spelled_as_the_folder_is() -> None:
    info = WindowsInfo(
        steam_path="c:/program files (x86)/steam",
        env={"ProgramFiles(x86)": r"C:\Program Files (x86)"},
    )
    assert steam_roots(info)[0] == r"C:\Program Files (x86)\Steam"
    assert steam_roots(WindowsInfo(steam_path="e:/games/steam"))[0] == r"e:\games\steam"


def test_a_steam_folder_without_the_game_is_not_an_install() -> None:
    disk = FakeDisk(PureWindowsPath).add_dir(
        r"C:\Program Files (x86)\Steam\steamapps\common\Football Manager 27"
    )
    assert find_install_dir("windows", WindowsInfo(), disk) is None


def test_epic_manifest() -> None:
    disk = FakeDisk(PureWindowsPath)
    manifests = r"C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests"
    for name, title, location in [
        ("A.item", "Football Manager 27 Editor", r"E:\Epic\FM27 Editor"),
        ("B.item", "Football Manager 26", r"E:\Epic\FM26"),
        ("C.item", "Football Manager 27", r"E:\Epic\Football Manager 27"),
    ]:
        disk.add_file(
            manifests + "\\" + name,
            json.dumps({"DisplayName": title, "InstallLocation": location}),
        )
        disk.add_dir(location + r"\fm_Data")
    disk.add_file(manifests + r"\broken.item", "{not json")
    assert find_install_dir("windows", WindowsInfo(), disk) == win(r"E:\Epic\Football Manager 27")


def test_xbox_app_folder_on_another_drive() -> None:
    info = WindowsInfo(drives=("C:\\", "F:\\"))
    disk = FakeDisk(PureWindowsPath).add_dir(r"F:\XboxGames\Football Manager 27\Content\fm_Data")
    assert find_install_dir("windows", info, disk) == win(
        r"F:\XboxGames\Football Manager 27\Content"
    )


def test_xbox_app_folder_without_a_content_level() -> None:
    """As reported for FM26's Game Pass install: XboxGames\\Football Manager 26\\shared\\..."""
    info = WindowsInfo(drives=("C:\\",))
    disk = FakeDisk(PureWindowsPath).add_dir(r"C:\XboxGames\Football Manager 27\fm_Data")
    assert find_install_dir("windows", info, disk) == win(r"C:\XboxGames\Football Manager 27")


def test_no_install_found() -> None:
    assert find_install_dir("windows", WindowsInfo(), FakeDisk(PureWindowsPath)) is None
    assert install_dir_candidates("linux", WindowsInfo(), FakeDisk(PurePosixPath)) == []


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Football Manager 27", True),
        ("Football Manager 2027", True),
        ("FOOTBALL MANAGER 27", True),
        ("Football Manager 27 Editor", False),
        ("Football Manager 26", False),
        ("Football Manager 2027 Touch", False),
        ("Football Manager 270", False),
    ],
)
def test_fm27_titles(title: str, expected: bool) -> None:
    assert is_fm27_title(title) is expected


# --- caching ----------------------------------------------------------------------------------


def test_detection_is_cached_and_a_missing_folder_is_looked_for_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    answers = [None, Path("/found")]

    def find(kind: str, info: WindowsInfo, disk: object) -> PurePath | None:
        calls.append(kind)
        return answers[len(calls) - 1] if len(calls) <= len(answers) else Path("/other")

    monkeypatch.setattr(fm_paths, "find_user_dir", find)
    monkeypatch.setattr(fm_paths, "system_info", lambda kind, disk: WindowsInfo())
    monkeypatch.setattr(fm_paths.platform_info, "current", lambda: "wsl")
    fm_paths.forget()
    now = [0.0]
    try:
        assert fm_paths.detect_user_dir(lambda: now[0]) is None
        now[0] = 10.0
        assert fm_paths.detect_user_dir(lambda: now[0]) is None  # not looked for again yet
        now[0] = fm_paths.RETRY_SECONDS + 1
        assert fm_paths.detect_user_dir(lambda: now[0]) == Path("/found")
        now[0] = 10_000.0
        assert fm_paths.detect_user_dir(lambda: now[0]) == Path("/found")  # kept
        assert len(calls) == 2
    finally:
        fm_paths.forget()
