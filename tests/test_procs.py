"""Other programs run without a console window (procs.py), and which system this is."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from facegen import procs
from facegen.platform_info import kind
from facegen.procs import (
    CREATE_NO_WINDOW,
    linux_process_running,
    quiet_flags,
    tasklist_command,
    tasklist_says_running,
    windows_tool,
)


@pytest.mark.parametrize(
    "platform,proc_version,expected",
    [
        ("win32", None, "windows"),
        ("linux", "Linux version 6.18.58.1-microsoft-standard-WSL2 (root@...)", "wsl"),
        ("linux", "Linux version 4.4.0-19041-Microsoft (Microsoft@Microsoft.com)", "wsl"),
        ("linux", "Linux version 6.8.0-45-generic (buildd@lcy02-amd64-075)", "linux"),
        ("linux", None, "linux"),
        ("darwin", None, "linux"),
    ],
)
def test_platform_kind(platform: str, proc_version: str | None, expected: str) -> None:
    assert kind(platform, proc_version) == expected


def test_tasklist_command_line() -> None:
    assert tasklist_command(r"C:\Windows\System32\tasklist.exe") == [
        r"C:\Windows\System32\tasklist.exe",
        "/FI",
        "IMAGENAME eq fm.exe",
        "/FO",
        "CSV",
        "/NH",
    ]


@pytest.mark.parametrize(
    "stdout,expected",
    [
        (b'"fm.exe","12345","Console","1","2,000,000 K"\r\n', True),
        (b'"FM.EXE","1","Console","1","1 K"\r\n', True),
        (b"INFO: No tasks are running which match the specified criteria.\r\n", False),
        ("정보: 지정된 조건에 일치하는 작업이 없습니다.".encode("cp949"), False),
        (b'"fm.exe.bak","1"', False),
    ],
)
def test_tasklist_output(stdout: bytes, expected: bool) -> None:
    assert tasklist_says_running(stdout) is expected


def test_linux_process_names() -> None:
    assert linux_process_running(["bash\n", "fm.exe\n"])
    assert not linux_process_running(["bash\n", "fm\n"])


def test_quiet_flags_per_platform() -> None:
    assert quiet_flags("nt") == CREATE_NO_WINDOW == 0x08000000
    assert quiet_flags("posix") == 0


def capture_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append({"args": args, **kwargs})
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_run_quiet_hides_the_console_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = capture_run(monkeypatch)
    procs.run_quiet(["tasklist.exe"], timeout=5, os_name="nt", capture_output=True)
    assert seen[0]["creationflags"] == CREATE_NO_WINDOW
    assert seen[0]["timeout"] == 5 and seen[0]["check"] is False
    procs.run_quiet(["x"], timeout=5, os_name="nt", creationflags=0x200)
    assert seen[1]["creationflags"] == 0x200 | CREATE_NO_WINDOW


def test_run_quiet_adds_nothing_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = capture_run(monkeypatch)
    procs.run_quiet(["true"], timeout=5, os_name="posix")
    assert "creationflags" not in seen[0]


def test_popen_quiet_hides_the_console_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: seen.append(kw) or "child")
    assert procs.popen_quiet(["python.exe"], os_name="nt", creationflags=0x200) == "child"
    assert seen[0]["creationflags"] == 0x200 | CREATE_NO_WINDOW


def test_popen_quiet_really_starts_a_process() -> None:
    child = procs.popen_quiet(
        [sys.executable, "-c", "print('ok')"], stdout=subprocess.PIPE, text=True
    )
    out, _ = child.communicate(timeout=30)
    assert out.strip() == "ok" and child.returncode == 0


@pytest.mark.parametrize("os_name", ["nt", "posix"])
@pytest.mark.parametrize("detached", [True, False])
def test_json_child_spawn_preserves_console_and_environment_policy(
    os_name: str, detached: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: seen.append(kw) or "child")
    env = {"HF_HUB_OFFLINE": "1", "CUSTOM": "inherited by caller"}
    assert procs.popen_json(["python"], detached=detached, env=env, os_name=os_name) == "child"
    options = seen[0]
    assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.PIPE
    assert options["text"] and options["encoding"] == "utf-8" and options["errors"] == "replace"
    assert options["bufsize"] == 1 and options["env"] == env
    if os_name == "nt":
        assert options["creationflags"] == procs.CREATE_NO_WINDOW | (0x200 if detached else 0)
        assert "start_new_session" not in options
    else:
        assert "creationflags" not in options
        assert options.get("start_new_session", False) is detached

    procs.popen_json(["python"], detached=False, stdin=subprocess.DEVNULL, os_name=os_name)
    assert seen[1]["stdin"] == subprocess.DEVNULL and seen[1]["env"] is None


def test_windows_tool_on_path_first() -> None:
    found = windows_tool("tasklist.exe", "wsl", {}, which=lambda n: f"/on/path/{n}")
    assert found == "/on/path/tasklist.exe"


def test_windows_tool_from_system32_on_windows() -> None:
    seen: list[str] = []
    found = windows_tool(
        "tasklist.exe",
        "windows",
        {"SystemRoot": r"D:\WIN"},
        which=lambda n: None,
        exists=lambda p: seen.append(p) or True,
    )
    assert found == r"D:\WIN\System32\tasklist.exe" == seen[0]


def test_windows_tool_from_system32_on_wsl() -> None:
    found = windows_tool(
        r"WindowsPowerShell\v1.0\powershell.exe",
        "wsl",
        {},
        which=lambda n: None,
        exists=lambda p: True,
    )
    assert found == "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def test_windows_tool_missing() -> None:
    assert (
        windows_tool("tasklist.exe", "wsl", {}, which=lambda n: None, exists=lambda p: False)
        is None
    )
    assert windows_tool("tasklist.exe", "linux", {}, which=lambda n: "/x") is None
