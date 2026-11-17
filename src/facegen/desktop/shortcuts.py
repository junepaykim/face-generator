"""Windows shortcuts to the desktop app (Phase 11): the desktop and the Start Menu
(`Start.bat --shortcut`, which runs `facegen shortcuts`), and Start with Windows.

A shortcut starts `pythonw.exe -m facegen.desktop app` from the repository's `.venv`, so no
console window opens (desktop.py). Start with Windows is the same shortcut in the Startup
folder with `--minimized`: facegen starts in the tray, without its window. It is off by
default and only ever changed from the tray's visible "Start with Windows" item, as it changes
how the PC starts. Turning it off deletes our own `facegen.lnk` there and nothing else.

The folders are Windows's Known Folders (Desktop, Programs, Startup), wherever they were moved
(OneDrive's desktop included). The shortcuts are written by WScript.Shell in one PowerShell
call, given as -EncodedCommand so that no path needs quoting on a command line.

The pure parts (the command a shortcut runs, the PowerShell text) take plain values; the
tests check them on any system.
"""

from __future__ import annotations

import base64
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from facegen.platform import procs

LINK_NAME = "facegen.lnk"
DESCRIPTION = "facegen: anime portraits for FM27's players"
FOLDER_IDS = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "programs": "A77F5D77-2E2B-44C3-A6A2-ABA601054A51",
    "startup": "B97D20BB-F46A-4C97-BA10-5E3608430854",
}
"""FOLDERID_Desktop, FOLDERID_Programs (the Start Menu's Programs) and FOLDERID_Startup."""
FALLBACK = {
    "desktop": ("USERPROFILE", "Desktop"),
    "programs": ("APPDATA", r"Microsoft\Windows\Start Menu\Programs"),
    "startup": ("APPDATA", r"Microsoft\Windows\Start Menu\Programs\Startup"),
}
"""Where each is when Windows cannot be asked: under an environment variable."""
WINDOW_MINIMIZED = 7
"""WScript.Shell's WindowStyle for "minimised"."""
POWERSHELL = r"WindowsPowerShell\v1.0\powershell.exe"
POWERSHELL_SECONDS = 60.0


class ShortcutError(RuntimeError):
    """A shortcut could not be written or removed."""


@dataclass(frozen=True, slots=True)
class Shortcut:
    path: str
    """Where the .lnk goes."""
    target: str
    arguments: str
    working_dir: str
    icon: str
    description: str = DESCRIPTION
    minimized: bool = False


def quote_arg(text: str) -> str:
    """One command-line argument, in double quotes when it has a space."""
    return f'"{text}"' if any(c.isspace() for c in text) or not text else text


def app_arguments(*, minimized: bool = False, config: str | None = None) -> str:
    """The arguments pythonw.exe is given: the desktop app, minimised to the tray or not."""
    parts = ["-m", "facegen.desktop"]
    if config:
        parts += ["--config", quote_arg(config)]
    parts.append("app")
    if minimized:
        parts.append("--minimized")
    return " ".join(parts)


def pythonw_beside(python: str) -> str:
    """pythonw.exe in the same folder as this python.exe (a venv's Scripts folder has both)."""
    path = PureWindowsPath(python)
    return str(path.with_name("pythonw.exe"))


def link_path(folder: str) -> str:
    """Our shortcut's path in `folder`: a Windows folder (C:\\...) in Windows's form, else this
    system's (the tests' folders)."""
    from facegen.platform.fm_paths import is_windows_drive_path

    if is_windows_drive_path(folder):
        return str(PureWindowsPath(folder, LINK_NAME))
    return str(Path(folder) / LINK_NAME)


def app_shortcuts(
    folders: Mapping[str, str], *, pythonw: str, root: str, icon: str, config: str | None = None
) -> list[Shortcut]:
    """The desktop's and the Start Menu's shortcut (each folder that is known)."""
    arguments = app_arguments(config=config)
    return [
        Shortcut(link_path(folders[name]), pythonw, arguments, root, icon)
        for name in ("desktop", "programs")
        if folders.get(name)
    ]


def startup_shortcut(
    folder: str, *, pythonw: str, root: str, icon: str, config: str | None = None
) -> Shortcut:
    """The Startup folder's shortcut: the app minimised to the tray."""
    return Shortcut(
        link_path(folder),
        pythonw,
        app_arguments(minimized=True, config=config),
        root,
        icon,
        minimized=True,
    )


def ps_string(text: str) -> str:
    """A PowerShell string literal that holds `text` as it is: single quotes, doubled inside."""
    return "'" + text.replace("'", "''") + "'"


def powershell_script(shortcuts: Sequence[Shortcut]) -> str:
    """The PowerShell that writes `shortcuts` with WScript.Shell."""
    lines = [
        "$ErrorActionPreference = 'Stop'",
        # Its messages in UTF-8 whatever the console's code page, and no progress records.
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        "$ProgressPreference = 'SilentlyContinue'",
        "$shell = New-Object -ComObject WScript.Shell",
    ]
    for s in shortcuts:
        lines += [
            f"$s = $shell.CreateShortcut({ps_string(s.path)})",
            f"$s.TargetPath = {ps_string(s.target)}",
            f"$s.Arguments = {ps_string(s.arguments)}",
            f"$s.WorkingDirectory = {ps_string(s.working_dir)}",
            f"$s.IconLocation = {ps_string(s.icon + ',0')}",
            f"$s.Description = {ps_string(s.description)}",
        ]
        if s.minimized:
            lines.append(f"$s.WindowStyle = {WINDOW_MINIMIZED}")
        lines.append("$s.Save()")
    return "\n".join(lines) + "\n"


def encoded_command(script: str) -> str:
    """PowerShell's -EncodedCommand form: base64 of the script in UTF-16LE."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def powershell_command(powershell: str, script: str) -> list[str]:
    return [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        encoded_command(script),
    ]


def windows_folders(
    env: Mapping[str, str] | None = None,
    ask: Callable[[str], str | None] | None = None,
) -> dict[str, str]:
    """The desktop, Start Menu Programs and Startup folders: Windows's Known Folders, else
    the usual places under the user's profile. Empty values are left out."""
    env = os.environ if env is None else env
    if ask is None:
        from facegen.platform.fm_paths import known_folder

        ask = known_folder
    found: dict[str, str] = {}
    for name, folder_id in FOLDER_IDS.items():
        try:
            path = ask(folder_id)
        except (OSError, AttributeError, ValueError):
            path = None
        if not path:
            var, sub = FALLBACK[name]
            base = env.get(var)
            path = str(PureWindowsPath(base, sub)) if base else None
        if path:
            found[name] = path
    return found


def write(shortcuts: Sequence[Shortcut], *, run: Callable[..., object] | None = None) -> None:
    """Write the shortcuts (one PowerShell call). Raises ShortcutError."""
    if not shortcuts:
        return
    powershell = procs.windows_tool(POWERSHELL, "windows", os.environ)
    if powershell is None:
        raise ShortcutError("PowerShell was not found, so no shortcut could be written")
    runner = run or procs.run_quiet
    try:
        done = runner(
            powershell_command(powershell, powershell_script(shortcuts)),
            timeout=POWERSHELL_SECONDS,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError) as exc:
        raise ShortcutError(f"PowerShell could not be started: {exc}") from exc
    code = getattr(done, "returncode", 1)
    if code != 0:
        text = (getattr(done, "stderr", "") or getattr(done, "stdout", "") or "").strip()
        last = text.splitlines()[-1] if text else f"exit code {code}"
        raise ShortcutError(f"the shortcut could not be written: {last}")


@dataclass(frozen=True, slots=True)
class AppPaths:
    """What a shortcut to this checkout of facegen needs."""

    root: str
    pythonw: str
    icon: str
    config: str | None = None


def this_app(data_dir: Path, config: Path | None = None) -> AppPaths:
    """This checkout: the repository folder, the venv's pythonw.exe, the icon (written into
    the data folder if it is not there yet)."""
    from facegen.desktop.app_icon import write_ico
    from facegen.locations import PROJECT_ROOT

    icon = write_ico(data_dir / "app")
    return AppPaths(
        str(PROJECT_ROOT),
        pythonw_beside(sys.executable),
        str(icon),
        str(config.resolve()) if config else None,
    )


class StartupShortcut:
    """Start with Windows (app_controller.StartupSwitch): our shortcut in the Startup folder."""

    def __init__(
        self,
        folder: str,
        app: AppPaths,
        *,
        writer: Callable[[Sequence[Shortcut]], None] = write,
    ) -> None:
        self.shortcut = startup_shortcut(
            folder, pythonw=app.pythonw, root=app.root, icon=app.icon, config=app.config
        )
        self._writer = writer

    @property
    def path(self) -> Path:
        return Path(self.shortcut.path)

    def is_on(self) -> bool:
        return self.path.is_file()

    def set(self, on: bool) -> None:
        if on:
            self._writer([self.shortcut])
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ShortcutError(f"could not remove {self.path}: {exc}") from exc
