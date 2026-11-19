"""Windows desktop, Start Menu and Startup shortcuts for the desktop app.

Shortcuts run `.venv`'s `facegen-app.exe -m facegen.desktop app` without a console. That
program is pythonw.exe's launcher under facegen's name and icon (app_exe.py); where it
cannot be made, they run pythonw.exe itself. Start with Windows adds `--minimized`; it is
off until enabled in the tray, which creates or deletes only our facegen.lnk.
Known Folders follow relocated directories, including OneDrive. WScript.Shell writes the
shortcuts through PowerShell's -EncodedCommand so paths need no command-line quoting.

The installer's shortcuts start Start.bat, as there is no .venv until the first setup.
`refresh` (run by `facegen launch` once that setup is done) points those, and the ones an
earlier version pointed at pythonw.exe, at the app, so later starts show no console and
carry facegen's name; a shortcut that targets anything else is left alone.
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
    """The arguments the program is given: the desktop app, minimised to the tray or not."""
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
    folders: Mapping[str, str], *, program: str, root: str, icon: str, config: str | None = None
) -> list[Shortcut]:
    """The desktop's and the Start Menu's shortcut (each folder that is known)."""
    arguments = app_arguments(config=config)
    return [
        Shortcut(link_path(folders[name]), program, arguments, root, icon)
        for name in ("desktop", "programs")
        if folders.get(name)
    ]


def startup_shortcut(
    folder: str, *, program: str, root: str, icon: str, config: str | None = None
) -> Shortcut:
    """The Startup folder's shortcut: the app minimised to the tray."""
    return Shortcut(
        link_path(folder),
        program,
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


def refresh_script(links: Sequence[Shortcut], replacing: Sequence[str]) -> str:
    """PowerShell that rewrites each of `links` that exists and starts one of `replacing`
    (compared without case), and prints the paths it rewrote."""
    targets = "@(" + ", ".join(ps_string(t) for t in replacing) + ")"
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        "$ProgressPreference = 'SilentlyContinue'",
        "$shell = New-Object -ComObject WScript.Shell",
    ]
    for s in links:
        lines += [
            f"if (Test-Path -LiteralPath {ps_string(s.path)}) {{",
            f"  $s = $shell.CreateShortcut({ps_string(s.path)})",
            f"  if ({targets} -icontains $s.TargetPath) {{",
            f"    $s.TargetPath = {ps_string(s.target)}",
            f"    $s.Arguments = {ps_string(s.arguments)}",
            f"    $s.WorkingDirectory = {ps_string(s.working_dir)}",
            f"    $s.IconLocation = {ps_string(s.icon + ',0')}",
            f"    $s.Description = {ps_string(s.description)}",
        ]
        if s.minimized:
            lines.append(f"    $s.WindowStyle = {WINDOW_MINIMIZED}")
        lines += [
            "    $s.Save()",
            f"    Write-Output {ps_string(s.path)}",
            "  }",
            "}",
        ]
    return "\n".join(lines) + "\n"


def refresh(
    links: Sequence[Shortcut],
    replacing: Sequence[str],
    *,
    run: Callable[..., object] | None = None,
    exists: Callable[[str], bool] = os.path.isfile,
) -> list[str]:
    """Point those of `links` that start one of `replacing` (Start.bat, an earlier
    pythonw.exe) at the app (module docstring). Returns the shortcuts rewritten; raises
    ShortcutError. Without any of `links` on disk, or nothing to replace, nothing runs."""
    present = [s for s in links if exists(s.path)]
    if not present or not replacing:
        return []
    powershell = procs.windows_tool(POWERSHELL, "windows", os.environ)
    if powershell is None:
        raise ShortcutError("PowerShell was not found, so no shortcut could be updated")
    runner = run or procs.run_quiet
    try:
        done = runner(
            powershell_command(powershell, refresh_script(present, replacing)),
            timeout=POWERSHELL_SECONDS,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError) as exc:
        raise ShortcutError(f"PowerShell could not be started: {exc}") from exc
    code = getattr(done, "returncode", 1)
    out = (getattr(done, "stdout", "") or "").strip()
    if code != 0:
        text = (getattr(done, "stderr", "") or out).strip()
        last = text.splitlines()[-1] if text else f"exit code {code}"
        raise ShortcutError(f"the shortcut could not be updated: {last}")
    return [line.strip() for line in out.splitlines() if line.strip()]


@dataclass(frozen=True, slots=True)
class AppPaths:
    """What a shortcut to this checkout of facegen needs."""

    root: str
    program: str
    """facegen-app.exe, else the venv's pythonw.exe."""
    icon: str
    config: str | None = None


def _version() -> str:
    import facegen

    try:
        return str(facegen.__version__)
    except Exception:  # noqa: BLE001 - only the copy's version information shows it
        return "0"


def this_app(data_dir: Path, config: Path | None = None) -> AppPaths:
    """This checkout: the repository folder, the program the shortcuts run (facegen-app.exe,
    made or updated here on Windows, else the venv's pythonw.exe) and the icon (written into
    the data folder if it is not there yet)."""
    from facegen.desktop import app_exe
    from facegen.desktop.app_icon import write_ico
    from facegen.locations import PROJECT_ROOT

    icon = write_ico(data_dir / "app")
    program = app_exe.ensure(Path(sys.executable).parent, icon, _version())
    return AppPaths(
        str(PROJECT_ROOT),
        str(program) if program else pythonw_beside(sys.executable),
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
            folder, program=app.program, root=app.root, icon=app.icon, config=app.config
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
