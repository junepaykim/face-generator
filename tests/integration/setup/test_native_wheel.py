"""The prebuilt Rust extension: which wheel a PC takes, its checksum, the install, and how
the extension's state is reported (native_wheel.py, native.py, `facegen native`). No network
and no uv: downloads come from file:// URLs or fakes, the install is a fake."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest
from support.paths import ROOT

from facegen.cli import cmd_native, main
from facegen.platform import native
from facegen.setup import native_wheel
from facegen.setup.native_wheel import (
    Host,
    Plan,
    WheelError,
    choose,
    fits,
    parse_sums,
    parse_wheel_name,
)

WIN = "facegen_native-0.1.0-cp312-abi3-win_amd64.whl"
LINUX = "facegen_native-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
WINDOWS_PC = Host(system="win32", machine="AMD64", implementation="cpython", python=(3, 12))
LINUX_PC = Host(
    system="linux", machine="x86_64", implementation="cpython", python=(3, 12), glibc=(2, 39)
)


def sums_text(**files: bytes) -> str:
    lines = ["# a comment", ""]
    lines += [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in files.items()]
    return "\n".join(lines) + "\n"


# --- names, tags and sums ----------------------------------------------------------------------


def test_wheel_names() -> None:
    win = parse_wheel_name(WIN)
    assert win is not None
    assert (win.version, win.python, win.abi, win.platforms) == (
        "0.1.0",
        "cp312",
        "abi3",
        ("win_amd64",),
    )
    linux = parse_wheel_name(LINUX)
    assert linux is not None
    assert linux.platforms == ("manylinux_2_17_x86_64", "manylinux2014_x86_64")
    assert parse_wheel_name("numpy-2.0.0-cp312-cp312-win_amd64.whl") is None
    assert parse_wheel_name("facegen_native-0.1.0.tar.gz") is None
    assert parse_wheel_name("SHA256SUMS.txt") is None


def test_sums_lists() -> None:
    digest = "AB" * 32
    text = f"# comment\n\n{digest}  {WIN}\n{'cd' * 32} *{LINUX}\n"
    assert parse_sums(text) == {WIN: digest.lower(), LINUX: "cd" * 32}
    for bad in (f"{digest} {WIN} extra", "abc  x.whl", f"{digest}{WIN}"):
        with pytest.raises(WheelError):
            parse_sums(bad)


def test_the_tracked_files_agree() -> None:
    """native/wheels.sha256 parses and lists only facegen-native wheels; the version facegen
    calls is the crate's and the Python package's."""
    listed = parse_sums((ROOT / "native" / "wheels.sha256").read_text(encoding="utf-8"))
    assert all(parse_wheel_name(name) is not None for name in listed)
    pyproject = (ROOT / "native" / "pyproject.toml").read_text(encoding="utf-8")
    assert native_wheel.native_version(pyproject) == native.VERSION
    with (ROOT / "native" / "Cargo.toml").open("rb") as f:
        assert tomllib.load(f)["package"]["version"] == native.VERSION
    with pytest.raises(WheelError):
        native_wheel.native_version("[project]\nname = 'x'\n")


def test_which_pc_takes_which_wheel() -> None:
    win, linux = parse_wheel_name(WIN), parse_wheel_name(LINUX)
    assert win is not None and linux is not None
    assert fits(WINDOWS_PC, win) and not fits(WINDOWS_PC, linux)
    assert fits(LINUX_PC, linux) and not fits(LINUX_PC, win)
    # abi3 for 3.12 serves later Pythons, not earlier ones.
    assert fits(Host("win32", "AMD64", "cpython", (3, 13)), win)
    assert not fits(Host("win32", "AMD64", "cpython", (3, 11)), win)
    assert not fits(Host("win32", "ARM64", "cpython", (3, 12)), win)
    assert not fits(Host("win32", "AMD64", "pypy", (3, 12)), win)
    # manylinux_2_17 needs glibc 2.17 or later; musl (no glibc) and ARM take neither.
    assert fits(Host("linux", "x86_64", "cpython", (3, 12), glibc=(2, 17)), linux)
    assert not fits(Host("linux", "x86_64", "cpython", (3, 12), glibc=(2, 12)), linux)
    assert not fits(Host("linux", "x86_64", "cpython", (3, 12), glibc=None), linux)
    assert not fits(Host("linux", "aarch64", "cpython", (3, 12), glibc=(2, 39)), linux)
    assert not fits(Host("darwin", "arm64", "cpython", (3, 12)), linux)
    # A wheel for one CPython only (not abi3).
    exact = parse_wheel_name("facegen_native-0.1.0-cp312-cp312-win_amd64.whl")
    assert exact is not None
    assert fits(WINDOWS_PC, exact) and not fits(Host("win32", "AMD64", "cpython", (3, 13)), exact)


def test_choose_takes_the_wanted_version_for_this_pc() -> None:
    old = WIN.replace("0.1.0", "0.0.9")
    sums = {old: "0" * 64, LINUX: "1" * 64, WIN: "2" * 64}
    assert choose(WINDOWS_PC, "0.1.0", sums).filename == WIN  # type: ignore[union-attr]
    assert choose(LINUX_PC, "0.1.0", sums).filename == LINUX  # type: ignore[union-attr]
    assert choose(WINDOWS_PC, "0.0.9", sums).filename == old  # type: ignore[union-attr]
    assert choose(WINDOWS_PC, "0.2.0", sums) is None
    assert choose(LINUX_PC, "0.1.0", {WIN: "2" * 64}) is None


def test_this_host_describes_this_python() -> None:
    host = native_wheel.this_host()
    assert host.system == sys.platform and host.python == sys.version_info[:2]
    if sys.platform.startswith("linux"):
        assert host.glibc is None or host.glibc >= (2, 0)


def test_the_release_address() -> None:
    url = native_wheel.release_url("0.1.0", WIN)
    assert url.startswith("https://github.com/") and url.endswith(f"/native-v0.1.0/{WIN}")


# --- plan, download and install ----------------------------------------------------------------


def native_dir(tmp_path: Path, version: str = "0.1.0", **files: bytes) -> Path:
    folder = tmp_path / "native"
    folder.mkdir(parents=True)
    (folder / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    if files:
        (folder / "wheels.sha256").write_text(sums_text(**files))
    return folder


def test_plan_reads_the_version_and_the_list(tmp_path: Path) -> None:
    folder = native_dir(tmp_path, **{WIN: b"windows wheel", LINUX: b"linux wheel"})
    planned = native_wheel.plan(WINDOWS_PC, native_dir=folder)
    assert planned.wheel is not None and planned.wheel.filename == WIN
    assert planned.sha256 == hashlib.sha256(b"windows wheel").hexdigest()
    assert planned.url == native_wheel.release_url("0.1.0", WIN)
    assert WIN in planned.describe()
    empty = native_wheel.plan(WINDOWS_PC, native_dir=native_dir(tmp_path / "x", "0.1.0"))
    assert empty.wheel is None and "No prebuilt facegen-native 0.1.0" in empty.describe()


def test_install_checks_the_sum_before_installing(tmp_path: Path) -> None:
    good = b"the real wheel"
    planned = native_wheel.plan(WINDOWS_PC, native_dir=native_dir(tmp_path, **{WIN: good}))
    fetched: list[str] = []
    installed: list[tuple[Path, bytes]] = []

    def fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        dest.write_bytes(good)

    def installer(path: Path) -> None:
        installed.append((path, path.read_bytes()))

    assert native_wheel.install(planned, fetch=fetch, installer=installer) == WIN
    assert fetched == [planned.url]
    assert installed[0][0].name == WIN and installed[0][1] == good
    assert not installed[0][0].exists()  # downloaded into a temporary folder

    downloads: list[Path] = []

    def tampered(url: str, dest: Path) -> None:
        downloads.append(dest)
        dest.write_bytes(b"something else")

    installed.clear()
    with pytest.raises(WheelError, match="SHA-256"):
        native_wheel.install(planned, fetch=tampered, installer=installer)
    assert installed == [] and not downloads[0].exists()


def test_install_from_a_folder(tmp_path: Path) -> None:
    planned = native_wheel.plan(WINDOWS_PC, native_dir=native_dir(tmp_path, **{WIN: b"w"}))
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    installed: list[Path] = []
    with pytest.raises(WheelError, match="does not exist"):
        native_wheel.install(planned, source=wheels, installer=installed.append)
    (wheels / WIN).write_bytes(b"w")

    def no_network(url: str, dest: Path) -> None:
        raise AssertionError("downloaded")

    native_wheel.install(planned, source=wheels, fetch=no_network, installer=installed.append)
    assert installed == [wheels / WIN]
    (wheels / WIN).write_bytes(b"not w")
    with pytest.raises(WheelError):
        native_wheel.install(planned, source=wheels, installer=installed.append)
    assert (wheels / WIN).exists()  # the user's own file is left alone
    with pytest.raises(WheelError, match="No prebuilt"):
        native_wheel.install(Plan("0.1.0", None, None, None), installer=installed.append)


def test_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "wheel.whl"
    source.write_bytes(b"x" * 1000)
    dest = tmp_path / "copy.whl"
    native_wheel.download(source.as_uri(), dest)
    assert dest.read_bytes() == b"x" * 1000
    with pytest.raises(WheelError, match="could not download"):
        native_wheel.download((tmp_path / "missing.whl").as_uri(), dest)
    monkeypatch.setattr(native_wheel, "MAX_BYTES", 100)
    with pytest.raises(WheelError, match="larger than"):
        native_wheel.download(source.as_uri(), dest)


def test_uv_install_runs_uv_pip_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uv = tmp_path / "uv"
    uv.write_text("")
    assert native_wheel.find_uv({"UV": str(uv)}) == str(uv)
    monkeypatch.setattr(native_wheel.shutil, "which", lambda name: None)
    assert native_wheel.find_uv({"UV": str(tmp_path / "gone")}) is None
    monkeypatch.setenv("UV", str(uv))
    calls: list[list[str]] = []
    answer = subprocess.CompletedProcess([], 0, "", "")

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert kwargs["timeout"]
        return answer

    monkeypatch.setattr(native_wheel.procs, "run_quiet", run)
    native_wheel.uv_install(tmp_path / WIN, python="py")
    assert calls[0][:3] == [str(uv), "pip", "install"]
    assert calls[0][-1] == str(tmp_path / WIN) and "--python" in calls[0] and "py" in calls[0]
    answer = subprocess.CompletedProcess([], 2, "", "resolving\nerror: no space left\n")
    with pytest.raises(WheelError, match="no space left"):
        native_wheel.uv_install(tmp_path / WIN)
    monkeypatch.delenv("UV")
    with pytest.raises(WheelError, match="uv is not on PATH"):
        native_wheel.uv_install(tmp_path / WIN)


@pytest.mark.parametrize("version", [native.VERSION, "0.0.0"])
def test_install_verification_loads_the_extension_in_a_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    (tmp_path / "facegen_native.py").write_text("", encoding="utf-8")
    metadata = tmp_path / f"facegen_native-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Name: facegen-native\nVersion: {version}\n", encoding="utf-8"
    )
    run = native_wheel.procs.run_quiet

    def isolated(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return run(
            [args[0], "-S", *args[1:]],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ROOT / "src")))},
            **kwargs,
        )

    monkeypatch.setattr(native_wheel.procs, "run_quiet", isolated)
    assert native_wheel.imports_cleanly() is (version == native.VERSION)


# --- native.py: which extension is used ---------------------------------------------------------


def test_only_the_wanted_version_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("facegen_native")
    monkeypatch.setitem(sys.modules, "facegen_native", fake)
    monkeypatch.setattr(native, "installed_version", lambda: native.VERSION)
    assert native._load() == (fake, "")
    # An older wheel is not loaded at all (Windows could not replace a loaded one).
    monkeypatch.setattr(native, "installed_version", lambda: "0.0.1")
    ext, reason = native._load()
    assert ext is None and "0.0.1 is installed" in reason and native.VERSION in reason
    monkeypatch.setitem(sys.modules, "facegen_native", None)  # import fails
    monkeypatch.setattr(native, "installed_version", lambda: None)
    assert native._load() == (None, "not installed")


def test_status_says_what_is_in_use() -> None:
    assert native.Status(True, "0.1.0", "wheel", "").describe() == (
        "in use (facegen-native 0.1.0, prebuilt wheel)"
    )
    assert native.Status(True, "0.1.0", "build", "").describe().endswith("built from native/)")
    assert native.Status(False, None, None, "not installed").describe() == (
        "not in use: the Python versions run (not installed)"
    )
    now = native.status()
    assert now.in_use == (native.ext is not None)
    assert now.in_use or now.reason


# --- `facegen native` --------------------------------------------------------------------------


def run_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *argv: str,
    in_use: bool = False,
    served: bytes = b"wheel",
    listed: bytes | None = b"wheel",
    loads: bool = True,
) -> tuple[int, str, list[str], list[Path]]:
    monkeypatch.setattr(
        native,
        "status",
        lambda: native.Status(in_use, "0.1.0" if in_use else None, None, "not installed"),
    )
    folder = native_dir(tmp_path, **({WIN: listed} if listed is not None else {}))
    fetched: list[str] = []
    installed: list[Path] = []

    def fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        dest.write_bytes(served)

    args = argparse.Namespace(install="--install" in argv, wheel_dir=None)
    out = io.StringIO()
    code = cmd_native(
        args,
        out,
        host=WINDOWS_PC,
        native_dir=folder,
        fetch=fetch,
        installer=installed.append,
        loads=lambda: loads,
    )
    return code, out.getvalue(), fetched, installed


def test_native_command_installs_the_listed_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, text, fetched, installed = run_native(tmp_path, monkeypatch, "--install")
    assert code == 0 and f"Installed {WIN} (SHA-256 checked)" in text
    assert fetched == [native_wheel.release_url("0.1.0", WIN)]
    assert [p.name for p in installed] == [WIN]


def test_native_command_without_install_only_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, text, fetched, installed = run_native(tmp_path, monkeypatch)
    assert code == 0 and fetched == [] and installed == []
    assert "not in use" in text and WIN in text and "facegen native --install" in text


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"served": b"tampered"}, "SHA-256"),
        ({"listed": None}, "No prebuilt facegen-native 0.1.0"),
        ({"loads": False}, "does not load"),
    ],
)
def test_native_command_failures_are_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kw: dict, says: str
) -> None:
    code, text, _, installed = run_native(tmp_path, monkeypatch, "--install", **kw)
    assert code == 3 and says in text
    if "loads" not in kw:
        assert installed == [] and "Python versions" in text


def test_native_command_leaves_an_extension_in_use_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, text, fetched, installed = run_native(tmp_path, monkeypatch, "--install", in_use=True)
    assert code == 0 and "Nothing to install" in text and fetched == [] and installed == []


def test_native_command_needs_no_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    opened: list[object] = []
    monkeypatch.setattr("facegen.cli.db.connect", lambda *a, **k: opened.append(a))
    out = io.StringIO()
    assert main(["native"], out=out) == 0
    assert "Rust code" in out.getvalue() and opened == []
