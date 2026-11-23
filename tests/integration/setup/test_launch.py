"""`facegen launch`, what Start.bat runs after `uv sync` (Phase 11): Start.bat stays thin and
its decisions are here."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

from facegen.cli import main
from facegen.desktop import instance
from facegen.setup import launch


def test_tee_writes_the_console_and_the_log_without_progress_lines() -> None:
    console, log = io.StringIO(), io.StringIO()
    tee = launch.Tee(console, log)
    print("Rust code   in use", file=tee)
    print("   42%  4.9 GB of 11.6 GB  file 3 of 24: unet", file=tee)
    tee.write("partial")
    tee.flush()
    assert console.getvalue().splitlines() == [
        "Rust code   in use",
        "   42%  4.9 GB of 11.6 GB  file 3 of 24: unet",
        "partial",
    ]
    lines = log.getvalue().splitlines()
    assert len(lines) == 1 and lines[0].endswith(" Rust code   in use")  # time-stamped


def test_a_large_log_is_set_aside(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "start.log"
    f = launch.open_log(path)
    assert f is not None
    f.write("x" * (launch.LOG_LIMIT + 1))
    f.close()
    f = launch.open_log(path)
    assert f is not None
    f.close()
    assert (tmp_path / "logs" / "start.log.1").stat().st_size > launch.LOG_LIMIT
    assert path.stat().st_size == 0
    assert launch.open_log(None) is None


def test_open_running_says_nothing_runs(tmp_path: Path) -> None:
    out = io.StringIO()
    assert launch.open_running(tmp_path, out) == launch.NOT_RUNNING
    assert out.getvalue() == "" and not (tmp_path / instance.OPEN_REQUEST_NAME).exists()


def test_open_running_asks_the_running_app_for_its_window(tmp_path: Path) -> None:
    lock = instance.InstanceLock(tmp_path)
    assert lock.acquire()
    taken = threading.Event()

    def running_app() -> None:  # what desktop.Refresher does every second
        for _ in range(100):
            if instance.take_open_request(tmp_path):
                taken.set()
                return
            time.sleep(0.05)

    threading.Thread(target=running_app).start()
    out = io.StringIO()
    try:
        assert launch.open_running(tmp_path, out, timeout=5) == launch.OPENED
    finally:
        lock.release()
    assert taken.is_set() and "its window opens" in out.getvalue()


def test_open_running_reports_an_app_that_does_not_answer(tmp_path: Path) -> None:
    lock = instance.InstanceLock(tmp_path)
    assert lock.acquire()
    out = io.StringIO()
    try:
        assert launch.open_running(tmp_path, out, timeout=0.2) == launch.NO_ANSWER
    finally:
        lock.release()
    assert "did not answer" in out.getvalue()


def test_prepare_goes_on_after_a_missing_wheel_and_declined_models() -> None:
    out = io.StringIO()
    steps: list[str] = []

    def native(o: io.StringIO) -> int:
        steps.append("native")
        return 3

    def models(o: io.StringIO) -> int:
        steps.append("models")
        return 3

    assert launch.prepare(out, native_step=native, models_step=models) == 0  # type: ignore[arg-type]
    assert steps == ["native", "models"]
    text = out.getvalue()
    assert "Warning: the prebuilt Rust extension is not installed" in text
    assert "The dashboard offers the download" in text


def test_an_unexpected_error_in_a_step_is_a_warning() -> None:
    out = io.StringIO()

    def broken(o: io.StringIO) -> int:
        raise ConnectionResetError("the network went away")

    assert launch.prepare(out, native_step=broken, models_step=lambda o: 0) == 0  # type: ignore[arg-type]
    assert "Warning: ConnectionResetError: the network went away" in out.getvalue()


def test_prepare_is_quiet_when_all_is_there() -> None:
    out = io.StringIO()
    assert launch.prepare(out, native_step=lambda o: 0, models_step=lambda o: 0) == 0
    assert "Warning" not in out.getvalue() and "not downloaded" not in out.getvalue()


def test_the_download_question_and_its_answer_reach_the_log() -> None:
    console, log = io.StringIO(), io.StringIO()
    tee = launch.Tee(console, log)
    ask = launch.asker(tee, read=lambda: "y")  # type: ignore[arg-type]
    assert ask("Download 11.6 GB from Hugging Face? [y/N] ") == "y"
    assert console.getvalue() == "Download 11.6 GB from Hugging Face? [y/N] "
    assert "answer: 'y'" in log.getvalue()


def test_facegen_launch_open_running_from_the_command_line(config_file: Path) -> None:
    out = io.StringIO()
    log = config_file.parent / "data" / "logs" / "start.log"
    code = main(
        ["--config", str(config_file), "launch", "--open-running", "--log", str(log)], out=out
    )
    assert code == launch.NOT_RUNNING
    assert log.is_file()  # opened for appending, nothing to say
    assert not (config_file.parent / "data" / "facegen.db").exists()  # no database needed


def test_facegen_launch_runs_the_wheel_and_the_models_steps(config_file: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from facegen.setup import operations as setup_operations

    seen: list[tuple[str, object]] = []

    def fake_native(*, install, report):  # type: ignore[no-untyped-def]
        seen.append(("native", install))
        report("Rust code   not in use")
        return 3

    def fake_models(*, model, download, confirmed, report, ask, interactive, run_download):  # type: ignore[no-untyped-def]
        seen.append(("models", (model, download, confirmed)))
        return 0

    monkeypatch.setattr(setup_operations, "prepare_native", fake_native)
    monkeypatch.setattr(setup_operations, "prepare_models", fake_models)
    log = config_file.parent / "start.log"
    out = io.StringIO()
    assert main(["--config", str(config_file), "launch", "--log", str(log)], out=out) == 0
    # The model chosen in Settings (the default: no database yet); it asks, never --yes.
    assert seen == [("native", True), ("models", ("animagine", True, False))]
    assert "Rust code   not in use" in log.read_text(encoding="utf-8")
    assert "Warning: the prebuilt Rust extension" in out.getvalue()
    assert not (config_file.parent / "data" / "facegen.db").exists()


def test_facegen_launch_offers_the_model_chosen_in_settings(config_file: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Phase 15e: only the selected model's files are offered."""
    from facegen.persistence import db
    from facegen.persistence.settings import load_settings, set_override
    from facegen.setup import operations as setup_operations

    conn = db.connect(load_settings(defaults_path=config_file).paths.db_path)
    set_override(conn, "style.model", "realvis", config_file)
    conn.close()
    chosen: list[str] = []
    monkeypatch.setattr(setup_operations, "prepare_native", lambda **kwargs: 0)
    monkeypatch.setattr(
        setup_operations, "prepare_models", lambda *, model, **kwargs: chosen.append(model) or 0
    )
    assert main(["--config", str(config_file), "launch"], out=io.StringIO()) == 0
    assert chosen == ["realvis"]


# -- a local file copied from WSL onto Windows ---------------------------------------------------

WSL_LOCAL = """# This PC's own folders.
[paths]
fm_user_dir = "/mnt/c/Users/me/Documents/Sports Interactive/Football Manager 27"
"""
WINDOWS_FORM = r"C:\Users\me\Documents\Sports Interactive\Football Manager 27"


class Console(io.StringIO):
    """stdin of a console window: a tty, with the user's answer typed in."""

    def isatty(self) -> bool:
        return True


def on_windows_with_a_wsl_local_file(config_file: Path, monkeypatch) -> Path:  # type: ignore[no-untyped-def]
    from facegen.setup import operations as setup_operations

    monkeypatch.setattr("facegen.platform.platform_info.current", lambda: "windows")
    monkeypatch.setattr(setup_operations, "prepare_native", lambda **kwargs: 0)
    monkeypatch.setattr(setup_operations, "prepare_models", lambda **kwargs: 0)
    local = config_file.with_name("local.toml")
    local.write_text(WSL_LOCAL, encoding="utf-8")
    return local


def test_wsl_paths_are_found_on_windows_only(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text(WSL_LOCAL, encoding="utf-8")
    found = {"fm_user_dir": (WSL_LOCAL.split('"')[1], WINDOWS_FORM)}
    assert launch.wsl_paths(local, "windows") == found
    assert launch.wsl_paths(local, "wsl") == {}
    local.write_text(f"[paths]\nfm_user_dir = '{WINDOWS_FORM}'\n", encoding="utf-8")
    assert launch.wsl_paths(local, "windows") == {}
    local.write_text("[paths\n", encoding="utf-8")
    assert launch.wsl_paths(local, "windows") == {}
    assert launch.wsl_paths(tmp_path / "missing.toml", "windows") == {}


def test_a_wsl_path_copied_to_windows_is_rewritten_after_one_yes(
    config_file: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    local = on_windows_with_a_wsl_local_file(config_file, monkeypatch)
    monkeypatch.setattr("sys.stdin", Console("\n"))  # Enter: the default, yes
    out = io.StringIO()
    assert main(["--config", str(config_file), "launch"], out=out) == 0
    text = out.getvalue()
    assert "names a folder as WSL sees it" in text and WINDOWS_FORM in text
    assert "Write it in Windows's form" in text and "Written in" in text
    written = local.read_text(encoding="utf-8")
    assert f"fm_user_dir = '{WINDOWS_FORM}'" in written
    assert written.startswith("# This PC's own folders.")  # the rest of the file kept
    assert "Traceback" not in text


def test_a_no_leaves_the_file_and_says_why_without_a_traceback(
    config_file: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    local = on_windows_with_a_wsl_local_file(config_file, monkeypatch)
    monkeypatch.setattr("sys.stdin", Console("n\n"))
    out = io.StringIO()
    assert main(["--config", str(config_file), "launch"], out=out) == 2
    assert local.read_text(encoding="utf-8") == WSL_LOCAL
    assert "Kitshot's settings cannot be read" in out.getvalue()
    assert f"on Windows write it as '{WINDOWS_FORM}'" in out.getvalue()


def test_without_a_console_nothing_is_asked_or_changed(config_file: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    local = on_windows_with_a_wsl_local_file(config_file, monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))  # not a tty
    out = io.StringIO()
    assert main(["--config", str(config_file), "launch"], out=out) == 2
    assert local.read_text(encoding="utf-8") == WSL_LOCAL
    assert "Write it in Windows's form" not in out.getvalue()


def test_open_running_says_nothing_about_settings_it_cannot_read(
    config_file: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    on_windows_with_a_wsl_local_file(config_file, monkeypatch)
    log = config_file.parent / "start.log"
    out = io.StringIO()
    code = main(
        ["--config", str(config_file), "launch", "--open-running", "--log", str(log)], out=out
    )
    assert code == launch.NOT_RUNNING and out.getvalue() == ""  # the next step explains it
    assert "settings cannot be read" in log.read_text(encoding="utf-8")


def test_the_apps_message_box_says_a_settings_error_plainly() -> None:
    from facegen.desktop import shell as desktop
    from facegen.persistence.settings import SettingsError

    error = SettingsError("config/local.toml, paths.fm_user_dir: write it as 'C:\\x'")
    assert desktop.explain(error) == "config/local.toml, paths.fm_user_dir: write it as 'C:\\x'"
