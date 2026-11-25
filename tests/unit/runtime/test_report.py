"""The Log page's report (Phase 17i, D11): what it says, the home folder hidden in every path,
and the worker's log oldest first with its levels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PureWindowsPath

import pytest

from kitshot.core.job_records import Job
from kitshot.core.settings_values import Settings
from kitshot.runtime.contracts import LogLine, PollerStatus, SyncStatus, WorkerStatus
from kitshot.runtime.report import ReportFacts, build, log_line, redact
from kitshot.setup.hardware import Hardware

HOME = r"C:\Users\Sam Lee"
FM = HOME + r"\Documents\Sports Interactive\Football Manager 27"


def job() -> Job:
    return Job(
        id=7,
        unique_id=1002,
        kind="initial",
        group="a",
        target_club_id=None,
        target_band=None,
        name="Mine Two",
        at_my_club=True,
        goalkeeper=False,
        height_cm=None,
        strength=None,
        age=None,
    )


def facts(settings: Settings, **kw: object) -> ReportFacts:
    paths = replace(
        settings.paths,
        fm_user_dir=PureWindowsPath(FM),  # type: ignore[arg-type]
        data_dir=PureWindowsPath(HOME + r"\AppData\Local\kitshot\data"),  # type: ignore[arg-type]
    )
    worker = WorkerStatus(
        state="running",
        detail="Mine Two: initial A",
        current=job(),
        paused=False,
        gpu_fault=False,
        generate_now=False,
        models_loaded=True,
        done=12,
        failed=1,
        last_error=None,
        log=(
            LogLine("18:55:01", "Loading the image model..."),
            LogLine("18:55:09", "  attempt 1 failed the check: blurry", "warning"),
            LogLine("18:55:10", f"Publishing failed: cannot write {FM}\\graphics", "error"),
            LogLine("18:55:11", "Published 1 picture(s)."),
        ),
        reason="drawing",
        values={"name": "Mine Two", "kind": "initial", "group": "A"},
        since="2026-11-23T09:55:01+00:00",
    )
    base: dict[str, object] = {
        "made_at": "2026-11-23 18:56:00",
        "version": "0.1.6",
        "install": "installed",
        "system": "Windows 11 (10.0.26100)",
        "python": "3.12.8",
        "card": Hardware("NVIDIA GeForce RTX 4080", 16 * 2**30, torch="2.9.0", cuda="13.2"),
        "card_asked": True,
        "settings": replace(settings, paths=paths),
        "models_ready": True,
        "career": "Sam Lee - Redwood",
        "worker": worker,
        "poller": PollerStatus(
            "synced",
            "newest save Sam Lee - 레드우드.fm (saved 2026-11-22 18:19) is synced",
            "18:55:10",
            "synced",
            {"name": "Sam Lee - 레드우드.fm", "saved": "2026-11-22 18:19"},
        ),
        "watching": True,
        "last_sync": {
            "save_file": "Sam Lee - 레드우드.fm",
            "finished_at": "2026-11-22T18:20:02",
            "build": "27.1.0",
            "game_date": "2027-08-01",
            "trigger": "poller",
            "error": None,
        },
        "sync": SyncStatus(message="Synced Sam Lee - 레드우드.fm", warnings=["2 kits unknown"]),
        "homes": [HOME],
    }
    base.update(kw)
    return ReportFacts(**base)  # type: ignore[arg-type]


def test_the_report_says_what_helps_and_hides_the_home_folder(settings: Settings) -> None:
    text = build(facts(settings))
    lines = text.splitlines()
    assert lines[0] == "Kitshot report, 2026-11-23 18:56:00"
    for expected in (
        "Version: 0.1.6 (installed)",
        "System: Windows 11 (10.0.26100); Python 3.12.8",
        "Graphics card: NVIDIA GeForce RTX 4080, 16.0 GiB (torch 2.9.0, CUDA 13.2)",
        "Image model: RealVisXL V5.0 (fp16, INT8 UNet blocks); files on this PC",
        "Drawn as: man",
        "Picture shape: 1:1, rendered at 1024x1024",
        "  Detailed (A): published 1024 px, 3 candidate(s), detail pass x1.5 (1536x1536)",
        "Drawn automatically: A, B, C",
        "Draw while FM27 runs: off",
        "VAE tiling: off",
        "Career: Sam Lee - Redwood",
        "  Drawing: Mine Two (1002), initial A",
        "  This run: 12 made, 1 sent to review; models loaded",
        "Last sync: Sam Lee - 레드우드.fm at 2026-11-22T18:20:02",
        "  Warning: 2 kits unknown",
        r"FM27 folder: %USERPROFILE%\Documents\Sports Interactive\Football Manager 27",
        r"Data folder: %USERPROFILE%\AppData\Local\kitshot\data",
    ):
        assert expected in lines, expected
    assert any(line.startswith("Save watcher: synced (checked 18:55:10)") for line in lines)
    # The Windows user's folder is gone from every path, the log's too; names stay.
    assert "Users" not in text and "%USERPROFILE%\\Documents" in text
    assert "Sam Lee - 레드우드.fm" in text
    # The worker's log last, oldest first, each line with its level.
    log = lines[lines.index("Worker log, oldest first (4 line(s)):") + 1 :]
    assert log == [
        "18:55:01 INFO    Loading the image model...",
        "18:55:09 WARNING   attempt 1 failed the check: blurry",
        (
            r"18:55:10 ERROR   Publishing failed: cannot write %USERPROFILE%\Documents\Sports"
            r" Interactive\Football Manager 27\graphics"
        ),
        "18:55:11 INFO    Published 1 picture(s).",
    ]
    assert ".png" not in text  # no pictures


def test_the_report_without_a_worker_card_or_sync(settings: Settings) -> None:
    text = build(
        facts(
            settings,
            card=None,
            card_asked=False,
            models_ready=None,
            worker=None,
            poller=None,
            watching=False,
            last_sync=None,
            sync=None,
            career=None,
        )
    )
    for expected in (
        "Graphics card: not asked in this process",
        "Image model: RealVisXL V5.0 (fp16, INT8 UNet blocks); files not checked in this process",
        "Worker: not in this process",
        "Save watcher: off",
        "Last sync: none yet",
        "Career: none yet",
        "Worker log, oldest first (0 line(s)):",
    ):
        assert expected in text.splitlines(), expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"C:\Users\Sam Lee\Documents\x.fm", r"%USERPROFILE%\Documents\x.fm"),
        ("C:/Users/Sam Lee/Documents", "%USERPROFILE%/Documents"),
        ("/mnt/c/Users/evan/AppData/Local", "%USERPROFILE%/AppData/Local"),
        ("cannot look at C:\\\\Users\\\\evan\\\\games", "cannot look at %USERPROFILE%\\\\games"),
        (r"in c:\users\evan", r"in %USERPROFILE%"),
        (r"D:\Games\Football Manager 27", r"D:\Games\Football Manager 27"),
        ("/home/joon/projects/kitshot/data", "~/projects/kitshot/data"),
        ("/home/joonp/x", "/home/joonp/x"),  # another folder that starts alike
    ],
)
def test_redaction(text: str, expected: str) -> None:
    assert redact(text, ["/home/joon", "/", "C:"]) == expected


def test_a_log_line_in_the_report() -> None:
    assert log_line(LogLine("09:00:00", "x", "error")) == "09:00:00 ERROR   x"
    assert str(LogLine("09:00:00", "x", "error")) == "09:00:00  x"
