"""The graphics card and its preset (Phase 10d): no GPU and no torch needed."""

from __future__ import annotations

import argparse
import builtins
import io
import json
import sys
import textwrap
from pathlib import Path

import pytest

from facegen.cli import cmd_hardware
from facegen.persistence.settings import load_defaults, load_settings, set_overrides
from facegen.setup import hardware
from facegen.setup.hardware import HIGH, LOW, MEDIUM, NONE, Hardware, changes, recommend

GIB = 2**30


@pytest.mark.parametrize(
    ("vram", "preset"),
    [
        (15.99, HIGH),  # an RTX 4080 "16 GB" as torch reports it
        (24.0, HIGH),
        (14.4, HIGH),
        (11.99, HIGH),  # a 12 GB card that reports a little less
        (11.6, HIGH),  # rounds to 12
        (11.4, MEDIUM),  # an 11 GB card
        (9.99, MEDIUM),  # a 10 GB card that reports a little less
        (9.0, LOW),  # "A 9 GB card falls in the low preset"
        (7.99, LOW),
        (6.0, LOW),
        (None, NONE),
    ],
)
def test_the_preset_for_each_card(vram: float | None, preset: hardware.Preset) -> None:
    assert recommend(vram) is preset


def test_the_presets_change_only_picture_settings() -> None:
    """Never the group thresholds (user, 2026-11-13): who gets a picture does not change."""
    for preset in (HIGH, MEDIUM, LOW):
        assert tuple(preset.changes) == hardware.PRESET_KEYS
    assert NONE.changes == {}
    assert LOW.changes == {
        "groups.a.candidates": 2,
        "groups.a.detail_size": "off",
        "render.vae_tiling": True,
    }
    assert MEDIUM.changes["groups.a.detail_size"] == "smaller"


def test_the_high_preset_is_the_defaults() -> None:
    defaults = load_defaults()
    assert {k: defaults[k] for k in HIGH.changes} == HIGH.changes
    assert changes(HIGH, load_settings().values) == []


def test_each_preset_makes_valid_settings(conn: object) -> None:
    for preset in (HIGH, MEDIUM, LOW):
        s = set_overrides(conn, preset.changes)  # type: ignore[arg-type]
        assert s.groups.a.candidates == preset.changes["groups.a.candidates"]


def test_changes_lists_only_what_differs() -> None:
    current = {
        "groups.a.candidates": 3,
        "groups.a.detail_size": "1344x1728",
        "render.vae_tiling": False,
    }
    assert [(c.key, c.before, c.after) for c in changes(LOW, current)] == [
        ("groups.a.candidates", 3, 2),
        ("groups.a.detail_size", "1344x1728", "off"),
        ("render.vae_tiling", False, True),
    ]
    assert [c.key for c in changes(MEDIUM, current)] == ["groups.a.detail_size"]


def test_a_card_as_json_and_back() -> None:
    card = Hardware("NVIDIA GeForce RTX 4080", 16 * GIB - 2**20, None, "2.15.0", "13.2")
    assert Hardware.from_json(json.loads(json.dumps(card.to_json()))) == card
    assert card.has_gpu and round(card.vram_gib or 0) == 16
    assert card.describe() == "NVIDIA GeForce RTX 4080, 16.0 GiB"
    none = Hardware(reason="no driver")
    assert not none.has_gpu and none.vram_gib is None
    assert none.describe() == "no CUDA graphics card (no driver)"


def test_detect_without_torch_is_an_answer_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_torch(name: str, *args: object, **kw: object) -> object:
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kw)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    found = hardware.detect()
    assert not found.has_gpu and "torch could not be loaded" in (found.reason or "")


def test_detect_with_a_torch_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake = types.SimpleNamespace(
        __version__="2.15.0+cpu",
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    found = hardware.detect()
    assert not found.has_gpu and "without CUDA" in (found.reason or "")
    assert found.torch == "2.15.0+cpu"


def test_detect_with_a_card(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    props = types.SimpleNamespace(name="NVIDIA GeForce RTX 3080", total_memory=10 * GIB)
    fake = types.SimpleNamespace(
        __version__="2.15.0",
        version=types.SimpleNamespace(cuda="13.2"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True, get_device_properties=lambda i: props
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    found = hardware.detect()
    assert found == Hardware("NVIDIA GeForce RTX 3080", 10 * GIB, None, "2.15.0", "13.2")
    assert recommend(found.vram_gib) is MEDIUM


# --- the probe: `facegen hardware --json` in a child process -----------------------------------


def program(tmp_path: Path, body: str) -> tuple[str, ...]:
    """A stand-in for `python -m facegen.cli`: it answers `... hardware --json`."""
    path = tmp_path / "fake_cli.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return (sys.executable, str(path))


def test_the_probe_reads_the_childs_answer(tmp_path: Path) -> None:
    prog = program(
        tmp_path,
        """
        import json, sys
        assert sys.argv[-2:] == ["hardware", "--json"], sys.argv
        print("a library's notice")
        print(json.dumps({"name": "Fake 8 GB", "total_bytes": 8 * 2**30}))
        """,
    )
    found = hardware.probe(prog)
    assert found.name == "Fake 8 GB" and recommend(found.vram_gib) is LOW


def test_a_probe_that_fails_or_hangs_is_an_answer(tmp_path: Path) -> None:
    broken = program(
        tmp_path, "import sys\nprint('Traceback: boom', file=sys.stderr)\nsys.exit(1)\n"
    )
    assert hardware.probe(broken).reason == "the check failed: Traceback: boom"
    slow = program(tmp_path, "import time\ntime.sleep(30)\n")
    assert "did not answer within 1 s" in (hardware.probe(slow, timeout=1).reason or "")
    missing = (str(tmp_path / "no-such-python"),)
    assert "could not be started" in (hardware.probe(missing).reason or "")


def test_the_probe_command_passes_the_config(tmp_path: Path) -> None:
    command = hardware.command(("py", "-m", "facegen.cli"), tmp_path / "x.toml")
    assert command == [
        "py",
        "-m",
        "facegen.cli",
        "--config",
        str(tmp_path / "x.toml"),
        "hardware",
        "--json",
    ]


def test_facegen_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    card = Hardware("Fake 8 GB", 8 * GIB, None, "2.15.0", "13.2")
    out = io.StringIO()
    current = dict(load_settings().values)
    code = cmd_hardware(argparse.Namespace(json=False), out, detect=lambda: card, current=current)
    text = out.getvalue()
    assert code == 0 and "Fake 8 GB, 8.0 GiB" in text and "under 10 GB" in text
    assert "would change groups.a.detail_size: x1.5 -> off" in text
    out = io.StringIO()
    assert cmd_hardware(argparse.Namespace(json=True), out, detect=lambda: card) == 0
    assert Hardware.from_json(json.loads(out.getvalue())) == card
    out = io.StringIO()
    none = Hardware(reason="no driver")
    assert cmd_hardware(argparse.Namespace(json=False), out, detect=lambda: none) == 1
