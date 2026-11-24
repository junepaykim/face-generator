"""Phase 8 on Windows: does a console window flash while `kitshot serve` runs?

Records every console window Windows shows (a WinEvent hook on EVENT_OBJECT_SHOW, so even a
window shown for a few milliseconds is caught) and the program that owns it.

    uv run python scripts/spikes/console_watch.py --selftest
    uv run python scripts/spikes/console_watch.py --name check --seconds 600

`--selftest` first proves the watcher sees a flash: a window-less Python (pythonw.exe) starts
tasklist.exe plainly, which opens a console, and then through `procs.run_quiet`, which must
not. The run starts `kitshot serve` with pythonw.exe (no console of its own, as the app shell
of Phase 11 will) on a sandbox in data/console/<name>/: its own config, database, data folder
and a stand-in FM27 folder with the soak test's made-up players queued, so the user's
database and FM27's real folders are never touched. The worker process it starts asks
tasklist.exe whether FM27 runs before each job, every 30 s while FM27 runs; with FM27 closed it
draws the made-up players on the GPU. At the end Kitshot and its worker process are ended
(taskkill) and the windows seen are listed, with whether the model loaded and the jobs' states.

Windows only. Console windows of other programs opened meanwhile are listed too, with their
program, so they can be told apart.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CONSOLE_CLASSES = {"ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS"}
"""conhost's window, and Windows Terminal's when it is the default terminal."""
EVENT_OBJECT_SHOW = 0x8002
WINEVENT_OUTOFCONTEXT = 0x0000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PM_REMOVE = 0x0001


def program_of(pid: int) -> str:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return "?"
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name
        return "?"
    finally:
        kernel32.CloseHandle(handle)


class Watcher:
    """Collects console windows shown while `pump` runs."""

    def __init__(self) -> None:
        self.seen: list[dict[str, object]] = []
        user32 = ctypes.windll.user32
        proto = ctypes.WINFUNCTYPE(
            None,
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.HWND,
            wintypes.LONG,
            wintypes.LONG,
            wintypes.DWORD,
            wintypes.DWORD,
        )

        def on_event(hook, event, hwnd, id_object, id_child, thread, ms):
            if id_object != 0 or not hwnd:
                return
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            if cls.value not in CONSOLE_CLASSES:
                return
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            self.seen.append(
                {
                    "at": time.strftime("%H:%M:%S"),
                    "class": cls.value,
                    "pid": pid.value,
                    "program": program_of(pid.value),
                }
            )

        self._callback = proto(on_event)  # kept alive for as long as the hook
        user32.SetWinEventHook.restype = wintypes.HANDLE
        self._hook = user32.SetWinEventHook(
            EVENT_OBJECT_SHOW, EVENT_OBJECT_SHOW, 0, self._callback, 0, 0, WINEVENT_OUTOFCONTEXT
        )
        if not self._hook:
            raise OSError("SetWinEventHook failed")

    def pump(self, seconds: float) -> None:
        """Deliver window events for this long (they arrive through this thread's messages)."""
        user32 = ctypes.windll.user32
        msg = wintypes.MSG()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            while user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, PM_REMOVE):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            time.sleep(0.005)

    def close(self) -> None:
        ctypes.windll.user32.UnhookWinEvent(self._hook)


def pythonw() -> str:
    path = Path(sys.executable).with_name("pythonw.exe")
    if not path.exists():
        raise SystemExit(f"no pythonw.exe beside {sys.executable}")
    return str(path)


def selftest() -> int:
    plain = "import subprocess; subprocess.run(['tasklist.exe'], capture_output=True)"
    quiet = (
        "from kitshot.platform import procs; procs.run_quiet(['tasklist.exe'], timeout=30,"
        " capture_output=True)"
    )
    results = {}
    for name, code in (("plain", plain), ("run_quiet", quiet)):
        w = Watcher()
        child = subprocess.Popen([pythonw(), "-c", code])
        while child.poll() is None:
            w.pump(0.1)
        w.pump(2.0)
        w.close()
        results[name] = w.seen
        print(f"{name}: {len(w.seen)} console window(s) shown {w.seen}", flush=True)
    ok = bool(results["plain"]) and not results["run_quiet"]
    print("self-test", "passed" if ok else "FAILED", flush=True)
    return 0 if ok else 1


def run(name: str, seconds: float, port: int) -> int:
    from factories import snapshot
    from soak_test import world, write_config

    from kitshot.persistence import db
    from kitshot.persistence.settings import load_settings
    from kitshot.pipeline.sync import run_sync

    sandbox = ROOT / "data" / "console" / name
    if sandbox.exists():
        raise SystemExit(f"{sandbox} exists: choose another --name")
    sandbox.mkdir(parents=True)
    config = write_config(sandbox)
    settings = load_settings(defaults_path=config)
    for p in (settings.paths.fm_user_dir, settings.paths.data_dir):
        assert p is not None and p.resolve().is_relative_to(sandbox.resolve()), p
    (settings.paths.fm_user_dir / "games").mkdir(parents=True)
    conn = db.connect(settings.paths.db_path)
    run_sync(conn, snapshot(world()), settings, save_file="soak.fm")
    conn.close()

    w = Watcher()
    log = (sandbox / "serve-output.txt").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [pythonw(), "-m", "kitshot.cli", "--config", str(config), "serve", "--port", str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    print(f"kitshot serve (pid {server.pid}) under pythonw.exe for {seconds:.0f} s", flush=True)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds and server.poll() is None:
            w.pump(1.0)
    finally:
        alive = server.poll() is None
        subprocess.run(
            ["taskkill", "/PID", str(server.pid), "/T", "/F"],
            capture_output=True,
            creationflags=0x08000000,
            check=False,
        )
        w.pump(2.0)
        w.close()
        log.close()
    facegen_log = settings.paths.data_dir / "logs" / "kitshot.log"
    text = facegen_log.read_text(encoding="utf-8", errors="replace") if facegen_log.exists() else ""
    conn = db.connect(settings.paths.db_path)
    try:
        jobs = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    finally:
        conn.close()
    summary = {
        "seconds": round(time.monotonic() - t0),
        "server_alive_at_end": alive,
        "console_windows": w.seen,
        "model_loaded": "loaded in" in text,
        "jobs": jobs,
        "log_tail": text.splitlines()[-15:],
    }
    (sandbox / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if alive and not w.seen else 1


def main() -> int:
    if os.name != "nt":
        raise SystemExit("Windows only")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.name:
        ap.error("--name is required for a run")
    return run(args.name, args.seconds, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
