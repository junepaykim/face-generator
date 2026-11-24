"""Phase 7 on the GPU: draw the same made-up players again and again through `kitshot serve`'s
worker process (WorkerProxy and the real renderer), and record faults, restarts, stalls,
time and memory.

    uv run python scripts/spikes/soak_test.py --name run1 --pictures 500
    uv run python scripts/spikes/soak_test.py --name check --pictures 30 --fault-on-job 4 \
        --pause-after 8

Everything lives in data/soak/<name>/: its own config, database, data folder and a stand-in
FM27 folder, so the user's database and FM27's real graphics folder are never touched. The
players are made up (tests/factories.py): 14 at my club (group A: best of 3 and the detail
pass), 3 group B and 3 group C newgens. Round 1 is their first pictures; then every round
asks for a new picture of all 20, alternating "new picture" (same face; at my club with the
current picture as a reference) and "new face".

The worker yields to FM27 as usual (it waits while fm.exe runs); time spent waiting is not
counted as a stall. With --generate-now it draws while FM27 runs, as the user had it when
the crash of 2026-11-13 happened. --fault-on-job N makes the worker cause a real CUDA error once, on its
N-th job (soak_child.py). --pause-after N pauses after N pictures and reads the card's
memory with the worker process gone, then resumes.

Writes data/soak/<name>/summary.json; the worker's log is data/soak/<name>/data/logs/kitshot.log
and each render's time and memory data/soak/<name>/calls.jsonl.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))  # the made-up players' builders

from factories import NEWGEN, RIVAL, player, snapshot

from kitshot.persistence import db
from kitshot.persistence.settings import DEFAULTS_PATH, load_settings
from kitshot.pipeline import manual
from kitshot.pipeline.manual import ManualError
from kitshot.pipeline.sync import run_sync
from kitshot.pipeline.worker import load_career
from kitshot.runtime.logging import log_to_file
from kitshot.runtime.worker_process import WorkerProxy, worker_command

CHILD = Path(__file__).resolve().parent / "soak_child.py"
LINES = {
    "fm_user_dir": 'fm_user_dir = ""',
    "fm_install_dir": 'fm_install_dir = ""',
    "data_dir": 'data_dir = "data"',
}


def write_config(sandbox: Path) -> Path:
    """The defaults with every path inside the sandbox. Given explicitly, the file is read
    alone: config/local.toml (this PC's real FM27 folder) is not laid over it."""
    text = DEFAULTS_PATH.read_text(encoding="utf-8")
    paths = {
        "fm_user_dir": sandbox / "fm",
        "fm_install_dir": sandbox / "fm-install",
        "data_dir": sandbox / "data",
    }
    for key, line in LINES.items():
        if line not in text:
            raise SystemExit(f"config/default.toml no longer has the line {line!r}")
        text = text.replace(line, f'{key} = "{paths[key].as_posix()}"')
    config = sandbox / "soak.toml"
    config.write_text(text, encoding="utf-8")
    return config


def world() -> list[Any]:
    mine = [
        player(
            1001 + i,
            name=f"Soak Mine {i + 1}",
            age=17 + (i * 5) % 19,
            join=date(2020 + i % 6, 7, 1),
            goalkeeper=i in (0, 7),
        )
        for i in range(14)
    ]
    b = [
        player(NEWGEN + 2 + 2 * i, name=f"Soak B {i + 1}", ca=90, pa=152, club_id=RIVAL, age=18)
        for i in range(3)
    ]
    c = [
        player(NEWGEN + 102 + 2 * i, name=f"Soak C {i + 1}", ca=120, pa=145, club_id=RIVAL, age=19)
        for i in range(3)
    ]
    return mine + b + c


def vram_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return int(out.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def counts(db_path: Path) -> dict[str, int]:
    conn = db.connect(db_path)
    try:
        rows = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
    finally:
        conn.close()
    return {r[0]: r[1] for r in rows}


def queue_round(db_path: Path, config: Path, round_no: int, uids: list[int]) -> str:
    conn = db.connect(db_path)
    try:
        settings = load_settings(conn, config)
        career = load_career(conn)
        assert career is not None
        if round_no % 2 == 0:
            manual.new_pictures(conn, career.id, uids, settings)
            return "new picture"
        for uid in uids:
            try:
                manual.new_face(conn, career.id, uid, settings)
            except ManualError as exc:
                print(f"  new face skipped for {uid}: {exc}", flush=True)
        return "new face"
    finally:
        conn.close()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarise_calls(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    calls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    out: dict[str, Any] = {"faults_caused": sum(1 for c in calls if c["kind"] == "fault")}
    for kind in ("render", "detail"):
        for ref in (False, True):
            group = [c for c in calls if c["kind"] == kind and c["reference"] is ref and c["ok"]]
            if not group:
                continue
            secs = [c["seconds"] for c in group if c["seconds"] < 120]
            peaks = [c["peak_reserved_gib"] for c in group if "peak_reserved_gib" in c]
            out[f"{kind}{' with reference' if ref else ''}"] = {
                "calls": len(group),
                "median_s": round(statistics.median(secs), 2) if secs else None,
                "p95_s": percentile(secs, 0.95),
                "max_peak_reserved_gib": max(peaks) if peaks else None,
            }
    out["failed_calls"] = sum(1 for c in calls if c["kind"] != "fault" and not c["ok"])
    out["processes"] = len({c["pid"] for c in calls})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--pictures", type=int, default=500)
    ap.add_argument("--fault-on-job", type=int, default=0)
    ap.add_argument("--pause-after", type=int, default=0)
    ap.add_argument("--hours", type=float, default=6.0, help="stop after this long regardless")
    ap.add_argument("--stall-seconds", type=float, default=180.0)
    ap.add_argument("--launch-blocking", action="store_true", help="CUDA_LAUNCH_BLOCKING=1")
    ap.add_argument(
        "--generate-now", action="store_true", help="draw while FM27 runs (7c, FM27 open)"
    )
    args = ap.parse_args()

    sandbox = ROOT / "data" / "soak" / args.name
    if sandbox.exists():
        raise SystemExit(f"{sandbox} exists: choose another --name")
    sandbox.mkdir(parents=True)
    config = write_config(sandbox)
    settings = load_settings(defaults_path=config)
    for p in (settings.paths.fm_user_dir, settings.paths.data_dir):
        if not p.resolve().is_relative_to(sandbox.resolve()):
            raise SystemExit(f"refusing: {p} is outside the sandbox")
    if args.launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    conn = db.connect(settings.paths.db_path)
    players = world()
    run_sync(conn, snapshot(players), settings, save_file="soak.fm")
    conn.close()
    uids = [p.unique_id for p in players]
    log_path = log_to_file(settings.paths.data_dir)
    command = worker_command(
        (sys.executable, str(CHILD), str(sandbox), str(args.fault_on_job)), config
    )

    baseline = vram_used_mib()
    print(f"Sandbox {sandbox}; card memory in use before: {baseline} MiB", flush=True)
    proxy = WorkerProxy(command, db_path=settings.paths.db_path)
    t0 = time.monotonic()
    proxy.start()
    if args.generate_now:
        proxy.generate_now()

    drawn_at: list[float] = []
    last_progress = t0
    stalls: list[dict[str, Any]] = []
    stalled = False
    pause: dict[str, Any] = {}
    loaded_vram: int | None = None
    round_no, round_kind = 1, "first pictures"
    seen_done = seen_failed = 0
    ended = "target reached"
    try:
        while True:
            time.sleep(0.5)
            now = time.monotonic()
            c = counts(settings.paths.db_path)
            done, failed = c.get("done", 0), c.get("failed", 0)
            status = proxy.status()
            if done + failed > seen_done + seen_failed:
                new = done + failed - seen_done - seen_failed
                drawn_at.extend([now] * new)
                last_progress, stalled = now, False
                seen_done, seen_failed = done, failed
                gap = now - drawn_at[-2] if len(drawn_at) > 1 else now - t0
                print(
                    f"[{(now - t0) / 60:6.1f} min] {done}/{args.pictures} drawn, {failed} to review,"
                    f" round {round_no} ({round_kind}), last {gap:5.1f} s, worker processes"
                    f" {proxy.starts}, state {status.state}",
                    flush=True,
                )
                if status.models_loaded and loaded_vram is None:
                    loaded_vram = vram_used_mib()
            if done >= args.pictures:
                break
            if now - t0 > args.hours * 3600:
                ended = f"time limit of {args.hours} h"
                break
            if status.state == "gpu-fault":
                ended = f"gave up: {status.detail}"
                break
            slow = now - last_progress > args.stall_seconds
            if status.state == "running" and slow and not stalled:
                stalled = True
                event = {
                    "after_picture": done + failed,
                    "seconds": round(now - last_progress),
                    "detail": status.detail,
                    "vram_mib": vram_used_mib(),
                }
                stalls.append(event)
                print(f"STALL: {event}", flush=True)
            if args.pause_after and not pause and done >= args.pause_after:
                pause["loaded_mib"] = vram_used_mib()
                p0 = time.monotonic()
                proxy.pause()
                while proxy.child_pid is not None and time.monotonic() - p0 < 300:
                    time.sleep(0.2)
                pause["seconds_to_exit"] = round(time.monotonic() - p0, 1)
                time.sleep(2)
                pause["paused_mib"] = vram_used_mib()
                pause["state"] = proxy.status().state
                print(f"Pause check: {pause}", flush=True)
                proxy.resume()
                last_progress = time.monotonic()
            if c.get("queued", 0) + c.get("running", 0) == 0 and status.state in (
                "idle",
                "running",
            ):
                round_no += 1
                round_kind = queue_round(settings.paths.db_path, config, round_no, uids)
                if args.generate_now:  # it ends when the queue is empty: ask again
                    proxy.generate_now()
                else:
                    proxy.wake()
    except KeyboardInterrupt:
        ended = "interrupted"
    finally:
        s0 = time.monotonic()
        proxy.stop(timeout=120)
        stop_seconds = time.monotonic() - s0
        time.sleep(2)
        after = vram_used_mib()

    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    gaps = [b - a for a, b in itertools.pairwise(drawn_at)]
    final = counts(settings.paths.db_path)
    summary = {
        "ended": ended,
        "minutes": round((time.monotonic() - t0) / 60, 1),
        "pictures": final.get("done", 0),
        "to_review": final.get("failed", 0),
        "rounds": round_no,
        "worker_processes": proxy.starts,
        "gpu_fault_lines": log_text.count("  GPU fault: "),
        "unexpected_ends": log_text.count("ended unexpectedly"),
        "restarts_scheduled": log_text.count("Starting a new worker process"),
        "attribute_errors": log_text.count("AttributeError"),
        "stalls": stalls,
        "gap_s": {
            "median": round(statistics.median(gaps), 1) if gaps else None,
            "p95": round(percentile(gaps, 0.95) or 0, 1),
            "max": round(max(gaps), 1) if gaps else None,
        },
        "vram_mib": {
            "before": baseline,
            "while_drawing": loaded_vram,
            **{f"pause_{k}": v for k, v in pause.items()},
            "after_quit": after,
        },
        "quit_seconds": round(stop_seconds, 1),
        "calls": summarise_calls(sandbox / "calls.jsonl"),
        "launch_blocking": args.launch_blocking,
        "generate_now": args.generate_now,
    }
    (sandbox / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
