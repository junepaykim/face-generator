"""A stand-in for `facegen worker` that speaks the pipe protocol (worker_process) by a plan,
so WorkerProxy can be tested without a worker loop.

    python fake_child.py STATE_DIR PLAN [--generate-now]

PLAN has one step per process start, comma-separated; the last step repeats:

    idle         reports "idle" and obeys commands: pause and stop end it, others are noted
    fault[:ID]   reports drawing job ID (default: no job), then ends with a GPU fault
    die[:ID]     reports drawing job ID, writes to stderr and exits with code 3, saying nothing
    error        ends with an error that pauses the worker
    freed        reports a loaded model, then ends having freed the GPU
    hang         reports "idle", reads commands and never ends (until killed, or after 30 s)
    junk         prints a line that is not a message, then behaves as idle

Each start appends its arguments to STATE_DIR/starts, and each command received to
STATE_DIR/commands.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from facegen.persistence.jobs import Job
from facegen.runtime.worker_process import encode, job_to_json


def send(message: dict) -> None:
    sys.stdout.write(encode(message))
    sys.stdout.flush()


def status(state: str, detail: str = "", job_id: int | None = None, **extra: object) -> None:
    job = None
    if job_id is not None:
        job = Job(
            job_id,
            1000 + job_id,
            "initial",
            "a",
            None,
            None,
            f"Player {job_id}",
            True,
            False,
            None,
            None,
            25,
        )
    send(
        {
            "type": "status",
            "status": {
                "state": state,
                "detail": detail,
                "current": job_to_json(job),
                "paused": False,
                "gpu_fault": False,
                "generate_now": "--generate-now" in sys.argv,
                "models_loaded": bool(extra.get("models", False)),
                "done": int(extra.get("done", 0)),  # type: ignore[call-overload]
                "failed": 0,
                "last_error": None,
            },
        }
    )


def exit_(reason: str, message: str | None = None, job_id: int | None = None) -> None:
    job = None
    if job_id is not None:
        job = {"id": job_id}  # only the ID: the parent falls back to the status's job
    send({"type": "exit", "reason": reason, "message": message, "job": job})


def obey(state: Path, *, hang: bool = False) -> None:
    for raw in sys.stdin:
        cmd = json.loads(raw)["cmd"]
        with (state / "commands").open("a") as f:
            f.write(cmd + "\n")
        if hang:
            continue
        if cmd == "pause":
            send({"type": "line", "time": "00:00:00", "text": "Paused."})
            exit_("paused")
            return
        if cmd == "stop":
            exit_("stopped")
            return
    if hang:
        time.sleep(30)  # the parent kills it first
    exit_("stopped")


def main() -> None:
    state, plan = Path(sys.argv[1]), sys.argv[2].split(",")
    state.mkdir(parents=True, exist_ok=True)
    starts = state / "starts"
    run = len(starts.read_text().splitlines()) if starts.exists() else 0
    with starts.open("a") as f:
        f.write(" ".join(sys.argv[3:]) + "\n")
    step = plan[min(run, len(plan) - 1)]
    kind, _, arg = step.partition(":")
    job_id = int(arg) if arg else None
    if kind == "junk":
        print("this is not a message", flush=True)
        kind = "idle"
    match kind:
        case "idle":
            status("idle", f"run {run + 1}", done=1)
            obey(state)
        case "fault":
            status("running", "drawing", job_id)
            exit_(
                "fault",
                "RuntimeError: CUDA error: an illegal memory access was encountered",
                job_id,
            )
        case "die":
            status("running", "drawing", job_id)
            print("Fatal Python error: Segmentation fault", file=sys.stderr, flush=True)
            os._exit(3)
        case "error":
            status("running", "drawing", 5)
            exit_("error", "Player 5: RuntimeError: CUDA out of memory", 5)
        case "freed":
            status("idle", "the queue is empty", models=True)
            exit_("freed")
        case "hang":
            status("idle", "hanging")
            threading.Thread(target=lambda: (time.sleep(30), os._exit(0)), daemon=True).start()
            obey(state, hang=True)
        case _:
            raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
