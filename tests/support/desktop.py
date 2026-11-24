"""Controller worker, sync, startup, and status stand-ins shared by desktop tests."""

from __future__ import annotations

import dataclasses

from kitshot.runtime.contracts import SyncStatus, WorkerStatus


def status(state: str = "idle", **kw: object) -> WorkerStatus:
    base: dict[str, object] = {
        "state": state,
        "detail": "",
        "current": None,
        "paused": state in ("paused", "error", "gpu-fault"),
        "gpu_fault": False,
        "generate_now": False,
        "models_loaded": False,
        "done": 0,
        "failed": 0,
        "last_error": None,
        "log": (),
    }
    base.update(kw)
    return WorkerStatus(**base)  # type: ignore[arg-type]


class FakeWorker:
    def __init__(self, s: WorkerStatus | None = None) -> None:
        self.s = s or status()
        self.calls: list[str] = []
        self.resume_ok = True

    def status(self) -> WorkerStatus:
        return self.s

    def pause(self) -> None:
        self.calls.append("pause")
        self.s = dataclasses.replace(self.s, paused=True, state="paused")

    def resume(self) -> bool:
        self.calls.append("resume")
        if self.resume_ok:
            self.s = dataclasses.replace(self.s, paused=False, state="idle")
        return self.resume_ok

    def generate_now(self) -> bool:
        return True

    def wake(self) -> None:
        pass

    def log(self, line: str) -> None:
        pass


class FakeSyncer:
    def __init__(self) -> None:
        self.s = SyncStatus()
        self.starts = 0

    def status(self) -> SyncStatus:
        return self.s

    def start(self) -> bool:
        if self.s.running:
            return False
        self.starts += 1
        self.s = SyncStatus(running=True, started="12:00:00", trigger="sync now")
        return True

    def finish(self, **kw: object) -> None:
        self.s = SyncStatus(running=False, started="12:00:00", trigger="sync now", **kw)  # type: ignore[arg-type]


class FakeStartup:
    def __init__(self, on: bool = False) -> None:
        self.on = on
        self.sets: list[bool] = []

    def is_on(self) -> bool:
        return self.on

    def set(self, on: bool) -> None:
        self.sets.append(on)
        self.on = on
