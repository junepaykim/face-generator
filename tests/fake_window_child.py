"""The window process with a stand-in for pywebview, for test_window_process.py: the real
`app_window.child_main` and pipe protocol, a fake window.

    python fake_window_child.py NOTES MODE --url URL [--hidden]

MODE: "ok"; "hide" (the user closes the window once it is up); "nowebview" (no WebView2);
"crash" (ends at once with exit code 1). What the window is asked is appended to NOTES.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facegen.app_window import child_main
from facegen.pipes import protocol_streams

notes = Path(sys.argv[1])
mode = sys.argv[2]


def note(text: str) -> None:
    with notes.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


class FakeWindow:
    def __init__(self, url: str, *, hidden: bool, on_hide) -> None:  # type: ignore[no-untyped-def]
        self.closed = threading.Event()
        self.on_hide = on_hide
        note(f"window {url} hidden={hidden}")

    def run(self) -> None:
        if mode == "hide":
            self.on_hide()
        self.closed.wait(30)
        note("closed")

    def show(self) -> None:
        note("show")

    def destroy(self) -> None:
        self.closed.set()

    def browse(self, start: str | None) -> str | None:
        note(f"browse {start}")
        return (
            None
            if start == "cancel"
            else r"C:\Users\me\Documents\Sports Interactive\Football Manager 27"
        )


if __name__ == "__main__":
    if mode == "crash":
        raise SystemExit(1)
    args = sys.argv[3:]
    url = args[args.index("--url") + 1]
    commands, out = protocol_streams(sys.stdin, sys.stdout)
    raise SystemExit(
        child_main(
            url,
            "--hidden" in args,
            commands,
            out,
            missing=lambda: mode == "nowebview",
            make=FakeWindow,  # type: ignore[arg-type]
        )
    )
