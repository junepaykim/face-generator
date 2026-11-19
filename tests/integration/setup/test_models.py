"""The model files and their download (Phase 10b): no network, no Hugging Face cache of the PC's.

The cache is a temporary folder laid out as huggingface_hub lays it out; the download process is
either `run_child` with a fake fetch, or a small script standing in for it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
from support.paths import ROOT

from facegen.cli import cmd_models
from facegen.setup import models
from facegen.setup.models import (
    FILES,
    HubCache,
    ModelDownload,
    ModelFile,
    child_env,
    gigabytes,
    missing,
    run_child,
    status,
    total_size,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def put(cache: Path, f: ModelFile, data: bytes | None = None) -> None:
    """A file in the cache as huggingface_hub leaves it: blob, snapshot entry, refs/main."""
    repo = cache / ("models--" + f.repo.replace("/", "--"))
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(COMMIT, encoding="utf-8")
    target = repo / "snapshots" / COMMIT / f.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if data is not None else b"x")


SMALL = (
    ModelFile("org/one", "config.json", 10),
    ModelFile("org/one", "unet/weights.safetensors", 1000),
    ModelFile("org/two", "model.safetensors", 500),
)


# --- the list ---------------------------------------------------------------------------------


def test_setup_downloads_the_files_required_by_the_three_models() -> None:
    # A missing tokenizer or encoder config can leave a completed download unusable offline.
    expected = {
        "cagliostrolab/animagine-xl-4.0": {
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/vocab.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        },
        "h94/IP-Adapter": {
            "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        },
        "SmilingWolf/wd-eva02-large-tagger-v3": {
            "config.json",
            "model.safetensors",
            "selected_tags.csv",
        },
    }
    assert list(models.REPOS) == list(expected)
    assert {f.label for f in FILES} == {
        f"{repo}/{filename}" for repo, filenames in expected.items() for filename in filenames
    }


def test_the_files_are_those_the_code_loads() -> None:
    from facegen import qc
    from facegen.render import renderer

    repos = {f.repo for f in FILES}
    assert {renderer.MODEL_REPO, renderer.IPA_REPO, qc.TAGGER_REPO} == repos
    names = {f.label for f in FILES}
    assert f"{renderer.IPA_REPO}/{renderer.IPA_SUBFOLDER}/{renderer.IPA_WEIGHT}" in names
    assert f"{renderer.IPA_REPO}/{renderer.IPA_ENCODER}/model.safetensors" in names
    assert f"{qc.TAGGER_REPO}/selected_tags.csv" in names


def test_the_whole_download_is_about_eleven_and_a_half_gigabytes() -> None:
    assert gigabytes(total_size(FILES)) == "11.6 GB"
    assert gigabytes(total_size(FILES)) in (ROOT / "README.md").read_text(encoding="utf-8")
    assert gigabytes(640_000_000) == "640 MB"
    assert len({f.label for f in FILES}) == len(FILES)


# --- the cache --------------------------------------------------------------------------------


def test_missing_is_pure() -> None:
    have = {SMALL[0]}
    assert missing(SMALL, lambda f: f in have) == [SMALL[1], SMALL[2]]


def test_the_cache_folder_follows_huggingface_hubs_rules_without_it(tmp_path: Path) -> None:
    """An install without the GPU group (CI's) has no huggingface_hub: the folder is worked out
    by its documented rules, and the model status still answers."""
    home = Path("~").expanduser()
    assert models.hub_cache_root({}) == home / ".cache" / "huggingface" / "hub"
    assert models.hub_cache_root({"XDG_CACHE_HOME": str(tmp_path)}) == (
        tmp_path / "huggingface" / "hub"
    )
    assert models.hub_cache_root({"HF_HOME": str(tmp_path / "hf")}) == tmp_path / "hf" / "hub"
    assert models.hub_cache_root(
        {"HF_HOME": str(tmp_path / "hf"), "HF_HUB_CACHE": str(tmp_path / "c")}
    ) == (tmp_path / "c")
    assert models.hub_cache_root({"HUGGINGFACE_HUB_CACHE": str(tmp_path / "old")}) == (
        tmp_path / "old"
    )


def test_without_huggingface_hub_the_status_still_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import fails, as without it
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    assert models.hub_cache_root() == tmp_path / "empty-cache"
    assert not models.status().ready


def test_with_huggingface_hub_the_folder_is_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    constants = pytest.importorskip("huggingface_hub.constants")
    assert models.hub_cache_root() == Path(constants.HF_HUB_CACHE)


def test_the_cache_finds_files_through_refs_main(tmp_path: Path) -> None:
    cache = HubCache(tmp_path)
    assert not cache.has(SMALL[0])  # no repository at all
    put(tmp_path, SMALL[0])
    assert cache.has(SMALL[0]) and not cache.has(SMALL[1])
    # A snapshot of another commit does not count: the models load refs/main.
    other = tmp_path / "models--org--one" / "snapshots" / "other" / SMALL[1].filename
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")
    assert not cache.has(SMALL[1])


def test_a_broken_link_into_blobs_is_missing(tmp_path: Path) -> None:
    put(tmp_path, SMALL[0])
    link = tmp_path / "models--org--one" / "snapshots" / COMMIT / SMALL[1].filename
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(tmp_path / "models--org--one" / "blobs" / "gone")
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows does not grant this process symlink creation privileges")
        raise
    assert not HubCache(tmp_path).has(SMALL[1])


def test_partial_bytes_count_unfinished_downloads(tmp_path: Path) -> None:
    blobs = tmp_path / "models--org--one" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"x" * 300)
    (blobs / "abc.1234abcd.incomplete").write_bytes(b"x" * 20)
    (blobs / "done").write_bytes(b"x" * 999)
    assert HubCache(tmp_path).partial_bytes("org/one") == 320
    assert HubCache(tmp_path).partial_bytes("org/none") == 0


def test_status_says_what_is_missing_and_the_space(tmp_path: Path) -> None:
    put(tmp_path, SMALL[0])
    s = status(HubCache(tmp_path), SMALL)
    assert not s.ready and s.missing == (SMALL[1], SMALL[2])
    assert s.missing_bytes == 1500 and s.total_bytes == 1510
    assert s.free_bytes is not None and s.enough_space
    for f in SMALL[1:]:
        put(tmp_path, f)
    assert status(HubCache(tmp_path), SMALL).ready


def test_free_space_is_asked_where_the_cache_will_be(tmp_path: Path) -> None:
    assert models.free_space(tmp_path / "not" / "yet") is not None


# --- the download process ---------------------------------------------------------------------


def fake_fetch(cache: Path, fail_on: str | None = None) -> Callable[[ModelFile], None]:
    def fetch(f: ModelFile) -> None:
        if f.filename == fail_on:
            raise ConnectionError("network is unreachable")
        put(cache, f, b"x" * f.size)

    return fetch


def messages(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_child_fetches_only_what_is_missing_and_says_how_far_it_is(tmp_path: Path) -> None:
    put(tmp_path, SMALL[0])  # finished before: skipped
    out = io.StringIO()
    code = run_child(out, cache=HubCache(tmp_path), files=SMALL, fetch=fake_fetch(tmp_path))
    assert code == 0
    sent = messages(out)
    assert sent[0] == {"type": "plan", "files": 2, "bytes": 1500}
    assert [m["name"] for m in sent if m["type"] == "file"] == [
        "org/one/unet/weights.safetensors",
        "org/two/model.safetensors",
    ]
    done = [m["done"] for m in sent if m["type"] == "progress"]
    assert done == sorted(done) and done[-1] == 1500
    assert sent[-1] == {"type": "done", "bytes": 1500}


def test_the_child_reports_an_error_and_a_second_run_resumes(tmp_path: Path) -> None:
    out = io.StringIO()
    fetch = fake_fetch(tmp_path, fail_on="model.safetensors")
    assert run_child(out, cache=HubCache(tmp_path), files=SMALL, fetch=fetch) == 1
    assert messages(out)[-1] == {
        "type": "error",
        "message": "ConnectionError: network is unreachable",
    }
    out = io.StringIO()
    assert run_child(out, cache=HubCache(tmp_path), files=SMALL, fetch=fake_fetch(tmp_path)) == 0
    assert messages(out)[0] == {"type": "plan", "files": 1, "bytes": 500}


def test_the_child_counts_a_partial_file_while_it_downloads(tmp_path: Path) -> None:
    import threading

    started, release = threading.Event(), threading.Event()

    def slow(f: ModelFile) -> None:
        blobs = tmp_path / "models--org--one" / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        (blobs / "x.incomplete").write_bytes(b"x" * 400)
        started.set()
        release.wait(5)
        (blobs / "x.incomplete").unlink()
        put(tmp_path, f, b"x")

    out = io.StringIO()
    files = (SMALL[1],)
    kwargs = {"cache": HubCache(tmp_path), "files": files, "fetch": slow, "interval": 0.02}
    t = threading.Thread(target=run_child, args=(out,), kwargs=kwargs)
    t.start()
    started.wait(5)
    for _ in range(200):
        if any(m.get("done") == 400 for m in messages(out) if m["type"] == "progress"):
            break
        threading.Event().wait(0.01)
    release.set()
    t.join(5)
    assert any(m.get("done") == 400 for m in messages(out) if m["type"] == "progress")


def test_the_child_runs_online_and_quiet() -> None:
    env = child_env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PATH": "/bin"})
    assert "HF_HUB_OFFLINE" not in env and "TRANSFORMERS_OFFLINE" not in env
    assert env["PATH"] == "/bin" and env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert models.child_command(("python", "-m", "facegen.cli"))[-2:] == ["models", "--child"]


# --- the parent's side ------------------------------------------------------------------------


def script(tmp_path: Path, body: str) -> list[str]:
    """A stand-in for the download process."""
    path = tmp_path / "child.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(path)]


def test_the_parent_follows_the_child_to_the_end(tmp_path: Path) -> None:
    command = script(
        tmp_path,
        """
        import json, os, sys
        def send(**m): print(json.dumps(m), flush=True)
        assert "HF_HUB_OFFLINE" not in os.environ
        send(type="plan", files=2, bytes=1000)
        send(type="file", index=1, name="org/one/a", size=600)
        print("a library's notice", file=sys.stderr)
        send(type="progress", done=600)
        print("not a message")
        send(type="file", index=2, name="org/one/b", size=400)
        send(type="progress", done=1000)
        send(type="done", bytes=1000)
        """,
    )
    woken = []
    download = ModelDownload(command, on_done=lambda: woken.append(1), env={"HF_HUB_OFFLINE": "1"})
    assert download.start()
    download.join(30)
    s = download.status()
    assert not s.running and s.finished and s.error is None
    assert (s.done_bytes, s.total_bytes, s.files, s.index, s.percent) == (1000, 1000, 2, 2, 100)
    assert woken == [1]


def test_a_child_that_fails_says_why(tmp_path: Path) -> None:
    command = script(
        tmp_path,
        """
        import json, sys
        print(json.dumps({"type": "plan", "files": 1, "bytes": 10}), flush=True)
        print("Traceback: something broke", file=sys.stderr)
        sys.exit(3)
        """,
    )
    download = ModelDownload(command)
    download.start()
    download.join(30)
    s = download.status()
    assert not s.running and not s.finished
    assert s.error == "the download stopped (Traceback: something broke)"
    assert download.start()  # it can be started again
    download.join(30)


def test_one_download_at_a_time(tmp_path: Path) -> None:
    command = script(tmp_path, "import time\ntime.sleep(30)\n")
    download = ModelDownload(command)
    assert download.start()
    assert not download.start()
    download.stop()
    download.join(30)
    assert not download.status().running


def test_a_child_that_cannot_be_started(tmp_path: Path) -> None:
    download = ModelDownload([str(tmp_path / "no-such-program")])
    download.start()
    download.join(30)
    assert "could not be started" in (download.status().error or "")


# --- facegen models ---------------------------------------------------------------------------


def args(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(**{"download": False, "yes": False, "child": False, **kw})


def all_files(cache: Path) -> None:
    for f in FILES:
        put(cache, f)


def test_status_lists_each_model_and_says_what_to_run(tmp_path: Path) -> None:
    out = io.StringIO()
    assert cmd_models(args(), out, cache=HubCache(tmp_path)) == 1
    text = out.getvalue()
    assert "cagliostrolab/animagine-xl-4.0" in text and "file(s) missing" in text
    assert "facegen models --download" in text
    all_files(tmp_path)
    out = io.StringIO()
    assert cmd_models(args(), out, cache=HubCache(tmp_path)) == 0
    assert "nothing to download" in out.getvalue()


def test_nothing_is_downloaded_without_the_confirmation(tmp_path: Path) -> None:
    """AGENTS.md hard rule 4: the download asks once; no answer, no download."""
    started: list[str] = []
    command = script(tmp_path, "open(r'%s', 'w').write('ran')\n" % (tmp_path / "ran"))
    out = io.StringIO()
    code = cmd_models(
        args(download=True), out, cache=HubCache(tmp_path / "hub"), command=command,
        interactive=lambda: False,
    )  # fmt: skip
    assert code == 3 and "--yes" in out.getvalue()
    out = io.StringIO()
    code = cmd_models(
        args(download=True), out, cache=HubCache(tmp_path / "hub"), command=command,
        interactive=lambda: True, ask=lambda q: started.append(q) or "n",
    )  # fmt: skip
    assert code == 3 and "Not downloaded." in out.getvalue()
    assert "11.6 GB" in started[0]
    assert not (tmp_path / "ran").exists()


def test_a_confirmed_download_runs_the_child(tmp_path: Path) -> None:
    command = script(
        tmp_path,
        """
        import json
        for m in ({"type": "plan", "files": 1, "bytes": 5}, {"type": "file", "index": 1,
                  "name": "org/one/a", "size": 5}, {"type": "done", "bytes": 5}):
            print(json.dumps(m), flush=True)
        """,
    )
    out = io.StringIO()
    code = cmd_models(
        args(download=True), out, cache=HubCache(tmp_path / "hub"), command=command,
        interactive=lambda: True, ask=lambda q: "y",
    )  # fmt: skip
    assert code == 0 and "All model files are here." in out.getvalue()
    out = io.StringIO()
    code = cmd_models(
        args(download=True, yes=True), out, cache=HubCache(tmp_path / "hub"), command=command,
        interactive=lambda: False,
    )  # fmt: skip
    assert code == 0


def test_not_enough_space_is_refused_before_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models, "free_space", lambda path: 10**9)
    out = io.StringIO()
    code = cmd_models(args(download=True, yes=True), out, cache=HubCache(tmp_path))
    assert code == 3 and "Not enough free space" in out.getvalue()
