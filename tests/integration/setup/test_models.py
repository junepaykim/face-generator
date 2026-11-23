"""The model files and their download (Phase 10b; one list per image model, Phase 15e): no
network, no Hugging Face cache of the PC's.

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
from unittest.mock import Mock

import pytest
from support.paths import ROOT

from facegen.cli import cmd_models
from facegen.core.settings_values import IMAGE_MODELS, MODEL_CHOICES
from facegen.persistence import db
from facegen.persistence.settings import load_settings, set_override
from facegen.setup import models
from facegen.setup.models import (
    HubCache,
    ModelDownload,
    ModelFile,
    child_env,
    files_for,
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


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> str:
    """A made-up model "small" whose files are SMALL."""
    monkeypatch.setitem(models.MODELS, "small", SMALL)
    return "small"


# --- the list ---------------------------------------------------------------------------------


SDXL_PARTS = {
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "vae/config.json",
}
SDXL_WEIGHTS = (
    "text_encoder/model{}.safetensors",
    "text_encoder_2/model{}.safetensors",
    "unet/diffusion_pytorch_model{}.safetensors",
    "vae/diffusion_pytorch_model{}.safetensors",
)
IP_ADAPTER = {
    "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
    "models/image_encoder/config.json",
    "models/image_encoder/model.safetensors",
}
TAGGER = {"config.json", "model.safetensors", "selected_tags.csv"}


def test_each_model_downloads_the_files_it_loads() -> None:
    # A missing tokenizer or encoder config can leave a completed download unusable offline.
    # Checked against a cache holding only these files: scripts/spikes/model_file_lists.py.
    expected = {
        "animagine": {
            "cagliostrolab/animagine-xl-4.0": SDXL_PARTS | {w.format("") for w in SDXL_WEIGHTS},
            "h94/IP-Adapter": IP_ADAPTER,
            "SmilingWolf/wd-eva02-large-tagger-v3": TAGGER,
        },
        "realvis": {  # the fp16 files, the VAE's included
            "SG161222/RealVisXL_V5.0": SDXL_PARTS | {w.format(".fp16") for w in SDXL_WEIGHTS},
            "h94/IP-Adapter": IP_ADAPTER,
            "SmilingWolf/wd-eva02-large-tagger-v3": TAGGER,
        },
        "zimage_turbo": {  # all from its own repository; no reference picture, no IP-Adapter
            "Tongyi-MAI/Z-Image-Turbo": {
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                *(f"text_encoder/model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)),
                "text_encoder/model.safetensors.index.json",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "transformer/config.json",
                *(
                    f"transformer/diffusion_pytorch_model-0000{i}-of-00003.safetensors"
                    for i in (1, 2, 3)
                ),
                "transformer/diffusion_pytorch_model.safetensors.index.json",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            },
            "SmilingWolf/wd-eva02-large-tagger-v3": TAGGER,
        },
    }
    assert list(models.MODELS) == list(expected)
    for model, repos in expected.items():
        files = files_for(model)
        assert [f.label for f in files] == list(dict.fromkeys(f.label for f in files))
        assert {f.label for f in files} == {
            f"{repo}/{filename}" for repo, filenames in repos.items() for filename in filenames
        }, model
        assert list(dict.fromkeys(f.repo for f in files)) == list(repos)  # the model first
    assert set(models.REPOS) == {repo for repos in expected.values() for repo in repos}
    with pytest.raises(ValueError, match="no image model"):
        files_for("flux")


def test_every_model_the_settings_offer_has_a_list() -> None:
    assert set(MODEL_CHOICES) <= set(models.MODELS)
    assert models.DEFAULT_MODEL == load_settings().style.model == "animagine"


def test_the_files_are_those_the_code_loads() -> None:
    from facegen import qc
    from facegen.render import renderer, zimage

    image_repo = {
        "animagine": renderer.ANIMAGINE.repo,
        "realvis": renderer.REALVIS.repo,
        "zimage_turbo": zimage.ZIMAGE_REPO,
    }
    for model, repo in image_repo.items():
        names = {f.label for f in files_for(model)}
        assert files_for(model)[0].repo == repo
        assert f"{qc.TAGGER_REPO}/selected_tags.csv" in names
        adapter = f"{renderer.IPA_REPO}/{renderer.IPA_SUBFOLDER}/{renderer.IPA_WEIGHT}"
        encoder = f"{renderer.IPA_REPO}/{renderer.IPA_ENCODER}/model.safetensors"
        takes_reference = IMAGE_MODELS[model].reference
        assert (adapter in names) is takes_reference and (encoder in names) is takes_reference
    # What the settings say of a model is what its renderer does (Phase 15g).
    assert zimage.ZImageRenderer.reference_adapter is IMAGE_MODELS["zimage_turbo"].reference
    assert zimage.ZImageRenderer.negative_prompt is IMAGE_MODELS["zimage_turbo"].negative
    assert not hasattr(renderer.SdxlRenderer, "reference_adapter")  # SDXL takes a reference
    assert renderer.REALVIS.variant == "fp16"
    weights = [f.filename for f in models.REALVIS_FILES if f.filename.endswith(".safetensors")]
    assert len(weights) == 4 and all(name.endswith(".fp16.safetensors") for name in weights)


def test_the_downloads_sizes() -> None:
    sizes = {model: gigabytes(total_size(files_for(model))) for model in models.MODELS}
    assert sizes == {"animagine": "11.6 GB", "realvis": "11.6 GB", "zimage_turbo": "34.1 GB"}
    # The README states the default model's download.
    assert sizes["animagine"] in (ROOT / "README.md").read_text(encoding="utf-8")
    assert total_size(models.ZIMAGE_TURBO_FILES) == 32_848_305_294  # "about 33 GB" (decisions)
    assert gigabytes(640_000_000) == "640 MB"


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
    assert not models.status("animagine").ready


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


def test_status_says_what_is_missing_and_the_space(tmp_path: Path, small: str) -> None:
    put(tmp_path, SMALL[0])
    s = status(small, HubCache(tmp_path))
    assert not s.ready and s.missing == (SMALL[1], SMALL[2])
    assert (s.model, s.files) == ("small", SMALL)
    assert s.missing_bytes == 1500 and s.total_bytes == 1510
    assert s.missing_repos() == ["org/one", "org/two"]
    assert s.free_bytes is not None and s.enough_space
    assert s.describe().startswith("small: 2 of 3 files missing (0 MB)")
    for f in SMALL[1:]:
        put(tmp_path, f)
    assert status(small, HubCache(tmp_path)).ready
    assert status(small, HubCache(tmp_path)).describe() == f"small: all 3 files are in {tmp_path}"


def test_only_the_selected_models_files_are_asked_for(tmp_path: Path) -> None:
    """Phase 15e: the IP-Adapter and the tagger are shared; another model's files are never
    missing, and never offered."""
    for f in files_for("animagine"):
        put(tmp_path, f)
    cache = HubCache(tmp_path)
    assert status("animagine", cache).ready
    realvis = status("realvis", cache)
    assert realvis.missing == models.REALVIS_FILES
    assert realvis.missing_repos() == [models.REALVIS]
    assert gigabytes(realvis.missing_bytes) == "6.9 GB"
    zimage = status("zimage_turbo", cache)
    assert zimage.missing == models.ZIMAGE_TURBO_FILES
    assert gigabytes(zimage.missing_bytes) == "32.8 GB"
    with pytest.raises(ValueError):
        status("flux", cache)


def test_ready_now_asks_about_the_model_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # the cache from HF_HUB_CACHE
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    for f in files_for("realvis"):
        put(tmp_path, f)
    assert models.ready_now("realvis")
    assert not models.ready_now("animagine") and not models.ready_now("zimage_turbo")
    assert models.ready_now("flux")  # no such model: loading it reports the real error


def test_status_reads_each_repository_reference_once_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = HubCache(tmp_path)
    snapshot = Mock(wraps=cache.snapshot)
    monkeypatch.setattr(cache, "snapshot", snapshot)
    files = (SMALL[1], SMALL[2], SMALL[0])  # interleave the two repositories
    monkeypatch.setitem(models.MODELS, "mixed", files)

    assert status("mixed", cache).missing == files
    assert sorted(c.args[0] for c in snapshot.call_args_list) == ["org/one", "org/two"]

    snapshot.reset_mock()
    for f in files:
        put(tmp_path, f)
    assert status("mixed", cache).ready
    assert sorted(c.args[0] for c in snapshot.call_args_list) == ["org/one", "org/two"]


def test_status_rechecks_files_reference_changes_and_disk_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small: str
) -> None:
    cache = HubCache(tmp_path)
    for f in SMALL:
        put(tmp_path, f)
    monkeypatch.setattr(models, "free_space", Mock(side_effect=[3000, 2000, 1000]))
    first = status(small, cache)
    assert first.ready and first.free_bytes == 3000

    repo = cache.repo_dir("org/one")
    (repo / "snapshots" / COMMIT / SMALL[0].filename).unlink()
    second = status(small, cache)
    assert second.missing == (SMALL[0],) and second.free_bytes == 2000

    other = repo / "snapshots" / "other"
    other.mkdir()
    (other / SMALL[0].filename).write_bytes(b"x")
    (repo / "refs" / "main").write_text("other", encoding="utf-8")
    third = status(small, cache)
    assert third.missing == (SMALL[1],) and third.free_bytes == 1000


def test_free_space_is_asked_where_the_cache_will_be(tmp_path: Path) -> None:
    assert models.free_space(tmp_path / "not" / "yet") is not None


def test_free_space_budget_includes_the_whole_interrupted_file(
    tmp_path: Path, small: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(tmp_path, SMALL[0])
    put(tmp_path, SMALL[2])
    blobs = tmp_path / "models--org--one" / "blobs"
    blobs.mkdir()
    (blobs / "etag.12345678.incomplete").write_bytes(b"x" * 800)
    # The pinned Hub client opens a new temp file on retry, even for the same ETag.
    monkeypatch.setattr(models, "free_space", lambda _path: models.SPACE_MARGIN + 200)
    s = status(small, HubCache(tmp_path))
    assert s.missing == (SMALL[1],) and s.missing_bytes == 1000 and not s.enough_space


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


def test_download_reports_only_current_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tqdm")
    for key in models.CHILD_ENV_DROP:
        monkeypatch.setenv(key, "1")
    blobs = tmp_path / "models--org--one" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "previous.12345678.incomplete").write_bytes(b"x" * 800)
    (blobs / "other.incomplete").write_bytes(b"x" * 300)
    f = SMALL[1]

    def download(repo: str, filename: str, *, revision: str, tqdm_class: type) -> None:
        assert (repo, filename, revision) == (f.repo, f.filename, "main")
        with tqdm_class(total=f.size) as bar:
            bar.update(400)
            bar.update_transfer(375)  # Xet's transfer bytes are a separate measure
            bar.set_transfer_postfix_str("speed", refresh=False)
            bar.update(500)
        put(tmp_path, f)

    monkeypatch.setitem(sys.modules, "huggingface_hub", Mock(hf_hub_download=download))
    out = io.StringIO()
    assert run_child(out, cache=HubCache(tmp_path), files=[f], interval=0) == 0
    assert [m["done"] for m in messages(out) if m["type"] == "progress"] == [400, 900, 1000]


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
        assert sys.argv[-2:] == ["--model", "realvis"], sys.argv  # the model agreed to
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
    with pytest.raises(ValueError):
        download.start("flux")  # no such model: nothing starts
    assert download.start("realvis")
    download.join(30)
    s = download.status()
    assert not s.running and s.finished and s.error is None and s.model == "realvis"
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
    download.start("animagine")
    download.join(30)
    s = download.status()
    assert not s.running and not s.finished
    assert s.error == "the download stopped (Traceback: something broke)"
    assert download.start("animagine")  # it can be started again
    download.join(30)


def test_one_download_at_a_time(tmp_path: Path) -> None:
    command = script(tmp_path, "import time\ntime.sleep(30)\n")
    download = ModelDownload(command)
    assert download.start("animagine")
    assert not download.start("realvis")
    download.stop()
    download.join(30)
    assert not download.status().running


def test_a_child_that_cannot_be_started(tmp_path: Path) -> None:
    download = ModelDownload([str(tmp_path / "no-such-program")])
    download.start("animagine")
    download.join(30)
    assert "could not be started" in (download.status().error or "")


# --- facegen models ---------------------------------------------------------------------------


def args(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "download": False,
            "yes": False,
            "child": False,
            "model": "animagine",
            "config": None,
            **kw,
        }
    )


def all_files(cache: Path, model: str = "animagine") -> None:
    for f in files_for(model):
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


def test_models_lists_only_the_chosen_models_repositories(tmp_path: Path) -> None:
    all_files(tmp_path)  # Animagine's, with the IP-Adapter and the tagger
    out = io.StringIO()
    assert cmd_models(args(model="zimage_turbo"), out, cache=HubCache(tmp_path)) == 1
    text = out.getvalue()
    assert "Image model  zimage_turbo" in text
    assert "Tongyi-MAI/Z-Image-Turbo" in text and "32.8 GB  18 file(s) missing" in text
    assert "SmilingWolf/wd-eva02-large-tagger-v3" in text and "ready" in text
    assert "h94/IP-Adapter" not in text and "animagine" not in text
    assert "To download (32.8 GB)" in text


def test_models_follows_the_model_saved_in_settings(tmp_path: Path, config_file: Path) -> None:
    """Without --model, `facegen models` checks the model chosen in Settings; before the first
    start there is no database, and it is not created."""
    db_path = load_settings(defaults_path=config_file).paths.db_path
    out = io.StringIO()
    cmd_models(args(model=None, config=config_file), out, cache=HubCache(tmp_path / "hub"))
    assert "Image model  animagine" in out.getvalue() and not db_path.exists()

    conn = db.connect(db_path)
    set_override(conn, "style.model", "realvis", config_file)
    conn.close()
    out = io.StringIO()
    cmd_models(args(model=None, config=config_file), out, cache=HubCache(tmp_path / "hub"))
    assert "Image model  realvis" in out.getvalue()
    assert "SG161222/RealVisXL_V5.0" in out.getvalue()


def test_the_child_downloads_the_model_it_is_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[ModelFile, ...]] = []
    monkeypatch.setattr(models, "run_child", lambda out, *, files: seen.append(files) or 0)
    assert cmd_models(args(child=True, model="realvis"), io.StringIO()) == 0
    assert seen == [files_for("realvis")]


def test_a_confirmed_download_is_for_the_model_shown(tmp_path: Path) -> None:
    command = script(
        tmp_path,
        """
        import json, sys
        assert sys.argv[-2:] == ["--model", "realvis"], sys.argv
        for m in ({"type": "plan", "files": 1, "bytes": 5}, {"type": "done", "bytes": 5}):
            print(json.dumps(m), flush=True)
        """,
    )
    asked: list[str] = []
    out = io.StringIO()
    code = cmd_models(
        args(download=True, model="realvis"), out, cache=HubCache(tmp_path / "hub"),
        command=command, interactive=lambda: True, ask=lambda q: asked.append(q) or "y",
    )  # fmt: skip
    assert code == 0 and "11.6 GB" in asked[0]
