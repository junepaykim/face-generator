"""Check that each model's download list (setup/models.py MODELS) is enough to load it offline.

For each model, a Hugging Face cache is laid out in a temporary folder holding only the listed
files: each snapshot entry links to this PC's cached copy (nothing is copied or downloaded).
A child process with HF_HUB_OFFLINE=1 and HF_HUB_CACHE set to that folder then loads, on the
CPU, what the renderer, IP-Adapter and tagger load. `--drop FILE` leaves one listed file out,
as a control: the load must then fail.

    uv run --no-sync python scripts/spikes/model_file_lists.py [MODEL ...] [--drop REPO/FILE]

Exit 0 when every model named (default: all) loaded. Phase 15e, docs/research.md section 20.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from kitshot.setup import models


def lay_out(root: Path, files: tuple[models.ModelFile, ...], drop: str | None) -> list[str]:
    """The listed files under `root`, linked to the real cache; returns problems found."""
    real = models.HubCache(models.hub_cache_root())
    problems = []
    for f in files:
        if f.label == drop:
            continue
        snapshot = real.snapshot(f.repo)
        source = snapshot / f.filename if snapshot else None
        if source is None or not source.is_file():
            problems.append(f"not in this PC's cache: {f.label}")
            continue
        size = source.stat().st_size
        if size != f.size:
            problems.append(f"{f.label}: listed {f.size:,} bytes, cached {size:,}")
        repo = root / ("models--" + f.repo.replace("/", "--"))
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(snapshot.name, encoding="utf-8")
        link = repo / "snapshots" / snapshot.name / f.filename
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source.resolve())
    return problems


def load(model: str) -> None:
    """In the child: load what production loads for `model`, on the CPU."""
    from kitshot.render.gpu import prepare_environment

    prepare_environment()
    import torch
    from huggingface_hub import constants

    print(f"  cache {constants.HF_HUB_CACHE}", flush=True)
    if model in ("animagine", "realvis"):
        from diffusers import StableDiffusionXLPipeline

        from kitshot.render import renderer as r

        spec = r.ANIMAGINE if model == "animagine" else r.REALVIS
        variant = {"variant": spec.variant} if spec.variant else {}
        pipe = StableDiffusionXLPipeline.from_pretrained(
            spec.repo, dtype=torch.float16, use_safetensors=True, **variant
        )
        print(f"  {spec.repo}: {type(pipe.unet).__name__}, vae {type(pipe.vae).__name__}")
        assert pipe.tokenizer("a test").input_ids and pipe.tokenizer_2("a test").input_ids
        pipe.load_ip_adapter(
            r.IPA_REPO,
            subfolder=r.IPA_SUBFOLDER,
            weight_name=r.IPA_WEIGHT,
            image_encoder_folder=r.IPA_ENCODER,
        )
        print(f"  {r.IPA_REPO}: image encoder {type(pipe.image_encoder).__name__}")
        del pipe
    elif model == "zimage_turbo":
        import json

        from accelerate import init_empty_weights
        from diffusers import AutoencoderKL, ZImagePipeline, ZImageTransformer2DModel
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        from kitshot.render.zimage import ZIMAGE_REPO

        def fetch(name: str) -> str:
            return hf_hub_download(ZIMAGE_REPO, f"transformer/{name}", local_files_only=True)

        # The transformer as zimage.load_transformer_nf4 reads it, without the weights.
        config = ZImageTransformer2DModel.load_config(
            ZIMAGE_REPO, subfolder="transformer", local_files_only=True
        )
        with init_empty_weights():
            transformer = ZImageTransformer2DModel.from_config(config)
        index = fetch("diffusion_pytorch_model.safetensors.index.json")
        weight_map = json.loads(Path(index).read_text("utf-8"))["weight_map"]
        keys: set[str] = set()
        for shard in sorted(set(weight_map.values())):
            with safe_open(fetch(shard), framework="pt") as f:
                keys |= set(f.keys())
        names = {n for n, _ in transformer.named_parameters()}
        assert keys == names == set(weight_map), "the transformer's weights do not match"
        print(f"  transformer: {len(names)} weights in {len(set(weight_map.values()))} shards")
        bf16 = torch.bfloat16
        pipe = ZImagePipeline.from_pretrained(
            ZIMAGE_REPO, transformer=None, vae=None, dtype=bf16, local_files_only=True
        )
        text = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": "a test"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        assert pipe.tokenizer(text).input_ids
        print(f"  text encoder {type(pipe.text_encoder).__name__}, scheduler ok")
        del pipe
        vae = AutoencoderKL.from_pretrained(
            ZIMAGE_REPO, subfolder="vae", dtype=bf16, local_files_only=True
        )
        print(f"  vae {type(vae).__name__}")
        del vae
    else:
        raise SystemExit(f"no loader for {model!r}")
    from kitshot.render.tagger import Eva02Tagger

    tagger = Eva02Tagger("cpu")
    print(f"  tagger: {len(tagger.names)} tags")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="*", default=list(models.MODELS))
    parser.add_argument("--drop", metavar="REPO/FILE", help="leave this listed file out")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        load(args.child)
        return 0
    failed = []
    for model in args.model:
        files = models.files_for(model)
        print(f"{model}: {len(files)} files, {models.gigabytes(models.total_size(files))}")
        with tempfile.TemporaryDirectory(prefix="kitshot-lists-") as tmp:
            problems = lay_out(Path(tmp), files, args.drop)
            for p in problems:
                print(f"  {p}")
            env = {
                **os.environ,
                "HF_HUB_CACHE": tmp,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
            env.pop("HF_HOME", None)
            code = subprocess.run(
                [sys.executable, __file__, "--child", model], env=env, check=False
            ).returncode
        ok = code == 0 and not problems
        print(f"  {'loaded' if ok else 'FAILED'}", flush=True)
        if not ok:
            failed.append(model)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
