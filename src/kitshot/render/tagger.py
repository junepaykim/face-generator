"""WD EVA02-Large inference, loaded lazily when the worker needs it."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from typing import TYPE_CHECKING

from kitshot.render.quality_checks import TagResult

if TYPE_CHECKING:
    from PIL.Image import Image

TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
_KEEP_FLOOR = 0.01  # probabilities below this are dropped from TagResult.tags


class Eva02Tagger:
    """WD EVA02-Large tagger v3 through timm, as in the bake-off, in fp16.

    With `offload` (the default) the model (0.6 GiB) waits in system memory and comes onto
    the GPU only while it tags, so it holds no VRAM while the UNet runs (gpu.on_gpu).
    """

    name = TAGGER_REPO

    def __init__(self, device: str = "cuda", *, offload: bool = True) -> None:
        from kitshot.render.gpu import prepare_environment

        prepare_environment()
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        self._torch = torch
        self.device = device
        self.offload = offload and device != "cpu"
        model = timm.create_model(f"hf-hub:{TAGGER_REPO}", pretrained=True)
        self.model = model.eval().to("cpu" if self.offload else device, torch.float16)
        config = timm.data.resolve_data_config(model.pretrained_cfg, model=model)
        self.transform = timm.data.create_transform(**config)
        with open(hf_hub_download(TAGGER_REPO, "selected_tags.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"].replace("_", " ") for r in rows]
        categories = [int(r["category"]) for r in rows]
        self.rating_idx = [i for i, c in enumerate(categories) if c == 9]
        self.general_idx = [i for i, c in enumerate(categories) if c == 0]

    def close(self) -> None:
        """Free the GPU memory the tagger holds."""
        import contextlib
        import gc

        self.model = None
        gc.collect()
        with contextlib.suppress(Exception):  # after a CUDA fault this fails too
            self._torch.cuda.empty_cache()

    def __call__(self, images: Sequence[Image]) -> list[TagResult]:
        from kitshot.render.gpu import on_gpu

        torch = self._torch
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images])
        x = batch[:, [2, 1, 0]].to(self.device, torch.float16)  # the model wants BGR
        with torch.no_grad(), on_gpu(self.model, device=self.device, enabled=self.offload):
            probs = torch.sigmoid(self.model(x)).float().cpu().tolist()
        out = []
        for row in probs:
            ratings = {self.names[i]: row[i] for i in self.rating_idx}
            tags = {self.names[i]: row[i] for i in self.general_idx if row[i] >= _KEEP_FLOOR}
            out.append(TagResult(ratings, tags))
        return out
