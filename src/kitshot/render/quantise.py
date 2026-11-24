"""Weight-only INT8 and NF4 storage for Linear layers.

Import after `kitshot.render.gpu.prepare_environment`: this module imports torch.
INT8 keeps one scale per output channel (SDXL's UNet blocks, Qwen3's layers). NF4 keeps
QLoRA's 16 NormalFloat values with one absmax per NF4_GROUP inputs, two codes per byte
(Z-Image-Turbo's transformer blocks). Both rebuild the weight in its dtype for each matmul;
activations and matmuls stay in the model's dtype, without torch.compile or warm-up.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import torch
import torch.nn.functional as F

INT8_MAX = 127.0

NF4_CODE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)  # fmt: skip
"""The 16 NormalFloat-4 values of QLoRA (Dettmers et al. 2023), as bitsandbytes uses them."""
NF4_GROUP = 64
"""Inputs per absmax: bitsandbytes' default block size."""


class Int8Linear(torch.nn.Module):
    """nn.Linear with INT8 weights and one scale per output channel, in the weight's dtype."""

    def __init__(self, linear: torch.nn.Linear, device: torch.device | str | None = None) -> None:
        super().__init__()
        w = linear.weight.detach().to(device or linear.weight.device)
        scale = (w.float().abs().amax(dim=1, keepdim=True) / INT8_MAX).clamp(min=1e-8)
        scale = scale.to(w.dtype)  # rounded first, so quantising and rebuilding agree
        q = (w.float() / scale.float()).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        self.register_buffer("weight_int8", q)
        self.register_buffer("weight_scale", scale)
        bias = linear.bias
        self.bias = (
            None
            if bias is None
            else torch.nn.Parameter(bias.detach().to(w.device), requires_grad=False)
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def weight(self) -> torch.Tensor:
        """The rebuilt weight: int8 times the per-channel scale, one kernel, in fp16."""
        return self.weight_int8 * self.weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)


class Nf4Linear(torch.nn.Module):
    """nn.Linear with 4-bit NF4 codes, one absmax per NF4_GROUP inputs in the weight's dtype,
    and two codes per byte, the low nibble first.

    Each code is the NormalFloat value nearest the weight divided by its group's absmax. The
    weight is rebuilt for each call through a 256-row table of byte to value pairs: one
    `F.embedding` lookup per byte is 3.4 times faster than indexing per nibble (0.65 against
    2.2 ms for a 10240x3840 layer on the RTX 4080, whose matmul at 4,500 tokens takes 5.5 ms;
    docs/research.md section 20).
    """

    def __init__(self, linear: torch.nn.Linear, device: torch.device | str | None = None) -> None:
        super().__init__()
        dtype = linear.weight.dtype
        w = linear.weight.detach().to(device or linear.weight.device, torch.float32)
        out_features, in_features = w.shape
        if in_features % NF4_GROUP:
            raise ValueError(f"NF4 needs inputs in groups of {NF4_GROUP}, got {in_features}")
        groups = w.reshape(out_features, in_features // NF4_GROUP, NF4_GROUP)
        # Rounded to the weight's dtype first, so quantising and rebuilding use the same scale.
        absmax = groups.abs().amax(dim=-1, keepdim=True).to(dtype)
        scale = absmax.float().clamp(min=torch.finfo(torch.float32).tiny)
        code = torch.tensor(NF4_CODE, device=w.device)
        # The nearest code: the midpoints between neighbouring codes bound each code's bucket.
        idx = torch.bucketize(groups / scale, (code[1:] + code[:-1]) / 2).to(torch.uint8)
        self.register_buffer("packed", (idx[..., 0::2] | (idx[..., 1::2] << 4)).contiguous())
        self.register_buffer("absmax", absmax)
        byte = torch.arange(256, device=w.device)
        self.register_buffer("pairs", torch.stack((code[byte & 15], code[byte >> 4]), -1).to(dtype))
        bias = linear.bias
        self.bias = (
            None
            if bias is None
            else torch.nn.Parameter(bias.detach().to(w.device), requires_grad=False)
        )
        self.in_features = in_features
        self.out_features = out_features

    def weight(self) -> torch.Tensor:
        """The rebuilt weight: each code's value times its group's absmax, in the weight's dtype."""
        packed = self.packed
        values = F.embedding(packed.int(), self.pairs).reshape(*packed.shape[:-1], NF4_GROUP)
        return (values * self.absmax).reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)


def _replace(
    model: torch.nn.Module, names: list[str], make: Callable[[torch.nn.Linear], torch.nn.Module]
) -> int:
    for name in names:
        parent, _, child = name.rpartition(".")
        holder = model.get_submodule(parent)
        setattr(holder, child, make(getattr(holder, child)))
    return len(names)


def quantise_linears(
    model: torch.nn.Module, scope: str, device: torch.device | str | None = None
) -> int:
    """Swap every nn.Linear whose name contains `scope` for an Int8Linear; returns how many.

    With `device`, each layer is quantised there (the GPU is much faster than the CPU) and
    its fp16 weights are never moved there whole; the rest of the model stays where it is.
    """
    names = [
        name
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and scope in name
    ]
    return _replace(model, names, lambda linear: Int8Linear(linear, device))


_NUMBERED_BLOCK = re.compile(r"(^|\.)\d+\.")
BLOCK_KEEP = ("adaLN", "norm", "modulation", "embed", "time_", "proj_out")
"""Name parts of the Linear layers inside blocks that keep the model's dtype: modulation,
norms, embedders and the output projection see one row per picture and cost little exact."""


def block_linears(model: torch.nn.Module, group: int = 1) -> list[str]:
    """The Linear layers inside numbered blocks ("layers.3.", "noise_refiner.0."), except
    BLOCK_KEEP's, whose inputs divide into `group`. 238 in Z-Image-Turbo's transformer."""
    return [
        name
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and _NUMBERED_BLOCK.search(name)
        and not any(k in name for k in BLOCK_KEEP)
        and m.in_features % group == 0
    ]


def quantise_blocks_nf4(model: torch.nn.Module, device: torch.device | str | None = None) -> int:
    """Swap `block_linears` for Nf4Linear layers, built on `device` one by one; returns how
    many. The rest of the model stays where it is."""
    names = block_linears(model, NF4_GROUP)
    return _replace(model, names, lambda linear: Nf4Linear(linear, device))
