"""INT8 weights for the UNet's Linear layers (docs/research.md, run `inference`).

GPU code only: this module imports torch, so import it after
`facegen.gpu.prepare_environment`, as `AnimagineRenderer` does.

Weight-only: each weight matrix is stored as INT8 with one scale per output channel and
rebuilt in fp16 just before its matmul, so only the INT8 copy stays in memory. The
activations stay in fp16 and the matmul is the ordinary fp16 one, so nothing needs
torch.compile and nothing warms up. Quantising activations as well (INT8 matmul) was as fast
as this only under torch.compile; run eagerly it was almost twice as slow as fp16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

INT8_MAX = 127.0


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
    for name in names:
        parent, _, child = name.rpartition(".")
        holder = model.get_submodule(parent)
        setattr(holder, child, Int8Linear(getattr(holder, child), device))
    return len(names)
