"""INT8 weights for Linear layers (facegen.quantise), on the CPU. Skipped without torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from facegen.quantise import Int8Linear, quantise_linears


def _linear(n_in: int = 64, n_out: int = 48, bias: bool = True) -> torch.nn.Linear:
    torch.manual_seed(0)
    lin = torch.nn.Linear(n_in, n_out, bias=bias)
    with torch.no_grad():
        lin.weight[3] *= 50  # one loud output channel must not cost the others precision
    return lin


def test_int8_linear_matches_the_float_layer() -> None:
    lin = _linear()
    q = Int8Linear(lin)
    x = torch.randn(10, 64)
    ref, out = lin(x), q(x)
    assert q.weight_int8.dtype == torch.int8
    assert q.weight_scale.shape == (48, 1)  # one scale per output channel
    assert torch.equal(q.bias, lin.bias)
    # Per-channel scales: every channel keeps about INT8's precision, the loud one included.
    err = (out - ref).abs().amax(dim=0) / ref.abs().amax(dim=0)
    assert err.max().item() < 0.02


def test_int8_linear_rebuilds_in_the_weight_dtype() -> None:
    lin = _linear(bias=False).to(torch.bfloat16)
    q = Int8Linear(lin)
    assert q.weight().dtype == torch.bfloat16
    assert q.bias is None
    assert q(torch.randn(4, 64, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_quantise_linears_swaps_only_layers_in_scope() -> None:
    model = torch.nn.Module()
    model.blocks = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)
    )
    model.head = torch.nn.Linear(4, 2)
    assert quantise_linears(model, "blocks") == 2
    assert isinstance(model.blocks[0], Int8Linear)
    assert isinstance(model.blocks[2], Int8Linear)
    assert type(model.head) is torch.nn.Linear
    assert model.head(model.blocks(torch.randn(3, 8))).shape == (3, 2)
