# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""INT8 and NF4 weights for Linear layers (render/quantise.py), on the CPU. Skipped without
torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kitshot.render.quantise import (
    NF4_CODE,
    NF4_GROUP,
    Int8Linear,
    Nf4Linear,
    block_linears,
    quantise_blocks_nf4,
    quantise_linears,
)


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


def test_nf4_linear_packs_two_codes_per_byte_and_rebuilds_code_values_exactly() -> None:
    """Weights that are NF4 values times a group absmax come back bit for bit, and the codes
    sit two to a byte, low nibble first, with one absmax per 64 inputs."""
    torch.manual_seed(1)
    n_out, n_in = 6, 3 * NF4_GROUP
    code = torch.tensor(NF4_CODE)
    idx = torch.randint(0, 16, (n_out, n_in))
    idx[:, ::NF4_GROUP] = 15  # every group holds 1.0, so its absmax is exactly its scale
    scale = torch.rand(n_out, n_in // NF4_GROUP, 1) + 0.1
    weight = (code[idx].reshape(n_out, -1, NF4_GROUP) * scale).reshape(n_out, n_in)
    lin = torch.nn.Linear(n_in, n_out)
    with torch.no_grad():
        lin.weight.copy_(weight)
    q = Nf4Linear(lin)
    assert q.packed.dtype == torch.uint8
    assert q.packed.shape == (n_out, n_in // NF4_GROUP, NF4_GROUP // 2)
    assert q.absmax.shape == (n_out, n_in // NF4_GROUP, 1)
    assert q.absmax.dtype == q.pairs.dtype == torch.float32
    assert q.packed[2, 1, 5].item() == idx[2, NF4_GROUP + 10] | idx[2, NF4_GROUP + 11] << 4
    assert torch.equal(q.weight(), lin.weight)
    assert torch.equal(q.bias, lin.bias)
    x = torch.randn(5, n_in)
    assert torch.allclose(q(x), lin(x), atol=1e-5)


def test_nf4_linear_error_stays_within_four_bits() -> None:
    """Normal-distributed weights, like a transformer's, lose about 9% to NF4 (docs/research.md
    section 20: 9.2-10.3% on Z-Image-Turbo's layers), against about 1% for INT8. A group's
    absmax is its own, so a loud output channel costs the others nothing."""
    torch.manual_seed(2)
    lin = torch.nn.Linear(8 * NF4_GROUP, 64)
    with torch.no_grad():
        lin.weight.normal_(0, 0.02)
        lin.weight[3] *= 50
    q = Nf4Linear(lin)
    err = (q.weight() - lin.weight).norm(dim=1) / lin.weight.norm(dim=1)
    assert err.max().item() < 0.12  # every output channel, the loud one included
    total = ((q.weight() - lin.weight).norm() / lin.weight.norm()).item()
    int8 = ((Int8Linear(lin).weight() - lin.weight).norm() / lin.weight.norm()).item()
    assert int8 < 0.02 < total < 0.11
    x = torch.randn(64, 8 * NF4_GROUP)
    ref = lin(x)
    assert ((q(x) - ref).norm() / ref.norm()).item() < 0.15


def test_nf4_linear_rebuilds_in_the_weight_dtype() -> None:
    lin = torch.nn.Linear(NF4_GROUP, 16, bias=False).to(torch.bfloat16)
    q = Nf4Linear(lin)
    assert q.weight().dtype == q.absmax.dtype == torch.bfloat16
    assert q.bias is None
    assert q(torch.randn(4, NF4_GROUP, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_nf4_linear_keeps_an_all_zero_group_at_zero() -> None:
    lin = torch.nn.Linear(2 * NF4_GROUP, 4).to(torch.float16)
    with torch.no_grad():
        lin.weight[:, :NF4_GROUP] = 0
    w = Nf4Linear(lin).weight()
    assert torch.isfinite(w).all()
    assert torch.equal(w[:, :NF4_GROUP], torch.zeros(4, NF4_GROUP, dtype=torch.float16))


def test_nf4_linear_needs_whole_groups() -> None:
    with pytest.raises(ValueError, match="groups of 64"):
        Nf4Linear(torch.nn.Linear(NF4_GROUP + 8, 4))


def _transformer_like() -> torch.nn.Module:
    """Z-Image-Turbo's naming: numbered blocks of attention and feed-forward, with modulation,
    norms and embedders around them."""

    def block(n_in: int = NF4_GROUP) -> torch.nn.Module:
        b = torch.nn.Module()
        b.attention = torch.nn.Module()
        b.attention.to_q = torch.nn.Linear(n_in, NF4_GROUP)
        b.attention.to_out = torch.nn.ModuleList([torch.nn.Linear(NF4_GROUP, n_in)])
        b.attention.norm_q = torch.nn.Linear(n_in, n_in)
        b.feed_forward = torch.nn.Module()
        b.feed_forward.w1 = torch.nn.Linear(n_in, 2 * NF4_GROUP)
        b.adaLN_modulation = torch.nn.ModuleList([torch.nn.Linear(NF4_GROUP, 4 * n_in)])
        return b

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([block(), block()])
    model.noise_refiner = torch.nn.ModuleList([block(48)])  # 48 wide: some layers not in groups
    model.t_embedder = torch.nn.Module()
    model.t_embedder.mlp = torch.nn.ModuleList([torch.nn.Linear(NF4_GROUP, NF4_GROUP)])
    model.cap_embedder = torch.nn.ModuleList([torch.nn.Linear(NF4_GROUP, NF4_GROUP)])
    model.final_layer = torch.nn.Module()
    model.final_layer.linear = torch.nn.Linear(NF4_GROUP, 16)
    return model


def test_block_scope_takes_block_attention_and_feed_forward_only() -> None:
    model = _transformer_like()
    expected = [
        f"{blocks}.attention.{name}"
        for blocks in ("layers.0", "layers.1")
        for name in ("to_q", "to_out.0")
    ]
    feed_forward = ["layers.0.feed_forward.w1", "layers.1.feed_forward.w1"]
    # The refiner's output projection has 64 inputs, so it is whole NF4 groups.
    expected = sorted([*expected, *feed_forward, "noise_refiner.0.attention.to_out.0"])
    assert sorted(block_linears(model, NF4_GROUP)) == expected
    # Without a group size the refiner's two 48-input layers count too.
    assert len(block_linears(model)) == len(expected) + 2

    assert quantise_blocks_nf4(model) == len(expected)
    assert isinstance(model.layers[1].attention.to_out[0], Nf4Linear)
    assert type(model.layers[0].adaLN_modulation[0]) is torch.nn.Linear
    assert type(model.layers[0].attention.norm_q) is torch.nn.Linear
    assert type(model.noise_refiner[0].attention.to_q) is torch.nn.Linear
    assert type(model.t_embedder.mlp[0]) is torch.nn.Linear
    assert type(model.final_layer.linear) is torch.nn.Linear
    x = torch.randn(3, NF4_GROUP)
    assert model.layers[0].feed_forward.w1(x).shape == (3, 2 * NF4_GROUP)
