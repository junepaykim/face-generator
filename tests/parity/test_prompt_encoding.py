# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""The renderer's cached prompt encoding against the encoding from before the cache, on the
CPU with two tiny randomly initialised CLIP text encoders. Skipped without torch.

Each text is encoded on its own and cached; a pair is padded with the empty text's chunk
afterwards (renderer.pair_embeddings). That must give the very bits that encoding the pair
together gave, or pictures would change.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from encoding_reference import encode_pair_before_cache

from kitshot.render.renderer import _encode_text, pair_embeddings

BOS, EOS = 997, 998


class Tokenizer:
    """A word-level stand-in for the CLIP tokenizers: one token per word."""

    model_max_length = 77
    bos_token_id = BOS
    eos_token_id = EOS

    def __init__(self, pad: int) -> None:
        self.pad_token_id = pad

    def __call__(self, text: str, add_special_tokens: bool, verbose: bool) -> Any:
        assert not add_special_tokens
        ids = [1 + sum(map(ord, w)) % 990 for w in text.replace(",", " ").split()]
        return SimpleNamespace(input_ids=ids)


def _pipe() -> Any:
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(0)
    config = {
        "vocab_size": 1000,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "max_position_embeddings": 77,
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "pad_token_id": EOS,
    }
    return SimpleNamespace(
        # The SDXL pair: the first pads with its end token, the second with "!" (token 0).
        tokenizer=Tokenizer(pad=EOS),
        text_encoder=CLIPTextModel(CLIPTextConfig(**config)).eval(),
        tokenizer_2=Tokenizer(pad=0),
        text_encoder_2=CLIPTextModelWithProjection(
            CLIPTextConfig(**config, projection_dim=16)
        ).eval(),
    )


def _words(prefix: str, n: int) -> str:
    return ", ".join(f"{prefix}{i}" for i in range(n))


@pytest.mark.parametrize(
    "positive_words,negative_words,chunks",
    [
        (200, 80, 3),  # the positive longer, as usual: the negative is padded
        (60, 160, 3),  # the negative longer: the positive is padded
        (75, 76, 2),  # both at the 75-token edge
        (10, 0, 1),  # an empty negative prompt
    ],
)
def test_cached_texts_give_the_pair_encoding_bit_for_bit(
    positive_words: int, negative_words: int, chunks: int
) -> None:
    pipe = _pipe()
    positive, negative = _words("p", positive_words), _words("n", negative_words)
    with torch.no_grad():
        expected = encode_pair_before_cache(pipe, positive, negative)
        # Encoded separately and in another order, as the cache would hold them.
        empty = _encode_text(pipe, "")
        neg = _encode_text(pipe, negative)
        pos = _encode_text(pipe, positive)
        got = pair_embeddings(pos, neg, empty)
    assert got[4] == expected[4] == chunks
    for name, a, b in zip(("positive", "pooled", "negative", "negative pooled"), got, expected):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert torch.equal(a, b), name
    assert got[0].shape == (1, 77 * chunks, 64)  # both encoders' features side by side


def test_a_text_is_kept_with_its_own_chunk_count_in_system_memory() -> None:
    pipe = _pipe()
    with torch.no_grad():
        encoded = _encode_text(pipe, _words("p", 100))
    assert encoded.chunks == 2
    assert encoded.hidden.shape == (1, 154, 64)
    assert encoded.pooled.shape == (1, 16)
    assert encoded.hidden.device.type == encoded.pooled.device.type == "cpu"


def test_tokenizers_that_disagree_on_chunks_are_refused() -> None:
    class Short(Tokenizer):
        model_max_length = 40

    pipe = _pipe()
    pipe.tokenizer_2 = Short(pad=0)  # 38-token chunks against the first one's 75
    with torch.no_grad(), pytest.raises(RuntimeError, match="chunks"):
        _encode_text(pipe, _words("p", 100))
