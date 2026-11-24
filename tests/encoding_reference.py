# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Prompt encoding as the renderer did it before the prompt cache (until 2026-11-12).

Kept as the reference for bit-for-bit checks: a positive and a negative prompt encoded
together, each padded with empty chunks to the pair's chunk count as it is encoded, on the
encoders' own device. test_prompt_encoding.py uses it on the CPU, test_gpu.py on the model.
"""

from __future__ import annotations

from typing import Any

from kitshot.render.renderer import _token_chunks


def _encode(pipe: Any, text: str, n_chunks: int) -> tuple[Any, Any]:
    import torch

    per_encoder = []
    pooled = None
    for tok, enc in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)):
        chunks = _token_chunks(tok, text)
        chunks += [[]] * (n_chunks - len(chunks))
        hidden = []
        for k, chunk in enumerate(chunks):
            row = [tok.bos_token_id, *chunk, tok.eos_token_id]
            row += [tok.pad_token_id] * (tok.model_max_length - len(row))
            out = enc(torch.tensor([row], device=enc.device), output_hidden_states=True)
            hidden.append(out.hidden_states[-2])
            if enc is pipe.text_encoder_2 and k == 0:
                pooled = out[0]
        per_encoder.append(torch.cat(hidden, dim=1))
    return torch.cat(per_encoder, dim=-1), pooled


def encode_pair_before_cache(pipe: Any, positive: str, negative: str) -> tuple[Any, ...]:
    """(prompt_embeds, pooled, negative_prompt_embeds, negative_pooled, chunks)."""
    n = max(len(_token_chunks(pipe.tokenizer, t)) for t in (positive, negative))
    pe, pp = _encode(pipe, positive, n)
    ne, npool = _encode(pipe, negative, n)
    return pe, pp, ne, npool, n
