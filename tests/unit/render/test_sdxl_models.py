"""The SDXL checkpoints' settings (render/renderer.py). No GPU and no weights."""

from __future__ import annotations

from kitshot.render.renderer import ANIMAGINE, REALVIS


def test_realvis_decodes_in_tiles_of_animagines_size() -> None:
    """RealVisXL's VAE is Animagine's fp16-fix VAE with another config: `sample_size` 1024
    made its tiles barely smaller than a 896x1152 picture, so the low preset saved little
    (Phase 15i, docs/research.md section 20). Animagine keeps its own config (512)."""
    assert REALVIS.vae_tile == 512 and REALVIS.vae_upcast_off
    assert ANIMAGINE.vae_tile is None and not ANIMAGINE.vae_upcast_off
