"""Phase 0.3 spike: anime model bake-off.

Renders the same character sheets on each candidate model, checks every picture with an
anime tagger (content rating, hair and eye colour, kit, one character), and writes a
local comparison page. Pictures stay on this PC (AGENTS.md hard rule 6): everything goes
to data/bakeoff/<run>/, and the page is a local file.

    uv run python scripts/spikes/bakeoff.py prompts --model noobai     # no GPU
    uv run python scripts/spikes/bakeoff.py render                     # all models
    uv run python scripts/spikes/bakeoff.py render --models animagine --cases S01a,S01b
    uv run python scripts/spikes/bakeoff.py tag
    uv run python scripts/spikes/bakeoff.py report
    uv run python scripts/spikes/bakeoff.py all

Cases: 20 character sheets, four per age band, each drawn twice ("a" and "b") with a
different kit, pose and seed, to judge whether the same character is recognisable; plus
an age ladder, two sheets drawn in all five bands with the same kit, pose and seed, to
judge whether the bands look distinct.

Rendering is resumable: a picture already on disk is skipped. Runs offline once the
models are in the Hugging Face cache. Close FM27 first; the GPU needs the room.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Never reach the network while rendering or tagging; the models are already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Less reserved-but-unused GPU memory. With Windows holding about 3 GB of the 16, the
# driver otherwise spills into system RAM and rendering slows down by half.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[2]
BAKEOFF_DIR = ROOT / "data" / "bakeoff"
SIZE = 1024  # SDXL's native square size
STEPS = 28
CFG = 5.5

# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    file: str | None  # single-file checkpoint; None means the repo is in diffusers layout
    family: str  # prompt conventions: "noob" (NoobAI, Illustrious) or "animagine"


MODELS = {
    m.key: m
    for m in (
        ModelSpec(
            "animagine",
            "Animagine XL 4.0 Opt",
            "cagliostrolab/animagine-xl-4.0",
            None,
            "animagine",
        ),
        ModelSpec(
            "illustrious",
            "Illustrious XL 2.0",
            "OnomaAIResearch/Illustrious-XL-v2.0",
            "Illustrious-XL-v2.0.safetensors",
            "noob",
        ),
        ModelSpec("noobai", "NoobAI-XL 1.1", "Laxhar/noobai-XL-1.1", None, "noob"),
    )
}
# The user chose Animagine (2026-11-10). NoobAI and Illustrious stay listed so the round
# 1 report can be rebuilt; Anima (round 2) was removed.
DEFAULT_MODELS = ("animagine",)

TAGGERS = {
    "eva02": "SmilingWolf/wd-eva02-large-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
}

# ------------------------------------------------------------------ prompt vocabulary
# Tags follow Danbooru conventions, which all three models were trained on.

# Fixed content terms (spec section 3). Every prompt carries them; nothing turns them off.
# Both model cards put the rating tag near the start of the prompt.
RATING = "safe"
CONTENT_POSITIVE = "sportswear, soccer uniform, soccer"
CONTENT_NEGATIVE = (
    "nsfw, explicit, questionable, sensitive, nude, nipples, underwear, lingerie, swimsuit, "
    "bikini, cleavage, midriff, navel, see-through, suggestive, revealing clothes, large breasts"
)
# Kit logos, brand marks, lettering and numbers are accepted (user, 2026-11-10), so none
# of them is in the negative prompt. These terms are about the picture, not the kit.
COMPOSITION_NEGATIVE = (
    "multiple girls, 2girls, multiple views, text, watermark, signature, username, artist name"
)
QUALITY = {
    "noob": (
        "masterpiece, best quality, newest, absurdres, highres",
        (
            "worst quality, old, early, low quality, lowres, bad hands, mutated hands, "
            "mammal, anthro, furry, ambiguous form, feral, semi-anthro"
        ),
    ),
    "animagine": (
        "masterpiece, high score, great score, absurdres",
        (
            "lowres, bad anatomy, bad hands, error, missing finger, extra digits, fewer digits, "
            "cropped, worst quality, low quality, low score, bad score, average score, blurry"
        ),
    ),
}
FRAMING = "upper body, looking at viewer, standing, simple background, grey background"

# Image-age bands (spec section 3). Neutral age wording only; this table is what the
# ladder tests.
BANDS = {
    1: "young teenage girl, teenage, youthful face",
    2: "teenage girl, teenage",
    3: "young adult woman",
    4: "adult woman, mature face",
    5: "mature female, adult woman, mature face",
}

# Round 2: band 1 should look younger (user, 2026-11-10). A variant replaces a band's
# wording and adds negative terms for it. Every variant's extra negative keeps adult
# features off the youngest band; that is part of the content rule, not a style choice.
YOUNG_NEGATIVE = "mature female, adult, mature face, breasts"
BAND_VARIANTS: dict[str, dict[int, tuple[str, str]]] = {
    "neg": {1: (BANDS[1], YOUNG_NEGATIVE)},
    "child": {1: ("child, young girl, youthful face, round face", YOUNG_NEGATIVE)},
    "aged": {1: ("aged down, young girl, youthful face, round face", YOUNG_NEGATIVE)},
}

# value -> (prompt tags, tags the tagger may report for it)
HAIR_COLOUR = {
    "black": ("black hair", {"black hair"}),
    "brown": ("brown hair", {"brown hair", "light brown hair"}),
    "blonde": ("blonde hair", {"blonde hair"}),
    "silver": ("silver hair", {"grey hair", "white hair"}),
    "pink": ("pink hair", {"pink hair"}),
    "blue": ("blue hair", {"blue hair", "light blue hair"}),
    "red": ("red hair", {"red hair"}),
}
EYE_COLOUR = {
    "brown": ("brown eyes", {"brown eyes"}),
    "blue": ("blue eyes", {"blue eyes", "aqua eyes"}),
    "green": ("green eyes", {"green eyes"}),
    "amber": ("amber eyes, yellow eyes", {"yellow eyes", "orange eyes"}),
    "violet": ("purple eyes", {"purple eyes"}),
}
HAIR_STYLE = {
    "long": "long hair, straight hair",
    "ponytail": "ponytail",
    "twintails": "twintails",
    "bob": "bob cut",
    "short": "short hair",
    "braid": "braid",
}
FRINGE = {"blunt": "blunt bangs", "parted": "parted bangs", "swept": "swept bangs"}
EYE_SHAPE = {"gentle": "tareme", "sharp": "tsurime", "round": ""}
SKIN = {"fair": "pale skin", "light": "", "tan": "tan", "dark": "dark skin, dark-skinned female"}
# Lightest to darkest. "slightly tanned skin" sits between light and tan, so a softened tan
# or dark skin keeps some of its colour. At my club, tan and dark skin are drawn
# `soften` steps lighter (user, 2026-11-11); fair and light are unchanged.
SKIN_SCALE = ["pale skin", "", "slightly tanned skin", "tan", "dark skin, dark-skinned female"]
SKIN_STEP = {"fair": 0, "light": 1, "tan": 3, "dark": 4}


def skin_words(skin: str, soften: int = 0) -> str:
    step = SKIN_STEP[skin]
    if step >= 3:
        step = max(step - soften, 1)
    return SKIN_SCALE[step]


ACCESSORY = {
    "none": "",
    "hairband": "hairband",
    "ribbon": "hair ribbon",
    "clip": "hairclip",
    "glasses": "glasses",
}
MARK = {"none": "", "freckles": "freckles", "mole": "mole under eye"}
# Banned traits are named by their sheet value; this gives their prompt words.
TRAIT_WORDS = {k: v for k, v in (ACCESSORY | MARK).items() if v}
EXPRESSION = {
    "cheerful": "smile, open mouth",
    "serious": "serious, closed mouth",
    "shy": "shy, light blush, closed mouth",
    "confident": "confident, smirk",
}
POSE = {
    "crossed": "crossed arms",
    "hips": "hands on own hips",
    "ball": "holding ball, soccer ball",
    "behind": "arms behind back",
    "thumb": "thumbs up",
    "v": "v",
}

STRIPES_V = {
    "vertical-striped shirt",
    "striped shirt",
    "vertical-striped clothes",
    "striped clothes",
}
STRIPES_H = {
    "horizontal-striped shirt",
    "striped shirt",
    "horizontal-striped clothes",
    "striped clothes",
}
GLOVES = {"gloves", "black gloves", "grey gloves"}

# kit -> (prompt tags, groups of tagger tags; one tag from each group must be present)
KITS: dict[str, tuple[str, list[set[str]]]] = {
    "blue_white_stripes": (
        "blue and white vertical-striped shirt, short sleeves, blue shorts",
        [STRIPES_V, {"blue shirt", "white shirt"}],
    ),
    "red_white": ("red shirt, white collar, short sleeves, white shorts", [{"red shirt"}]),
    "yellow_green_hoops": (
        "yellow and green horizontal-striped shirt, short sleeves, green shorts",
        [STRIPES_H, {"yellow shirt", "green shirt"}],
    ),
    "black_white_stripes": (
        "black and white vertical-striped shirt, short sleeves, black shorts",
        [STRIPES_V, {"black shirt", "white shirt"}],
    ),
    "sky_blue": ("light blue shirt, white collar, short sleeves, white shorts", [{"blue shirt"}]),
    "claret_blue": (
        "dark red shirt, light blue sleeves, short sleeves, white shorts",
        [{"red shirt"}],
    ),
    "orange_black": ("orange shirt, black trim, short sleeves, black shorts", [{"orange shirt"}]),
    "white_navy": (
        "white shirt, navy blue collar, short sleeves, navy blue shorts",
        [{"white shirt"}],
    ),
    "gk_green": ("goalkeeper, green shirt, long sleeves, black gloves", [{"green shirt"}, GLOVES]),
    "gk_orange": (
        "goalkeeper, orange shirt, long sleeves, black gloves",
        [{"orange shirt"}, GLOVES],
    ),
    "gk_purple": (
        "goalkeeper, purple shirt, long sleeves, grey gloves",
        [{"purple shirt"}, GLOVES],
    ),
}

# ----------------------------------------------------------------------------- cases


@dataclass(frozen=True)
class Sheet:
    id: str
    band: int
    hair: str
    style: str
    fringe: str
    eyes: str
    eye_shape: str
    skin: str
    accessory: str
    mark: str
    expression: str
    kits: tuple[str, str]
    poses: tuple[str, str]


# Four per band; varied on purpose. Three are goalkeepers (S03, S07, S13).
# Columns: id, band, hair, style, fringe, eyes, eye shape, skin, accessory, mark,
# expression, kits (a, b), poses (a, b).
# fmt: off
SHEETS = [
    Sheet("S01", 1, "black", "long", "blunt", "brown", "gentle", "light", "hairband", "none",
          "cheerful", ("blue_white_stripes", "red_white"), ("ball", "behind")),
    Sheet("S02", 1, "blonde", "twintails", "parted", "blue", "round", "fair", "ribbon",
          "freckles", "shy", ("yellow_green_hoops", "sky_blue"), ("v", "hips")),
    Sheet("S03", 1, "brown", "short", "swept", "green", "sharp", "tan", "none", "none",
          "confident", ("gk_green", "gk_orange"), ("crossed", "thumb")),
    Sheet("S04", 1, "pink", "bob", "blunt", "violet", "round", "light", "clip", "mole",
          "cheerful", ("white_navy", "claret_blue"), ("thumb", "ball")),
    Sheet("S05", 2, "silver", "ponytail", "swept", "blue", "sharp", "fair", "none", "none",
          "serious", ("black_white_stripes", "orange_black"), ("crossed", "hips")),
    Sheet("S06", 2, "black", "braid", "parted", "amber", "gentle", "dark", "glasses", "none",
          "cheerful", ("red_white", "blue_white_stripes"), ("v", "ball")),
    Sheet("S07", 2, "blue", "short", "blunt", "amber", "round", "light", "hairband",
          "freckles", "confident", ("gk_purple", "gk_green"), ("hips", "crossed")),
    Sheet("S08", 2, "red", "long", "swept", "green", "gentle", "fair", "ribbon", "none", "shy",
          ("sky_blue", "white_navy"), ("behind", "thumb")),
    Sheet("S09", 3, "brown", "ponytail", "parted", "brown", "sharp", "tan", "none", "mole",
          "serious", ("claret_blue", "yellow_green_hoops"), ("hips", "ball")),
    Sheet("S10", 3, "blonde", "bob", "swept", "green", "round", "light", "glasses", "none",
          "cheerful", ("orange_black", "black_white_stripes"), ("thumb", "v")),
    Sheet("S11", 3, "black", "twintails", "blunt", "violet", "sharp", "fair", "ribbon", "none",
          "confident", ("white_navy", "red_white"), ("crossed", "behind")),
    Sheet("S12", 3, "pink", "long", "parted", "blue", "gentle", "dark", "clip", "freckles",
          "cheerful", ("blue_white_stripes", "sky_blue"), ("ball", "hips")),
    Sheet("S13", 4, "black", "short", "swept", "brown", "sharp", "dark", "none", "none",
          "serious", ("gk_orange", "gk_purple"), ("crossed", "hips")),
    Sheet("S14", 4, "silver", "long", "blunt", "violet", "gentle", "fair", "hairband", "mole",
          "shy", ("yellow_green_hoops", "claret_blue"), ("behind", "ball")),
    Sheet("S15", 4, "brown", "braid", "parted", "amber", "round", "light", "glasses",
          "freckles", "cheerful", ("red_white", "orange_black"), ("v", "thumb")),
    Sheet("S16", 4, "blue", "ponytail", "blunt", "green", "sharp", "tan", "none", "none",
          "confident", ("black_white_stripes", "white_navy"), ("hips", "crossed")),
    Sheet("S17", 5, "blonde", "long", "parted", "blue", "sharp", "light", "none", "mole",
          "serious", ("sky_blue", "red_white"), ("crossed", "ball")),
    Sheet("S18", 5, "red", "bob", "swept", "brown", "gentle", "fair", "clip", "freckles",
          "cheerful", ("claret_blue", "blue_white_stripes"), ("thumb", "behind")),
    Sheet("S19", 5, "black", "ponytail", "parted", "green", "round", "dark", "ribbon", "none",
          "confident", ("orange_black", "yellow_green_hoops"), ("hips", "v")),
    Sheet("S20", 5, "brown", "short", "blunt", "violet", "sharp", "tan", "glasses", "none",
          "serious", ("white_navy", "black_white_stripes"), ("ball", "crossed")),
]

# Age ladder: two contrasting looks, same kit, pose and seed in every band.
LADDER = [
    Sheet("L1", 0, "black", "long", "blunt", "brown", "gentle", "light", "none", "none",
          "cheerful", ("blue_white_stripes", "blue_white_stripes"), ("crossed", "crossed")),
    Sheet("L2", 0, "blonde", "ponytail", "parted", "blue", "sharp", "fair", "none", "none",
          "serious", ("red_white", "red_white"), ("hips", "hips")),
]
# Two more looks for the round 3 age wordings, so the choice does not rest on two faces.
AGE_SHEETS = LADDER + [
    Sheet("L3", 0, "brown", "short", "swept", "green", "sharp", "tan", "none", "freckles",
          "confident", ("orange_black", "orange_black"), ("ball", "ball")),
    Sheet("L4", 0, "pink", "twintails", "blunt", "violet", "round", "dark", "ribbon", "none",
          "shy", ("white_navy", "white_navy"), ("behind", "behind")),
]
# fmt: on

# Round 3 (user, 2026-11-10): ten age wordings, youngest first, for the user to pick five
# as bands 1-5. The user bounded the range: no younger than round 2's band 1 "child" and
# "aged" wordings, no older than round 1's band 5. Wordings meant to look under 18 carry
# YOUNG_NEGATIVE (content rule). key -> (wording, extra negative, where it came from)
AGE_WORDINGS: dict[str, tuple[str, str, str]] = {
    "W01": ("child, young girl, youthful face, round face", YOUNG_NEGATIVE, "r2 band 1 'child'"),
    "W02": ("aged down, young girl, youthful face, round face", YOUNG_NEGATIVE, "r2 band 1 'aged'"),
    "W03": ("young girl, youthful face, round face", YOUNG_NEGATIVE, "new"),
    "W04": ("young teenage girl, teenage, youthful face, round face", YOUNG_NEGATIVE, "new"),
    "W05": (
        "young teenage girl, teenage, youthful face",
        YOUNG_NEGATIVE,
        "r1 band 1, with the young negative",
    ),
    "W06": ("teenage girl, teenage", YOUNG_NEGATIVE, "r1 band 2, with the young negative"),
    "W07": ("young adult woman", "", "r1 band 3"),
    "W08": ("young woman, adult", "", "new"),
    "W09": ("adult woman, mature face", "", "r1 band 4"),
    "W10": ("mature female, adult woman, mature face", "", "r1 band 5"),
}


@dataclass(frozen=True)
class Case:
    id: str  # e.g. "S01a", "L1-b3"
    sheet: Sheet
    band: int
    kit: str
    pose: str
    seed: int
    group: str  # "pair", "ladder" or "ages"
    expect: dict[str, list[list[str]]] = field(default_factory=dict)
    wording: tuple[str, str] | None = None  # (age wording, extra negative) instead of a band


def base_seed(sheet_id: str) -> int:
    """Seeded from the sheet ID, as the real tool seeds from the player's unique ID."""
    return int.from_bytes(hashlib.sha256(f"bakeoff:{sheet_id}".encode()).digest()[:4], "big")


def make_cases() -> list[Case]:
    cases = []
    for s in SHEETS:
        for i, variant in enumerate("ab"):
            cases.append(
                Case(
                    f"{s.id}{variant}",
                    s,
                    s.band,
                    s.kits[i],
                    s.poses[i],
                    base_seed(s.id) + i,
                    "pair",
                    expectations(s, s.kits[i]),
                )
            )
    for s in LADDER:
        for band in BANDS:
            cases.append(
                Case(
                    f"{s.id}-b{band}",
                    s,
                    band,
                    s.kits[0],
                    s.poses[0],
                    base_seed(s.id),
                    "ladder",
                    expectations(s, s.kits[0]),
                )
            )
    return cases


def make_age_cases() -> list[Case]:
    """Round 3: every age wording on every age sheet; seed, kit and pose fixed per sheet."""
    return [
        Case(
            f"{s.id}-{key}",
            s,
            0,
            s.kits[0],
            s.poses[0],
            base_seed(s.id),
            "ages",
            expectations(s, s.kits[0]),
            (words, negative),
        )
        for s in AGE_SHEETS
        for key, (words, negative, _) in AGE_WORDINGS.items()
    ]


def expectations(s: Sheet, kit: str) -> dict[str, list[list[str]]]:
    return {
        "hair": [sorted(HAIR_COLOUR[s.hair][1])],
        "eyes": [sorted(EYE_COLOUR[s.eyes][1])],
        "kit": [sorted(group) for group in KITS[kit][1]],
    }


def _join(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


def build_prompt(
    case: Case,
    family: str,
    variant: str | None = None,
    soften: int = 0,
    banned: frozenset[str] = frozenset(),
) -> tuple[str, str]:
    """`soften` and `banned` are the my-club adjustments (settings `my_club`)."""
    s = case.sheet
    band_words, band_negative = case.wording or (BANDS[case.band], "")
    if variant and case.band in BAND_VARIANTS[variant]:
        band_words, band_negative = BAND_VARIANTS[variant][case.band]
    look = _join(
        HAIR_COLOUR[s.hair][0],
        HAIR_STYLE[s.style],
        FRINGE[s.fringe],
        EYE_COLOUR[s.eyes][0],
        EYE_SHAPE[s.eye_shape],
        skin_words(s.skin, soften),
        "" if s.accessory in banned else ACCESSORY[s.accessory],
        "" if s.mark in banned else MARK[s.mark],
    )
    body = _join(
        band_words,
        look,
        KITS[case.kit][0],
        POSE[case.pose],
        EXPRESSION[s.expression],
        FRAMING,
        CONTENT_POSITIVE,
    )
    quality, quality_negative = QUALITY[family]
    if family == "animagine":  # subject, rating, tags, quality last (model card)
        positive = _join("1girl, solo", RATING, body, quality)
    else:  # quality and rating prefix first (model card)
        positive = _join(quality, RATING, "1girl, solo", body)
    banned_words = _join(*(TRAIT_WORDS[x] for x in sorted(banned)))
    negative = _join(
        CONTENT_NEGATIVE, band_negative, banned_words, COMPOSITION_NEGATIVE, quality_negative
    )
    return positive, negative


# ------------------------------------------------------------------------ rendering


# One VAE for every model: madebyollin/sdxl-vae-fp16-fix, which Animagine ships as its
# own. NoobAI and Illustrious ship the original SDXL VAE, which overflows in fp16, so the
# pipeline upcasts it to fp32 for every decode; on this card that pushed peak memory past
# what Windows leaves free. The fp16-fix VAE is a fine-tune of that same VAE.
FP16_VAE = ("cagliostrolab/animagine-xl-4.0", "vae")


def load_pipeline(spec: ModelSpec):
    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(FP16_VAE[0], subfolder=FP16_VAE[1], dtype=torch.float16)
    if spec.file:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(spec.repo, spec.file)
        pipe = StableDiffusionXLPipeline.from_single_file(
            path, vae=vae, dtype=torch.float16, config="stabilityai/stable-diffusion-xl-base-1.0"
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            spec.repo, vae=vae, dtype=torch.float16, use_safetensors=True
        )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, spec: ModelSpec, positive: str, negative: str, seed: int):
    """One picture. Returns the image and the settings to store with it."""
    import torch

    generator = torch.Generator("cuda").manual_seed(seed)
    with torch.no_grad():
        n_chunks = max(len(_token_chunks(pipe.tokenizer, t)) for t in (positive, negative))
        pe, pp = encode(pipe, positive, n_chunks)
        ne, npool = encode(pipe, negative, n_chunks)
        image = pipe(
            prompt_embeds=pe,
            pooled_prompt_embeds=pp,
            negative_prompt_embeds=ne,
            negative_pooled_prompt_embeds=npool,
            width=SIZE,
            height=SIZE,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            generator=generator,
        ).images[0]
    settings = {
        "steps": STEPS,
        "cfg": CFG,
        "sampler": "euler_a",
        "vae": "/".join(FP16_VAE),
        "prompt_chunks": n_chunks,
    }
    return image, settings


def _token_chunks(tokenizer, text: str) -> list[list[int]]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    size = tokenizer.model_max_length - 2
    return [ids[i : i + size] for i in range(0, len(ids), size)] or [[]]


def encode(pipe, text: str, n_chunks: int):
    """SDXL prompt embedding without the 77-token cut-off.

    Splits the prompt into 75-token chunks, encodes each through both text encoders and
    joins them, as ComfyUI and A1111 do. The stock pipeline silently drops everything
    after token 77, which would cut the kit and the content terms from a long prompt.
    """
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


def column_name(model: str, variant: str | None) -> str:
    """Output folder and report column: the model, or "model~variant"."""
    return f"{model}~{variant}" if variant else model


def render(
    run_dir: Path,
    model_keys: list[str],
    cases: list[Case],
    variant: str | None = None,
    column: str | None = None,
    soften: int = 0,
    banned: frozenset[str] = frozenset(),
) -> None:
    import torch
    from PIL.PngImagePlugin import PngInfo

    log_path = run_dir / "renders.jsonl"
    models_path = run_dir / "models.json"
    model_stats = json.loads(models_path.read_text()) if models_path.exists() else {}

    for model_key in model_keys:
        spec = MODELS[model_key]
        key = column_name(model_key, column or variant)
        out_dir = run_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [c for c in cases if not (out_dir / f"{c.id}.png").exists()]
        if not todo:
            print(f"{key}: nothing to render")
            continue
        print(f"{key}: loading {spec.label}")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pipe = load_pipeline(spec)
        load_s = time.perf_counter() - t0
        weights_gb = torch.cuda.memory_allocated() / 2**30
        print(f"{key}: loaded in {load_s:.0f} s, {weights_gb:.1f} GiB of weights on the GPU")

        for n, case in enumerate(todo, 1):
            positive, negative = build_prompt(case, spec.family, variant, soften, banned)
            t1 = time.perf_counter()
            image, settings = generate(pipe, spec, positive, negative, case.seed)
            seconds = time.perf_counter() - t1
            record = {
                "model": model_key,
                "variant": variant,
                "model_label": spec.label,
                "repo": spec.repo,
                "file": spec.file,
                "case": case.id,
                "seed": case.seed,
                **settings,
                "width": SIZE,
                "height": SIZE,
                "prompt": positive,
                "negative": negative,
                "seconds": round(seconds, 2),
            }
            meta = PngInfo()
            meta.add_text("facegen", json.dumps(record))
            tmp = out_dir / f".{case.id}.png.tmp"
            image.save(tmp, format="PNG", pnginfo=meta)
            tmp.replace(out_dir / f"{case.id}.png")
            with log_path.open("a") as f:
                f.write(json.dumps({**record, "column": key}) + "\n")
            print(f"  [{n}/{len(todo)}] {case.id} {seconds:.1f} s")

        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        reserved_gb = torch.cuda.max_memory_reserved() / 2**30
        model_stats[key] = {
            "label": spec.label,
            "load_seconds": round(load_s, 1),
            "weights_gib": round(weights_gb, 2),
            "peak_allocated_gib": round(peak_gb, 2),
            "peak_reserved_gib": round(reserved_gb, 2),
        }
        models_path.write_text(json.dumps(model_stats, indent=2))
        print(f"{key}: peak {peak_gb:.1f} GiB allocated, {reserved_gb:.1f} GiB reserved")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


# -------------------------------------------------------------------------- tagging


def load_tagger(key: str):
    import timm
    import torch
    from huggingface_hub import hf_hub_download

    repo = TAGGERS[key]
    model = timm.create_model(f"hf-hub:{repo}", pretrained=True).eval().to("cuda", torch.float16)
    config = timm.data.resolve_data_config(model.pretrained_cfg, model=model)
    transform = timm.data.create_transform(**config)
    with open(hf_hub_download(repo, "selected_tags.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"].replace("_", " ") for r in rows]
    categories = [int(r["category"]) for r in rows]
    return model, transform, names, categories


# Kept at any probability, as a rough signal of apparent age for the band tuning.
PROBE_TAGS = ("child", "aged down", "mature female", "dark skin", "tan", "freckles")


def tag(run_dir: Path, tagger_keys: list[str], threshold: float = 0.35) -> None:
    import torch
    from PIL import Image

    images = sorted(p for p in run_dir.glob("*/*.png") if p.parent.name.partition("~")[0] in MODELS)
    if not images:
        sys.exit(f"no pictures under {run_dir}; run 'render' first")
    for key in tagger_keys:
        model, transform, names, categories = load_tagger(key)
        rating_idx = [i for i, c in enumerate(categories) if c == 9]
        general_idx = [i for i, c in enumerate(categories) if c == 0]
        probe_idx = [i for i, n in enumerate(names) if n in PROBE_TAGS]
        results = {}
        t0 = time.perf_counter()
        for start in range(0, len(images), 8):
            batch = images[start : start + 8]
            tensors = [transform(Image.open(p).convert("RGB")) for p in batch]
            x = torch.stack(tensors)[:, [2, 1, 0]].to("cuda", torch.float16)  # RGB -> BGR
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).float().cpu()
            for p, row in zip(batch, probs, strict=True):
                ratings = {names[i]: round(float(row[i]), 4) for i in rating_idx}
                general = {
                    names[i]: round(float(row[i]), 3) for i in general_idx if row[i] >= threshold
                }
                results[f"{p.parent.name}/{p.stem}"] = {
                    "rating": max(ratings, key=ratings.get),
                    "ratings": ratings,
                    "tags": dict(sorted(general.items(), key=lambda kv: -kv[1])),
                    "probes": {names[i]: round(float(row[i]), 3) for i in probe_idx},
                }
        seconds = time.perf_counter() - t0
        out = {
            "tagger": TAGGERS[key],
            "threshold": threshold,
            "seconds_per_image": round(seconds / len(images), 3),
            "images": results,
        }
        (run_dir / f"tags_{key}.json").write_text(json.dumps(out, indent=1))
        print(f"{key}: tagged {len(images)} pictures, {seconds / len(images):.2f} s each")
        del model
        gc.collect()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- report


def check(expect: dict[str, list[list[str]]], tags: dict[str, float]) -> dict[str, bool]:
    result = {
        name: all(any(t in tags for t in group) for group in groups)
        for name, groups in expect.items()
    }
    # Exactly one character, and a girl: "solo" alone also fires on pictures of nobody.
    result["solo"] = (
        "solo" in tags
        and "1girl" in tags
        and not any(
            t in tags for t in ("no humans", "multiple girls", "2girls", "3girls", "multiple views")
        )
    )
    return result


def report(run_dir: Path) -> Path:
    cases = {c.id: c for c in make_cases()}
    renders = {}
    if (run_dir / "renders.jsonl").exists():
        for line in (run_dir / "renders.jsonl").read_text().splitlines():
            r = json.loads(line)
            renders[(r.get("column", r["model"]), r["case"])] = r  # last write wins
    models_stats = (
        json.loads((run_dir / "models.json").read_text())
        if (run_dir / "models.json").exists()
        else {}
    )
    taggers = {
        k: json.loads((run_dir / f"tags_{k}.json").read_text())
        for k in TAGGERS
        if (run_dir / f"tags_{k}.json").exists()
    }
    primary = "eva02" if "eva02" in taggers else next(iter(taggers), None)
    order = list(MODELS)
    models = sorted(
        (
            d.name
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.partition("~")[0] in MODELS and any(d.glob("*.png"))
        ),
        key=lambda n: (order.index(n.partition("~")[0]), n),
    )

    def label(col: str) -> str:
        base, _, variant = col.partition("~")
        return MODELS[base].label + (f" · band 1 '{variant}'" if variant else "")

    # Per picture: checks from the primary tagger, ratings from every tagger.
    rows = []
    for m in models:
        for cid, case in cases.items():
            if not (run_dir / m / f"{cid}.png").exists():
                continue
            row = {
                "model": m,
                "case": cid,
                "group": case.group,
                "band": case.band,
                "seconds": renders.get((m, cid), {}).get("seconds"),
            }
            for tk, data in taggers.items():
                info = data["images"].get(f"{m}/{cid}")
                if info:
                    row[f"rating_{tk}"] = info["rating"]
                    row[f"general_{tk}"] = info["ratings"].get("general")
            if primary and (info := taggers[primary]["images"].get(f"{m}/{cid}")):
                row["tags"] = info["tags"]
                row.update(check(case.expect, info["tags"]))
                for probe, value in info.get("probes", {}).items():
                    row[f"p_{probe.replace(' ', '_')}"] = value
            rows.append(row)

    with (run_dir / "summary.csv").open("w", newline="") as f:
        keys = ["model", "case", "group", "band", "seconds", "hair", "eyes", "kit", "solo"]
        keys += [f"{a}_{t}" for t in TAGGERS for a in ("rating", "general")]
        keys += [f"p_{probe.replace(' ', '_')}" for probe in PROBE_TAGS]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def pct(values: list[bool]) -> str:
        return f"{100 * sum(values) / len(values):.0f}%" if values else "–"

    summary = []
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        secs = [r["seconds"] for r in mine if r["seconds"]]
        stats = models_stats.get(m, {})
        line = {
            "Model": label(m),
            "Pictures": str(len(mine)),
            "s / picture": f"{statistics.median(secs):.1f}" if secs else "–",
            "Peak VRAM": f"{stats['peak_reserved_gib']:.1f} GiB" if stats else "–",
            "Load": f"{stats['load_seconds']:.0f} s" if stats else "–",
        }
        for tk in taggers:
            line[f"General ({tk})"] = pct([r.get(f"rating_{tk}") == "general" for r in mine])
        if primary:
            for name in ("hair", "eyes", "kit", "solo"):
                line[name.capitalize()] = pct([r.get(name, False) for r in mine])
            for band, probe in ((1, "p_child"), (5, "p_mature_female")):
                vals = [r[probe] for r in mine if r["band"] == band and probe in r]
                name = f"Band {band} {probe[2:].replace('_', ' ')}"
                line[name] = f"{statistics.mean(vals):.2f}" if vals else "–"
        summary.append(line)

    def cell(m: str, cid: str) -> str:
        row = next((r for r in rows if r["model"] == m and r["case"] == cid), None)
        note = ""
        if row is None and "~" in m:  # a band variant only renders that band; show the base
            base = m.partition("~")[0]
            row = next((r for r in rows if r["model"] == base and r["case"] == cid), None)
            m, note = base, '<span class="b">same as base</span>'
        if row is None:
            return '<div class="pic missing">not rendered</div>'
        badges = [note] if note else []
        for tk in taggers:
            rating, g = row.get(f"rating_{tk}"), row.get(f"general_{tk}")
            if rating:
                cls = "ok" if rating == "general" else "bad"
                badges.append(
                    f'<span class="b {cls}" title="{tk}: p(general)={g}">'
                    f"{tk} {html.escape(rating)}</span>"
                )
        for name in ("hair", "eyes", "kit", "solo"):
            if name in row:
                cls = "ok" if row[name] else "bad"
                badges.append(f'<span class="b {cls}">{name}</span>')
        if "p_child" in row:
            badges.append(f'<span class="b">child {row["p_child"]:.2f}</span>')
        tags = ", ".join(list(row.get("tags", {}))[:40])
        r = renders.get((m, cid), {})
        title = html.escape(
            f"{cid} · seed {r.get('seed')} · {row['seconds']} s\n\n"
            f"{r.get('prompt', '')}\n\nTagger: {tags}"
        )
        return (
            f'<figure class="pic"><a href="{m}/{cid}.png" target="_blank">'
            f'<img src="{m}/{cid}.png" loading="lazy" title="{title}" alt="{cid}"></a>'
            f"<figcaption>{''.join(badges)}</figcaption></figure>"
        )

    head = "".join(f"<th>{html.escape(k)}</th>" for k in (summary[0] if summary else {}))
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in s.values()) + "</tr>"
        for s in summary
    )
    sections = [
        f"<h2>Summary</h2><div class='scroll'><table><tr>{head}</tr>{body}</table></div>",
        (
            f"<p class='note'>Checks use the {primary or 'no'} tagger. A picture passes the "
            "content rule only if its top rating is <b>general</b>. Hover a picture for its "
            "prompt and tags; click to open it full size.</p>"
        ),
    ]

    sections.append(
        "<h2>Age ladder</h2><p class='note'>Same sheet, kit, pose and seed; only "
        "the band changes. The bands should look visibly different.</p>"
    )
    for s in LADDER:
        for m in models:
            pics = "".join(
                f"<td><div class='band'>Band {b}</div>{cell(m, f'{s.id}-b{b}')}</td>" for b in BANDS
            )
            sections.append(
                f"<h3>{s.id} · {html.escape(label(m))}</h3>"
                f"<div class='scroll'><table class='grid'><tr>{pics}</tr></table></div>"
            )

    base_cols = [m for m in models if "~" not in m]
    if len(base_cols) < len(models):
        heads = "".join(f"<th>{html.escape(label(m))}</th>" for m in models)
        sections.append(
            "<h2>Band 1 wording</h2><p class='note'>Band 1 pictures under each wording "
            f"variant. Variants: {html.escape(json.dumps(BAND_VARIANTS))}. Same seeds, so "
            "only the wording differs.</p>"
        )
        trs = []
        for s in (x for x in SHEETS if x.band == 1):
            tds = "".join(
                f"<td><div class='pair'>{cell(m, s.id + 'a')}{cell(m, s.id + 'b')}</div></td>"
                for m in models
            )
            trs.append(f"<tr><td class='desc'>{html.escape(s.id)}</td>{tds}</tr>")
        sections.append(
            f"<div class='scroll'><table class='grid'><tr><th></th>{heads}</tr>"
            f"{''.join(trs)}</table></div>"
        )

    base_heads = "".join(f"<th>{html.escape(label(m))}</th>" for m in base_cols)
    sections.append(
        "<h2>Character sheets</h2><p class='note'>Each cell shows variant a and "
        "variant b: different kit, pose and seed. Is it the same character?</p>"
    )
    for band, wording in BANDS.items():
        sections.append(f"<h3>Band {band}: {html.escape(wording)}</h3>")
        trs = []
        for s in (x for x in SHEETS if x.band == band):
            desc = html.escape(
                f"{s.id}: {s.hair} {s.style}, {s.fringe} fringe, {s.eyes} {s.eye_shape} eyes, "
                f"{s.skin} skin, {s.accessory}, {s.mark}, {s.expression}. "
                f"a: {s.kits[0]} / {s.poses[0]}; b: {s.kits[1]} / {s.poses[1]}"
            )
            tds = "".join(
                f"<td><div class='pair'>{cell(m, s.id + 'a')}{cell(m, s.id + 'b')}</div></td>"
                for m in base_cols
            )
            trs.append(f"<tr><td class='desc'>{desc}</td>{tds}</tr>")
        sections.append(
            f"<div class='scroll'><table class='grid'><tr><th></th>{base_heads}"
            f"</tr>{''.join(trs)}</table></div>"
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model bake-off</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4; --card:#fff;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2c323c; --card:#1c2027;
          --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px 24px 48px; background:var(--bg); color:var(--fg);
        font:14px/1.45 system-ui, sans-serif; }}
h1 {{ margin:0 0 4px; font-size:22px; }} h2 {{ margin:32px 0 8px; font-size:18px; }}
h3 {{ margin:20px 0 6px; font-size:15px; color:var(--muted); font-weight:600; }}
.note {{ color:var(--muted); max-width:80ch; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }} th, td {{ padding:6px 10px; border-bottom:1px solid
  var(--line); text-align:left; vertical-align:top; white-space:nowrap; }}
.grid td {{ border:none; padding:4px; }}
.desc {{ white-space:normal; width:220px; min-width:220px; color:var(--muted); font-size:12px; }}
.pair {{ display:flex; gap:4px; }}
.pic {{ margin:0; width:208px; background:var(--card); border:1px solid var(--line);
        border-radius:6px; overflow:hidden; }}
.pic img {{ display:block; width:208px; height:208px; }}
.pic.missing {{ height:208px; display:grid; place-items:center; color:var(--muted); }}
figcaption {{ display:flex; flex-wrap:wrap; gap:3px; padding:4px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; color:var(--muted);
      background:var(--line); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
.band {{ font-size:12px; color:var(--muted); margin-bottom:2px; }}
</style></head><body>
<h1>Model bake-off · {html.escape(run_dir.name)}</h1>
<p class="note">{SIZE}×{SIZE}, {STEPS} steps, CFG {CFG}, Euler a. "child" is the tagger's
probability for that tag, a rough age signal. Local file only; these pictures do not leave
this PC.</p>
{"".join(sections)}
</body></html>"""
    path = run_dir / "index.html"
    path.write_text(page)
    return path


def ages_report(run_dir: Path) -> Path:
    """One row per age wording, youngest first; one column per character."""
    tags = json.loads((run_dir / "tags_eva02.json").read_text())["images"]
    rows = []
    for key, (words, negative, origin) in AGE_WORDINGS.items():
        cells, child, mature = [], [], []
        for s in AGE_SHEETS:
            cid = f"{s.id}-{key}"
            info = tags.get(f"animagine/{cid}", {})
            probes = info.get("probes", {})
            child.append(probes.get("child", 0.0))
            mature.append(probes.get("mature female", 0.0))
            rating = info.get("rating", "?")
            cls = "ok" if rating == "general" else "bad"
            cells.append(
                f'<td><figure class="pic"><a href="animagine/{cid}.png" target="_blank">'
                f'<img src="animagine/{cid}.png" loading="lazy" alt="{cid}"></a><figcaption>'
                f'<span class="b {cls}">{html.escape(rating)}</span>'
                f'<span class="b">child {probes.get("child", 0):.2f}</span></figcaption>'
                "</figure></td>"
            )
        neg = f"<br>extra negative: {html.escape(negative)}" if negative else ""
        rows.append(
            f"<tr><td class='desc'><b class='key'>{key}</b><br>{html.escape(words)}"
            f"<br><i>{html.escape(origin)}</i>{neg}<br>mean child "
            f"{statistics.mean(child):.2f}, mature female {statistics.mean(mature):.2f}</td>"
            f"{''.join(cells)}</tr>"
        )
    heads = "".join(f"<th>{s.id}: {s.hair} {s.style}, {s.skin} skin</th>" for s in AGE_SHEETS)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Age wordings</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4; --card:#fff;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2c323c; --card:#1c2027;
          --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px 24px 48px; background:var(--bg); color:var(--fg);
        font:14px/1.45 system-ui, sans-serif; }}
h1 {{ margin:0 0 4px; font-size:22px; }}
.note {{ color:var(--muted); max-width:90ch; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }}
th {{ text-align:left; font-weight:600; padding:4px; color:var(--muted); font-size:12px; }}
td {{ padding:4px; vertical-align:top; border-bottom:1px solid var(--line); }}
.desc {{ width:230px; min-width:230px; font-size:12px; color:var(--muted); }}
.key {{ font-size:18px; color:var(--fg); }}
.pic {{ margin:0; width:240px; background:var(--card); border:1px solid var(--line);
        border-radius:6px; overflow:hidden; }}
.pic img {{ display:block; width:240px; height:240px; }}
figcaption {{ display:flex; gap:3px; padding:4px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; color:var(--muted);
      background:var(--line); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
</style></head><body>
<h1>Age wordings · {html.escape(run_dir.name)}</h1>
<p class="note">Ten wordings, youngest first. Each column is one character with the same
seed, kit and pose in every row, so only the age wording changes. Pick five for bands
1&ndash;5 (12&ndash;14, 15&ndash;18, 19&ndash;21, 22&ndash;25, 26&ndash;30). Animagine XL 4.0
Opt, {SIZE}&times;{SIZE}, {STEPS} steps, CFG {CFG}. "child" is the tagger's probability for
that tag, a rough age signal. Local file only.</p>
<div class="scroll"><table><tr><th></th>{heads}</tr>{"".join(rows)}</table></div>
</body></html>"""
    path = run_dir / "ages.html"
    path.write_text(page)
    return path


MY_CLUB_BANNED = frozenset({"freckles"})


def myclub_preview(run_dir: Path) -> Path:
    """r5: sheets with tan or dark skin or freckles, drawn as my-club players.

    Level 1 for all of them, level 2 for the tan and dark ones; round 2's "a" picture of
    each, drawn without the adjustments, is the original.
    """
    import shutil

    sheets = [s for s in SHEETS if s.skin in ("tan", "dark") or s.mark in MY_CLUB_BANNED]
    pairs = {c.id: c for c in make_cases()}
    original = run_dir / "animagine~original"
    original.mkdir(parents=True, exist_ok=True)
    for s in sheets:
        dst = original / f"{s.id}a.png"
        if not dst.exists():
            shutil.copy2(BAKEOFF_DIR / "r2" / "animagine" / f"{s.id}a.png", dst)
    level1 = [pairs[f"{s.id}a"] for s in sheets]
    level2 = [pairs[f"{s.id}a"] for s in sheets if s.skin in ("tan", "dark")]
    render(run_dir, ["animagine"], level1, column="club1", soften=1, banned=MY_CLUB_BANNED)
    render(run_dir, ["animagine"], level2, column="club2", soften=2, banned=MY_CLUB_BANNED)
    tag(run_dir, ["eva02"])
    tags = json.loads((run_dir / "tags_eva02.json").read_text())["images"]

    def cell(col: str, s: Sheet) -> str:
        key = f"{col}/{s.id}a"
        if key not in tags:
            return "<td></td>"
        info = tags[key]
        pr = info.get("probes", {})
        rating = info["rating"]
        badges = [
            f'<span class="b {"ok" if rating == "general" else "bad"}">{rating}</span>',
            f'<span class="b">dark skin {pr.get("dark skin", 0):.2f}</span>',
            f'<span class="b">tan {pr.get("tan", 0):.2f}</span>',
            (
                f'<span class="b {"bad" if pr.get("freckles", 0) >= 0.35 else ""}">'
                f"freckles {pr.get('freckles', 0):.2f}</span>"
            ),
        ]
        return (
            f'<td><figure class="pic"><a href="{key}.png" target="_blank"><img src="{key}.png" '
            f'loading="lazy" alt=""></a><figcaption>{"".join(badges)}</figcaption></figure></td>'
        )

    cols = ["animagine~original", "animagine~club1", "animagine~club2"]
    heads = ["original (round 2)", "my club, level 1", "my club, level 2"]
    rows = []
    for s in sheets:
        desc = html.escape(
            f"{s.id}: {s.skin} skin" + (", freckles" if s.mark == "freckles" else "")
        )
        lv1 = skin_words(s.skin, 1) or "(no skin term: light)"
        lv2 = skin_words(s.skin, 2) or "(no skin term: light)"
        words = html.escape(f"original: {skin_words(s.skin) or '(none)'}; 1: {lv1}; 2: {lv2}")
        rows.append(
            f"<tr><td class='desc'><b>{desc}</b><br>{words}</td>"
            + "".join(cell(c, s) for c in cols)
            + "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>My-club skin preview</title>
<style>
:root {{ --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4; --card:#fff;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2c323c; --card:#1c2027;
          --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; }} }}
body {{ margin:0; padding:16px 24px 48px; background:var(--bg); color:var(--fg);
        font:14px/1.45 system-ui, sans-serif; }}
h1 {{ margin:0 0 4px; font-size:22px; }}
.note {{ color:var(--muted); max-width:90ch; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }}
th {{ text-align:left; font-weight:600; padding:4px; color:var(--muted); font-size:12px; }}
td {{ padding:4px; vertical-align:top; border-bottom:1px solid var(--line); }}
.desc {{ width:220px; min-width:220px; font-size:12px; color:var(--muted); }}
.pic {{ margin:0; width:260px; background:var(--card); border:1px solid var(--line);
        border-radius:6px; overflow:hidden; }}
.pic img {{ display:block; width:260px; height:260px; }}
figcaption {{ display:flex; flex-wrap:wrap; gap:3px; padding:4px; }}
.b {{ font-size:11px; padding:1px 5px; border-radius:3px; color:var(--muted);
      background:var(--line); }}
.b.ok {{ color:var(--ok); background:var(--okbg); }}
.b.bad {{ color:var(--bad); background:var(--badbg); }}
</style></head><body>
<h1>My-club preview · {html.escape(run_dir.name)}</h1>
<p class="note">Players at my club: freckles left out of the prompt and added to its negative;
tan and dark skin drawn one (level 1) or two (level 2) steps lighter on the scale pale, light,
slightly tanned, tan, dark. Same seed, kit, pose and age wording as the original. The
numbers are the tagger's probabilities for those tags. Local file only.</p>
<div class="scroll"><table><tr><th></th>{"".join(f"<th>{h}</th>" for h in heads)}</tr>
{"".join(rows)}</table></div>
</body></html>"""
    path = run_dir / "myclub.html"
    path.write_text(page)
    return path


# ------------------------------------------------------------------------------ CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", help="run name under data/bakeoff/ (default r1; ages: r3)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts", help="print every prompt for one model; no GPU")
    p.add_argument("--model", choices=MODELS, default="animagine")
    p.add_argument("--variant", choices=BAND_VARIANTS)
    for name in ("render", "all"):
        p = sub.add_parser(name)
        p.add_argument("--models", default=",".join(DEFAULT_MODELS))
        p.add_argument("--cases", default="", help="comma-separated case IDs; default all")
        p.add_argument("--variant", choices=BAND_VARIANTS, help="band wording variant")
        p.add_argument("--band", type=int, help="only cases in this band")
    sub.add_parser("ages", help="round 3: render the age wordings, tag, write ages.html")
    sub.add_parser("myclub", help="r5: preview the my-club skin and banned-trait rules")
    p = sub.add_parser("tag")
    p.add_argument("--taggers", default=",".join(TAGGERS))
    sub.add_parser("report")
    args = parser.parse_args()

    default_run = {"ages": "r3", "myclub": "r5"}.get(args.command, "r1")
    run_dir = BAKEOFF_DIR / (args.run or default_run)
    if args.command == "prompts":
        for case in make_cases():
            positive, negative = build_prompt(case, MODELS[args.model].family, args.variant)
            print(f"{case.id} seed={case.seed}\n  + {positive}\n  - {negative}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "myclub":
        print("myclub:", myclub_preview(run_dir))
        return
    if args.command == "ages":
        render(run_dir, ["animagine"], make_age_cases())
        tag(run_dir, ["eva02"])
        print("ages:", ages_report(run_dir))
        return
    if args.command in ("render", "all"):
        models = [m for m in args.models.split(",") if m]
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            sys.exit(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
        wanted = {c for c in args.cases.split(",") if c}
        cases = [
            c
            for c in make_cases()
            if (not wanted or c.id in wanted) and (not args.band or c.band == args.band)
        ]
        render(run_dir, models, cases, args.variant)
    if args.command in ("tag", "all"):
        keys = args.taggers.split(",") if args.command == "tag" else list(TAGGERS)
        tag(run_dir, keys)
    if args.command in ("report", "all"):
        print("report:", report(run_dir))


if __name__ == "__main__":
    main()
