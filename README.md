# FM27 Face Generator

**Every newgen deserves a face.**

This is a local tool. It reads your Football Manager 27 save, picks the players you will
actually look at, draws an anime-style portrait of each one on your own GPU, and puts the
pictures into FM27. It needs no cloud service, and you don't copy any files by hand.

<p align="center">
  <img src="docs/images/hero.jpg" alt="Four generated anime portraits of made-up players in football kits" width="760">
  <br>
  <sub>Four made-up test players, drawn with the same setup the tool uses today.</sub>
</p>

## The problem

Football Manager invents new players every season. Fans call them *newgens*. They arrive
with no photo. A few seasons in, your league is full of faceless
wonderkids, and so is your own academy.

You *can* give them faces by hand: find a picture, name it after the player's ID, add a
line to `config.xml` and reload the skin. That's fine for one player and a chore for
hundreds. It also never ends, because the game keeps making more.

So this tool does the chore:

1. **It reads the save.** It copies your newest save and reads every player, club, CA
   and PA from the copy.
2. **It picks who matters.** You don't look at 20,000 players. You look at your squad,
   your league rivals and the stars. The tool sorts those players into groups A, B and C
   and skips everyone else.
3. **It draws on your GPU.** Each player gets a portrait in their club's kit, made on
   your own graphics card.
4. **It publishes.** It writes the pictures and a `config.xml` into FM27's graphics
   folder. Press *Reload Skin* in FM27 and the faces appear.

## Highlights

- **A queue instead of brute force.** The GPU is shared with the game, so the tool spends
  it where you'll look. Group A (your club and the superstars) gets the best of 3 tries
  plus a detail pass. Group C (decent players in your league) gets one try. Everyone else
  gets nothing.
- **Faces that grow up.** Each player gets a saved *character sheet* (hair, eyes, skin,
  accessory and so on). When a tracked player changes club or gets older, the tool
  re-draws them from the same sheet. It also shows the model a crop of the old face
  through **IP-Adapter**, a small add-on that lets the model look at a reference picture.
  The result is the same girl in a new kit, a bit older.
- **It fits next to the game.** It uses INT8 weights, keeps its helper models in system
  RAM between uses and turns on fp16 accumulation. Peak VRAM went from
  **10.95 to 6.88 GiB (−37%)** and one render from **5.1 to 4.4 s (−14%)**. You can't
  see any difference in the pictures.
- **Nothing leaves your PC.** Models load offline, the web UI answers only on
  `127.0.0.1`, and saves are only ever copied.

## How it fits together

```mermaid
flowchart LR
    save[("FM27 save")] -->|"copy"| snap["Snapshot copy"]
    snap -->|"fmsave reader"| players["Players<br/>(~22k at men's clubs)"]
    players --> who{"Who will you<br/>actually look at?"}
    who -->|"my club, superstars"| A["Group A<br/>best of 3 + detail pass"]
    who -->|"academy grads, league stars,<br/>top prospects"| B["Group B<br/>1 try"]
    who -->|"good players in my league"| C["Group C<br/>1 try"]
    who -.->|"everyone else"| skip["No picture"]
    A --> queue[["Job queue"]]
    B --> queue
    C --> queue
    ui["Web UI<br/>hand picks"] --> queue
    queue --> gpu["GPU worker<br/>Animagine XL 4.0, INT8"]
    gpu --> check{"Checks: content,<br/>hair, eyes, one person"}
    check -->|"pass"| pub["graphics/facegen/<br/>+ config.xml"]
    check -.->|"fails 3 rounds"| review["Review page"]
    pub --> fm["FM27: Reload Skin"]
```

The parts talk to each other only through one SQLite database:

- A **poller** watches the saves folder, so every save or autosave starts a sync.
- A **worker** owns the GPU. It runs in a process of its own, so a graphics driver error
  costs one restart rather than the session, and quitting or pausing hands every byte of
  video memory back.
- A small **FastAPI page** lets you change thresholds and ask for pictures by hand.

By default the worker waits while FM27 is running and frees the GPU when you start the
game. Its log is in `data/logs/facegen.log`.

## The constraint

**One RTX 4080 (16 GB), shared with other programs.** Sixteen gigabytes sounds like
plenty, but Windows keeps about 3 GB of it, and FM27 and a browser need room too. The goal
is that a full group A job peaks **under 8 GiB**. An opt-in GPU test fails if the peak
reaches 8.0 GiB.

## Design choices

### 1. Fitting in 8 GB: INT8, offloading and fp16 accumulation

First, a quick map. The image model (from the Stable Diffusion XL family) starts from
noise and cleans it up in 28 steps. The part that does the cleaning is the **UNet**, and it
runs at every step. Around it sit some **helpers**: two text encoders that read the prompt,
a tagger that checks the finished picture, and an image encoder for reference faces. Each
helper works for about a second per job and sits idle the rest of the time.

The setup uses three tricks, each simple on its own.

**Trick 1: store the weights as INT8 and unpack one layer at a time.** The UNet has 700
large matrix layers in its transformer blocks. Each one is stored as INT8 (1 byte per
number instead of 2), with one scale factor per row, so rows with small and large values
both keep their detail. Right before a layer runs, only that layer is unpacked back to
fp16. The layer does its math and the fp16 copy is thrown away. It works like
vacuum-packed clothes: you unpack one outfit when you need it, and the suitcase stays
small.

This takes the UNet weights from **4.78 to 2.75 GiB**. The math itself is still ordinary
fp16, so there are no special kernels and no warm-up.

```mermaid
flowchart LR
    w["Layer weights in INT8<br/>+ one scale per row<br/>(stay on the GPU)"] -->|"unpack only this layer"| t["fp16 copy<br/>(freed right after)"]
    t --> mm(("matmul<br/>fp16 accumulation"))
    x["Input from the<br/>previous layer"] --> mm
    mm --> y["Output to the<br/>next layer"]
```

**Trick 2: keep the helpers in system RAM.** The text encoders (1.5 GiB), the tagger and
the face encoder wait in normal memory and move onto the GPU only for the moment they're
needed. Each trip takes time: 0.7 s for the text encoders, 0.2 s for the tagger. So the
text encoders don't make a trip per job. The worker encodes the prompts of the next 32
jobs in one trip and keeps the results (a few MB) in RAM. The model gets exactly the same
numbers as before, so the pictures don't change. What's left is the tagger's two trips, about 0.45 s per job.

**Trick 3: fp16 accumulation.** A matrix multiply adds up a lot of products. By default,
PyTorch adds them up in fp32, and on GeForce cards that runs the fp16 tensor cores at half
speed. One flag (`allow_fp16_accumulation`) makes the GPU add them up in fp16 instead. One
UNet step went from **184 to 145 ms**. The model's output changed by 0.04%, which is less
than switching the whole model to bf16 changes it. The flag is on only while the image
pipeline runs; the text encoders and the tagger keep fp32.

**The result, one step at a time:**

| Setup | One render | Group A player¹ | Peak VRAM² |
|---|---|---|---|
| Plain fp16 | 5.1 s | 19.9 s | 10.95 GiB |
| + fp16 accumulation | 4.1 s | 16.4 s | 10.95 GiB |
| + INT8 weights | 4.4 s | 17.2 s | 9.00 GiB |
| **+ helpers in system RAM (what runs today)** | **4.4 s** | **≈ 17.7 s** | **6.88 GiB** |

¹ Three candidate renders, a detail pass and the checks. The last row adds the ~0.45 s
per job that moving the helpers still costs (~1 s before prompts were encoded 32 jobs at
a time).<br>
² The memory PyTorch reserves at the peak of a whole group A job. fp16 accumulation
doesn't change memory, so the first two rows share one measurement. A re-draw with a
reference face goes from 11.82 to 7.70 GiB (−35%).

To be honest about it, **the big win is memory (−37%)**. Speed is a bonus. Renders are
14% faster, but moving the helpers takes back about 0.45 s per job, so a whole group A
player is about 11% faster. INT8 here is about memory, not speed: unpacking costs a little
time, and fp16 accumulation pays for it.

### 2. Why INT8, and not FP8, torch.compile or SageAttention

I tried every reasonable option on the same 10 made-up players, with the same seeds and
the same prompts, changing only the UNet's arithmetic. "Closeness" measures how near each
picture is to the plain fp16 picture (PSNR; higher means closer). bf16 serves as the
yardstick, because it's a precision change nobody would call harmful.

| Option | Speed-up (group A) | Closeness to fp16 | UNet weights | Start-up wait | Checks passed |
|---|---|---|---|---|---|
| Plain fp16 | 1.00× | (baseline) | 4.78 GiB | none | 20/20 |
| bf16 (the yardstick) | not timed | 24.9 dB | 4.78 GiB | none | 20/20 |
| fp16 accumulation | 1.24× | 30.9 dB | 4.78 GiB | none | 20/20 |
| + torch.compile | 1.26× | 31.7 dB | 4.78 GiB | 112 s | 20/20 |
| + SageAttention | 1.30× | 22.5 dB | 4.78 GiB | none | 20/20 |
| FP8 + torch.compile | 1.36× | 20.8 dB | 2.75 GiB | 192 s | 18/20 |
| INT8 weights and activations + torch.compile | 1.38× | 23.4 dB | 2.75 GiB | 184 s | 20/20 |
| **INT8 weights + fp16 accumulation (chosen)** | **1.16×** | **25.9 dB** | **2.75 GiB** | **none** | **20/20** |

The chosen setup is not the fastest. It won for four reasons.

**1. You can't see the difference.** Its pictures are closer to fp16 than bf16's are.
Here are two of the test players side by side:

| Plain fp16 | INT8 + fp16 accumulation |
|---|---|
| <img src="docs/images/fp16-910008.jpg" width="200" alt="Test player drawn in plain fp16"> | <img src="docs/images/int8-910008.jpg" width="200" alt="The same test player drawn with INT8 weights"> |
| <img src="docs/images/fp16-910010.jpg" width="200" alt="Second test player drawn in plain fp16"> | <img src="docs/images/int8-910010.jpg" width="200" alt="The second test player drawn with INT8 weights"> |

**2. The theory agrees.** INT8 with one scale per row keeps about 7 bits of precision,
while FP8 (E4M3) has 3 mantissa bits. On the 4080, INT8 math is at least as fast as FP8
(on paper they have the same peak), so FP8 would add error without adding speed. The test
showed exactly that: FP8 was as fast as INT8 with twice the error, and it lost one
player's hair colour.

**3. There's no warm-up.** The fastest rows need `torch.compile`, which takes about
3 minutes every time the worker starts. The worker restarts often, because it releases
the GPU whenever you open FM27. At about 3 s saved per player, it would have to draw more
than 60 group A players in one sitting before the wait paid off.

**4. There are no extra dependencies.** SageAttention is a little faster, but it adds a
package and moves the pictures further from fp16 than bf16 does.

### 3. Picking the model: 8 contestants, 1 judge

Before building the real pipeline, I held a bake-off. Every model drew the same 50 test
characters. A tagger model (WD EVA02-Large) then checked each picture for content rating,
hair colour, eye colour, kit colours and "exactly one person". The table shows each
model's version of the same character: black twin tails, purple eyes, a white and navy
kit, arms crossed.

| Model | Sample | All checks | Time / picture³ | Why it lost |
|---|---|---|---|---|
| **Animagine XL 4.0 Opt** ✅ | <img src="docs/images/models/animagine.jpg" width="120" alt="Animagine sample"> | 82–86% | 5.3 s | **The winner.** Tight upper-body framing, and none of its 476 bake-off pictures was flagged for content |
| Illustrious XL 2.0 | <img src="docs/images/models/illustrious.jpg" width="120" alt="Illustrious sample"> | 42% | 5.3 s | Loves to draw a crowd ("multiple views" in 16 of 50); 6 of 50 failed the content check |
| NoobAI-XL 1.1 | <img src="docs/images/models/noobai.jpg" width="120" alt="NoobAI sample"> | 64% | 5.7 s | Extra figures in 7 of 50; behind Animagine on every check |
| Anima Aesthetic v1.1 | <img src="docs/images/models/anima.jpg" width="120" alt="Anima sample"> | 78% | 18.1 s | 3× slower, and clearly worse to the judge's eye |
| Z-Image (6B, FP8) | <img src="docs/images/models/zimage.jpg" width="120" alt="Z-Image sample"> | 70% | 44.3 s | 8× slower; frames from the thighs up, so faces are smaller; once drew a bare midriff that the tagger still rated "general" |
| Z-Image-Turbo (FP8) | <img src="docs/images/models/zimage-turbo.jpg" width="120" alt="Z-Image-Turbo sample"> | 76% | 5.9 s | Has no negative prompt; faces look alike across players |
| WAI-illustrious v17 | <img src="docs/images/models/wai.jpg" width="120" alt="WAI-illustrious sample"> | 84% | 6.6 s | Close on the numbers, but frames from the thighs up twice as often, and failed one content check |
| Anima-2.9B (preview) | <img src="docs/images/models/anima-2.9b.jpg" width="120" alt="Anima-2.9B sample"> | 86% | 24.5 s | 4× slower, needs ComfyUI, and drew two players as boys |

³ On the 4080 at 1024×1024, each model at its own recommended settings, before the
speed-ups above.

The numbers narrowed the field, and a **human annotator (n = 1)** made the final call: me,
looking at the pictures. Animagine won on style. No other model matched its clean content
record at its speed.

### 4. Spend the GPU where you look: groups A, B and C

A save holds tens of thousands of players (21,834 at men's clubs in my test career), but
you mostly look at three kinds: **your own squad**, **the teams you play every week**, and
**the stars** who keep turning up in the news and in scouting. Those players get pictures,
the ones you see most often get the best pictures, and everyone else is skipped.

| Group | Who (default thresholds) | Tries | Detail pass | Published size |
|---|---|---|---|---|
| **A** | Your first team and your staff; any newgen with CA > 155 or PA > 170 | best of 3 | yes | 768×1024 |
| **B** | Your other teams (reserves, under-23s, under-18s); your academy graduates after they leave; newgens with CA > 130 or PA > 150 who are in your league or a continental cup; any newgen with CA > 140 or PA > 155; journalists from your club's country and your press officers | 1 | no | 576×768 |
| **C** | Newgens in your league with CA > 115 or PA > 140; the other clubs' managers in your league who have no photo | 1 | no | 384×512 |
| none | Everyone else | no picture, unless you pick them by hand in the web UI | | |

- **Real players without a photo count as newgens.** FM27 ships photos for some leagues
  only; everyone else shows a silhouette. At each sync facegen reads the game's list of the
  photos it ships and the `config.xml` of any facepack you installed, and a real player with
  neither is picked by the same thresholds as a newgen (the academy rule aside). Settings
  turns this off.
- **Queue order:** your hand-picked requests come first, then group A (newest arrivals at
  your club first, so youth intake day goes to the front), then B, then C, each sorted by
  PA.
- **Cost:** a group A player takes about 18 s. A B or C player takes a single render
  (~4.4 s) plus the check.
- **Why B and C still render at full size:** image models draw *worse*, not just smaller,
  below their native size. So every group renders at full size. The groups differ in how
  many tries they get, whether they get the detail pass, and how big the published file
  is.
- **Smaller graphics cards:** Settings → "Auto-configure for my GPU" asks the card and
  proposes picture settings for it: below 10 GB, best of 2 with no detail pass and the
  picture decoded in tiles; 10 and 11 GB, a smaller detail pass (1152×1472 instead of
  1344×1728). It lists each change first, saves them as one change that one Undo reverts,
  and never touches who gets a picture. On the RTX 4080 a group A job peaks at about
  6.9 GiB of PyTorch memory with the defaults, 5.8 GiB with the 10–11 GB preset and 4.5 GiB
  with the under-10 GB one (7.7, 6.7 and 5.0 GiB for a re-draw with a reference picture);
  they are not yet tried on smaller cards.
- **A new photo shoot each time.** The character sheet keeps a player's face, hair and
  eyes; each picture draws the rest. The kit is worn with short, long or layered sleeves,
  a collar, or as club training wear: a track jacket, open over the shirt or zipped to a
  high collar. The in-game date sets the season, so winter brings more long sleeves and
  jackets. Each player has a favourite style, and seniors now and then wear an armband.
  Poses combine arms and an angle (front, a turn, a head tilt, contrapposto), and the
  backdrop is a plain colour, a gradient or two tones, never the kit's or hair's colour.
- **Your staff too.** Coaches, physios, scouts, the board and your own manager get group A
  pictures, dressed for the job: suits and overcoats for the board, a suit or a bench coat
  for the manager, training wear and a whistle for coaches, lab coats for doctors, with the
  club's colours on a tie, scarf or jacket. Their age wording has its own table of
  very wide, seven-year bands, so most staff fall in the middle of the range.
- **The people around the club.** Journalists and press officers get one picture each, in
  work clothes with a press pass or a lanyard. The other clubs' managers in your league
  get one too if FM27 has no photo of them. Pin someone you see often to group A and they
  are kept up to date like your own staff.
- **Thresholds are live.** Each setting is saved as you change it in the web UI, the page
  says who moved and what was queued, and Undo puts it back, several steps at once if you
  like. Newly eligible players join the queue, and existing pictures are never deleted.

### 5. Faces that grow up

A player who changes club or gets older should still look like themselves. Three things
make that work:

1. **A character sheet for each player:** hair colour and style, fringe, eye colour and
   shape, skin, accessory, mark and default expression. All randomness is seeded from the
   career, the player's ID and a version number, so the same inputs always give the same
   sheet and the same prompt. A fun extra: personality nudges the expression. High
   *Pressure* leans confident, and high *Professionalism* leans serious.
2. **Age wording:** the player's real age picks one of ten age descriptions for the
   prompt. When a tracked player's age moves into the next band, they're re-drawn.
3. **The old face as a reference:** when a player at your club is re-drawn, the tool crops
   the face from their current picture and gives it to **IP-Adapter plus-face** at a
   gentle strength of 0.1. I tested strengths from 0.1 to 0.8. From 0.4 up, kit accuracy
   dropped by 5–10 points, and an adapter that looked at the whole picture carried the old
   club's shirt into 30% of the new pictures. 0.1–0.2 gave the closest faces with no loss
   in kit accuracy. The reference stays gentle because any reference holds back the aging
   a little.

| Age 22 | Age 27 | Age 31 |
|---|---|---|
| <img src="docs/images/age-22.jpg" width="170" alt="Test character at age 22"> | <img src="docs/images/age-27.jpg" width="170" alt="The same character at age 27"> | <img src="docs/images/age-31.jpg" width="170" alt="The same character at age 31"> |

<sub>One test character drawn at three ages, each from the same sheet with her original
picture as the reference.</sub>

## Numbers

Everything was measured on the RTX 4080 under WSL2, with Windows using the GPU at the same
time. Because the shared GPU's speed drifted by up to 40% within minutes, the speed tests
alternated the baseline and the variant in turn and took the median.

| What | Result |
|---|---|
| Loading the model from disk (incl. INT8 conversion) | ~6 s |
| One render, 896×1152, 28 steps | 4.4 s |
| Detail pass, 1344×1728 | 4.0 s |
| Content and quality check | 0.03 s per picture |
| One group A player (3 renders + detail pass + checks) | ~18 s |
| Peak VRAM for a group A job | 6.9 GiB (7.7 GiB with a reference face) |
| GPU memory held between jobs | ~3.8 GiB; all freed after 5 idle minutes or when FM27 starts |
| First real squad run (Colchester, 52 players) | 154 of 156 candidates passed; all 52 players passed in the first round (19 min, before the speed-ups) |
| Reading and sorting a whole save | ~10 s |
| Web UI players list, 33,000-player database | 5–40 ms per page |

**Tests**

Tests are organized by tier and domain: `tests/unit/`, `tests/integration/`,
`tests/parity/` and `tests/opt_in/`. See [the test suite guide](tests/AGENTS.md) for
shared doubles, targeted commands and isolation rules. The default suite needs no GPU
or save file; native/Python and CPU encoding parity run when their libraries are installed.

Run the prepared environment offline, without synchronizing dependencies:

```bash
env -u FACEGEN_TEST_GPU -u FACEGEN_TEST_SAVE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync pytest -ra
```

The real GPU tests need cached models and explicit opt-in. They check rendering, reference
faces, encoding equivalence and the 8 GiB peak VRAM limit:

```bash
FACEGEN_TEST_GPU=1 uv run --no-sync pytest tests/opt_in/test_gpu.py
```

## Run it yourself

**The quick way, on Windows:** download this repository (*Code → Download ZIP*), unpack it
anywhere you can write to, and double-click **`Start.bat`**. It installs
[`uv`](https://docs.astral.sh/uv/) if you don't have it, sets up Python and the libraries
(about 3 GB the first time, mostly PyTorch with CUDA), asks once before downloading the
models (11.6 GB), and opens facegen in a window of its own. A tray icon offers Pause, Sync
and Quit; closing the window keeps facegen drawing in the background. Double-click
`Start.bat` again to bring the window back. `Start.bat --shortcut` also puts a shortcut on
the desktop and in the Start Menu, and the tray menu can start facegen with Windows. The
window needs Microsoft Edge WebView2, which Windows 11 has built in; without it the dashboard
opens in your browser. Start.bat keeps a log in `data\logs\start.log`.

The steps below are the manual way, and the way in WSL.

**You need:**

- A Windows PC with FM27. facegen runs on Windows itself, or in WSL2 (Ubuntu); the same
  commands work in both.
- An NVIDIA GPU with about 8 GB of free VRAM. It was tested on an RTX 4080 16 GB.
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/). uv fetches Python if you have none.
- Optionally, a small Rust extension that republishes a pack about 9× faster. In WSL with
  Rust from [rustup](https://rustup.rs/), `uv sync` builds it; without Rust there, install
  with `uv sync --no-group native`. On Windows, or anywhere without Rust,
  `uv run facegen native --install` installs a prebuilt build of it from this repo's
  releases, checked against the SHA-256 listed in `native/wheels.sha256`; after that, update
  with `uv sync --inexact`, since a plain `uv sync` removes it. Without it everything works
  the same, only slower to publish. `uv run facegen native` says whether it is in use.
- About 11 GB of disk for the models, plus a few GB for PyTorch with CUDA. On Windows uv
  takes PyTorch from PyTorch's own CUDA index, since PyPI's Windows build has no CUDA.

**1. Install.** From the cloned repo:

```bash
uv sync
```

**2. FM27's folders.** facegen finds FM27's user folder (in your Documents folder, OneDrive
included) and its Steam install (in any Steam library) by itself; `uv run facegen status`
shows what it found. If it cannot find the user folder, the web UI asks for it when you start
it (step 5) and saves it in `config/local.toml`. You can also write that file yourself. It is
not in git, so your Windows user name stays out of the repo. Write a Windows path in single
quotes; in WSL it works too.

```toml
[paths]
fm_user_dir = 'C:\Users\<you>\Documents\Sports Interactive\Football Manager 27'
```

**3. Download the models once** (~11.6 GB). After this, everything runs offline. Either let
facegen do it (it shows the size and asks once; the dashboard offers the same button):

```bash
uv run facegen models --download
```

or fetch them yourself:

```bash
uv run hf download cagliostrolab/animagine-xl-4.0 --include "model_index.json" --include "scheduler/*" --include "text_encoder*/*" --include "tokenizer*/*" --include "unet/*" --include "vae/*"
```

```bash
uv run hf download h94/IP-Adapter sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors models/image_encoder/config.json models/image_encoder/model.safetensors
```

```bash
uv run hf download SmilingWolf/wd-eva02-large-tagger-v3 config.json model.safetensors selected_tags.csv
```

**4. Look before you draw.** This command copies your newest save, reads it and prints
who would get a picture. It stores nothing.

```bash
uv run facegen sync --dry-run
```

**5. Start it:**

```bash
uv run facegen serve
```

Open <http://127.0.0.1:8765/>. The dashboard shows the queue and the worker. The worker
draws while FM27 is closed; press **Generate now** to draw anyway. Then, in FM27, choose
**Settings → Reload Skin**.

Other handy commands:

| Command | What it does |
|---|---|
| `uv run facegen run --group a --my-club --dry-run` | Prints each job's character sheet and prompt without drawing |
| `uv run facegen run --group a --my-club` | Draws your squad from the command line, without the web UI (close FM27 first, or add `--now`) |
| `uv run facegen status` | Shows careers, group counts and the queue |
| `uv run facegen serve --no-worker` | Runs only the web UI, with no GPU work |
| `uv run facegen app` | On Windows: the dashboard in a window, with a tray icon (what Start.bat opens) |
| `uv run facegen tray` | On Windows: only the tray icon; Open Dashboard opens the browser |

Club kit colours are in `config/kits.toml`. A club that isn't listed takes the colours
estimated from FM27's own image of its home shirt: facegen ships these for the 832 clubs
FM27 has an image for (`src/facegen/data/kit_catalog.json`, colour names only), so there is
nothing to set up. After a game update you can estimate them again on your PC with
`scripts/spikes/kit_colour_probe.py --install-dir <FM27 install> --all-home` (it needs
UnityPy, and writes to `data/kit-colour-probe/`); a newer estimate there comes first, and
`scripts/build_kit_catalog.py` turns it into the bundled file. A club in neither gets a
neutral kit.

## Known limits

- **Works on my machine™.** It was built and tested on one PC: Windows with WSL2, an
  RTX 4080 and FM27 build 27.0.0, and since then on the same PC's native Windows, where a
  picture takes about 8% longer than in WSL. Other GPUs, plain Linux and macOS are untested. The
  check for a running FM27 relies on Windows' `tasklist.exe`. facegen finds FM27's folders in
  the usual places only; the Epic and Game Pass versions are untested.
- **Only one model is supported.** Animagine XL 4.0 is built in: its settings, its
  tag-style prompts, and the tagger checks that go with them. There's no setting to switch
  models yet. For a realistic style, **Z-Image-Turbo** looks like the best next step. It
  already ran on this 4080 in FP8 at about 6 s per picture. It would need a prompt builder
  that writes sentences instead of tags. It also has no negative prompt, so the content
  rule would rest on the checker alone.
- **The save reader is alpha software.** [`fmsave`](https://github.com/rhiever/fmsave) is
  pinned at 0.11.0, and a game patch can break reading until fmsave catches up. A save the
  tool can't read is skipped and reported, and existing pictures stay.
- **Kits are approximate.** Colours come from a hand-made table or are estimated from
  FM27's kit images (colours only, no pattern), stripes and hoops are hit-and-miss, and
  there are no real badges or sponsors.
- **Faces are similar, not identical.** A re-drawn face stays close to the old one but
  isn't a copy. Faces also don't match the in-game 3D model, because the save doesn't
  expose a newgen's hair or skin.
- **Continental cups aren't wired up yet.** The competition IDs of the Champions League,
  Europa League and Conference League are still unknown, because my test saves load no
  European league. Until they're added, group B's "continental cup" rule matches nobody.
- **Some features are still waiting for sign-off.** Automatic save watching and re-draws
  of tracked players are built and unit-tested, but haven't had their final in-game test.

## Learn more

[AGENTS.md](AGENTS.md) has the house rules for coding agents in this repo.

```
config/          default settings and the kit colour table
docs/images/     the sample pictures on this page
native/          the optional Rust extension, and its released wheels' checksums
scripts/         the kit colour bundle's builder; spikes/: the experiments behind the numbers
src/facegen/     the Python package
tests/           unit tests (no GPU or save needed)
```

Built with Python 3.12, PyTorch, diffusers, FastAPI, SQLite and
[`fmsave`](https://github.com/rhiever/fmsave).

The code is under the [MIT License](LICENSE).

<sub>Not affiliated with Sports Interactive or SEGA. Model weights are not included, and
each model has its own licence.</sub>
