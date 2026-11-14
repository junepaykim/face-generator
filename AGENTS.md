# AGENTS.md

Guidance for coding agents working in this repository.

## Project

A local tool that reads a Football Manager 27 save, selects players (everyone at the
managed club, qualifying newgens and photo-less players elsewhere, and manual picks),
generates an anime-style portrait for each on the local GPU, and publishes the pictures
into FM27's graphics folder. It has a small web UI for thresholds and manual requests.

Status: Phases 0.2, 0.3, 1, 2 and 3 have passed: the Colchester squad's pictures show in
FM27, and the web UI (`facegen serve`) works with the GPU worker. Phases 4 (automation) and
5 (updates for tracked players) are built and wait for the user's exit tests. Pictures are
3:4 portraits since 2026-11-11. Since 2026-11-12 my club's staff get pictures too, the
settings page saves each change as it is made, with undo, and pre-existing players with no photo
in FM27 are picked like newgens (`no_photo.enabled`). Players' pictures are shoots since
2026-11-12: the kit in several looks, poses of arms and angle, coloured backdrops, the season
from the in-game date; clubs outside the kit table take colours from FM27's own kit images.
Since 2026-11-13 the plan's phases 7 to 13 turn the tool into a Windows program. Phase 7
(GPU fault recovery) is done: `facegen serve` runs the GPU worker in a child process
(`facegen worker`) that is started again after a GPU fault. Phase 8 waits for the user.
Still open from Phase 0.1: the continental competition IDs.
Build the rest phase by phase from `docs/implementation-plan.md`.

## Read first

| File | Use it for |
|---|---|
| `docs/spec.md` | Required behaviour. The source of truth. |
| `docs/architecture.md` | Module boundaries, database tables, file locations. |
| `docs/implementation-plan.md` | The current phase and its task list. |
| `docs/research.md` | Verified facts about the save format, `fmsave` and this PC. |
| `docs/decisions.md` | What the user decided, what is only proposed, open questions. |

If code and `docs/spec.md` disagree, the spec wins; raise the conflict rather than
quietly changing either.

These docs live on this PC only: `docs/*.md` is git-ignored and was removed from the
public repository's history (user, 2026-11-12). Never add them back to git.

## Environment

- WSL2 on the Windows PC that runs FM27. Python 3.12, `uv`.
- Rust (stable, from rustup, with rustfmt and clippy) for the optional extension in
  `native/`; `uv sync` builds it with maturin. `uv sync --no-group native` leaves it out
  and the Python versions run instead.
- GPU: RTX 4080, 16 GB VRAM, visible from WSL.
- FM27 user folder: `/mnt/c/Users/<Windows user>/Documents/Sports Interactive/Football Manager 27`.
  This PC's real path is in `config/local.toml`, which is not in git. Never write the
  Windows user name into a tracked file.
- FM27 install: `/mnt/c/Program Files (x86)/Steam/steamapps/common/Football Manager 27`
- File-change events do not work on `/mnt/c`. Poll instead.

## Commands

```bash
uv sync                 # install dependencies
uv run pytest           # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run facegen sync --dry-run   # read the newest save (via a snapshot copy), store nothing
uv run facegen serve            # web UI on http://127.0.0.1:8765/; the GPU worker in a child process
uv run facegen serve --no-worker   # the UI only: no GPU work
(cd native && cargo test && cargo clippy --all-targets && cargo fmt --check)   # the Rust side
```

`FACEGEN_TEST_SAVE=data/snapshots/<file>.fm uv run pytest` also runs the one test that
reads a real save copy.

The Rust extension is rebuilt by the next `uv sync` or `uv run` after a change under
`native/src/`. Everything in it has a Python version; `tests/test_native.py` checks that
the two agree, and the geometry (crop boxes, sizes, thresholds) stays in Python.

## Hard rules

1. **Never modify the user's saves.** Do not write, move or delete anything in the FM27
   `games/` folder. Copy a save to `data/snapshots/` and parse the copy.
2. **Stay in our own graphics folder.** Under the FM27 user folder, write only inside
   `graphics/facegen/`. Never edit or delete other packs.
3. **The content rule is fixed.** Every picture is a fully clothed, non-sexual sports
   portrait (spec section 3). Do not add settings, prompt options, wardrobe or pose
   entries that weaken it, and never publish a picture that failed the content check.
4. **Ask before large downloads.** Model weights and GPU libraries are several
   gigabytes. Get the user's go-ahead before adding them.
5. **`fmsave` stays pinned** to an exact version in `pyproject.toml`. It is alpha and
   changes often. Bump it on purpose, re-run the Phase 0.1 checks, and note the result
   in `docs/research.md`.
6. **Nothing leaves this PC.** Do not upload saves, player data or pictures.
7. **Keep generated data out of git:** `data/`, model files, pictures, save files. The
   one exception is `docs/images/`: a few small sample pictures of made-up test
   characters for the README (user, 2026-11-12). The docs' Markdown files stay out
   of git too.

## Conventions

- `src/` layout, package `facegen`. Type hints throughout.
- Keep the pure modules pure: classification, age mapping, character sheet sampling,
  prompt assembly and planning take plain records and return plain values. They must be
  testable with no GPU and no save file.
- All randomness is seeded from the career, the player's unique ID and a version
  number. The same inputs must give the same character sheet and prompt.
- Access `fmsave` only through the `DataSource` adapter.
- Read settings only through the settings accessor. Defaults live in
  `config/default.toml`; the UI's overrides live in the database.
- Store the full prompt, seed and model name with every generated picture.

## Workflow

- Work on the current phase only. Tick tasks in `docs/implementation-plan.md` as they
  land, and update "Current phase" when an exit test passes.
- Record new decisions in `docs/decisions.md`. Move a proposal to "Decided by the user"
  only when the user confirms it.
- When you verify something listed as unverified in `docs/research.md`, update it there
  with how it was checked.
- Do not commit unless asked.

## Planned layout

```
config/default.toml       defaults, including group thresholds
src/facegen/
  settings.py             defaults plus database overrides
  settings_history.py     the UI's settings saves, with undo and redo
  db.py                   SQLite schema and migrations
  datasource/             DataSource protocol, plain records, fmsave adapter
  identity.py             newgen boundary, career fingerprint
  leagues.py              my league and continental membership
  classify.py             group rules
  photos.py               who FM27 shows a photo for: its catalog and facepacks, scanned per sync
  age.py                  real age to age wording (the age band)
  picture.py              the 3:4 shape, published sizes, the reference face crop
  traits.py               character sheet sampling
  wardrobe.py             my club's staff's clothes, by job
  shoot.py                a player's picture: kit look, extra, arms and angle, backdrop
  prompt.py               prompt assembly
  planner.py              snapshot diff to jobs
  sync.py                 snapshot into the database, classify, plan
  manual.py               the UI's requests: generate, new face, pin, remove, retry
  worker.py               GPU job runner
  quantise.py             INT8 weights for the UNet's Linear layers
  prompt_cache.py         prompt embeddings in system memory; the next jobs' prompts in one trip
  service.py              the worker loop, "sync now" and the save watcher
  worker_process.py       the worker as a child process: pipe protocol, child, WorkerProxy
  supervision.py          restart, give-up and review policy after a GPU fault
  qc.py                   quality and content checks
  figure.py               the figure scan behind qc.head_top and picture.face_box
  native.py               loads the optional Rust extension, or None
  publish.py              pictures and config.xml into FM27
  poller.py               save-folder watcher
  web/                    FastAPI app, queries, templates, static CSS and JS
  cli.py                  command-line entry points
native/                   optional Rust extension (PyO3, maturin): figure scan, PNG encoder
scripts/spikes/           throwaway Phase 0 experiments
tests/                    unit tests for the pure modules
data/                     runtime state, not in git
```
