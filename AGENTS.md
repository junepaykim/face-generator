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
(`facegen worker`) that is started again after a GPU fault. Phase 8 (native Windows: FM27's
folders found on the PC, no console windows, PyTorch's CUDA wheels on Windows) is built and
waits for the user's exit test. Phase 9 (kit colours bundled in the package, prebuilt wheels
of the Rust extension with `facegen native --install`) is built. The `native-v0.1.0` release
and checksums are published (2026-11-15); installation from its downloaded Windows wheel
and all 53 native tests pass, and the CLI's download of both wheels from the release was
checked from WSL. A native Windows run of `facegen native --install` (online, no `--from`)
and the user's acceptance remain.
Since 2026-11-13 my club's youth teams are group B, and journalists and press officers get
pictures (read from the save's bytes by `datasource/people.py`, as fmsave does not). Since
2026-11-14 everyone is drawn as a woman again (staff and media were drawn as men for a day),
journalists start at band W04, and one pinned to A is tracked as a regular at my club.
Since 2026-11-15 the first-team managers of the other clubs in my league are read too; those
with no photo get one group C picture, untracked. Each published picture is listed under
both of FM27's IDs for the person (the number and its `r-` form).
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

- WSL2 on the Windows PC that runs FM27 (the development route), or native Windows (since
  Phase 8). Python 3.12, `uv`. Every test runs on Linux; Windows-only parts sit behind
  functions the tests replace (`platform_info`, `fm_paths`, `procs`).
- Rust (stable, from rustup, with rustfmt and clippy) for the optional extension in
  `native/`; `uv sync` builds it with maturin, except on Windows. `uv sync --no-group
  native` leaves it out and the Python versions run instead. Elsewhere (Windows, no Rust)
  `facegen native --install` installs the prebuilt wheel from the GitHub Release
  `native-v<version>`, only if its SHA-256 is listed in `native/wheels.sha256`; then use
  `uv sync --inexact`, as a plain `uv sync` removes it. `native.VERSION` must equal
  `native/pyproject.toml`'s version; bump both together, tag `native-v<version>`, and add
  the release's `SHA256SUMS.txt` to `native/wheels.sha256`.
- The kit colours for clubs outside `config/kits.toml` ship in
  `src/facegen/data/kit_catalog.json` (club IDs and colour names only). Refresh after an
  FM27 update: `scripts/spikes/kit_colour_probe.py --all-home`, then
  `scripts/build_kit_catalog.py`. `.gitignore` ignores `/data/` at the root only.
- On Windows `uv sync` takes torch from PyTorch's CUDA index (`pytorch-cu132`): PyPI's
  Windows torch has no CUDA.
- GPU: RTX 4080, 16 GB VRAM, visible from WSL.
- FM27 user folder: `<Documents>/Sports Interactive/Football Manager 27`, found on the PC
  (`fm_paths.py`), or set in `config/local.toml`, which is not in git (a Windows path there
  works from WSL too). Never write the Windows user name into a tracked file.
- FM27 install: Steam's `steamapps/common/Football Manager 27` in any Steam library, found
  the same way.
- File-change events do not work on `/mnt/c`. Poll instead.
- Start other programs only through `procs.run_quiet` or `procs.popen_quiet`, so no console
  window flashes on Windows.

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
uv run facegen native [--install [--from DIR]]   # Rust extension in use? install its wheel
uv run python scripts/build_kit_catalog.py       # probe report -> bundled kit colours
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
  settings.py             defaults plus database overrides; FM27's folders
  platform_info.py        native Windows, WSL or Linux
  fm_paths.py             where FM27's user and install folders are on this PC
  procs.py                subprocesses without a console window; tasklist
  settings_history.py     the UI's settings saves, with undo and redo
  db.py                   SQLite schema and migrations
  datasource/             DataSource protocol, plain records, fmsave adapter;
                          people.py: gender flags, journalists, press officers from game_db
  identity.py             newgen boundary, career fingerprint
  leagues.py              my league and continental membership
  classify.py             group rules
  photos.py               who FM27 shows a photo for: its catalog and facepacks, scanned per sync
  age.py                  real age to age wording (the age band)
  picture.py              the 3:4 shape, published sizes, the reference face crop
  traits.py               character sheet sampling
  kits.py                 kit colours: config/kits.toml, a probe report or the bundle
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
  native.py               loads the optional Rust extension at the wanted version, or None
  native_wheel.py         the prebuilt extension: which wheel, its checksum, install
  data/kit_catalog.json   bundled kit colours per club ID (scripts/build_kit_catalog.py)
  publish.py              pictures and config.xml into FM27
  poller.py               save-folder watcher
  web/                    FastAPI app, queries, templates, static CSS and JS
  cli.py                  command-line entry points
native/                   optional Rust extension (PyO3, maturin): figure scan, PNG encoder;
                          wheels.sha256: the released wheels' checksums
scripts/build_kit_catalog.py   reduces a kit colour probe report to the bundled catalog
scripts/spikes/           throwaway Phase 0 experiments
tests/                    unit tests for the pure modules
.github/workflows/ci.yml  ruff and pytest on Ubuntu and Windows, no GPU group
.github/workflows/wheels.yml   the extension's wheels (Windows, manylinux), tested; a
                          `native-v*` tag releases them with SHA-256 sums
data/                     runtime state, not in git
```
