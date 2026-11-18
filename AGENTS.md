# AGENTS.md

## Project

Local FM27 portrait generator: snapshot a save, select players/staff/media, draw anime
portraits on the GPU, check them, then publish an FM graphics pack. FastAPI serves the
settings/manual-request UI; Windows adds a tray and pywebview window.

## Project references and workflow

Read the relevant sections before changing behavior:

| Local document | Authority |
|---|---|
| `docs/spec.md` | Source of truth over code; raise any conflict. |
| `docs/architecture.md` | Data flow, database and file formats. |
| `docs/implementation-plan.md` | Current phase, tasks and exit tests; keep status here. |
| `docs/research.md` | Verified save-format, model and platform facts. |
| `docs/decisions.md` | User decisions versus proposals and open questions. |

- Work on the current phase or the user's explicitly requested task. Tick tasks as they
  land; advance the phase only after its exit test passes.
- Record new decisions; mark proposals as decided only when the user confirms them.
  Update research when an unverified fact is checked, with how it was checked.
- Do not commit unless asked.

## Repository map

Local guides add only directory-specific instructions; commands below use the repo root.

| Path | Responsibility / local guide |
|---|---|
| `src/facegen/` | Python implementation; [guide](src/facegen/AGENTS.md). |
| `src/facegen/core/` | Plain records, classification, sheets, prompts and planning. |
| `src/facegen/{persistence,pipeline,render}/` | Storage, application operations, image generation/QC. |
| `src/facegen/{runtime,platform,setup,desktop}/` | Services/processes, OS adapters, installation and Windows shell. |
| `src/facegen/web/` | HTTP/templates/JS/CSS; [guide](src/facegen/web/AGENTS.md). |
| `src/facegen/datasource/` | Save, photo and kit adapters; [guide](src/facegen/datasource/AGENTS.md). |
| `native/` | Optional Rust/PyO3 CPU extension; [guide](native/AGENTS.md). |
| `tests/` | Test tiers, fixtures and doubles; [guide](tests/AGENTS.md). |
| `config/` | Defaults and kit table; `local.toml` holds this PC's paths. |
| `installer/` | Inno Setup script, payload staging and the pinned uv checksum (Phase 13). |
| `scripts/spikes/`, `scripts/archive/` | Active probes and completed experiments. |
| `.github/workflows/` | Python CI, native wheel builds and the installer release. |
| `data/` | Local runtime state and snapshots. |

## Environment and setup

- Python 3.12+, `uv`; WSL2/Linux for development/tests, native Windows for the desktop.
  Stable Rust with rustfmt/clippy builds the optional extension via maturin.
- Light setup in a new environment: `uv sync --locked --no-group gpu --no-group native`.
  Default `uv sync` includes the large GPU group; hard rule 4 applies. Use `--inexact`
  when syncing an environment with a separately installed native wheel to retain it.
- Windows torch uses the configured `pytorch-cu132` index. Desktop dependencies install
  only on Windows; Linux/WSL uses `facegen serve`. This PC has an RTX 4080 (16 GB).
- FM folders are discovered by `platform/fm_paths.py` or set in ignored
  `config/local.toml`; `--config X` uses `local.toml` beside X. Windows paths work in WSL.
  Never hardcode a Windows user name in tracked files.
- `Start.bat` stays ASCII/CRLF and thin; launch decisions belong in `setup/launch.py`.
  From WSL, launch native Windows GUI checks with PowerShell `Start-Process`.

## Cross-module rules

- `core/` imports only the standard library and `facegen.core`, including lazy and
  type-only imports. Policy takes plain records; adapters own I/O. Keep type hints.
- Seed all randomness from career, unique ID and version; identical inputs reproduce
  the sheet and prompt. Store the full prompt, seed and model name with each picture.
- Access `fmsave` only through `DataSource`. Read settings through the accessor:
  `config/default.toml` defaults plus database overrides.
- Launch subprocesses through `platform.procs.run_quiet` / `popen_quiet` so Windows
  shows no console. Poll files on `/mnt/c`; change events do not work there.

## Full-project verification

Use the prepared environment; targeted checks live in the local guides. The default
suite must run offline. These are Bash/WSL commands; CI sets the same offline variables
on Windows. Keep real-GPU/save opt-ins unset for the final default gate.

```bash
env -u FACEGEN_TEST_GPU -u FACEGEN_TEST_SAVE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync pytest -ra
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
cargo test --locked --manifest-path native/Cargo.toml
cargo clippy --locked --manifest-path native/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path native/Cargo.toml --check
git diff --check
```

Run the Rust checks when changing the extension; Python changes need the affected tests
and lint. Report actual results and skips; automated checks do not imply in-game or
Windows GUI acceptance. Do not synchronize dependencies just to run a check.

## Hard rules

1. **Never modify saves.** Do not write, move or delete anything in FM27's `games/`.
   Copy a save to `data/snapshots/` and parse the copy.
2. **Stay in our graphics folder.** Under the FM27 user folder, write only inside
   `graphics/facegen/`; never edit/delete other packs.
3. **Content is fixed.** Fully clothed, non-sexual sports portraits only. No setting,
   prompt, wardrobe or pose may weaken this; never publish an image that failed QC.
4. **Ask before large downloads.** Get approval before model weights or GPU libraries.
5. **Pin `fmsave` exactly** in `pyproject.toml`. An intentional bump requires Phase 0.1
   checks and a result in `docs/research.md`.
6. **Nothing leaves this PC.** Do not upload saves, player data or pictures.
7. **Keep generated data out of git:** root `data/`, models, pictures and saves.
   Only small made-up README samples in `docs/images/` are excepted. `docs/*.md` stays
   ignored and must never be added back to git; bundled `src/facegen/data/` is source data.
