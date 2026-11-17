# Test suite guide

Preserve existing cases, parameter sets, content checks, deterministic outputs and parity
tolerances. Never weaken or bypass contract/DI/parity assertions to make an implementation
change pass without explicit user approval. Fix the implementation or raise the conflict.

## Placement and imports

Choose the tier by what a test exercises, then mirror the canonical `src/facegen/` domain:

| Directory | Put tests here when… |
|---|---|
| `unit/core/` | Testing classification, identity, age, traits, prompts, shoots, wardrobe or planning on plain records. |
| `unit/render/`, `unit/runtime/`, `unit/datasource/` | Testing image geometry/QC/cache policy, restart policy or synthetic byte readers without storage, processes or hardware. |
| `integration/persistence/`, `integration/pipeline/` | Using SQLite (including in-memory databases), stored sheets, settings history, sync/manual operations, worker or publishing. |
| `integration/datasource/`, `integration/platform/`, `integration/setup/` | Testing adapters, FM folder discovery, temporary filesystem trees, process I/O, setup or fake downloads/installers. |
| `integration/runtime/`, `integration/desktop/`, `integration/web/` | Testing service/controller wiring, fake child processes, tray/window stand-ins or HTTP flows. CLI tests live at `integration/test_cli.py`. |
| `integration/contracts/` | Testing dependency/DI boundaries, entry points, compatibility exports and repository resource contracts. Includes `test_di.py`. |
| `parity/` | Comparing all 53 native/Python cases, reference prompt encodings or CPU quantization against their reference behavior. |
| `opt_in/` | Requiring a real CUDA GPU/model, a real save copy or native OS behavior. |

Keep mixed storage/wiring files in integration; extract isolated cases when useful. A name
containing “GPU” or “Windows” does not determine the tier: `unit/render/test_gpu_faults.py`
uses synthetic exceptions; tray/window stand-ins run on Linux. CPU tensor parity constructs
tiny random encoders and needs installed libraries, never model weights.

Root `conftest.py` assigns exactly one directory marker (`unit`, `integration`, `parity`,
`opt_in`) and adds `di` to contracts. Explicit `gpu`, `real_save`, `windows_only` markers
describe opt-in requirements; **markers do not enable or skip tests**. Preserve the actual
environment/OS gates. Windows pipe coverage runs on native Windows and skips on Linux/WSL;
there is currently no real GUI pytest case. Future real GUI tests need an explicit opt-in
gate as well as `windows_only`.

Pytest uses importlib collection and adds `tests/` to its Python path. Import shared helpers
as `from factories import ...`, `from fakes import ...`, or `from support.<module> import ...`.
Do not import another `test_*.py` or `conftest.py`. Use `support.paths.ROOT` / `TESTS` for
repository resources / child scripts; do not count parents from a relocated test file.
Shared infrastructure stays at the root for diagnostic scripts that already import it.

## Test doubles catalog

Reuse these factories and fakes, including their dependency-injection seams, instead of
inventing ad-hoc `unittest.mock.patch` stacks. Extend a shared double for a reusable behavior;
use a focused `monkeypatch` only where the platform or import boundary itself is under test.

| Helper | Exact purpose |
|---|---|
| `factories.py` | Plain player/staff/media/club records, classification facts and synthetic snapshots, with stable IDs, league fixtures and career/newgen filler records. No save file. |
| `fakes.py` | `FakeRenderer` records prompts, seeds, references, encoding trips and detail settings; creates synthetic figures and can fail on a chosen call. `FakeTagger` returns scripted QC verdicts. Also supplies backdrop/figure pixels for geometry and parity. |
| `fake_child.py` | Executable worker-protocol simulator with planned idle/fault/crash/error/freed/hang/junk states; records starts and commands for `WorkerProxy` supervision tests. No real worker loop. |
| `fake_window_child.py` | Runs the real window child protocol with a fake window; simulates hide, browse/cancel, missing WebView2 and crashes without opening a GUI. |
| `gpu_child.py` | Runs the real CLI worker loop with fake renderer/tagger; injects one fatal render/cleanup failure on job three, records it in a temporary marker, then succeeds after restart. **No GPU required.** |
| `encoding_reference.py` | Frozen pre-cache prompt-encoding algorithm for bit-for-bit CPU and opt-in GPU comparisons. Keep independent of the optimized encoding path. |
| `support/config.py`, `support/paths.py` | Temporary settings writer and stable repository/helper paths. Root fixtures expose `config_file`, in-memory `conn`, `settings`, and disable real folder autodetection. |
| `support/desktop.py`, `support/fmsave.py`, `support/people.py`, `support/photos.py` | Shared controller doubles, fmsave-shaped records, synthetic person-database bytes, and photo catalog/XML builders. |

## Offline and isolation invariants

The default suite must pass with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, with **zero network
downloads and zero mutations to real user state, saves or caches**. Loopback HTTP servers
for integration tests are allowed. Use `tmp_path`, in-memory SQLite, fake disks/fetchers and
already installed packages. Never open the real app, install a wheel, fetch a model, run
`uv sync`, or let `uv run` synchronize dependencies as a test side effect. Use `--no-sync`.
Never point test configuration at the real FM folders, runtime database or model cache.

Real hardware/save checks require separate authorization and explicit selection:
`FACEGEN_TEST_GPU=1` uses already cached models; `FACEGEN_TEST_SAVE` must point at a snapshot
copy, never a save in FM's `games/`. Do not set these variables for the final default gate.
Keep `FACEGEN_REQUIRE_NATIVE=1` in wheel verification: a missing/broken extension must fail
there, not silently skip. Without optional native/torch libraries, parity reports skips;
those skips do not establish equivalence.

## Three verification tiers

Use the prepared environment; do not download or synchronize dependencies to run a check.

1. **Fast loop:** target the changed domain/file or individual case, for example
   `uv run --no-sync pytest tests/unit/core/test_classify.py -q` or
   `uv run --no-sync pytest tests/integration/pipeline/test_manual.py -q`.
2. **Contract loop:** run the focused architecture/DI and full parity suites:
   `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync pytest tests/integration/contracts tests/parity -q`.
   For static DI alone: `uv run --no-sync pytest tests/integration/contracts/test_di.py -q`.
   Directory selection avoids collecting unrelated suites; `-m 'di or parity'` selects the
   same contracts from a full collection. Never trim parity cases to meet a timing target.
3. **Final gate (Linux):** run these exact commands after meaningful changes settle:

   ```bash
   env -u FACEGEN_TEST_GPU -u FACEGEN_TEST_SAVE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     uv run --no-sync pytest -ra
   uv run --no-sync ruff check .
   uv run --no-sync ruff format --check .
   git diff --check
   ```

Report actual counts, skips and timings. The pre-migration prepared Linux baseline is
**1,507 passed, 6 skipped**; this migration adds two static DI cases. The six skips are four
real-GPU cases, one real-save case and one Windows pipe-handle case. Preserve all original
coverage; this gate does not imply real GPU, save or Windows GUI acceptance. Do not commit
unless asked.
