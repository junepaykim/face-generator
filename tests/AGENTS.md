# Test suite

## Verification Commands

Run from the repository root in the prepared environment:

```bash
uv run --no-sync pytest tests/unit/core/test_classify.py -q
uv run --no-sync pytest tests/integration/pipeline/test_manual.py -q
uv run --no-sync pytest tests/integration/contracts/test_di.py -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync pytest tests/integration/contracts tests/parity -q
```

Use the changed test file or case for the fast loop; the final default gate is in the
root guide. Preserve cases, parameter sets, content checks, deterministic outputs and
parity tolerances. Never weaken/bypass contract, DI or parity assertions to make a change
pass without explicit user approval; fix the implementation or raise the conflict.

## Placement and imports

| Tier | What belongs here |
|---|---|
| `unit/<domain>/` | Plain records, synthetic bytes, image geometry and restart policy. |
| `integration/<domain>/` | SQLite (including in-memory), files, process/HTTP/application wiring. |
| `integration/contracts/` | Import boundaries, dependency injection, entry points and compatibility. |
| `parity/` | Native/Python equivalence and CPU tensor reference comparisons. |
| `opt_in/` | Real GPU/models, real save copy or native OS behavior. |

- Mirror the canonical source domain; CLI cases are in `integration/test_cli.py`.
  Keep mixed storage/wiring files in integration; a GPU/Windows name alone is not a tier.
- `conftest.py` assigns one directory tier marker and adds `di` to contracts.
  Markers do not enable/skip tests: preserve the actual environment and OS gates.
- Pytest uses importlib collection with `tests/` on its path. Import `factories`, `fakes`
  or `support.*`, never other `test_*.py` files or `conftest.py`.
  Use `support.paths.ROOT` / `TESTS`, not parent-counting from relocated files.

## Fixtures and doubles

- Root fixtures: `conn` is in-memory SQLite, `config_file` writes under `tmp_path`,
  `settings` loads defaults; autouse `no_folder_detection` blocks real FM discovery.
- `factories.py`: plain people/club/snapshot records, stable IDs and career/newgen filler.
- `fakes.py`: `FakeRenderer`, `FakeTagger`, synthetic figures/backdrops and failure injection.
- `fake_child.py`: worker protocol/fault/crash/hang scenarios. `gpu_child.py` runs the
  real worker loop with fake rendering and fault injection; it requires no GPU.
- `fake_window_child.py`: real window protocol with a fake GUI.
- `sync_child.py`: real `facegen sync-process` with a fake reader by plan (synced,
  unsupported, crash, slow read/transaction/publication for Quit).
- `support/config.py`, `desktop.py`, `fmsave.py`, `people.py`, `photos.py`: temporary
  settings, controller doubles, reader-shaped records, byte fixtures and photo catalogs.
- Reuse/extend these doubles and injection seams before ad-hoc patch stacks. Focused
  monkeypatches belong at platform/import boundaries. Keep helpers at the test root
  for diagnostic consumers; `encoding_reference.py` stays independent of optimized code.

## Isolation and opt-ins

- Tests must not download, sync dependencies, install wheels, open the real app or mutate
  real user state/caches. Use `tmp_path`, fake disks/fetchers and installed packages;
  loopback HTTP servers are allowed. Never configure real FM folders or runtime databases.
- Real GPU/save tests require separate authorization and explicit selection:
  `FACEGEN_TEST_GPU=1` uses cached models; `FACEGEN_TEST_SAVE` names a snapshot copy.
- The Windows pipe test runs on native Windows and skips on Linux/WSL. There is no real
  GUI pytest case; future GUI cases need an explicit opt-in gate plus `windows_only`.
- CPU encoding/quantization parity uses tiny random models and installed tensor libraries,
  never weights. Optional-library skips do not establish equivalence; report them.
