# Python implementation

## Verification Commands

From the repository root, choose the affected domain:

```bash
uv run --no-sync pytest tests/unit/core tests/integration/contracts -q
uv run --no-sync pytest tests/integration/persistence tests/integration/pipeline -q
uv run --no-sync pytest tests/unit/runtime tests/integration/runtime -q
uv run --no-sync pytest tests/integration/platform tests/integration/desktop tests/integration/setup -q
uv run --no-sync pytest tests/unit/render -q
```

Use the web/datasource guides for HTTP, adapters and bundled kit data; use
`native/AGENTS.md` for extension integration and `tests/AGENTS.md` for shared doubles.

## Package boundaries

- Implement in the domain packages. Top-level `db.py`, `manual.py`, `hardware.py`, etc.
  are compatibility exports for remaining consumers; preserve their canonical object
  identity until those consumers move. Do not put new behavior in those facades.
- `persistence/` owns SQLite, settings loading and image/sheet storage; `pipeline/`
  orchestrates sync, manual actions, rendering jobs and publication. Keep UI/CLI imports
  out of those layers; runtime composition connects them to the front ends.
- `runtime/contracts.py` holds plain service protocols/records; dependency factories in
  `runtime/dependencies.py` defer reader/renderer/tagger construction until use.
- Package initializers, `desktop/shell.py` and `setup/venv_check.py` must import with only
  the standard library available. Entry-point `--help` must not create runtime state.
- Use `locations.py` for repository/config/native paths, not parent-counting in moved
  modules. Load packaged resources through `importlib.resources`.
- `setup/update.py` defines the release payload for both the installer and in-place
  updates. An update runs the *installed* version's code, so keep the selection an
  exclusion list (new top-level files must still ship) and never write `data/`, `.venv/`,
  `.bin/` or `config/local.toml`. Tests never reach GitHub (root `conftest.py`).

## Storage and job invariants

- `persistence.db.connect` migrates immediately and uses autocommit; group writes with
  `db.transaction` (`BEGIN IMMEDIATE`). Use temporary databases for validation.
- Older installations can share a database. Setting renames belong in
  `persistence.db.RENAMED_SETTINGS`, including history conversion, not a new schema version.
- `persistence/jobs.py` owns queue transitions. Claims revalidate revision and eligibility
  under the write transaction; do not render a stale selection. Persist sheet allocation
  before GPU work and retain it across interruptions.
- Sync ingestion/planning commits before publication. A later failure cannot roll back
  committed sync state or already published files. Threshold edits reclassify stored facts.
- When holding both locks, take the publish lock before the database transaction.
  Restore commits its sheet/image selection before publishing. Keep both numeric and
  `r-` FM portrait mappings for each published person.

## Runtime and rendering gotchas

- `runtime.composition.build_runtime` constructs without starting; serve/app/tray share
  its lifecycle. Stop downloads, watcher, worker, then the desktop server and GUI.
- Worker, window and each sync own a child process. `SyncService` runs the sync in the
  calling thread only for a test's fake source; the real reader goes to `facegen
  sync-process`. A stop may end that child only before its commit (`on_committed`).
  A fatal CUDA fault requires worker exit/restart, not another render in the same context.
- Child stdout carries JSON lines only; diagnostics go to stderr. Use
  `platform.pipes.JsonWriter` for atomic write/flush; status read-and-send share its lock.
  Windows pipe reads must poll, not block during DLL loads.
- Keep torch/diffusers/tagger imports lazy. `render.gpu.prepare_environment` runs before
  torch imports; `render/quality_checks.py` must remain usable without model libraries.
- Prompt embeddings wait in CPU memory; text encoders and the IP-Adapter encoder leave
  the GPU between uses. Do not keep them resident alongside the UNet.
- `render/picture.py` owns 3:4 geometry. The worker checks the detail pass too; a failed
  detail pass falls back to the passing candidate. Renderer/tagger interfaces accept fakes.
