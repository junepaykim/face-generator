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
- `src/facegen/` (beside this package) holds only `facegen.cli` and `python -m
  facegen.desktop`, which forward here: copies updated in place keep a 0.1.x uninstaller
  and taskbar pins that start them. Add nothing there; never import it from `kitshot`.
- Package initializers, `desktop/shell.py` and `setup/venv_check.py` must import with only
  the standard library available. Entry-point `--help` must not create runtime state.
- Use `locations.py` for repository/config/native paths, not parent-counting in moved
  modules. Load packaged resources through `importlib.resources`.
- `setup/release_payload.py` selects files for both the installer and updates. An update
  runs the installed version's code, so new top-level files must still ship. Keep user
  data, environments, tools and local configuration out of the payload.
  `setup/install_transaction.py` recovers interrupted replacements. Keep its standalone
  recovery entry point usable while the installed package is only partly replaced.
  Update tests use temporary installations and injected release responses.

## Storage and job invariants

- `persistence.db.connect` migrates immediately and uses autocommit; group writes with
  `db.transaction` (`BEGIN IMMEDIATE`). Use temporary databases for validation.
- Older installations can share a database. Setting renames belong in
  `persistence.db.RENAMED_SETTINGS`, including history conversion, not a new schema version.
- `persistence/jobs.py` owns queue transitions. Claims revalidate revision and eligibility
  under the write transaction; do not render a stale selection. Persist sheet allocation
  before GPU work and retain it across interruptions.
- `pipeline/worker.py` prepares the same portrait prompt for dry-run previews, encoding
  upcoming jobs and drawing. Preview reads or samples a sheet without storing it. A
  running job uses the sheet assigned by `persistence/jobs.py`.
- Sync ingestion, planning and manual actions use the settings current inside the write
  transaction (`read_settings`). Settings changes, Undo and Redo reclassify stored facts in
  `pipeline/reclassification.py`; a manual action re-plans only its own players (`only=`).
- Commit the pending publication with the sync. In `pipeline/publish.py`,
  `publish_active` clears that marker only after publishing the active pack succeeds.
  The CLI publish command, worker and restore action call `publish_pack` for a selected career.
- When holding both locks, take the publish lock before the database transaction.
  Restore commits its sheet/image selection before publishing. Keep both numeric and
  `r-` FM portrait mappings for each published person.
- `datasource/photos.py` scans FM files. Read stored photo counts and scan status from
  `persistence/photo_index.py`, the source for the `kitshot.photos` compatibility exports.

## Runtime and rendering gotchas

- `runtime.composition.build_runtime` constructs without starting. Desktop, tray and
  development server share its lifecycle. Quit signals the worker immediately, then
  stops downloads, the active sync and watcher, reaps the worker and closes the server and UI.
- The GPU worker, WebView2 window and each production sync own a child process.
  Tests opt into inline sync explicitly. Snapshot access belongs in `datasource/snapshots.py`,
  orchestration in `pipeline/sync_operation.py`, and shutdown timing in
  `runtime/sync_process.py`. A committed sync gets a publication grace period on Quit.
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
