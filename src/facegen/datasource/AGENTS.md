# FM data adapters

## Verification Commands

Run from the repository root:

```bash
uv run --no-sync pytest tests/integration/datasource tests/unit/datasource -q
uv run --no-sync pytest tests/integration/datasource/test_fmsave_source.py -q
uv run --no-sync ruff check src/facegen/datasource
uv run --no-sync python scripts/spikes/kit_colour_probe.py --help
```

## Boundaries

- Convert reader-specific values into `core.records.Snapshot`; do not import CLI/UI,
  renderer or runtime services here.
- `fmsave_source.py` imports fmsave inside `load`, translates its exceptions and always
  closes the save.
- `records.py` and package record exports alias `core.records`; do not redefine records.
- Club UIDs are save-local: normalize through `uid_map`. Competition keys prefer the
  database ID and encode absent IDs as `-(local_id + 1)`.
- Unknown potential stays unknown; missing/repeated player unique IDs are skipped.

## Gotchas

- Staff/media reader failure must not become an empty successful read: preserve
  `staff_read`/`media_read` flags so sync leaves existing people intact.
  Optional competition/people failures warn without aborting player syncs.
- `people.py` parses synthetic-testable bytes, not fmsave objects. Reject chance byte
  matches with the existing ID/name/date/personality checks.
- `league_rules.py` (the save's division lists, levels and regional groups) parses the same
  way. A record with an unknown value type is unreadable: skip it rather than guess its size.
  A field's record is the nearest one whose own fields start there; keep the look-back limits.
- `PhotoIndex.game is None` means unknown; an empty set means known absent.
  `PhotoScan.index is None` keeps stored IDs. An incomplete external pack scan records a
  warning and retries; `persistence/photo_index.py` owns the SQLite state and counts.
- Photo scans exclude our own pack and reuse folder/catalog fingerprints; avoid
  rescanning hundreds of thousands of unchanged picture files.
- Kit values belong in `core.kit_values`; this layer loads config/catalogs.
  A local report overrides the bundle only when newer.
- Do not hand-edit `src/facegen/data/kit_catalog.json`. After an FM update, refresh it
  with `scripts/spikes/kit_colour_probe.py` using `--install-dir` and `--all-home`, then
  `scripts/build_kit_catalog.py`. The probe needs UnityPy, numpy and Pillow (or `--deps`).
  The bundle keeps club IDs and colour names, not images/texture data; partial probes
  are rejected by the builder.
