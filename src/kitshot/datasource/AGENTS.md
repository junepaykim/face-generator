# FM data adapters

## Verification Commands

Run from the repository root:

```bash
uv run --no-sync pytest tests/integration/datasource tests/unit/datasource -q
uv run --no-sync pytest tests/integration/datasource/test_fmsave_source.py -q
uv run --no-sync ruff check src/kitshot/datasource
uv run --no-sync python scripts/spikes/kit_colour_probe.py --help
uv run --no-sync python scripts/spikes/kit_coverage.py --help
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
  `staff_read`/`media_read` flags so sync leaves existing people intact. Likewise a failed
  competitions or fixtures read sets `competitions_read` false and carries no competition
  records, so sync keeps the stored leagues and managers; a tables failure alone does not.
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
- `game_kits.py` reads FM27's own kits from the install folder's database (read only):
  both club tables are walked to their exact end, rows joined by position and checked by
  the club ID; anything unexpected drops the whole source with a log warning, never a guess
  or an error that stops the app. Its cache (`data/game-kits.json`) is keyed by the file's
  path, size and time and `READER_VERSION`; raise that when what it reads changes. Never
  bundle what it reads (decisions 2026-11-13: nothing of the game's in the repository).
  Import zstd inside functions: `kitshot.cli --help` runs without site-packages.
  Shirt styles and colour names are `core.kit_values` (`SHIRT_STYLES`, `colour_name`);
  `scripts/spikes/kit_coverage.py` reports what a database covers.
- Do not hand-edit `src/kitshot/data/kit_catalog.json`. After an FM update, refresh it
  with `scripts/spikes/kit_colour_probe.py` using `--install-dir` and `--all-home`, then
  `scripts/build_kit_catalog.py`. The probe needs UnityPy, numpy and Pillow (or `--deps`).
  The bundle keeps club IDs and colour names, not images/texture data; partial probes
  are rejected by the builder.
