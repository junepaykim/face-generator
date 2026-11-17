# Web UI

## Verification Commands

Run from the repository root:

```bash
uv run --no-sync pytest tests/integration/web -q
uv run --no-sync pytest tests/integration/contracts/test_message_catalog.py tests/unit/i18n -q
uv run --no-sync pytest tests/integration/web/test_web.py -k 'csrf or origin or host or token or pictures or redirects' -q
uv run --no-sync ruff check src/facegen/web
```

## Boundaries

- FastAPI/Jinja pages, query read models, form parsing and plain JS/CSS; no Node build.
- `app.py` composes fresh router factories and shared dependencies/presentation per app.
  Routers never import `app.py`; shared helpers never import routers.
- `AppContext` injects worker, sync, model status, download, hardware and folder browsing.
  Routes delegate to these services and pipeline operations; do not load models, import
  torch/fmsave, or render/parse saves in request handlers.
- Each request or blocking work function owns and closes its SQLite connection.
  Put blocking operations behind `run_in_threadpool`; never share a request connection
  with long-lived services.
- `queries.py` builds read models; mutations use pipeline/settings-history APIs.
  Wake the worker after changes that affect its queue.

## Gotchas

- Keep loopback-host checks, exact-origin checks and per-app CSRF tokens in `security.py`.
  Every POST form needs `csrf_token`; fetch requests use `X-CSRF-Token`.
- HTTP tests need a loopback `base_url` and a token obtained from the app's HTML.
  The default TestClient host is rejected; do not disable middleware to make tests pass.
- Picture routes resolve paths beneath masters; thumbnails go in the thumb cache.
  Redirect targets must pass `_safe_next`.
- Settings autosaves are serialized and send only dirty fields. Preserve edits made while
  a save is in flight; undo/redo/reset wait for pending saves.
- Keep template `data-*` attributes aligned with `static/app.js`; a live part has
  `data-refresh`, `data-topic` and `data-rev` (`web/live.py`), and the career marker reloads
  the page after the first sync. Asset URLs include `static_version`.
- `/events` never ends by itself: the test client reads to the end, so tests shorten
  `live.STREAM_SECONDS`; a server must call `close_streams` before stopping.
- Tests cover HTML/HTTP contracts, not browser behavior; inspect JS/CSS interactions
  when those change.
- No words in templates or routes: use catalog keys (`t("…")` in templates, `tr("…")` for
  attributes, a request `Translator` in routes) and add each key to both
  `facegen/i18n/en.toml` and `ko.toml`. Keep English wording stable; tests match it.
- Keys built from values (`"kind." ~ job.kind`) are listed in `test_message_catalog.py`;
  add new vocabularies there. Catalog text may hold markup; values are always escaped.
- Name groups as {a}/{b}/{c} in catalog text (chips on pages, names elsewhere). Keep page text
  minimal: no notes for obvious features; longer help goes in `_icons.html`'s `info` tip.
- Errors never reach a page raw: show `messages.problem` with `_problem.html` (raw text folded)
  or `problem_line`; a new error kind goes in `core/problems.py` and both catalogs.
- The worker's state reaches pages as `WorkerStatus.reason` + `values`; word it with
  `messages.worker_sentence`. A new reason goes in `contracts.WORKER_REASONS` and both catalogs.
- Colours are `:root` custom properties with dark overrides for both `prefers-color-scheme`
  and `html[data-theme=dark]`; add new colours to all three blocks.
