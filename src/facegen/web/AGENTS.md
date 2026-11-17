# Web UI

## Verification Commands

Run from the repository root:

```bash
uv run --no-sync pytest tests/integration/web -q
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
- Keep template `data-*` attributes aligned with `static/app.js`; status polling uses
  the career marker to reload after the first sync. Asset URLs include `static_version`.
- Tests cover HTML/HTTP contracts, not browser behavior; inspect JS/CSS interactions
  when those change.
