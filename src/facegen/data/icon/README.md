# Kitshot's icon

`kitshot-<size>.png`: the user's icon, `src/facegen/web/static/kitshot.svg` (Phase 17c), drawn
at each size Windows asks an `.ico` for (16, 20, 24, 32, 40, 48, 64, 128 and 256 pixels) by
librsvg 2.58 with cairo, called through ctypes on Ubuntu 24.04 (WSL2), on 2026-11-23:

    uv run --no-sync python scripts/build_icon.py

The desktop app loads them in `desktop/app_icon.py`. Make them again whenever the SVG changes.
