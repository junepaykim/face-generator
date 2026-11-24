"""Kitshot's name before Phase 17o: only the entry points that older files still start.

The package is `kitshot`. A copy updated in place keeps its older uninstaller, which runs
`python -m facegen.cli remove-pack --yes`, and a taskbar pin made by an older version starts
`facegen-app.exe -m facegen.desktop app`; both forward to `kitshot`. Nothing else lives
here, and nothing in `kitshot` imports it. Standard library only, like `kitshot/__init__`.
"""
