# Native extension

## Verification Commands

Run from the repository root with the prepared Rust/Python environment:

```bash
cargo test --locked --manifest-path native/Cargo.toml figure::tests
cargo test --locked --manifest-path native/Cargo.toml resample::tests
cargo test --locked --manifest-path native/Cargo.toml portrait::tests
FACEGEN_REQUIRE_NATIVE=1 uv run --no-sync pytest tests/parity/test_native.py -q
uv run --no-sync pytest tests/integration/setup/test_native_wheel.py -q
```

After Rust edits, rebuild before parity on Linux/WSL:
`uv sync --locked --inexact --no-group gpu` rebuilds the workspace extension and keeps
installed GPU packages. `--no-sync` alone tests the previously installed binary.
Full Rust lint/format checks are in the root guide.

## Boundary and performance

- PyO3 module `facegen_native` (abi3 Python 3.12+) supplies CPU figure scans,
  Pillow-compatible resizing and portrait PNG encoding. No game policy, database,
  model loading, CLI or UI dependencies.
- Python supplies crop boxes, target sizes and scan thresholds; do not duplicate
  those decisions in Rust. Every operation retains a Python fallback in
  `src/facegen/render/figure.py` or `src/facegen/pipeline/publish.py`.
- Figure resize/scan pixels match Pillow/Python exactly. Encoder parity permits the
  existing small fixed-point rounding difference; do not widen that tolerance.
- `Grid` accepts packed RGB bytes; preserve length/size validation and Python errors.
- Expensive resize/encoding releases the GIL with `py.detach`. Rayon batch encoding
  returns results in input order. Resize uses a separate pool capped at eight threads;
  sharing the global encoder pool changes contention.

## Packaging gotchas

- Keep `native/Cargo.toml`, `native/pyproject.toml` and
  `src/facegen/platform/native.py:VERSION` equal.
- Maturin's cache keys cover Rust sources and Cargo files; preserve them when changing
  build configuration so source edits invalidate the installed build.
- Windows uses prebuilt wheels via `facegen native --install`. Release tags are
  `native-v<version>`; copy released `SHA256SUMS.txt` entries into `native/wheels.sha256`.
  The installer accepts only listed hashes; see `.github/workflows/wheels.yml`.
- Keep `FACEGEN_REQUIRE_NATIVE=1` in wheel verification: a missing/broken extension
  must fail instead of silently skipping parity.
