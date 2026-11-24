"""Stage the installer's folder: the release payload, uv.exe and the icon (Phase 13).

    git archive --format=zip --prefix=kitshot/ -o build/source.zip <tag>
    uv run --no-sync python installer/stage.py --archive build/source.zip \
        --uv-zip build/uv-x86_64-pc-windows-msvc.zip --out build/stage --tag <tag>
    iscc /DVersion=<version> /DStage=build\\stage /DOutput=build\\dist installer\\kitshot.iss

The payload is exactly what an update in place writes (kitshot.setup.update.unpack_payload),
so an installed folder and an updated one hold the same files. uv.exe comes from uv's release
zip after its SHA-256 matches installer/uv.sha256, and goes to .bin\\uv.exe, where Start.bat
looks for it. The icon, kitshot.ico, is made from the bundled pictures
(desktop/app_icon.py, Phase 17c). With --tag, the tag must be v<the payload's version>. Prints
`version=<version>` last, for the workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UV_SUMS = HERE / "uv.sha256"
UV_ZIP = "uv-x86_64-pc-windows-msvc.zip"
ICO = "kitshot.ico"
"""The installer's icon in the stage (SetupIconFile and the shortcuts, installer/kitshot.iss)."""


def expected_sum(name: str, sums: Path | None = None) -> str:
    sums = sums or UV_SUMS
    for line in sums.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](\S+)", line.strip())
        if m and m[2] == name:
            return m[1]
    raise SystemExit(f"{sums.name} lists no sum for {name}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_uv(uv_zip: Path, dest: Path) -> Path:
    """uv.exe from uv's release zip, checked first; into dest/.bin/uv.exe."""
    found, wanted = sha256_of(uv_zip), expected_sum(UV_ZIP)
    if found != wanted:
        raise SystemExit(f"{uv_zip.name}: SHA-256 {found} is not the {wanted} in {UV_SUMS.name}")
    with zipfile.ZipFile(uv_zip) as zf:
        member = next((n for n in zf.namelist() if n.rsplit("/", 1)[-1] == "uv.exe"), None)
        if member is None:
            raise SystemExit(f"{uv_zip.name} has no uv.exe")
        target = dest / ".bin" / "uv.exe"
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, target.open("wb") as out:
            shutil.copyfileobj(src, out)
    return target


def main(argv: list[str] | None = None) -> int:
    from kitshot.desktop.app_icon import write_ico
    from kitshot.setup import update

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archive", type=Path, required=True, help="git archive zip (one top folder)"
    )
    parser.add_argument("--uv-zip", type=Path, help=f"uv's {UV_ZIP} (left out: no uv.exe)")
    parser.add_argument(
        "--out", type=Path, required=True, help="the staging folder (emptied first)"
    )
    parser.add_argument("--tag", help="the release tag, checked against the payload's version")
    args = parser.parse_args(argv)

    if args.out.exists():
        shutil.rmtree(args.out)
    files = update.unpack_payload(args.archive, args.out)
    version = update.project_version((args.out / "pyproject.toml").read_text(encoding="utf-8"))
    if update.parse_version(version) is None:
        raise SystemExit(f"pyproject.toml's version {version!r} is not <major>.<minor>.<patch>")
    if args.tag is not None and args.tag != f"v{version}":
        raise SystemExit(f"the tag {args.tag} must be v{version}, pyproject.toml's version")
    if args.uv_zip is not None:
        stage_uv(args.uv_zip, args.out)
    write_ico(args.out, ICO)
    print(f"staged {len(files)} files of Kitshot {version} in {args.out}", file=sys.stderr)
    print(f"version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
