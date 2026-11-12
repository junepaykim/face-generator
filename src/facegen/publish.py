"""Publishing pictures into FM27 (docs/research.md section 4, the recipe from Phase 0.2).

Each career has its own pack, `<FM27 user dir>/graphics/facegen/<career key>/`: one square
PNG per player named `<unique_id>.png`, plus a classic facepack `config.xml` that maps each
ID to its picture. FM27 loads changes on Settings -> Reload Skin.

AGENTS.md hard rule 2: under the FM27 user folder this module writes only inside
`graphics/facegen/`. Every path is checked before anything is written or deleted.

Writes are atomic (a temporary file, then a rename), and nothing is rewritten when it is
already in place, so publishing twice changes nothing.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

RELOAD_HINT = "In FM27: Settings -> Reload Skin (스킨 다시 불러오기) to load the pictures."


class PublishError(RuntimeError):
    """A path would leave graphics/facegen/, or a picture is missing."""


def packs_root(fm_user_dir: Path) -> Path:
    return fm_user_dir / "graphics" / "facegen"


def pack_dir(fm_user_dir: Path, career_key: str) -> Path:
    """This career's pack folder. Refuses any key that would leave graphics/facegen/."""
    root = packs_root(fm_user_dir)
    if not career_key or career_key in (".", "..") or any(c in career_key for c in "/\\:"):
        raise PublishError(f"bad career key {career_key!r}")
    folder = root / career_key
    _guard(folder, fm_user_dir)
    return folder


def _guard(path: Path, fm_user_dir: Path) -> None:
    root = packs_root(fm_user_dir).resolve()
    if not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise PublishError(f"refusing to write outside {root}: {path}")


def config_xml(ids: Iterable[int]) -> str:
    """The classic facepack mapping (verified in FM27, Phase 0.2)."""
    records = "".join(
        f'\t\t<record from="{i}" to="graphics/pictures/person/{i}/portrait"/>\n'
        for i in sorted(set(ids))
    )
    return (
        "<record>\n"
        '\t<boolean id="preload" value="false"/>\n'
        '\t<boolean id="amap" value="false"/>\n'
        '\t<list id="maps">\n'
        f"{records}"
        "\t</list>\n"
        "</record>\n"
    )


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_config(folder: Path, ids: Iterable[int], fm_user_dir: Path) -> bool:
    """Write config.xml if its content would change. Returns whether it was written."""
    path = folder / "config.xml"
    _guard(path, fm_user_dir)
    text = config_xml(ids).encode("utf-8")
    if path.is_file() and path.read_bytes() == text:
        return False
    folder.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, text)
    return True


def published_bytes(master: Path, size: int) -> bytes:
    """The master downscaled to a square of `size` pixels, as PNG bytes."""
    import io

    from PIL import Image

    with Image.open(master) as im:
        img = im.convert("RGB")
    if img.width != img.height:
        raise PublishError(f"{master} is not square ({img.width}x{img.height})")
    if img.width != size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def publish_picture(
    master: Path, folder: Path, unique_id: int, size: int, fm_user_dir: Path
) -> Path:
    dest = folder / f"{unique_id}.png"
    _guard(dest, fm_user_dir)
    if not master.is_file():
        raise PublishError(f"master picture missing: {master}")
    folder.mkdir(parents=True, exist_ok=True)
    _write_atomic(dest, published_bytes(master, size))
    return dest


@dataclass(frozen=True, slots=True)
class PublishReport:
    folder: Path
    written: int
    unchanged: int
    missing: tuple[int, ...]
    """Players whose master picture is missing; left unmapped."""
    config_changed: bool
    mapped: int

    @property
    def changed(self) -> bool:
        return bool(self.written or self.config_changed)


def publish_career(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    fm_user_dir: Path,
    sizes: dict[str, int],
    *,
    data_dir: Path,
) -> PublishReport:
    """Bring the career's pack in line with its current pictures.

    `sizes` maps group to published size in pixels. A picture already published to its
    place is left alone; config.xml is rewritten only if the mapping changes.
    """
    folder = pack_dir(fm_user_dir, career_key)
    rows = conn.execute(
        "SELECT id, unique_id, group_name, master_path, published_path FROM images"
        " WHERE career_id = ? AND is_current = 1 ORDER BY unique_id",
        (career_id,),
    ).fetchall()
    written = unchanged = 0
    missing: list[int] = []
    mapped: list[int] = []
    for r in rows:
        dest = folder / f"{r['unique_id']}.png"
        if r["published_path"] == str(dest) and dest.is_file():
            unchanged += 1
            mapped.append(r["unique_id"])
            continue
        master = Path(r["master_path"]) if r["master_path"] else None
        if master is not None and not master.is_absolute():
            master = data_dir / master
        if master is None or not master.is_file():
            missing.append(r["unique_id"])
            continue
        publish_picture(master, folder, r["unique_id"], sizes[r["group_name"]], fm_user_dir)
        conn.execute("UPDATE images SET published_path = ? WHERE id = ?", (str(dest), r["id"]))
        written += 1
        mapped.append(r["unique_id"])
    config_changed = (
        write_config(folder, mapped, fm_user_dir) if mapped or folder.exists() else False
    )
    return PublishReport(folder, written, unchanged, tuple(missing), config_changed, len(mapped))


def unpublish(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    unique_id: int,
    fm_user_dir: Path,
) -> bool:
    """Remove a player's published picture and mapping; FM27 then shows its own photo again.

    The picture stops being current; the master and the database history are kept. The
    planner would queue an eligible player again, so the UI's "remove picture" (Phase 3)
    pairs this with a "never" pin. Returns whether anything was removed.
    """
    folder = pack_dir(fm_user_dir, career_key)
    dest = folder / f"{unique_id}.png"
    _guard(dest, fm_user_dir)
    existed = dest.is_file()
    dest.unlink(missing_ok=True)
    conn.execute(
        "UPDATE images SET published_path = NULL, is_current = 0"
        " WHERE career_id = ? AND unique_id = ?",
        (career_id, unique_id),
    )
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT unique_id FROM images WHERE career_id = ? AND is_current = 1"
            " AND published_path IS NOT NULL",
            (career_id,),
        )
    ]
    changed = write_config(folder, ids, fm_user_dir) if folder.exists() else False
    return existed or changed
