"""Publish portraits only inside <FM27 user dir>/graphics/facegen/<career key>/.

Each pack has <unique_id>.png portraits and config.xml mappings for numeric and r- IDs.
Portrait height is the group's published size; non-3:4 masters are cropped centrally.
FM27 loads changes on Reload Skin or restart.

Newgen IDs repeat across careers, so only the career with the newest save maps pictures.
Other packs keep their PNGs with empty mappings. `publish_pack` and `publish_active` hold
`publish_lock` to serialize worker/sync writes. Every path is checked before writing or
deleting; atomic replacement and skipping unchanged files make publishing idempotent.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from facegen.persistence.db import active_career_id, transaction
from facegen.platform import native
from facegen.platform.locking import file_lock
from facegen.render.picture import Box, aspect_box, crop_to_aspect, published_size

if TYPE_CHECKING:
    from facegen.core.settings_values import Settings

RELOAD_HINT = "In FM27: Settings -> Reload Skin (스킨 다시 불러오기) to load the pictures."


class PublishError(RuntimeError):
    """A path would leave graphics/facegen/, or a picture is missing."""


class FolderNotSetError(PublishError):
    """FM27's user folder is neither set nor found: there is nowhere to publish to."""

    def __init__(self) -> None:
        from facegen.persistence.settings import folder_not_set_message

        super().__init__(folder_not_set_message())


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
    """Map each picture in both FM person namespaces.

    Database people use the numeric ID; generated players and staff use ``r-<ID>``.
    The save adapter does not expose an authoritative graphics-ID flag, and the
    eligibility ``is_newgen`` flag is not one (staff can be first seen mid-career).
    Keep both aliases to the same PNG rather than guess from age, role or ID range.
    ``from`` is always the numeric filename, without an ``r-`` prefix.
    """
    records = "".join(
        f'\t\t<record from="{i}" to="graphics/pictures/person/{prefix}{i}/portrait"/>\n'
        for i in sorted(set(ids))
        for prefix in ("", "r-")
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


_CHUNK = 8
"""Pictures the Rust encoder takes at once: enough to keep the cores busy, few enough that
the decoded masters (about 7 MB each) stay small in memory."""


def published_bytes(master: Path, long_edge: int, *, rust: bool = True) -> bytes:
    """The master as a 3:4 portrait `long_edge` pixels high, as PNG bytes.

    The Rust encoder is used when it is built (see `published_pictures`), unless `rust` is
    False.
    """
    if rust and native.ext is not None:
        return native.ext.encode_portrait(master, _crop_box(master), published_size(long_edge))
    return _pillow_bytes(master, long_edge)


def published_pictures(items: Sequence[tuple[Path, int]], *, rust: bool = True) -> Iterator[bytes]:
    """`published_bytes` for each (master, long edge), in order.

    With the Rust extension (native.py), a chunk of pictures at a time is decoded, cropped,
    scaled with Lanczos-3 and saved as an optimised 8-bit RGB PNG on all cores. Without it,
    Pillow does the same one picture at a time. The pictures are not byte-identical: on the
    58 real masters, 99.98% of channel values were within one level of Pillow's (fixed-point
    rounding), and a few pixels on hard edges differed by up to 15, because the Rust resize
    runs its vertical pass before the horizontal one and so clips Lanczos's overshoot at a
    different stage (docs/research.md). The quality and content checks look at the master,
    not at this copy.
    """
    if not rust or native.ext is None:
        for master, long_edge in items:
            yield _pillow_bytes(master, long_edge)
        return
    for start in range(0, len(items), _CHUNK):
        jobs = [(m, _crop_box(m), published_size(e)) for m, e in items[start : start + _CHUNK]]
        yield from native.ext.encode_portraits(jobs)


def _crop_box(master: Path) -> Box:
    """The 3:4 box of the master, from its header alone."""
    from PIL import Image

    with Image.open(master) as im:
        return aspect_box(im.width, im.height)


def _pillow_bytes(master: Path, long_edge: int) -> bytes:
    """The Python version of the encoder."""
    import io

    from PIL import Image

    with Image.open(master) as im:
        img = crop_to_aspect(im.convert("RGB"))
    size = published_size(long_edge)
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
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
    """Players config.xml maps; 0 for a pack that is not active."""
    active: bool = True
    """Whether this is the active career's pack, the one FM27 is shown."""
    deactivated: tuple[str, ...] = ()
    """Other careers' packs whose mappings were just removed."""

    @property
    def changed(self) -> bool:
        return bool(self.written or self.config_changed or self.deactivated)


def publish_career(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    fm_user_dir: Path,
    sizes: dict[str, int],
    *,
    data_dir: Path,
    active: bool = True,
) -> PublishReport:
    """Bring the career's pack in line with its current pictures.

    `sizes` maps group to published size (the picture's height) in pixels. A picture already
    published to its place is left alone; config.xml is rewritten only if the mapping changes.
    A pack that is not `active` gets its pictures but a config.xml that maps nothing.
    """
    folder = pack_dir(fm_user_dir, career_key)
    rows = conn.execute(
        "SELECT id, unique_id, group_name, master_path, published_path FROM images"
        " WHERE career_id = ? AND is_current = 1 ORDER BY unique_id",
        (career_id,),
    ).fetchall()
    unchanged = 0
    missing: list[int] = []
    mapped: list[int] = []
    todo: list[tuple[int, Path, Path, int]] = []  # image id, master, destination, size
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
        _guard(dest, fm_user_dir)
        todo.append((r["id"], master, dest, sizes[r["group_name"]]))
        mapped.append(r["unique_id"])
    # Encoded together, so the Rust encoder can work on several pictures at once.
    if todo:
        folder.mkdir(parents=True, exist_ok=True)
    pictures = published_pictures([(master, size) for _, master, _, size in todo])
    for (image_id, _, dest, _), data in zip(todo, pictures, strict=True):
        _write_atomic(dest, data)
        conn.execute("UPDATE images SET published_path = ? WHERE id = ?", (str(dest), image_id))
    written = len(todo)
    shown = mapped if active else []
    config_changed = (
        write_config(folder, shown, fm_user_dir) if mapped or folder.exists() else False
    )
    return PublishReport(
        folder, written, unchanged, tuple(missing), config_changed, len(shown), active
    )


def deactivate_packs(fm_user_dir: Path, keep: str | None) -> tuple[str, ...]:
    """Make every pack but `keep` map nothing; their pictures stay. Returns those changed.

    Only folders under graphics/facegen/ that have a config.xml are touched.
    """
    root = packs_root(fm_user_dir)
    if not root.is_dir():
        return ()
    changed = []
    for folder in sorted(root.iterdir()):
        if folder.name == keep or folder.is_symlink() or not folder.is_dir():
            continue
        if (folder / "config.xml").is_file() and write_config(folder, (), fm_user_dir):
            changed.append(folder.name)
    return tuple(changed)


@contextmanager
def publish_lock(data_dir: Path) -> Iterator[None]:
    """One writer of FM27's packs at a time, across threads and processes.

    Take it before any database transaction that publishing runs in, never inside one.
    """
    with file_lock(data_dir / "publish.lock"):
        yield


def publish_pack(
    conn: sqlite3.Connection, settings: Settings, career_id: int, career_key: str
) -> PublishReport:
    """Publish one career's pictures. If it is the active career, its config.xml maps them
    and every other pack is unmapped; otherwise its config.xml maps nothing. Raises
    FolderNotSetError without FM27's folder."""
    fm_user_dir = settings.paths.fm_user_dir
    if fm_user_dir is None:
        raise FolderNotSetError()
    with publish_lock(settings.paths.data_dir), transaction(conn):
        active = active_career_id(conn) == career_id
        report = publish_career(
            conn,
            career_id,
            career_key,
            fm_user_dir,
            settings.groups.published_sizes(),
            data_dir=settings.paths.data_dir,
            active=active,
        )
        if active:
            report = replace(report, deactivated=deactivate_packs(fm_user_dir, career_key))
    return report


def publish_active(conn: sqlite3.Connection, settings: Settings) -> PublishReport | None:
    """Show FM27 the active career's pack and no other; None if there is no career yet.

    Run after every sync: a save from another career switches the pack.
    """
    row = conn.execute(
        "SELECT id, key FROM careers WHERE id = ?", (active_career_id(conn),)
    ).fetchone()
    if row is None:
        return None
    return publish_pack(conn, settings, row["id"], row["key"])


def unpublish(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    unique_id: int,
    fm_user_dir: Path | None,
    *,
    active: bool = True,
) -> bool:
    """Remove a player's published picture and mapping; FM27 then shows its own photo again.

    The picture stops being current; the master and the database history are kept. The
    planner would queue an eligible player again, so the UI's "remove picture"
    pairs this with a "never" pin. A pack that is not `active` keeps mapping nothing.
    Returns whether anything was removed. Callers hold `publish_lock`. Raises
    FolderNotSetError without FM27's folder, before anything changes.
    """
    if fm_user_dir is None:
        raise FolderNotSetError()
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
    changed = write_config(folder, ids if active else (), fm_user_dir) if folder.exists() else False
    return existed or changed
