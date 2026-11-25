"""Publish portraits only inside <FM27 user dir>/graphics/kitshot/<career key>/.

Each pack has <unique_id>.png portraits and config.xml mappings for numeric and r- IDs.
Portrait height is the group's published size; non-3:4 masters are cropped centrally.
FM27 loads changes on Reload Skin or restart.

Before Phase 17o the packs were in graphics/facegen/. The next publish of the active pack
writes every picture into graphics/kitshot/ (the stored published paths no longer match),
then removes that old folder of ours (`remove_legacy_packs`).

Newgen IDs repeat across careers, so only the career with the newest save maps pictures.
Other packs keep their PNGs with empty mappings. `publish_pack` and `publish_active` hold
`publish_lock` to serialize worker/sync writes. Under it they write the pictures that are due
first (`stage_pictures`, each chunk recorded in a short transaction), then bring the pack in
line in one transaction that holds SQLite's write lock only for checks and mappings. Every path
is checked before writing or deleting; atomic replacement and skipping unchanged files make
publishing idempotent.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from kitshot.core.settings_values import ASPECTS
from kitshot.persistence import publication
from kitshot.persistence.db import active_career_id, transaction
from kitshot.persistence.image_store import mark_pictures_changed
from kitshot.platform import native
from kitshot.platform.locking import file_lock
from kitshot.render.picture import ASPECT, Box, aspect_box, crop_to_aspect, published_size

if TYPE_CHECKING:
    from kitshot.core.settings_values import Settings

RELOAD_HINT = "In FM27: Settings -> Reload Skin (스킨 다시 불러오기) to load the pictures."


class PublishError(RuntimeError):
    """Publishing needs a valid destination under FM27's graphics/kitshot/."""


class FolderNotSetError(PublishError):
    """FM27's user folder is neither set nor found: there is nowhere to publish to."""

    def __init__(self) -> None:
        from kitshot.persistence.settings import folder_not_set_message

        super().__init__(folder_not_set_message())


PACKS = "kitshot"
"""Our folder in FM27's graphics folder (AGENTS.md rule 2)."""
LEGACY_PACKS = "facegen"
"""Our folder's name before Phase 17o: removed once the active pack is in PACKS."""


def packs_root(fm_user_dir: Path) -> Path:
    return fm_user_dir / "graphics" / PACKS


def legacy_packs_root(fm_user_dir: Path) -> Path:
    return fm_user_dir / "graphics" / LEGACY_PACKS


def pack_dir(fm_user_dir: Path, career_key: str) -> Path:
    """This career's pack folder. Refuses any key that would leave graphics/kitshot/."""
    root = packs_root(fm_user_dir)
    if not career_key or career_key in (".", "..") or any(c in career_key for c in "/\\:"):
        raise PublishError(f"bad career key {career_key!r}")
    folder = root / career_key
    _guard(folder, fm_user_dir)
    return folder


def _guard(path: Path, fm_user_dir: Path) -> None:
    # Resolve the configured FM directory, then keep writes under its graphics/kitshot/.
    root = packs_root(fm_user_dir.resolve())
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
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
    # The publish lock is held. Discard any interrupted write before opening a fresh file.
    tmp.unlink(missing_ok=True)
    stream = tmp.open("xb")
    try:
        with stream:
            stream.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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


def published_bytes(
    master: Path, long_edge: int, aspect: str = "3:4", *, rust: bool = True
) -> bytes:
    """The master in its shape (`aspect`: a 3:4 portrait or a 1:1 square) `long_edge` pixels
    high, as PNG bytes.

    The Rust encoder is used when it is built (see `published_pictures`), unless `rust` is
    False.
    """
    ratio = ASPECTS[aspect]
    if rust and native.ext is not None:
        return native.ext.encode_portrait(
            master, _crop_box(master, ratio), published_size(long_edge, ratio)
        )
    return _pillow_bytes(master, long_edge, ratio)


def published_pictures(
    items: Sequence[tuple[Path, int] | tuple[Path, int, str]], *, rust: bool = True
) -> Iterator[bytes]:
    """`published_bytes` for each (master, long edge[, shape]), in order; 3:4 by default.

    With the Rust extension (native.py), a chunk of pictures at a time is decoded, cropped,
    scaled with Lanczos-3 and saved as an optimised 8-bit RGB PNG on all cores. Without it,
    Pillow does the same one picture at a time. The pictures are not byte-identical: on the
    58 real masters, 99.98% of channel values were within one level of Pillow's (fixed-point
    rounding), and a few pixels on hard edges differed by up to 15, because the Rust resize
    runs its vertical pass before the horizontal one and so clips Lanczos's overshoot at a
    different stage (docs/research.md). The quality and content checks look at the master,
    not at this copy.
    """
    shaped = [(item[0], item[1], ASPECTS[item[2] if len(item) > 2 else "3:4"]) for item in items]
    if not rust or native.ext is None:
        for master, long_edge, ratio in shaped:
            yield _pillow_bytes(master, long_edge, ratio)
        return
    for start in range(0, len(shaped), _CHUNK):
        jobs = [
            (m, _crop_box(m, ratio), published_size(e, ratio))
            for m, e, ratio in shaped[start : start + _CHUNK]
        ]
        yield from native.ext.encode_portraits(jobs)


def _crop_box(master: Path, ratio: tuple[int, int] = ASPECT) -> Box:
    """The box of the master in its shape, from its header alone."""
    from PIL import Image

    with Image.open(master) as im:
        return aspect_box(im.width, im.height, ratio)


def _pillow_bytes(master: Path, long_edge: int, ratio: tuple[int, int] = ASPECT) -> bytes:
    """The Python version of the encoder."""
    import io

    from PIL import Image

    with Image.open(master) as im:
        img = crop_to_aspect(im.convert("RGB"), ratio)
    size = published_size(long_edge, ratio)
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


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
    legacy_removed: bool = False
    """Our pack folder from before Phase 17o (graphics/facegen/) was just removed."""

    @property
    def changed(self) -> bool:
        return bool(self.written or self.config_changed or self.deactivated or self.legacy_removed)


@dataclass(frozen=True, slots=True)
class _Outstanding:
    """A current picture whose published copy is missing or out of date, with what its row
    said it is drawn from."""

    image_id: int
    master: Path
    dest: Path
    size: int
    aspect: str
    master_path: str
    group: str


@dataclass(frozen=True, slots=True)
class _Plan:
    """What a career's pack needs, from one read of its current pictures."""

    in_place: tuple[int, ...]
    """Image IDs whose published copy is in place."""
    missing: tuple[int, ...]
    mapped: tuple[int, ...]
    outstanding: tuple[_Outstanding, ...]


def _plan(
    conn: sqlite3.Connection,
    career_id: int,
    folder: Path,
    fm_user_dir: Path,
    sizes: dict[str, int],
    data_dir: Path,
) -> _Plan:
    rows = conn.execute(
        "SELECT id, unique_id, group_name, master_path, published_path, aspect FROM images"
        " WHERE career_id = ? AND is_current = 1 ORDER BY unique_id",
        (career_id,),
    ).fetchall()
    in_place: list[int] = []
    missing: list[int] = []
    mapped: list[int] = []
    outstanding: list[_Outstanding] = []
    for r in rows:
        dest = folder / f"{r['unique_id']}.png"
        if r["published_path"] == str(dest) and dest.is_file():
            in_place.append(r["id"])
            mapped.append(r["unique_id"])
            continue
        master = Path(r["master_path"]) if r["master_path"] else None
        if master is not None and not master.is_absolute():
            master = data_dir / master
        if master is None or not master.is_file():
            missing.append(r["unique_id"])
            continue
        _guard(dest, fm_user_dir)
        size = sizes[r["group_name"]]
        outstanding.append(
            _Outstanding(
                r["id"], master, dest, size, r["aspect"], r["master_path"], r["group_name"]
            )
        )
        mapped.append(r["unique_id"])
    return _Plan(tuple(in_place), tuple(missing), tuple(mapped), tuple(outstanding))


def _encoded(items: Sequence[_Outstanding]) -> Iterator[tuple[_Outstanding, bytes]]:
    """Each picture with its published bytes, encoded together so the Rust encoder can work on
    several at once."""
    pictures = published_pictures([(i.master, i.size, i.aspect) for i in items])
    return zip(items, pictures, strict=True)


def stage_pictures(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    fm_user_dir: Path,
    sizes: dict[str, int],
    *,
    data_dir: Path,
) -> frozenset[int]:
    """Write the career's pictures that are not in place yet, outside any long transaction;
    returns the IDs of the images written and recorded.

    The caller holds `publish_lock` and no transaction. Encoding takes up to 0.4 s a picture
    without the Rust encoder, and a whole pack can be due at once (1490 pictures after the
    Phase 17o move): inside the publish transaction that held SQLite's write lock for ten
    minutes, and syncs waiting behind it failed after their busy timeout (2026-11-25). Each
    chunk is recorded in a short transaction of its own, and only while its row still asks for
    the picture that was written: one picked or restored meanwhile stays due, and
    `publish_career` puts it in place. Files written but not recorded (an error in the chunk)
    are written again by the next publish.
    """
    assert not conn.in_transaction, "staging records each chunk in a transaction of its own"
    folder = pack_dir(fm_user_dir, career_key)
    plan = _plan(conn, career_id, folder, fm_user_dir, sizes, data_dir)
    if not plan.outstanding:
        return frozenset()
    folder.mkdir(parents=True, exist_ok=True)
    recorded: set[int] = set()
    chunk: list[_Outstanding] = []

    def record() -> None:
        with transaction(conn):
            now = {
                item.image_id
                for item in chunk
                if conn.execute(
                    "UPDATE images SET published_path = ? WHERE id = ? AND is_current = 1"
                    " AND master_path IS ? AND group_name IS ? AND aspect IS ?",
                    (str(item.dest), item.image_id, item.master_path, item.group, item.aspect),
                ).rowcount
            }
            if now:
                mark_pictures_changed(conn, career_id)
        recorded.update(now)
        chunk.clear()

    for item, data in _encoded(plan.outstanding):
        _write_atomic(item.dest, data)
        chunk.append(item)
        if len(chunk) == _CHUNK:
            record()
    if chunk:
        record()
    return frozenset(recorded)


def publish_career(
    conn: sqlite3.Connection,
    career_id: int,
    career_key: str,
    fm_user_dir: Path,
    sizes: dict[str, int],
    *,
    data_dir: Path,
    active: bool = True,
    staged: Collection[int] = (),
) -> PublishReport:
    """Bring the career's pack in line with its current pictures.

    `sizes` maps group to published size (the picture's height) in pixels. A picture already
    published to its place is left alone; config.xml is rewritten only if the mapping changes.
    A pack that is not `active` gets its pictures but a config.xml that maps nothing.
    `staged` are images `stage_pictures` wrote for this publish: in place now, counted as
    written.
    """
    folder = pack_dir(fm_user_dir, career_key)
    plan = _plan(conn, career_id, folder, fm_user_dir, sizes, data_dir)
    if plan.outstanding:
        folder.mkdir(parents=True, exist_ok=True)
    for item, data in _encoded(plan.outstanding):
        _write_atomic(item.dest, data)
        conn.execute(
            "UPDATE images SET published_path = ? WHERE id = ?", (str(item.dest), item.image_id)
        )
    if plan.outstanding:
        mark_pictures_changed(conn, career_id)
    restaged = len(set(plan.in_place).intersection(staged))
    shown = plan.mapped if active else ()
    config_changed = (
        write_config(folder, shown, fm_user_dir) if plan.mapped or folder.exists() else False
    )
    return PublishReport(
        folder,
        len(plan.outstanding) + restaged,
        len(plan.in_place) - restaged,
        plan.missing,
        config_changed,
        len(shown),
        active,
    )


def deactivate_packs(fm_user_dir: Path, keep: str | None) -> tuple[str, ...]:
    """Make every pack but `keep` map nothing; their pictures stay. Returns those changed.

    Only folders under graphics/kitshot/ that have a config.xml are touched.
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


def remove_packs(fm_user_dir: Path, data_dir: Path) -> bool:
    """Delete graphics/kitshot/ and everything in it, which the uninstaller offers: FM27
    shows its own photos again from its next start or Reload Skin. Our folder from before
    Phase 17o, graphics/facegen/, goes too. Nothing else in graphics/ is touched (AGENTS.md
    rule 2). False when there was nothing to remove. A later publish writes every picture
    again, as their files are gone.

    Refuses (PublishError) when FM27's graphics folder leads elsewhere (a junction or a
    symbolic link): publishing refuses to write there too, so a folder of ours found there
    is not ours. A link or junction with our name is removed, not what it leads to."""
    with publish_lock(data_dir):
        removed = _remove_root(packs_root(fm_user_dir), fm_user_dir)
        return _remove_root(legacy_packs_root(fm_user_dir), fm_user_dir) or removed


def remove_legacy_packs(fm_user_dir: Path) -> bool:
    """Delete our pack folder from before Phase 17o (graphics/facegen/), as `remove_packs`
    would; False when there is none. Callers hold `publish_lock` and have just published the
    active pack into graphics/kitshot/."""
    return _remove_root(legacy_packs_root(fm_user_dir), fm_user_dir)


def _remove_root(root: Path, fm_user_dir: Path) -> bool:
    if not root.exists() and not _is_link(root):
        return False
    graphics, expected = root.parent, fm_user_dir.resolve() / "graphics"
    if graphics.resolve() != expected:
        raise PublishError(
            f"refusing to delete outside {expected}: {graphics} leads to {graphics.resolve()}"
        )
    if _is_link(root):
        root.unlink()
        return True
    shutil.rmtree(root)
    return True


def _is_link(path: Path) -> bool:
    """A symbolic link or (on Windows) a junction: removing it leaves what it leads to."""
    return path.is_symlink() or path.is_junction()


@contextmanager
def publish_lock(data_dir: Path) -> Iterator[None]:
    """One writer of FM27's packs at a time, across threads and processes.

    Take it before any database transaction that publishing runs in, never inside one.
    """
    with file_lock(data_dir / "publish.lock"):
        yield


def _fm_user_dir(settings: Settings) -> Path:
    fm_user_dir = settings.paths.fm_user_dir
    if fm_user_dir is None:
        raise FolderNotSetError()
    return fm_user_dir


def _stage(
    conn: sqlite3.Connection, settings: Settings, career_id: int, career_key: str
) -> frozenset[int]:
    """`stage_pictures` for a career with these settings, before the publish transaction."""
    return stage_pictures(
        conn,
        career_id,
        career_key,
        _fm_user_dir(settings),
        settings.groups.published_sizes(),
        data_dir=settings.paths.data_dir,
    )


def _publish_locked(
    conn: sqlite3.Connection,
    settings: Settings,
    career_id: int,
    career_key: str,
    *,
    active: bool,
    staged: Collection[int] = (),
) -> PublishReport:
    """The pack brought in line inside the publish transaction: pictures staged before it are
    already in place, so it reads, checks and writes mappings, and writes only pictures that
    became due since."""
    fm_user_dir = _fm_user_dir(settings)
    report = publish_career(
        conn,
        career_id,
        career_key,
        fm_user_dir,
        settings.groups.published_sizes(),
        data_dir=settings.paths.data_dir,
        active=active,
        staged=staged,
    )
    if active:
        report = replace(report, deactivated=deactivate_packs(fm_user_dir, career_key))
    return report


def _remove_legacy_after_publish(report: PublishReport, settings: Settings) -> PublishReport:
    """`remove_legacy_packs` once the active pack is in place and committed, still under the
    publish lock but outside the transaction: removing a whole old pack takes a while. A
    failure is only logged: the new pack is published, and the next publish tries again."""
    if not report.active:
        return report
    fm_user_dir = _fm_user_dir(settings)
    try:
        removed = remove_legacy_packs(fm_user_dir)
    except (OSError, PublishError) as exc:
        logging.getLogger("kitshot").warning(
            "the old pack folder %s could not be removed: %s", legacy_packs_root(fm_user_dir), exc
        )
        return report
    return replace(report, legacy_removed=removed)


def publish_pack(
    conn: sqlite3.Connection, settings: Settings, career_id: int, career_key: str
) -> PublishReport:
    """Publish one career's pictures. If it is active, unmap every other career's pack.

    The pictures are written first, outside the transaction (`stage_pictures`); the
    transaction then holds SQLite's write lock only to check them and write the mappings."""
    with publish_lock(settings.paths.data_dir):
        staged = _stage(conn, settings, career_id, career_key)
        with transaction(conn):
            active = active_career_id(conn) == career_id
            report = _publish_locked(
                conn, settings, career_id, career_key, active=active, staged=staged
            )
        return _remove_legacy_after_publish(report, settings)


def publish_active(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    read_settings: Callable[[], Settings] | None = None,
) -> PublishReport | None:
    """Show FM27 the active career's pack and no other; None if there is no career yet.

    Run after every sync: a save from another career switches the pack. Its pictures are
    written first, outside the transaction, as in `publish_pack`.
    """

    def active_career() -> sqlite3.Row | None:
        return conn.execute(
            "SELECT id, key FROM careers WHERE id = ?", (active_career_id(conn),)
        ).fetchone()

    with publish_lock(settings.paths.data_dir):
        before = active_career()
        staged: frozenset[int] = frozenset()
        if before is not None:
            current = read_settings() if read_settings is not None else settings
            staged = _stage(conn, current, before["id"], before["key"])
        with transaction(conn):
            row = active_career()
            if row is None:
                return None
            # Settings edits and career changes cannot race the writer while it applies and
            # acknowledges the committed sync. An ordinary worker publishing an older job
            # does not acknowledge this marker with its own snapshot of the settings. A
            # career that became active while pictures were staged has its own written here.
            if read_settings is not None:
                settings = read_settings()
            if before is None or row["id"] != before["id"]:
                staged = frozenset()
            report = _publish_locked(
                conn, settings, row["id"], row["key"], active=True, staged=staged
            )
            publication.finish(conn)
        return _remove_legacy_after_publish(report, settings)


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
