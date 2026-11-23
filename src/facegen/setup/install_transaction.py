"""Replace a release's files as one recoverable transaction.

The old contents are all backed up before publishing a journal. Only then can the
installation change. Restoring that journal is idempotent, including if repair itself
is interrupted. A standalone copy in data/ can repair even a partially replaced Python
package. Callers hold the installation lock throughout recovery and replacement.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __name__ == "__main__" and not __package__:
    # Start.bat runs the copies kept together during a transaction.
    import locking
    import release_payload as payload
else:
    from facegen.platform import locking
    from facegen.setup import release_payload as payload


@dataclass(frozen=True, slots=True)
class Applied:
    written: int
    unchanged: int
    removed: int
    backup: Path


def _repair_dir(root: Path) -> Path:
    return root / "data" / "update-recovery"


def _copy_into(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f".{target.name}.update-part")
    try:
        shutil.copy2(source, part)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)


def _same(source: Path, target: Path) -> bool:
    return target.is_file() and filecmp.cmp(source, target, shallow=False)


def _finish(root: Path) -> None:
    folder = _repair_dir(root)
    (folder / "pending.json").unlink()
    # Once the journal is gone, cleanup is optional. Never remove unrelated data.
    for name in ("repair.py", "locking.py", "release_payload.py"):
        try:
            (folder / name).unlink(missing_ok=True)
        except OSError:
            pass


def _journal_paths(entries: list[str]) -> list[PurePosixPath]:
    paths = [payload.safe_relative(name) for name in entries]
    if any(not rel.parts or payload.never_written(rel) for rel in paths):
        raise payload.UpdateError("the repair journal names a protected path")
    return paths


def recover_update(root: Path) -> bool:
    marker = _repair_dir(root) / "pending.json"
    if not marker.exists():
        return False
    try:
        journal = json.loads(marker.read_text(encoding="utf-8"))
        backup = Path(journal["backup"])
        originals = _journal_paths(journal["originals"])
        created = _journal_paths(journal["created"])
        # A release file can have been a directory (or a parent file) in the
        # previous version. Delete only files the transaction actually introduced.
        for rel in (*created, *originals):
            target = root.joinpath(*rel.parts)
            part = target.with_name(f".{target.name}.update-part")
            if part.is_file():
                part.unlink()
        for rel in created:
            target = root.joinpath(*rel.parts)
            if target.is_file():
                target.unlink()
        _remove_empty_dirs(root, created)
        for rel in originals:
            original = backup.joinpath(*rel.parts)
            target = root.joinpath(*rel.parts)
            if not _same(original, target):
                _copy_into(original, target)
        _finish(root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise payload.UpdateError(f"could not restore the interrupted update: {exc}") from exc
    return True


def _prepare_journal(
    root: Path, backup: Path, originals: list[PurePosixPath], created: list[PurePosixPath]
) -> None:
    folder = _repair_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    for module, name in (
        (__file__, "repair.py"),
        (locking.__file__, "locking.py"),
        (payload.__file__, "release_payload.py"),
    ):
        _copy_into(Path(module), folder / name)
    temporary = folder / "pending.tmp"
    temporary.write_text(
        json.dumps(
            {
                "backup": str(backup.resolve()),
                "originals": [str(rel) for rel in originals],
                "created": [str(rel) for rel in created],
            }
        ),
        encoding="utf-8",
    )
    temporary.replace(folder / "pending.json")


def apply_payload(
    root: Path, staged: Path, files: Sequence[PurePosixPath], backup: Path
) -> Applied:
    recover_update(root)
    new = set(files)
    if any(not payload.in_payload(rel) for rel in new):
        raise payload.UpdateError("the new files include excluded paths")
    gone = [rel for rel in payload.managed_files(root) if rel not in new]
    version = PurePosixPath("pyproject.toml")
    changed = [
        rel
        for rel in sorted(new, key=lambda r: (r == version, r))
        if not _same(staged.joinpath(*rel.parts), root.joinpath(*rel.parts))
    ]
    originals = [rel for rel in changed if root.joinpath(*rel.parts).is_file()]
    existing = set(originals)
    created = [rel for rel in changed if rel not in existing]
    originals.extend(gone)
    if backup.exists():
        shutil.rmtree(backup)
    for rel in originals:
        target = backup.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root.joinpath(*rel.parts), target)
    _prepare_journal(root, backup, originals, created)
    try:
        for rel in gone:
            root.joinpath(*rel.parts).unlink()
        _remove_empty_dirs(root, gone)
        # The project version is the final write, after the entire new payload is in place.
        for rel in changed:
            _copy_into(staged.joinpath(*rel.parts), root.joinpath(*rel.parts))
        _finish(root)
    except OSError as exc:
        recover_update(root)
        raise payload.UpdateError(
            f"the files could not be replaced, so nothing changed: {exc}"
        ) from exc
    return Applied(len(changed), len(new) - len(changed), len(gone), backup)


def _remove_empty_dirs(root: Path, removed: Iterable[PurePosixPath]) -> None:
    dirs = {rel.parent for rel in removed}
    for rel in sorted(dirs, key=lambda d: len(d.parts), reverse=True):
        path = root.joinpath(*rel.parts)
        while len(rel.parts) > 1 and rel.parts[0] in payload.MANAGED_DIRS:
            try:
                path.rmdir()
            except OSError:
                break
            rel, path = rel.parent, path.parent


if __name__ == "__main__":
    install_root = Path(sys.argv[1]).resolve()
    try:
        with locking.installation_lock(install_root):
            recover_update(install_root)
    except (OSError, locking.LockBusy, payload.UpdateError) as error:
        sys.exit(f"Cannot repair Kitshot while the installation is in use or unavailable: {error}")
