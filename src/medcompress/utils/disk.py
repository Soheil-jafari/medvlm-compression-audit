"""Disk space on Kaggle (2026-11-13): RC-SMOKE's A1 job stopped with "No space left on device".

``/kaggle/working`` is the saved output of a Kaggle session (20 GB by Kaggle's documentation as
it is commonly quoted; the agent could not open kaggle.com to confirm it). Every large save
checks the free space first and stops with a clear message (:func:`require_free_disk`); hand-off
folders hard-link the checkpoint files instead of copying them (:func:`link_or_copy`), so a
pruned model is on disk once, not twice.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

GIB = 2**30
DISK_MARGIN = GIB // 2
"""Free space kept beyond a save (the manifests, logs and the allocator of the filesystem)."""


class DiskSpaceError(RuntimeError):
    """A save would not fit on the disk; nothing was written."""


def _existing(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def free_bytes(path: Path | str) -> int:
    """Free bytes on the filesystem that holds ``path`` (or its nearest existing parent)."""
    return int(shutil.disk_usage(_existing(Path(path))).free)


def tree_bytes(path: Path | str) -> int:
    """Bytes of a file or of every file below a folder (links followed once, not counted)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def require_free_disk(path: Path | str, need: int, what: str) -> None:
    """Stop before writing ``need`` bytes under ``path`` when the disk cannot hold them."""
    free = free_bytes(path)
    if free < need + DISK_MARGIN:
        raise DiskSpaceError(
            f"{what}: needs {need / GIB:.2f} GiB at {path}, {free / GIB:.2f} GiB free; nothing "
            "was written (Kaggle's /kaggle/working holds about 20 GB: run fewer runs per session)"
        )


def same_device(a: Path | str, b: Path | str) -> bool:
    """True when both paths (or their nearest existing parents) are on one filesystem."""
    return os.stat(_existing(Path(a))).st_dev == os.stat(_existing(Path(b))).st_dev


def link_or_copy(source: Path | str, target: Path | str) -> None:
    """Hard-link a file, or every file of a folder, to ``target`` (copy across filesystems)."""
    source, target = Path(source), Path(target)
    files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
    for file in files:
        dest = target if source.is_file() else target / file.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(file.resolve(), dest)
        except OSError:  # another filesystem, or links not allowed: copy
            shutil.copy2(file, dest)
