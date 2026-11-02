"""Find the datasets attached to a Kaggle notebook by their manifests, not by fixed paths.

Kaggle mounts a dataset at a path that depends on who owns it and how it was made (for our
private datasets ``/kaggle/input/datasets/<user>/<slug>/...``, sometimes one or two folders
deeper). So the cells and the card runner look for manifests instead:

* **the shared data** (RC-D1): the folder of a ``bundle_manifest.json`` that has a
  ``private/`` folder next to it;
* **a hand-off** (``E0_teacher_adapter``, ``E0_teacher_cache``, ``E2_seed1_adapter`` ...): the
  folder of a ``bundle_manifest.json`` whose path contains the hand-off's name. Case and ``-``
  against ``_`` do not matter, so the dataset slug ``e0-teacher-adapter`` matches
  ``E0_teacher_adapter`` too.

Exactly one folder must match; none or several stop with a message that says what to attach.
The root is ``/kaggle/input`` (``$MEDCOMPRESS_INPUT_ROOT`` overrides it, for tests).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from medcompress.utils.bundles import MANIFEST_NAME, BundleError

INPUT_ROOT_ENV = "MEDCOMPRESS_INPUT_ROOT"
DEFAULT_INPUT_ROOT = Path("/kaggle/input")
PRIVATE_DIR = "private"
SHARED_DATA_NAME = "medcompress-data"


class InputNotFound(BundleError):
    """No attached dataset (or more than one) matches what a run needs."""


def input_root(root: str | Path | None = None) -> Path:
    """Where Kaggle mounts the attached datasets."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get(INPUT_ROOT_ENV) or DEFAULT_INPUT_ROOT)


def bundle_folders(root: str | Path | None = None) -> list[Path]:
    """Every folder under the input root that holds a ``bundle_manifest.json`` (sorted)."""
    base = input_root(root)
    if not base.is_dir():
        return []
    found = [Path(d) for d, _, files in os.walk(base) if MANIFEST_NAME in files]
    return sorted(found)


def is_shared_data(folder: Path) -> bool:
    """The shared data carry the private RC-D1 files next to their manifest."""
    return (Path(folder) / PRIVATE_DIR).is_dir()


def _normalise(text: str) -> str:
    return text.lower().replace("-", "_")


def _one(found: Sequence[Path], what: str, attach: str, root: Path) -> Path:
    if not found:
        raise InputNotFound(f"{what} is not attached: no {MANIFEST_NAME} for it under {root}. "
                            f"Add Input: {attach}")  # fmt: skip
    if len(found) > 1:
        listed = ", ".join(str(p) for p in found)
        raise InputNotFound(f"{what} is attached more than once ({listed}): keep one")
    return found[0]


def find_shared_data(root: str | Path | None = None) -> Path:
    """The folder of the shared data's manifest (one ``private/`` next to it)."""
    base = input_root(root)
    found = [f for f in bundle_folders(base) if is_shared_data(f)]
    return _one(found, "the shared data", f"the private dataset {SHARED_DATA_NAME}", base)


def find_handoff(name: str, root: str | Path | None = None) -> Path:
    """The folder of the manifest whose path (below the input root) contains ``name``."""
    base = input_root(root)
    wanted = _normalise(name)
    found = [
        f for f in bundle_folders(base)
        if not is_shared_data(f) and wanted in _normalise(f.relative_to(base).as_posix())
    ]  # fmt: skip
    return _one(found, f"the hand-off {name}", f"the private dataset made from {name}", base)


def find_handoffs(root: str | Path | None = None) -> list[Path]:
    """Every attached hand-off (every manifest folder that is not the shared data)."""
    return [f for f in bundle_folders(root) if not is_shared_data(f)]
