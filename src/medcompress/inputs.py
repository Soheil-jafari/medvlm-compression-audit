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

Not every manifest folder is a hand-off: the shared dataset also carries RC-D1's own bundles
(``rc_d1_embed``, kind ``near_duplicates``), which no card reads as a hand-off. A folder is a
hand-off only if its manifest's kind is a hand-off kind and it does not lie inside the shared
dataset (:func:`find_handoffs`); the others are skipped with a note. Which hand-offs a card then
attaches is the runner's choice (``card_runner.select_handoffs``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from medcompress.utils.bundles import HANDOFF_KINDS, MANIFEST_NAME, BundleError

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


def manifest_kind(folder: Path) -> str | None:
    """The ``kind`` a manifest declares (None if it cannot be read); the files are not checked."""
    try:
        document = json.loads((Path(folder) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    kind = document.get("kind") if isinstance(document, dict) else None
    return str(kind) if kind is not None else None


def _inside(folder: Path, parents: Sequence[Path]) -> bool:
    return any(folder != p and folder.is_relative_to(p) for p in parents)


def classify_bundles(root: str | Path | None = None) -> tuple[list[Path], list[str]]:
    """``(hand-offs, notes)``: every manifest folder that is a hand-off, and why each other one
    is not (the shared data itself, a bundle inside the shared dataset, or a kind no card reads
    as a hand-off, such as RC-D1-embed's ``near_duplicates``). An unreadable manifest is kept
    as a hand-off, so its check reports the problem."""
    folders = bundle_folders(root)
    shared = [f for f in folders if is_shared_data(f)]
    shared_datasets = [f.parent for f in shared]  # RC-D1 puts its bundles next to the data
    handoffs, notes = [], []
    for folder in folders:
        if folder in shared:
            continue
        kind = manifest_kind(folder)
        if kind is not None and kind not in HANDOFF_KINDS:
            notes.append(f"{folder}: kind {kind!r} is not a hand-off; skipped")
        elif _inside(folder, shared) or (kind is None and _inside(folder, shared_datasets)):
            notes.append(f"{folder}: part of the shared dataset, not a hand-off; skipped")
        else:
            handoffs.append(folder)
    return handoffs, notes


def find_handoffs(root: str | Path | None = None) -> list[Path]:
    """Every attached hand-off (see :func:`classify_bundles`)."""
    return classify_bundles(root)[0]
