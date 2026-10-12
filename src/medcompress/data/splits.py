"""Patient-level splits and nested calibration subsets.

Splits are made once, with the single seed recorded in the preregistration
(``seeds.data_split``), at patient level so no patient appears in two splits.
Kvasir-VQA-x1 and Kvasir-SEG have no patient IDs, so the grouping key falls back to
the image (``image_path``); near-duplicate images are handled separately by
``data/dedup.py``. For this project the official Kvasir-VQA-x1 test split is halved
by image into validation and test (D02).

Calibration subsets for pruning-unit scoring are drawn from the training split
only and are nested (the 32-image subset is inside the 64, and so on) so that
ablation A4 varies size alone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medcompress.data.base import VQASample

CALIBRATION_SIZES: tuple[int, ...] = (32, 64, 128, 256)


def group_key(sample: VQASample) -> str:
    """Return the split-grouping key: the patient ID, or the image if there is none."""
    if sample.patient_id is not None:
        return f"patient:{sample.patient_id}"
    return f"image:{sample.image_path}"


def make_patient_splits(
    samples: Sequence[VQASample], ratios: Mapping[str, float], *, seed: int
) -> dict[str, list[VQASample]]:
    """Assign every sample to a split so that each patient is in exactly one split.

    ``ratios`` maps split names (e.g. train/val/test, or val/test) to fractions of
    **groups** (patients, or images when there are no patient IDs) and must sum to
    1. Groups are shuffled with ``seed`` after sorting, so the result does not
    depend on input order. Returned samples carry their ``split`` name. Datasets
    with official patient-disjoint splits use those instead (recorded in the data
    card).
    """
    if not ratios or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must be non-empty and sum to 1.")
    if any(r < 0 for r in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    groups: dict[str, list[VQASample]] = {}
    for sample in samples:
        groups.setdefault(group_key(sample), []).append(sample)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    names = list(ratios)
    out: dict[str, list[VQASample]] = {name: [] for name in names}
    start = 0
    cumulative = 0.0
    for i, name in enumerate(names):
        cumulative += ratios[name]
        end = n if i == len(names) - 1 else round(cumulative * n)
        for key in keys[start:end]:
            out[name].extend(dataclasses.replace(s, split=name) for s in groups[key])
        start = end
    return out


def check_patient_disjoint(splits: Mapping[str, Sequence[VQASample]]) -> list[str]:
    """Return grouping keys (patients, or images) found in more than one split (empty = OK)."""
    seen: dict[str, set[str]] = {}
    for name, members in splits.items():
        for sample in members:
            seen.setdefault(group_key(sample), set()).add(name)
    return sorted(key for key, names in seen.items() if len(names) > 1)


def make_calibration_subset(
    train: Sequence[VQASample], size: int, *, seed: int, stratify_by_finding: bool = True
) -> list[VQASample]:
    """Return a calibration subset of the training split for pruning-unit scoring.

    The subset holds ``size`` distinct **images** (one representative sample per
    image). Images are ordered once per seed; with ``stratify_by_finding`` the
    first positive image of every finding (in sorted finding order) is moved to the
    front, so every finding with a positive in the training split is covered when
    ``size`` allows. Because the order does not depend on ``size``, subsets for
    different sizes are nested (A4).

    Used by: E5, E6, E7, A1-A6 (``scripts/04_score_pruning_units.py``).
    """
    if any(s.split not in (None, "train") for s in train):
        raise ValueError("Calibration images must come from the training split only.")
    by_image: dict[str, VQASample] = {}
    for sample in sorted(train, key=lambda s: (s.image_path, s.sample_id)):
        by_image.setdefault(sample.image_path, sample)
    images = sorted(by_image)
    if size <= 0 or size > len(images):
        raise ValueError(f"size must be in 1..{len(images)}, got {size}")
    random.Random(seed).shuffle(images)
    if stratify_by_finding:
        findings = sorted(
            {f for s in by_image.values() for f, present in s.findings.items() if present}
        )
        front: list[str] = []
        for finding in findings:
            if any(by_image[image].findings.get(finding) for image in front):
                continue
            first = next(image for image in images if by_image[image].findings.get(finding))
            front.append(first)
        front_set = set(front)
        images = front + [image for image in images if image not in front_set]
    return [by_image[image] for image in images[:size]]


def split_fingerprint(splits: Mapping[str, Sequence[VQASample]]) -> str:
    """Return a SHA-256 over sorted (split, sample_id) pairs, recorded in the preregistration."""
    lines = sorted(
        f"{name}\t{sample.sample_id}" for name, members in splits.items() for sample in members
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
