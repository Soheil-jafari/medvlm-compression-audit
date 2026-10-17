"""Image-disjoint splits and nested calibration subsets.

Splits are made once, with the single seed recorded in the preregistration
(``seeds.data_split``). Kvasir-VQA-x1 and Kvasir-SEG have no patient IDs, and the official
Kvasir-VQA-x1 split is by question, not by image (3,821 of its 4,058 test images are also
training images; RC-D1 run 1). Since the D02 amendment of 2026-10-18 all images of both
official splits are re-split by image: near-duplicate images are grouped first
(``dedup.near_duplicate_groups``) and whole groups are assigned, stratified by finding and
by Kvasir-SEG mask (:func:`make_image_disjoint_splits`), 70 / 15 / 15 % of the images.

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


def assign_groups_stratified(
    group_sizes: Mapping[str, int],
    group_labels: Mapping[str, frozenset[str]],
    ratios: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits, stratified by labels and balanced by images.

    The method (D02 amendment 2026-10-18), a group version of iterative stratification:

    1. Groups are put in one order: sorted, then shuffled with ``seed``.
    2. Labels are taken from the rarest (fewest groups) to the most common. A label carried by
       at least as many groups as there are splits first gets one group in every split that
       has none yet, smallest split first (validation, test, then train). Its remaining
       unassigned groups each go to the split whose count of that label is furthest below
       ``ratio x the label's groups``; ties go to the split furthest below its image target,
       then to the first split named.
    3. Groups without a label fill the image targets: each goes to the split furthest below
       ``ratio x all images``, relative to that target.

    ``group_sizes`` is the number of images in each group, ``group_labels`` its labels (for
    example the answer values of the finding question and "masked"). Every group lands in
    exactly one split; the result depends only on the inputs and ``seed``.
    """
    if not ratios or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must be non-empty and sum to 1.")
    if any(r <= 0 for r in ratios.values()):
        raise ValueError("Split ratios must be positive.")
    names = list(ratios)
    order = sorted(group_sizes)
    random.Random(seed).shuffle(order)
    total = sum(group_sizes.values())
    target = {s: ratios[s] * total for s in names}
    images = dict.fromkeys(names, 0)
    by_label: dict[str, list[str]] = {}
    for group in order:
        for label in sorted(group_labels.get(group, ())):
            by_label.setdefault(label, []).append(group)
    count = {(label, s): 0 for label in by_label for s in names}
    assigned: dict[str, str] = {}

    def place(group: str, split: str) -> None:
        assigned[group] = split
        images[split] += group_sizes[group]
        for label in group_labels.get(group, ()):
            count[label, split] += 1

    smallest_first = sorted(names, key=lambda s: (ratios[s], names.index(s)))
    for label in sorted(by_label, key=lambda lab: (len(by_label[lab]), lab)):
        total_label = len(by_label[label])
        if total_label >= len(names):
            for split in smallest_first:
                pending = [g for g in by_label[label] if g not in assigned]
                if count[label, split] == 0 and pending:
                    place(pending[0], split)
        for group in by_label[label]:
            if group in assigned:
                continue
            split = max(
                names,
                key=lambda s: (
                    ratios[s] * total_label - count[label, s],
                    target[s] - images[s],
                    -names.index(s),
                ),
            )
            place(group, split)
    for group in order:
        if group not in assigned:
            split = max(names, key=lambda s: ((target[s] - images[s]) / target[s], -names.index(s)))
            place(group, split)
    return assigned


def make_image_disjoint_splits(
    samples: Sequence[VQASample],
    image_group: Mapping[str, str],
    group_labels: Mapping[str, frozenset[str]],
    ratios: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, list[VQASample]]:
    """Split every sample by its image's near-duplicate group (:func:`assign_groups_stratified`).

    ``image_group`` maps each image path to its group (``dedup.near_duplicate_groups``); every
    question of an image, and every image of a group, lands in the same split. The samples
    keep their order and carry their new ``split`` name.
    """
    images = sorted({s.image_path for s in samples})
    missing = [img for img in images if img not in image_group]
    if missing:
        raise ValueError(f"{len(missing)} images have no group (first: {missing[0]})")
    sizes: dict[str, int] = {}
    for image in images:
        sizes[image_group[image]] = sizes.get(image_group[image], 0) + 1
    assigned = assign_groups_stratified(sizes, group_labels, ratios, seed=seed)
    out: dict[str, list[VQASample]] = {name: [] for name in ratios}
    for sample in samples:
        name = assigned[image_group[sample.image_path]]
        out[name].append(dataclasses.replace(sample, split=name))
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
