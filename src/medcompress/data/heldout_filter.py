"""The held-out filter of D02 amendment 6 (2026-10-28; post hoc, after the rule sweep of
amendment 5 found no clean rule within the chaining guard).

The groups and the split stay those of the swept rule ``full:0.92:or_crop:0.80``. After the
split, every validation and test image with a training image that is too similar is removed.
An image is removed if one of these holds:
- its nearest training image by the full-view cosine has a cosine of at least F;
- its nearest training image by the crop correlation has one of at least 0.70;
- a training image within 4 dHash bits has a crop correlation of at least 0.50. This is the
  agent's reading of "a dHash distance <= 4 bits together with crop >= 0.50": the same
  training image must meet both.

The filter never moves an image between splits and never touches the training split. Pairs
between validation and test are not filtered: they are only counted
(:func:`count_matching_pairs`).

The similarities of each held-out image to the training split (:func:`heldout_neighbours`)
depend on the split only, not on F, so the filter sweep computes them once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from medcompress.data.dedup import CropSettings, cross_similarity_matrix, distances_to

HELDOUT_SPLITS: tuple[str, ...] = ("val", "test")
FILTER_CROP = 0.70
FILTER_BITS = 4
FILTER_BITS_CROP = 0.50
FILTER_THRESHOLDS: tuple[float, ...] = (0.90, 0.88, 0.86, 0.84, 0.82, 0.80)
MIN_TEST_KEPT = 0.50
"""The selection's floor: the filtered test keeps at least half of the unfiltered test images."""
REASONS: tuple[str, ...] = ("cosine", "crop", "dhash")


@dataclass(frozen=True)
class HeldoutNeighbours:
    """For each validation and test image, its similarity to the training split.

    ``max_cosine`` is the highest full-view cosine; ``max_crop`` the highest crop correlation;
    ``close_crop`` the highest crop correlation among training images within ``bits`` dHash
    bits (``-inf`` if there is none). ``split_images`` is the split the values were computed
    on.
    """

    split_images: Mapping[str, tuple[str, ...]]
    max_cosine: Mapping[str, np.float32]
    max_crop: Mapping[str, np.float32]
    close_crop: Mapping[str, np.float32]
    bits: int

    def fits(self, split_images: Mapping[str, Sequence[str]], bits: int) -> bool:
        """Whether these values belong to this split (the same images in each split)."""
        return bits == self.bits and all(
            tuple(split_images.get(name, ())) == self.split_images.get(name, ())
            for name in ("train", *HELDOUT_SPLITS)
        )


def heldout_neighbours(
    split_images: Mapping[str, Sequence[str]],
    fingerprints: Mapping[str, str],
    signatures: Mapping[str, np.ndarray],
    crop: CropSettings,
    embeddings: Mapping[str, np.ndarray],
    images: Sequence[str],
    *,
    bits: int = FILTER_BITS,
) -> HeldoutNeighbours:
    """The similarities of every validation and test image to the training split."""
    index = {p: i for i, p in enumerate(images)}
    train = list(split_images.get("train", ()))
    train_fp = [fingerprints[p] for p in train]
    full = embeddings["full"]
    max_cos: dict[str, np.float32] = {}
    max_crop: dict[str, np.float32] = {}
    close: dict[str, np.float32] = {}
    for name in HELDOUT_SPLITS:
        held = list(split_images.get(name, ()))
        if not held:
            continue
        if not train:
            for path in held:
                max_cos[path] = max_crop[path] = close[path] = np.float32(-np.inf)
            continue
        crops = cross_similarity_matrix(
            [signatures[p] for p in train], [signatures[p] for p in held], crop
        )
        cosines = full[[index[p] for p in train]] @ full[[index[p] for p in held]].T
        for col, path in enumerate(held):
            near = distances_to(train_fp, fingerprints[path]) <= bits
            max_cos[path] = np.float32(cosines[:, col].max())
            max_crop[path] = np.float32(crops[:, col].max())
            close[path] = np.float32(crops[near, col].max()) if near.any() else np.float32(-np.inf)
    frozen = {name: tuple(split_images.get(name, ())) for name in ("train", *HELDOUT_SPLITS)}
    return HeldoutNeighbours(frozen, max_cos, max_crop, close, bits)


@dataclass(frozen=True)
class HeldoutFilter:
    """Remove a validation or test image whose training neighbour is too similar (module
    docstring). ``threshold`` is F, the full-view cosine level."""

    threshold: float
    crop: float = FILTER_CROP
    bits: int = FILTER_BITS
    bits_crop: float = FILTER_BITS_CROP

    def reasons(self, neighbours: HeldoutNeighbours, path: str) -> dict[str, bool]:
        """Which of the three conditions remove ``path`` (compared in float32, as the rule)."""
        return {
            "cosine": bool(neighbours.max_cosine[path] >= np.float32(self.threshold)),
            "crop": bool(neighbours.max_crop[path] >= np.float32(self.crop)),
            "dhash": bool(neighbours.close_crop[path] >= np.float32(self.bits_crop)),
        }

    def removed(self, neighbours: HeldoutNeighbours) -> dict[str, tuple[str, ...]]:
        """The images to remove, per held-out split, in path order."""
        return {
            name: tuple(
                p for p in sorted(neighbours.split_images.get(name, ()))
                if any(self.reasons(neighbours, p).values())
            )
            for name in HELDOUT_SPLITS
        }  # fmt: skip

    def joins(self, cosine: np.ndarray, crop: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """Which pairs meet one of the three conditions (for the val/test count)."""
        return (
            (cosine >= np.float32(self.threshold))
            | (crop >= np.float32(self.crop))
            | ((bits <= self.bits) & (crop >= np.float32(self.bits_crop)))
        )

    def code(self) -> dict[str, float]:
        """The filter as numbers (results files)."""
        return {
            "threshold": float(self.threshold),
            "crop": float(self.crop),
            "bits": float(self.bits),
            "bits_crop": float(self.bits_crop),
        }

    @property
    def label(self) -> str:
        """``0.86`` (used in the result metric names and the sheet names)."""
        return f"{self.threshold:.2f}"


def count_matching_pairs(
    first: Sequence[str],
    second: Sequence[str],
    heldout: HeldoutFilter,
    fingerprints: Mapping[str, str],
    signatures: Mapping[str, np.ndarray],
    crop: CropSettings,
    embeddings: Mapping[str, np.ndarray],
    images: Sequence[str],
) -> int:
    """How many (first, second) pairs meet one of the filter's conditions: the validation and
    test pairs that the filter does not touch (reported only)."""
    if not first or not second:
        return 0
    index = {p: i for i, p in enumerate(images)}
    full = embeddings["full"]
    crops = cross_similarity_matrix(
        [signatures[p] for p in first], [signatures[p] for p in second], crop
    )
    cosines = full[[index[p] for p in first]] @ full[[index[p] for p in second]].T
    first_fp = [fingerprints[p] for p in first]
    bits = np.stack([distances_to(first_fp, fingerprints[p]) for p in second], axis=1)
    return int(heldout.joins(cosines, crops, bits).sum())


def kept_fraction(kept: int, total: int) -> float:
    """``kept / total`` (nan if ``total`` is 0)."""
    return kept / total if total else math.nan


__all__ = [
    "FILTER_BITS",
    "FILTER_BITS_CROP",
    "FILTER_CROP",
    "FILTER_THRESHOLDS",
    "HELDOUT_SPLITS",
    "MIN_TEST_KEPT",
    "REASONS",
    "HeldoutFilter",
    "HeldoutNeighbours",
    "count_matching_pairs",
    "heldout_neighbours",
    "kept_fraction",
]
