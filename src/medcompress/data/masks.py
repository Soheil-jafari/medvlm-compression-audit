"""Lesion masks: area, size strata and area-matched random regions.

Lesion area (fraction of image pixels) defines the size strata used by the
primary endpoint ``small_finding_recall``: stratum 0 is the smallest quartile.
The quartile thresholds are computed once on the training split and frozen in
the preregistration. Area-matched random regions are the control condition of
the lesion-mask audit (``evaluation/image_reliance.py``): the lesion's own shape,
moved to a random place that does not overlap the lesion, so area and shape match
exactly.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Region:
    """A binary region in image coordinates (bounding box + pixel count)."""

    x0: int
    y0: int
    x1: int
    y1: int
    n_pixels: int


def region_of(mask: Any) -> Region:
    """Return the bounding box (x1, y1 exclusive) and pixel count of a binary mask."""
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        raise ValueError("Empty mask has no region.")
    return Region(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(ys.size))


def load_mask(path: str) -> Any:
    """Load a binary lesion mask as an HxW bool array (pixels above 127 are lesion).

    Pillow is imported lazily.
    """
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def lesion_area_fraction(mask: Any) -> float:
    """Return the lesion area as a fraction of all image pixels."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        raise ValueError("Empty array.")
    return float(m.sum() / m.size)


def size_strata_thresholds(areas: Sequence[float], n_strata: int = 4) -> list[float]:
    """Return the ``n_strata - 1`` area quantile thresholds (computed on train, then frozen)."""
    data = np.asarray(areas, dtype=float)
    if n_strata < 2 or data.size < n_strata:
        raise ValueError("Need n_strata >= 2 and at least n_strata areas.")
    qs = [k / n_strata for k in range(1, n_strata)]
    return [float(q) for q in np.quantile(data, qs)]


def assign_size_stratum(area: float, thresholds: Sequence[float]) -> int:
    """Return the stratum index of one lesion area (0 = smallest).

    An area equal to a threshold goes to the upper stratum.
    """
    if list(thresholds) != sorted(thresholds):
        raise ValueError("Thresholds must be sorted.")
    return bisect.bisect_right(list(thresholds), area)


def area_matched_random_region(mask: Any, *, seed: int, max_tries: int = 100) -> Any:
    """Return a random mask with the same area as ``mask`` that does not overlap it.

    The lesion's own shape (its bounding-box crop) is placed at a random position
    inside the image. The first placement without overlap is returned; if none of
    ``max_tries`` placements is disjoint, the one with the least overlap is
    returned (the caller records the overlap with :func:`overlap_fraction`).
    """
    m = np.asarray(mask, dtype=bool)
    region = region_of(m)
    crop = m[region.y0 : region.y1, region.x0 : region.x1]
    h, w = crop.shape
    height, width = m.shape
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_overlap = np.inf
    for _ in range(max_tries):
        y = int(rng.integers(0, height - h + 1))
        x = int(rng.integers(0, width - w + 1))
        candidate = np.zeros_like(m)
        candidate[y : y + h, x : x + w] = crop
        overlap = int(np.sum(candidate & m))
        if overlap < best_overlap:
            best, best_overlap = candidate, overlap
        if overlap == 0:
            break
    assert best is not None
    return best


def overlap_fraction(region: Any, lesion: Any) -> float:
    """Return the fraction of ``region`` pixels that fall on the lesion."""
    r = np.asarray(region, dtype=bool)
    total = r.sum()
    return float(np.sum(r & np.asarray(lesion, dtype=bool)) / total) if total else 0.0
