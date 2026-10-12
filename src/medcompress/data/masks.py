"""Lesion masks: area, size strata and area-matched random regions.

Lesion area (fraction of image pixels) defines the size strata used by the
primary endpoint ``small_finding_recall``: stratum 0 is the smallest quartile.
The quartile thresholds are computed once on the training split and frozen in
the preregistration. Area-matched random regions are the control condition of
the lesion-mask audit (``evaluation/image_reliance.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Region:
    """A binary region in image coordinates (bounding box + pixel count)."""

    x0: int
    y0: int
    x1: int
    y1: int
    n_pixels: int


def load_mask(path: str) -> Any:
    """Load a binary lesion mask as an array (imaging libraries are imported lazily)."""
    raise NotImplementedError("TODO: read mask, binarise, return HxW bool array.")


def lesion_area_fraction(mask: Any) -> float:
    """Return the lesion area as a fraction of all image pixels."""
    raise NotImplementedError("TODO: mask.sum() / mask.size.")


def size_strata_thresholds(areas: Sequence[float], n_strata: int = 4) -> list[float]:
    """Return the ``n_strata - 1`` area quantile thresholds (computed on train, then frozen)."""
    raise NotImplementedError("TODO: quantile thresholds of lesion area.")


def assign_size_stratum(area: float, thresholds: Sequence[float]) -> int:
    """Return the stratum index of one lesion area (0 = smallest)."""
    raise NotImplementedError("TODO: bisect area into frozen thresholds.")


def area_matched_random_region(mask: Any, *, seed: int, max_tries: int = 100) -> Any:
    """Return a random mask with the same area as ``mask`` that does not overlap it.

    Control condition of the lesion-mask audit; falls back to minimal overlap if
    no disjoint placement exists and records that in the run metadata.
    """
    raise NotImplementedError("TODO: area-matched, lesion-disjoint random region.")
