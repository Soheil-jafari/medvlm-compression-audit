"""Small-finding recall stratified by lesion area (PRIMARY ENDPOINT at B2).

"Small" = smallest quartile of lesion area from masks, with thresholds frozen in
advance (``data/masks.size_strata_thresholds``). Inputs are one entry per positive
(lesion present, mask available): whether the model detected it and its size
stratum.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_by_size_stratum(
    detected: Sequence[bool], strata: Sequence[int], n_strata: int = 4
) -> dict[int, float]:
    """Return recall of positive findings per size stratum (0 = smallest).

    A stratum without positives gets NaN.

    Used by: all experiments with mask-annotated test data.
    """
    if len(detected) != len(strata):
        raise ValueError("detected and strata must have the same length.")
    hits = [0] * n_strata
    totals = [0] * n_strata
    for hit, stratum in zip(detected, strata, strict=True):
        if not 0 <= stratum < n_strata:
            raise ValueError(f"Stratum {stratum} outside 0..{n_strata - 1}.")
        totals[stratum] += 1
        hits[stratum] += bool(hit)
    return {s: (hits[s] / totals[s] if totals[s] else math.nan) for s in range(n_strata)}


def small_finding_recall(
    detected: Sequence[bool], strata: Sequence[int], n_strata: int = 4
) -> float:
    """Return recall in stratum 0 (primary endpoint: E7 vs best of E5/E6 at B2)."""
    return recall_by_size_stratum(detected, strata, n_strata)[0]
