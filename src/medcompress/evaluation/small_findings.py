"""Small-finding recall stratified by lesion area (PRIMARY ENDPOINT at B2).

"Small" = smallest quartile of lesion area from masks, with thresholds frozen in
advance (``data/masks.size_strata_thresholds``).
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_by_size_stratum(
    detected: Sequence[bool], strata: Sequence[int], n_strata: int = 4
) -> dict[int, float]:
    """Return recall of positive findings per size stratum (0 = smallest).

    Used by: all experiments with mask-annotated test data.
    """
    raise NotImplementedError("TODO: recall per lesion-size stratum.")


def small_finding_recall(detected: Sequence[bool], strata: Sequence[int]) -> float:
    """Return recall in stratum 0 (primary endpoint: E7 vs best of E5/E6 at B2)."""
    raise NotImplementedError("TODO: recall on smallest-quartile lesions (primary endpoint).")
