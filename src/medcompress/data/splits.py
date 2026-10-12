"""Patient-level splits and nested calibration subsets.

Splits are made once, with the single seed recorded in the preregistration
(``seeds.data_split``), at patient level so no patient appears in two splits.
Calibration subsets for pruning-unit scoring are drawn from the training split
only and are nested (the 32-sample subset is inside the 64, and so on) so that
ablation A4 varies size alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medcompress.data.base import VQASample

CALIBRATION_SIZES: tuple[int, ...] = (32, 64, 128, 256)


def make_patient_splits(
    samples: Sequence[VQASample], ratios: Mapping[str, float], *, seed: int
) -> dict[str, list[VQASample]]:
    """Assign every sample to a split so that each patient is in exactly one split.

    ``ratios`` maps split names (train/val/test) to fractions of patients and must
    sum to 1. Datasets with official patient-disjoint splits use those instead
    (recorded in the data card).
    """
    raise NotImplementedError("TODO: group by patient_id, shuffle with seed, cut by ratios.")


def check_patient_disjoint(splits: Mapping[str, Sequence[VQASample]]) -> list[str]:
    """Return patient IDs that occur in more than one split (empty list = OK)."""
    raise NotImplementedError("TODO: patient leakage check.")


def make_calibration_subset(
    train: Sequence[VQASample], size: int, *, seed: int, stratify_by_finding: bool = True
) -> list[VQASample]:
    """Return a calibration subset of the training split for pruning-unit scoring.

    Subsets for different sizes drawn with the same seed must be nested (A4).
    With ``stratify_by_finding`` every finding in the domain list has at least one
    positive image where the training split allows it.

    Used by: E5, E6, E7, A1-A6 (``scripts/04_score_pruning_units.py``).
    """
    raise NotImplementedError("TODO: nested, finding-stratified calibration subset.")


def split_fingerprint(splits: Mapping[str, Sequence[VQASample]]) -> str:
    """Return a SHA-256 over sorted (split, sample_id) pairs, recorded in the preregistration."""
    raise NotImplementedError("TODO: stable hash of the split assignment.")
