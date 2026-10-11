"""Statistics: paired bootstrap CIs, McNemar per finding, Holm correction, seed aggregation.

Analysis plan (pre-registered in docs/preregistration.md):
* paired bootstrap 95% CIs over test images for every metric difference,
* McNemar's test per finding for paired correct/incorrect outcomes,
* Holm correction across findings (and across the two primary endpoints),
* mean and SD over 3 seeds for training arms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def paired_bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (difference, ci_low, ci_high) of metric(a) - metric(b), resampling images jointly.

    Used by: all comparisons; primary endpoints E7 vs best of E5/E6 at B2.
    """
    raise NotImplementedError("TODO: paired bootstrap CI.")


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> tuple[float, float]:
    """Return (statistic, p-value) of McNemar's test (exact when discordant counts are small)."""
    raise NotImplementedError("TODO: McNemar test per finding.")


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the input order."""
    raise NotImplementedError("TODO: Holm step-down correction.")


def seed_mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return mean and sample SD over seeds."""
    raise NotImplementedError("TODO: mean and SD over seeds.")


def aggregate_results(results_root: Path, out_dir: Path) -> None:
    """Stage entry point for ``scripts/08_aggregate_results.py``.

    Collects every ``metrics.csv`` / ``efficiency.csv`` under ``results_root``,
    computes seed means/SDs, CIs and Holm-corrected tests, and writes summary
    CSVs plus LaTeX tables for ``paper/tables/``.

    Used by: all experiments (final step).
    """
    raise NotImplementedError("TODO: aggregate per-run results into summary tables.")
