"""Statistics: paired bootstrap CIs, McNemar per finding, Holm correction, seed aggregation.

Analysis plan (pre-registered in docs/preregistration.md):
* paired bootstrap 95% CIs over test images for every metric difference,
* McNemar's test per finding for paired correct/incorrect outcomes,
* Holm correction across findings (and across the two primary endpoints),
* mean and SD over 3 seeds for training arms,
* Kendall's tau between rankings of arms by different metrics (H4), with a
  bootstrap CI over images.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

EXACT_MCNEMAR_MAX_DISCORDANT = 25
"""Below this many discordant pairs McNemar uses the exact binomial test."""


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return math.nan, math.nan
    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


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

    ``outcomes_a`` and ``outcomes_b`` hold per-image outcomes of two models on the
    **same** images, aligned along the first axis (any trailing shape, e.g.
    ``(n_images,)`` correctness or ``(n_images, 2)`` label/prediction pairs). The
    same resampled image indices are used for both models, so the CI reflects the
    paired design. Percentile interval; resamples where the metric is undefined
    (NaN, e.g. a recall with no positives drawn) are dropped.

    Used by: all comparisons; primary endpoints E7 vs best of E5/E6 at B2.
    """
    a = np.asarray(outcomes_a)
    b = np.asarray(outcomes_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Paired outcomes need the same number of images: {a.shape} vs {b.shape}")
    if a.shape[0] == 0:
        raise ValueError("Need at least one image.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    observed = float(metric(a)) - float(metric(b))
    indices = _resample_indices(a.shape[0], n_resamples, seed)
    diffs = np.fromiter(
        (float(metric(a[idx])) - float(metric(b[idx])) for idx in indices),
        dtype=float,
        count=n_resamples,
    )
    low, high = _percentile_ci(diffs, alpha)
    return observed, low, high


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    outcomes: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (value, ci_low, ci_high) of metric(outcomes) by resampling images."""
    data = np.asarray(outcomes)
    if data.shape[0] == 0:
        raise ValueError("Need at least one image.")
    observed = float(metric(data))
    indices = _resample_indices(data.shape[0], n_resamples, seed)
    values = np.fromiter((float(metric(data[idx])) for idx in indices), float, n_resamples)
    low, high = _percentile_ci(values, alpha)
    return observed, low, high


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> tuple[float, float]:
    """Return (statistic, p-value) of McNemar's test (exact when discordant counts are small).

    ``b`` = pairs where only A is correct, ``c`` = pairs where only B is correct.
    With fewer than :data:`EXACT_MCNEMAR_MAX_DISCORDANT` discordant pairs the exact
    two-sided binomial test is used and the statistic is ``min(b, c)``; otherwise the
    chi-square test with continuity correction, ``(|b - c| - 1)^2 / (b + c)``.
    """
    from scipy import stats

    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    if a.shape != b_arr.shape:
        raise ValueError("Paired outcomes must have the same shape.")
    only_a = int(np.sum(a & ~b_arr))
    only_b = int(np.sum(~a & b_arr))
    discordant = only_a + only_b
    if discordant == 0:
        return 0.0, 1.0
    if discordant < EXACT_MCNEMAR_MAX_DISCORDANT:
        k = min(only_a, only_b)
        p_value = min(1.0, 2.0 * float(stats.binom.cdf(k, discordant, 0.5)))
        return float(k), p_value
    statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
    return float(statistic), float(stats.chi2.sf(statistic, df=1))


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the input order.

    Step-down: sort ascending, multiply the i-th smallest (0-based) by ``m - i``,
    enforce monotonicity with a running maximum, cap at 1.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("p-values must be finite and in [0, 1].")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def seed_mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return mean and sample SD (ddof=1) over seeds; SD is NaN for a single seed."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("Need at least one value.")
    mean = float(np.mean(data))
    sd = float(np.std(data, ddof=1)) if data.size > 1 else math.nan
    return mean, sd


def kendall_tau(scores_x: Sequence[float], scores_y: Sequence[float]) -> float:
    """Return Kendall's tau-b between two score lists over the same arms (H4)."""
    from scipy import stats

    x = np.asarray(scores_x, dtype=float)
    y = np.asarray(scores_y, dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Need two score lists of equal length >= 2.")
    return float(stats.kendalltau(x, y)[0])


def bootstrap_kendall_tau(
    metric_x: Callable[[np.ndarray], float],
    metric_y: Callable[[np.ndarray], float],
    outcomes_by_arm: Mapping[str, np.ndarray],
    *,
    n_resamples: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (tau, ci_low, ci_high) between arm rankings by two metrics (H4).

    ``outcomes_by_arm`` maps each arm to its per-image outcomes on the same images
    (aligned on the first axis). Images are resampled jointly for all arms, both
    metrics are recomputed per arm, and tau is taken between the two score lists.
    """
    arms = list(outcomes_by_arm)
    if len(arms) < 2:
        raise ValueError("Need at least two arms.")
    data = [np.asarray(outcomes_by_arm[arm]) for arm in arms]
    n = data[0].shape[0]
    if any(d.shape[0] != n for d in data):
        raise ValueError("All arms must be scored on the same images.")

    def tau_for(idx: np.ndarray | slice) -> float:
        xs = [metric_x(d[idx]) for d in data]
        ys = [metric_y(d[idx]) for d in data]
        return kendall_tau(xs, ys)

    observed = tau_for(slice(None))
    indices = _resample_indices(n, n_resamples, seed)
    taus = np.fromiter((tau_for(idx) for idx in indices), float, n_resamples)
    low, high = _percentile_ci(taus, alpha)
    return observed, low, high


def aggregate_results(results_root: Path, out_dir: Path) -> None:
    """Stage entry point for ``scripts/08_aggregate_results.py``.

    Collects every ``metrics.csv`` / ``efficiency.csv`` under ``results_root``,
    computes seed means/SDs, CIs and Holm-corrected tests, and writes summary
    CSVs plus LaTeX tables for ``paper/tables/``.

    Used by: all experiments (final step).
    """
    raise NotImplementedError("TODO: aggregate per-run results into summary tables.")
