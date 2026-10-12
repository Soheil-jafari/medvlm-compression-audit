"""Paired comparisons between arms, written as rows of the results schema.

Comparisons need per-image outcomes of both arms on the same images. Those stay
where the predictions are (the GPU session, git-ignored); only the resulting rows
go to ``results/comparisons/<A>_vs_<B>/<budget>/comparison.csv``.

Seeds (proposal for the pre-registration): for a training arm with several seeds,
per-image outcomes are averaged over seeds first, then images are bootstrapped. The
primary comparison is E7 against the best of E5/E6 at B2; "best" is chosen on the
validation split by :func:`pick_best_arm`, and the two primary p-values are
Holm-corrected together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from medcompress.evaluation.statistics import holm_correction, paired_bootstrap_test

Metric = Callable[[np.ndarray], float]


def average_over_seeds(per_seed: Sequence[np.ndarray]) -> np.ndarray:
    """Return the per-image mean over seeds (all seeds scored on the same images)."""
    if not per_seed:
        raise ValueError("Need at least one seed.")
    arrays = [np.asarray(x, dtype=float) for x in per_seed]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("All seeds must have outcomes for the same images.")
    return np.mean(np.stack(arrays), axis=0)


def pick_best_arm(validation_scores: Mapping[str, float]) -> str:
    """Return the arm with the highest validation score (ties broken by name).

    Used for "best of E5 and E6" before the test-split comparison.
    """
    if not validation_scores:
        raise ValueError("Need at least one arm.")
    return sorted(validation_scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def compare_arms(
    name: str,
    budget: str,
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    metrics: Mapping[str, Metric],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Return ``diff:<metric>`` and ``p:<metric>`` rows for arm A minus arm B."""
    rows: list[dict[str, object]] = []
    for metric_name, metric in metrics.items():
        diff, low, high, p_value = paired_bootstrap_test(
            metric, outcomes_a, outcomes_b, n_resamples=n_resamples, alpha=alpha, seed=seed
        )
        rows.append(
            {
                "experiment": name,
                "budget": budget,
                "seed": 0,
                "metric": f"diff:{metric_name}",
                "value": diff,
                "ci_low": low,
                "ci_high": high,
            }
        )
        rows.append(
            {
                "experiment": name,
                "budget": budget,
                "seed": 0,
                "metric": f"p:{metric_name}",
                "value": p_value,
            }
        )
    return rows


def add_holm_rows(
    rows: Sequence[Mapping[str, object]], metric_names: Sequence[str]
) -> list[dict[str, object]]:
    """Return ``rows`` plus ``p_holm:<metric>`` rows, correcting the named p-values together."""
    by_metric = {str(r["metric"]): r for r in rows}
    p_rows = [by_metric[f"p:{m}"] for m in metric_names]
    adjusted = holm_correction([float(r["value"]) for r in p_rows])
    extra = [
        {**dict(r), "metric": f"p_holm:{m}", "value": adj}
        for r, m, adj in zip(p_rows, metric_names, adjusted, strict=True)
    ]
    return [dict(r) for r in rows] + extra
