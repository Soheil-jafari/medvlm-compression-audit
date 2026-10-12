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
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

EXACT_MCNEMAR_MAX_DISCORDANT = 25
"""Below this many discordant pairs McNemar uses the exact binomial test."""


def _cluster_groups(clusters: Sequence[object] | np.ndarray, n: int) -> list[np.ndarray]:
    """Row indices of each cluster, clusters in order of first appearance."""
    labels = np.asarray(clusters)
    if labels.shape[0] != n:
        raise ValueError(f"clusters has {labels.shape[0]} labels for {n} rows")
    order: dict[object, list[int]] = {}
    for i, label in enumerate(labels.tolist()):
        order.setdefault(label, []).append(i)
    return [np.asarray(rows, dtype=int) for rows in order.values()]


def resample_indices(
    n: int,
    n_resamples: int,
    seed: int,
    clusters: Sequence[object] | np.ndarray | None = None,
) -> Iterator[np.ndarray]:
    """Yield ``n_resamples`` row-index arrays, one resample at a time.

    Without ``clusters`` each resample draws ``n`` rows with replacement. With ``clusters``
    (one label per row, e.g. the image a probe was asked about) whole clusters are drawn
    with replacement and all rows of a drawn cluster are kept, so rows of one image move
    together: the bootstrap then resamples images, as pre-registered, even when an image
    contributes several probes. Draws are generated lazily, so memory stays at one
    resample however large ``n_resamples * n`` is.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        for _ in range(n_resamples):
            yield rng.integers(0, n, size=n)
        return
    groups = _cluster_groups(clusters, n)
    k = len(groups)
    for _ in range(n_resamples):
        yield np.concatenate([groups[g] for g in rng.integers(0, k, size=k)])


def percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    """The ``(alpha/2, 1 - alpha/2)`` percentiles of the finite bootstrap values (NaN if none)."""
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
    clusters: Sequence[object] | np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return (difference, ci_low, ci_high) of metric(a) - metric(b), resampling images jointly.

    ``outcomes_a`` and ``outcomes_b`` hold per-image outcomes of two models on the
    **same** images, aligned along the first axis (any trailing shape, e.g.
    ``(n_images,)`` correctness or ``(n_images, 2)`` label/prediction pairs). The
    same resampled image indices are used for both models, so the CI reflects the
    paired design. Percentile interval; resamples where the metric is undefined
    (NaN, e.g. a recall with no positives drawn) are dropped. When rows are probes and an
    image has several, pass the image of each row as ``clusters`` (see
    :func:`resample_indices`).

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
    indices = resample_indices(a.shape[0], n_resamples, seed, clusters)
    diffs = np.fromiter(
        (float(metric(a[idx])) - float(metric(b[idx])) for idx in indices),
        dtype=float,
        count=n_resamples,
    )
    low, high = percentile_ci(diffs, alpha)
    return observed, low, high


def paired_bootstrap_test(
    metric: Callable[[np.ndarray], float],
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    clusters: Sequence[object] | np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return (difference, ci_low, ci_high, p_value) for metric(a) - metric(b).

    Same resampling as :func:`paired_bootstrap_ci` (including ``clusters``). The two-sided
    p-value is ``min(1, 2 * min(P(d* <= 0), P(d* >= 0)))`` over the bootstrap differences
    ``d*`` (with the +1 correction so it is never exactly 0).
    """
    a = np.asarray(outcomes_a)
    b = np.asarray(outcomes_b)
    if a.shape[0] != b.shape[0] or a.shape[0] == 0:
        raise ValueError("Paired outcomes need the same, non-zero number of images.")
    observed = float(metric(a)) - float(metric(b))
    indices = resample_indices(a.shape[0], n_resamples, seed, clusters)
    diffs = np.fromiter(
        (float(metric(a[idx])) - float(metric(b[idx])) for idx in indices), float, n_resamples
    )
    low, high = percentile_ci(diffs, alpha)
    return observed, low, high, bootstrap_p_value(diffs)


def bootstrap_p_value(diffs: np.ndarray) -> float:
    """Two-sided p-value of "no difference" from bootstrap differences ``d*``.

    ``min(1, 2 * min(P(d* <= 0), P(d* >= 0)))`` with the +1 correction, so it is never exactly
    0; undefined resamples (NaN) are dropped, and with none left the p-value is NaN.
    """
    finite = np.asarray(diffs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    n = finite.size
    below = (np.sum(finite <= 0) + 1) / (n + 1)
    above = (np.sum(finite >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(below, above)))


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    outcomes: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    clusters: Sequence[object] | np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return (value, ci_low, ci_high) of metric(outcomes) by resampling images.

    Rows are resampled one by one, or by cluster when ``clusters`` gives each row's image.
    """
    data = np.asarray(outcomes)
    if data.shape[0] == 0:
        raise ValueError("Need at least one image.")
    observed = float(metric(data))
    indices = resample_indices(data.shape[0], n_resamples, seed, clusters)
    values = np.fromiter((float(metric(data[idx])) for idx in indices), float, n_resamples)
    low, high = percentile_ci(values, alpha)
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
    clusters: Sequence[object] | np.ndarray | None = None,
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
    indices = resample_indices(n, n_resamples, seed, clusters)
    taus = np.fromiter((tau_for(idx) for idx in indices), float, n_resamples)
    low, high = percentile_ci(taus, alpha)
    return observed, low, high


def aggregate_results(
    results_root: Path,
    out_dir: Path,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    verdicts_path: Path | None = None,
    headline: str = "overall_accuracy",
) -> list[Path]:
    """Stage entry point for ``scripts/08_aggregate_results.py``.

    Validates and collects every results CSV under ``results_root`` (runs,
    comparisons, selection, data summary), writes ``all_rows.csv`` and
    ``by_arm.csv`` (seed means and SDs, per split) to ``out_dir``; then, from the test-split
    rows and the rows not tied to a split only (never validation rows), the paper tables and
    ``numbers.tex`` to ``tables_dir`` (default ``paper/tables``), figures F1-F5 to
    ``figures_dir`` (default ``paper/figures``) and the H1-H4 verdicts to
    ``verdicts_path`` (default ``docs/results/VERDICTS.md``); defaults are relative
    to the repository. Paired CIs and Holm-corrected p-values come from the
    comparison rows (``evaluation/compare_stage``). ``headline`` is the pre-registered headline
    accuracy (F1, F4). Returns the paths written.

    Used by: all experiments (final step).
    """
    from medcompress.config import repo_root
    from medcompress.evaluation.aggregate import (
        load_results,
        paper_rows,
        summarise_over_seeds,
        write_summary,
    )
    from medcompress.evaluation.figures import write_figures
    from medcompress.evaluation.report import write_tables
    from medcompress.evaluation.verdicts import write_verdicts

    root = repo_root()
    rows = load_results(Path(results_root))
    written = write_summary(rows, Path(out_dir))
    rows = paper_rows(rows)  # the paper reports the test split, never validation rows
    summary = summarise_over_seeds(rows)
    written += write_tables(rows, summary, Path(tables_dir or root / "paper" / "tables"))
    figures = Path(figures_dir or root / "paper" / "figures")
    written += write_figures(summary, figures, headline=headline)
    written.append(
        write_verdicts(summary, Path(verdicts_path or root / "docs" / "results" / "VERDICTS.md"))
    )
    return written
