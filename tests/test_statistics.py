"""Statistics used for every reported comparison (synthetic inputs only)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from medcompress.evaluation.statistics import (
    bootstrap_ci,
    bootstrap_kendall_tau,
    holm_correction,
    kendall_tau,
    mcnemar_test,
    paired_bootstrap_ci,
    seed_mean_sd,
)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def test_paired_bootstrap_identical_models() -> None:
    a = np.array([1, 0, 1, 1, 0, 1], dtype=float)
    assert paired_bootstrap_ci(_mean, a, a.copy(), n_resamples=200) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_constant_difference() -> None:
    diff, low, high = paired_bootstrap_ci(_mean, np.ones(50), np.zeros(50), n_resamples=200)
    assert (diff, low, high) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_contains_observed_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    a = rng.random(300) < 0.7
    b = rng.random(300) < 0.6
    first = paired_bootstrap_ci(_mean, a, b, n_resamples=500, seed=3)
    assert first == paired_bootstrap_ci(_mean, a, b, n_resamples=500, seed=3)
    diff, low, high = first
    assert low <= diff <= high
    assert math.isclose(diff, a.mean() - b.mean())


def test_paired_bootstrap_uses_the_same_images_for_both_models() -> None:
    # b equals a shifted by a constant on every image: the paired CI must be a point.
    a = np.linspace(0, 1, 40)
    diff, low, high = paired_bootstrap_ci(_mean, a + 0.1, a, n_resamples=300)
    assert math.isclose(diff, 0.1) and math.isclose(low, 0.1) and math.isclose(high, 0.1)


def test_paired_bootstrap_two_column_outcomes_and_nan_resamples() -> None:
    # columns: label, prediction; recall is undefined when no positive is drawn
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    pred_a = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    pred_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    def recall(x: np.ndarray) -> float:
        pos = x[:, 0] == 1
        return float(x[pos, 1].mean()) if pos.any() else math.nan

    a = np.stack([labels, pred_a], axis=1)
    b = np.stack([labels, pred_b], axis=1)
    diff, low, high = paired_bootstrap_ci(recall, a, b, n_resamples=300)
    assert (diff, low, high) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="same number"):
        paired_bootstrap_ci(_mean, np.ones(3), np.ones(4))


def test_bootstrap_ci_single_model() -> None:
    value, low, high = bootstrap_ci(_mean, np.array([0.0, 1.0] * 50), n_resamples=500)
    assert value == 0.5 and low < 0.5 < high


@pytest.mark.parametrize(
    ("only_a", "only_b", "expected"),
    [(0, 10, (0.0, 2 * 0.5**10)), (3, 3, (3.0, 1.0)), (0, 0, (0.0, 1.0))],
)
def test_mcnemar_exact(only_a: int, only_b: int, expected: tuple[float, float]) -> None:
    both = 5
    a = [True] * both + [True] * only_a + [False] * only_b
    b = [True] * both + [False] * only_a + [True] * only_b
    statistic, p_value = mcnemar_test(a, b)
    assert statistic == expected[0]
    assert math.isclose(p_value, expected[1])


def test_mcnemar_chi_square_with_continuity_correction() -> None:
    a = [True] * 30 + [False] * 10
    b = [False] * 30 + [True] * 10
    statistic, p_value = mcnemar_test(a, b)
    assert math.isclose(statistic, (20 - 1) ** 2 / 40)
    assert math.isclose(p_value, stats.chi2.sf(statistic, 1))


def test_holm_correction_known_values() -> None:
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm_correction([]) == []
    with pytest.raises(ValueError):
        holm_correction([1.5])


def test_holm_is_monotone_in_sorted_order() -> None:
    p = np.random.default_rng(0).random(20)
    adjusted = np.asarray(holm_correction(p))
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= p)


def test_seed_mean_sd() -> None:
    assert seed_mean_sd([1.0, 2.0, 3.0]) == (2.0, 1.0)
    mean, sd = seed_mean_sd([0.4])
    assert mean == 0.4 and math.isnan(sd)


def test_kendall_tau_extremes() -> None:
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_bootstrap_kendall_tau_agreeing_metrics() -> None:
    rng = np.random.default_rng(2)
    arms = {f"arm{k}": (rng.random(400) < 0.5 + 0.1 * k).astype(float) for k in range(4)}
    tau, low, high = bootstrap_kendall_tau(_mean, _mean, arms, n_resamples=100)
    assert tau == pytest.approx(1.0) and low == pytest.approx(1.0) and high == pytest.approx(1.0)


def test_cluster_bootstrap_resamples_whole_images() -> None:
    from medcompress.evaluation.statistics import resample_indices

    # three images with 1, 2 and 3 probes: every resample keeps whole images
    clusters = np.array(["a", "b", "b", "c", "c", "c"])
    for idx in resample_indices(6, 50, seed=1, clusters=clusters):
        for label in "abc":
            rows = np.flatnonzero(clusters == label)
            taken = np.isin(idx, rows)
            # an image is either absent or present with all its rows, the same number of times
            counts = [int(np.sum(idx == r)) for r in rows]
            assert taken.sum() == 0 or len(set(counts)) == 1
    with pytest.raises(ValueError, match="labels"):
        next(resample_indices(6, 1, seed=0, clusters=["a", "b"]))


def test_cluster_bootstrap_widens_the_interval_for_correlated_probes() -> None:
    # 40 images x 10 probes; within an image all probes share the outcome (fully correlated)
    rng = np.random.default_rng(0)
    per_image = rng.random(40) < 0.5
    outcomes = np.repeat(per_image, 10).astype(float)
    clusters = np.repeat(np.arange(40), 10)
    _, low, high = bootstrap_ci(np.mean, outcomes, n_resamples=2000, seed=0)
    _, c_low, c_high = bootstrap_ci(np.mean, outcomes, n_resamples=2000, seed=0, clusters=clusters)
    # resampling probes pretends there are 400 independent answers; images give the honest width
    assert (c_high - c_low) > 2.5 * (high - low)


def test_lazy_resampling_matches_the_old_batched_draws() -> None:
    from medcompress.evaluation.statistics import resample_indices

    batched = np.random.default_rng(7).integers(0, 13, size=(20, 13))
    lazy = np.stack(list(resample_indices(13, 20, seed=7)))
    assert np.array_equal(batched, lazy)
