"""Paired comparisons and the primary-endpoint rows (synthetic outcomes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medcompress.evaluation.aggregate import validate_results_tree
from medcompress.evaluation.comparisons import (
    add_holm_rows,
    average_over_seeds,
    compare_arms,
    pick_best_arm,
)
from medcompress.evaluation.statistics import paired_bootstrap_test
from medcompress.utils.io import append_results


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def test_bootstrap_test_p_values() -> None:
    rng = np.random.default_rng(0)
    a = (rng.random(400) < 0.8).astype(float)
    b = (rng.random(400) < 0.5).astype(float)
    diff, low, high, p = paired_bootstrap_test(_mean, a, b, n_resamples=1000)
    assert low <= diff <= high and p < 0.01
    same = paired_bootstrap_test(_mean, a, a.copy(), n_resamples=200)
    assert same[0] == 0.0 and same[3] == 1.0


def test_average_over_seeds_and_best_arm() -> None:
    avg = average_over_seeds([np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([1, 0, 0])])
    assert avg.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3])
    with pytest.raises(ValueError):
        average_over_seeds([np.ones(3), np.ones(4)])
    assert pick_best_arm({"E5:taylor": 0.71, "E6": 0.74, "E5:magnitude": 0.74}) == "E5:magnitude"


def test_compare_arms_rows_validate_in_the_results_tree(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    a = (rng.random(300) < 0.7).astype(float)
    b = (rng.random(300) < 0.6).astype(float)
    rows = compare_arms(
        "E7_vs_E6",
        "B2",
        a,
        b,
        {"small_finding_recall": _mean, "lesion_mask_effect": _mean},
        n_resamples=300,
    )
    rows = add_holm_rows(rows, ["small_finding_recall", "lesion_mask_effect"])
    metrics = [r["metric"] for r in rows]
    assert metrics.count("p_holm:small_finding_recall") == 1
    holm = {r["metric"]: r["value"] for r in rows}
    assert holm["p_holm:small_finding_recall"] >= holm["p:small_finding_recall"]
    append_results(rows, tmp_path / "comparisons" / "E7_vs_E6" / "B2" / "comparison.csv")
    assert validate_results_tree(tmp_path) == []
