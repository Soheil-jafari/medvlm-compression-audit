"""Calibration metrics on yes/no probe probabilities: ECE, Brier score, risk-coverage."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _check(a: Sequence[float], b: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Inputs must be 1-D and of equal length.")
    if x.size == 0:
        raise ValueError("Need at least one prediction.")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("Probabilities and confidences must be in [0, 1].")
    return x, y


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 15
) -> float:
    """Expected calibration error with equal-width bins (bin count pre-registered).

    ECE = sum_b (n_b / N) |accuracy_b - mean confidence_b|; bins are
    ``[0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1]``.

    Used by: all experiments.
    """
    conf, ok = _check(confidences, correct)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(ok[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 15
) -> list[tuple[int, float, float, float]]:
    """``(bin, share of answers, accuracy, mean confidence)`` of every non-empty bin.

    The bins are those of :func:`expected_calibration_error` (the points of a reliability
    diagram). Used by: the appendix reliability plot.
    """
    conf, ok = _check(confidences, correct)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return [
        (b, float((bins == b).mean()), float(ok[bins == b].mean()), float(conf[bins == b].mean()))
        for b in range(n_bins)
        if (bins == b).any()
    ]


RISK_COVERAGE_LEVELS: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
"""Coverage levels (percent) at which the risk-coverage curve is reported."""


def risk_at_coverage(
    confidences: Sequence[float],
    correct: Sequence[bool],
    levels: Sequence[int] = RISK_COVERAGE_LEVELS,
) -> dict[int, float]:
    """Error rate among the most confident ``level`` percent of answers, for each level."""
    coverage, risk = risk_coverage_curve(confidences, correct)
    n = len(risk)
    return {level: risk[max(1, math.ceil(n * level / 100)) - 1] for level in levels}


def brier_score(prob_positive: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error between predicted P(yes) and the label. Used by: all."""
    p, y = _check(prob_positive, labels)
    return float(np.mean((p - y.astype(float)) ** 2))


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> tuple[list[float], list[float]]:
    """Return (coverage, risk) points for selective prediction by confidence.

    Answers are kept in order of decreasing confidence (stable for ties); point k
    keeps the k most confident answers: coverage k/N, risk = error rate among them.
    """
    conf, ok = _check(confidences, correct)
    order = np.argsort(-conf, kind="stable")
    errors = np.cumsum(~ok[order])
    k = np.arange(1, conf.size + 1)
    return (k / conf.size).tolist(), (errors / k).tolist()


def area_under_risk_coverage(coverage: Sequence[float], risk: Sequence[float]) -> float:
    """Area under the risk-coverage curve (lower is better).

    Discrete AURC: the mean of the risk over the N coverage points.
    """
    if len(coverage) != len(risk):
        raise ValueError("coverage and risk must have the same length.")
    return float(np.mean(risk)) if len(risk) else math.nan
