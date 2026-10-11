"""Calibration metrics on yes/no probe probabilities: ECE, Brier score, risk-coverage."""

from __future__ import annotations

from collections.abc import Sequence


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 15
) -> float:
    """Expected calibration error with equal-width bins (bin count pre-registered).

    Used by: all experiments.
    """
    raise NotImplementedError("TODO: ECE.")


def brier_score(prob_positive: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error between predicted P(yes) and the label. Used by: all."""
    raise NotImplementedError("TODO: Brier score.")


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> tuple[list[float], list[float]]:
    """Return (coverage, risk) points for selective prediction by confidence."""
    raise NotImplementedError("TODO: risk-coverage curve.")


def area_under_risk_coverage(coverage: Sequence[float], risk: Sequence[float]) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    raise NotImplementedError("TODO: AURC.")
