"""Clinical metrics: per-finding sensitivity / specificity, macro-F1, overall accuracy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def parse_yes_no(answer: str) -> str | None:
    """Map a free-text answer to "yes", "no" or None (unparseable).

    Parsing rules are pre-registered; unparseable answers count as errors.
    """
    raise NotImplementedError("TODO: pre-registered yes/no answer parser.")


def per_finding_sensitivity_specificity(
    y_true: Mapping[str, Sequence[bool]], y_pred: Mapping[str, Sequence[bool]]
) -> dict[str, dict[str, float]]:
    """Return ``{finding: {"sensitivity": .., "specificity": .., "n_pos": .., "n_neg": ..}}``.

    Used by: all experiments.
    """
    raise NotImplementedError("TODO: per-finding sensitivity and specificity.")


def macro_f1(y_true: Mapping[str, Sequence[bool]], y_pred: Mapping[str, Sequence[bool]]) -> float:
    """Unweighted mean of per-finding F1. Used by: all experiments."""
    raise NotImplementedError("TODO: macro-F1 over findings.")


def overall_accuracy(correct: Sequence[bool]) -> float:
    """Fraction of correctly answered questions (the 'headline' metric). Used by: all."""
    raise NotImplementedError("TODO: overall accuracy.")
