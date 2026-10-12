"""Clinical metrics: per-finding sensitivity / specificity, macro-F1, overall accuracy.

Unparseable answers (``None`` predictions) always count as errors: a missed
positive or a false positive. Findings whose F1 is undefined (no positives and no
predicted positives) are left out of macro-F1. Both rules are proposals for the
pre-registration (``docs/preregistration.md`` section 6).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

_YES_NO = re.compile(r"^[^a-z]*(yes|no)\b")


def parse_yes_no(answer: str) -> str | None:
    """Map a free-text answer to "yes", "no" or None (unparseable).

    Rule: after lower-casing and dropping leading non-letters, the answer must
    start with the word "yes" or "no" ("Yes.", "no, there is none" parse;
    "there is a polyp" does not). Parsing rules are pre-registered; unparseable
    answers count as errors.
    """
    match = _YES_NO.match(answer.strip().lower())
    return match.group(1) if match else None


def _confusion(truth: Sequence[bool], pred: Sequence[bool | None]) -> tuple[int, int, int, int]:
    if len(truth) != len(pred):
        raise ValueError("Labels and predictions must have the same length.")
    tp = fn = tn = fp = 0
    for t, p in zip(truth, pred, strict=True):
        if t:
            if p is True:
                tp += 1
            else:
                fn += 1
        elif p is False:
            tn += 1
        else:
            fp += 1
    return tp, fn, tn, fp


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


def per_finding_sensitivity_specificity(
    y_true: Mapping[str, Sequence[bool]], y_pred: Mapping[str, Sequence[bool | None]]
) -> dict[str, dict[str, float]]:
    """Return ``{finding: {"sensitivity": .., "specificity": .., "n_pos": .., "n_neg": ..}}``.

    Sensitivity (specificity) is NaN for a finding without positives (negatives).

    Used by: all experiments.
    """
    if set(y_true) != set(y_pred):
        raise ValueError("Labels and predictions must cover the same findings.")
    out: dict[str, dict[str, float]] = {}
    for finding in y_true:
        tp, fn, tn, fp = _confusion(y_true[finding], y_pred[finding])
        out[finding] = {
            "sensitivity": _ratio(tp, tp + fn),
            "specificity": _ratio(tn, tn + fp),
            "n_pos": float(tp + fn),
            "n_neg": float(tn + fp),
        }
    return out


def macro_f1(
    y_true: Mapping[str, Sequence[bool]], y_pred: Mapping[str, Sequence[bool | None]]
) -> float:
    """Unweighted mean of per-finding F1 (undefined F1 left out). Used by: all experiments."""
    if set(y_true) != set(y_pred):
        raise ValueError("Labels and predictions must cover the same findings.")
    scores = []
    for finding in y_true:
        tp, fn, _tn, fp = _confusion(y_true[finding], y_pred[finding])
        if 2 * tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else math.nan


def overall_accuracy(correct: Sequence[bool | None]) -> float:
    """Fraction of correctly answered questions (the 'headline' metric). Used by: all.

    ``None`` (unparseable) counts as wrong.
    """
    if len(correct) == 0:
        raise ValueError("Need at least one answer.")
    return sum(1 for c in correct if c is True) / len(correct)
