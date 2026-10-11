"""Hallucination, omission and negation errors.

* hallucination: model asserts a finding that is absent,
* omission:      model fails to mention / denies a finding that is present,
* negation:      model output flips polarity (e.g. "no polyp" vs "polyp") relative
                 to the reference in free-text answers.
Exact rules for free-text answers are pre-registered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum


class ErrorType(str, Enum):
    """Answer error categories."""

    HALLUCINATION = "hallucination"
    OMISSION = "omission"
    NEGATION = "negation"


def classify_answer_errors(
    predicted: str, reference: str, finding_labels: Mapping[str, bool]
) -> list[ErrorType]:
    """Return the error types present in one answer. Used by: all experiments."""
    raise NotImplementedError("TODO: rule-based error classification.")


def error_rates(errors: Sequence[Sequence[ErrorType]]) -> dict[str, float]:
    """Return the rate of each error type over a set of answers."""
    raise NotImplementedError("TODO: per-type error rates.")
