"""Hallucination, omission and negation errors.

* hallucination: model asserts a finding that is absent,
* omission:      model fails to assert a finding that is present and that the
                 reference answer asserts,
* negation:      model output flips polarity (e.g. "no polyp" vs "polyp") relative
                 to the reference in free-text answers.

Rule-based detection (a proposal for the pre-registration): a finding is
*mentioned* when its name appears as whole words (case-insensitive); a mention is
*negated* when one of :data:`NEGATION_CUES` appears in the few words before it,
within the same clause (clauses end at punctuation, "but" or "however").
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum

NEGATION_CUES: tuple[str, ...] = (
    "no",
    "not",
    "without",
    "absent",
    "absence of",
    "free of",
    "negative for",
    "none",
)
NEGATION_WINDOW_WORDS = 4


class ErrorType(str, Enum):
    """Answer error categories."""

    HALLUCINATION = "hallucination"
    OMISSION = "omission"
    NEGATION = "negation"


def finding_polarity(text: str, finding: str) -> bool | None:
    """Return True (asserted), False (negated) or None (not mentioned) for a finding."""
    lowered = text.lower()
    pattern = re.compile(r"\b" + re.escape(finding.lower()) + r"s?\b")
    polarity: bool | None = None
    for match in pattern.finditer(lowered):
        clause = re.split(r"[.,;:!?]|\bbut\b|\bhowever\b", lowered[: match.start()])[-1]
        before = re.findall(r"[a-z]+", clause)[-NEGATION_WINDOW_WORDS:]
        window = " " + " ".join(before) + " "
        negated = any(f" {cue} " in window for cue in NEGATION_CUES)
        if not negated:
            return True
        polarity = False
    return polarity


def classify_answer_errors(
    predicted: str, reference: str, finding_labels: Mapping[str, bool]
) -> list[ErrorType]:
    """Return the error types present in one answer. Used by: all experiments.

    ``finding_labels`` maps finding names (as they appear in text) to whether the
    finding is present in the image.
    """
    errors: set[ErrorType] = set()
    for finding, present in finding_labels.items():
        pred = finding_polarity(predicted, finding)
        ref = finding_polarity(reference, finding)
        if pred is True and not present:
            errors.add(ErrorType.HALLUCINATION)
        if present and ref is True and pred is not True:
            errors.add(ErrorType.OMISSION)
        if pred is not None and ref is not None and pred != ref:
            errors.add(ErrorType.NEGATION)
    return sorted(errors, key=lambda e: e.value)


def error_rates(errors: Sequence[Sequence[ErrorType]]) -> dict[str, float]:
    """Return the rate of each error type over a set of answers."""
    if not errors:
        raise ValueError("Need at least one answer.")
    return {
        kind.value: sum(kind in answer for answer in errors) / len(errors) for kind in ErrorType
    }
