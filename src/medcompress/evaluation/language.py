"""Free-text answer quality for open-ended questions.

Which text metrics to report is undecided (TODO: choose after checking what the
chosen QA dataset's benchmark uses). Clinical correctness is measured separately
in clinical.py; these metrics only describe answer text quality.
"""

from __future__ import annotations

from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation and whitespace (rules pre-registered)."""
    raise NotImplementedError("TODO: answer normalisation.")


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Fraction of normalised predictions equal to the reference. Used by: all."""
    raise NotImplementedError("TODO: exact match.")


def token_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean token-overlap F1 between prediction and reference. Used by: all."""
    raise NotImplementedError("TODO: token-level F1.")


def answer_length_stats(predictions: Sequence[str]) -> dict[str, float]:
    """Mean / median answer length in tokens (compression can shorten answers)."""
    raise NotImplementedError("TODO: answer length statistics.")
