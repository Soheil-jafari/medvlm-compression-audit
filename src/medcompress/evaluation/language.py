"""Free-text answer quality for open-ended questions.

Clinical correctness is measured separately in clinical.py; these metrics only
describe answer text quality. Normalisation (a proposal for the pre-registration):
NFKC, lower case, every character other than letters, digits, spaces and ``;``
becomes a space, whitespace collapsed, and multi-value answers (Kvasir-VQA-x1 joins
values with ``;``) are compared as sorted, de-duplicated lists so that
"red; pink" equals "pink; red".
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Sequence

_NON_WORD = re.compile(r"[^0-9a-z; ]+")
_SPACES = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation and whitespace, sort ``;``-separated values."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_WORD.sub(" ", text)
    parts = [_SPACES.sub(" ", part).strip() for part in text.split(";")]
    parts = sorted({part for part in parts if part})
    return "; ".join(parts)


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).replace(";", " ").split()


def _paired(predictions: Sequence[str], references: Sequence[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError("Need one reference per prediction.")
    if not predictions:
        raise ValueError("Need at least one prediction.")


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Fraction of normalised predictions equal to the reference. Used by: all."""
    _paired(predictions, references)
    hits = sum(
        normalize_answer(p) == normalize_answer(r)
        for p, r in zip(predictions, references, strict=True)
    )
    return hits / len(predictions)


def _f1(pred: list[str], ref: list[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean token-overlap F1 between prediction and reference. Used by: all."""
    _paired(predictions, references)
    scores = [_f1(_tokens(p), _tokens(r)) for p, r in zip(predictions, references, strict=True)]
    return sum(scores) / len(scores)


def answer_length_stats(predictions: Sequence[str]) -> dict[str, float]:
    """Mean / median answer length in whitespace tokens (compression can shorten answers)."""
    if not predictions:
        raise ValueError("Need at least one prediction.")
    lengths = [len(p.split()) for p in predictions]
    return {
        "mean": float(statistics.fmean(lengths)),
        "median": float(statistics.median(lengths)),
    }
