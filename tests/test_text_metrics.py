"""Free-text answer metrics and rule-based error classification (synthetic text)."""

from __future__ import annotations

import pytest

from medcompress.evaluation.hallucination import (
    ErrorType,
    classify_answer_errors,
    error_rates,
    finding_polarity,
)
from medcompress.evaluation.language import (
    answer_length_stats,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_normalize_answer_sorts_multi_values() -> None:
    assert normalize_answer("  Red;  PINK. ") == "pink; red"
    assert normalize_answer("center; upper-center") == "center; upper center"
    assert normalize_answer("Colonoscopy!") == "colonoscopy"


def test_exact_match_and_token_f1() -> None:
    assert exact_match(["pink; red", "No"], ["red; pink", "yes"]) == 0.5
    assert token_f1(["a small polyp"], ["a polyp"]) == pytest.approx(0.8)
    assert token_f1([""], [""]) == 1.0
    assert token_f1(["x"], [""]) == 0.0
    with pytest.raises(ValueError):
        exact_match(["a"], [])


def test_answer_length_stats() -> None:
    assert answer_length_stats(["one two", "one two three four"]) == {"mean": 3.0, "median": 3.0}


@pytest.mark.parametrize(
    ("text", "polarity"),
    [
        ("There is a polyp in the image.", True),
        ("No polyp is visible.", False),
        ("There are no signs of polyps here.", False),
        ("Two polyps.", True),
        ("Normal mucosa.", None),
        ("No ulcer, but a polyp is seen.", True),
    ],
)
def test_finding_polarity(text: str, polarity: bool | None) -> None:
    assert finding_polarity(text, "polyp") is polarity


def test_classify_answer_errors() -> None:
    labels = {"polyp": True, "ulcer": False}
    assert classify_answer_errors("A polyp is seen.", "One polyp.", labels) == []
    assert classify_answer_errors("Polyp and ulcer.", "One polyp.", labels) == [
        ErrorType.HALLUCINATION
    ]
    assert classify_answer_errors("No polyp.", "One polyp.", labels) == [
        ErrorType.NEGATION,
        ErrorType.OMISSION,
    ]
    assert classify_answer_errors("Normal.", "One polyp.", labels) == [ErrorType.OMISSION]


def test_error_rates() -> None:
    rates = error_rates([[ErrorType.HALLUCINATION], [], [ErrorType.OMISSION, ErrorType.NEGATION]])
    assert rates == pytest.approx({"hallucination": 1 / 3, "omission": 1 / 3, "negation": 1 / 3})
