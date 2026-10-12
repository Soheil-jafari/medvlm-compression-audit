"""Clinical, small-finding, rare-finding, calibration and flip-rate metrics (synthetic data)."""

from __future__ import annotations

import math

import pytest

from medcompress.evaluation.calibration import (
    area_under_risk_coverage,
    brier_score,
    expected_calibration_error,
    risk_coverage_curve,
)
from medcompress.evaluation.clinical import (
    macro_f1,
    overall_accuracy,
    parse_yes_no,
    per_finding_sensitivity_specificity,
)
from medcompress.evaluation.image_reliance import answer_flip_rate, lesion_mask_effect
from medcompress.evaluation.rare_findings import recall_by_prevalence_group
from medcompress.evaluation.small_findings import recall_by_size_stratum, small_finding_recall


@pytest.mark.parametrize(
    ("answer", "parsed"),
    [
        ("Yes", "yes"),
        ("yes.", "yes"),
        ("  No, there is none", "no"),
        ("**Yes** - a polyp", "yes"),
        ("NO", "no"),
        ("not sure", None),
        ("nope", None),
        ("There is a polyp.", None),
        ("", None),
    ],
)
def test_parse_yes_no(answer: str, parsed: str | None) -> None:
    assert parse_yes_no(answer) == parsed


def test_sensitivity_specificity_and_unparseable_answers() -> None:
    truth = {"polyp": [True, True, False, False], "ulcer": [False, False, False, False]}
    pred = {"polyp": [True, None, False, None], "ulcer": [False, True, False, False]}
    out = per_finding_sensitivity_specificity(truth, pred)
    assert out["polyp"] == {"sensitivity": 0.5, "specificity": 0.5, "n_pos": 2.0, "n_neg": 2.0}
    assert math.isnan(out["ulcer"]["sensitivity"])
    assert out["ulcer"]["specificity"] == 0.75


def test_macro_f1_skips_undefined_findings() -> None:
    truth = {"a": [True, False], "b": [False, False]}
    pred = {"a": [True, False], "b": [False, False]}
    assert macro_f1(truth, pred) == 1.0
    pred_bad = {"a": [False, True], "b": [False, False]}
    assert macro_f1(truth, pred_bad) == 0.0
    with pytest.raises(ValueError):
        macro_f1(truth, {"a": [True, False]})


def test_overall_accuracy() -> None:
    assert overall_accuracy([True, False, None, True]) == 0.5
    with pytest.raises(ValueError):
        overall_accuracy([])


def test_recall_by_size_stratum_and_small_finding_recall() -> None:
    detected = [True, False, True, True, False, True]
    strata = [0, 0, 1, 2, 3, 3]
    assert recall_by_size_stratum(detected, strata) == {0: 0.5, 1: 1.0, 2: 1.0, 3: 0.5}
    assert small_finding_recall(detected, strata) == 0.5
    assert math.isnan(recall_by_size_stratum([True], [3])[0])
    with pytest.raises(ValueError):
        recall_by_size_stratum([True], [4])


def test_recall_by_prevalence_group_pooling_rules() -> None:
    detected = {"polyp": [True] * 9 + [False], "ulcer": [False, True], "bleed": []}
    groups = {"polyp": "common", "ulcer": "rare", "bleed": "rare"}
    pooled = recall_by_prevalence_group(detected, groups)
    assert pooled == {"common": 0.9, "rare": 0.5}
    per_finding = recall_by_prevalence_group(
        {"a": [True], "b": [False, False, False]},
        {"a": "rare", "b": "rare"},
        pooling="mean_per_finding",
    )
    assert per_finding == {"rare": 0.5}
    with pytest.raises(ValueError):
        recall_by_prevalence_group({"x": [True]}, {})


def test_ece_perfect_and_overconfident() -> None:
    assert expected_calibration_error([1.0, 1.0, 0.5], [True, True, True]) == pytest.approx(0.5 / 3)
    assert expected_calibration_error([0.9] * 10, [True] * 9 + [False]) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 10, [False] * 10) == pytest.approx(0.9)


def test_brier_score() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == 0.0
    assert brier_score([0.5, 0.5], [True, False]) == 0.25
    with pytest.raises(ValueError):
        brier_score([1.2], [True])


def test_risk_coverage_and_aurc() -> None:
    coverage, risk = risk_coverage_curve([0.9, 0.2, 0.6], [True, False, True])
    assert coverage == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert risk == pytest.approx([0.0, 0.0, 1 / 3])
    assert area_under_risk_coverage(coverage, risk) == pytest.approx(1 / 9)


def test_flip_rate_and_lesion_mask_effect() -> None:
    assert answer_flip_rate(["yes", "Yes", "no", "no"], ["no", "yes ", "no", "yes"]) == 0.5
    assert lesion_mask_effect([True, True, False, True], [False, False, False, True]) == 0.5
    with pytest.raises(ValueError):
        answer_flip_rate(["yes"], [])
    with pytest.raises(ValueError):
        lesion_mask_effect([], [True])
