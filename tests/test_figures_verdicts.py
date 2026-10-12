"""Figures render from a synthetic summary; verdict rules decide as documented."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from medcompress.evaluation.figures import FIGURE_WRITERS, family_of, write_figures
from medcompress.evaluation.verdicts import (
    INCONCLUSIVE,
    NOT_SUPPORTED,
    PENDING,
    SUPPORTED,
    all_verdicts,
    verdict_h3,
    verdicts_markdown,
)

NAN = math.nan


def _row(
    exp: str, budget: str, metric: str, mean: float, low: float = NAN, high: float = NAN
) -> dict:
    return {
        "experiment": exp,
        "budget": budget,
        "metric": metric,
        "n_seeds": 3,
        "mean": mean,
        "sd": 0.01,
        "min": mean,
        "max": mean,
        "ci_low": low,
        "ci_high": high,
    }


def synthetic_summary() -> pd.DataFrame:
    rows = [
        _row("E2", "", m, v)
        for m, v in [
            ("overall_accuracy", 0.80),
            ("small_finding_recall", 0.70),
            ("lesion_mask_effect", 0.40),
            ("peak_memory_bytes", 1e9),
            ("rare_recall:rare", 0.60),
            ("ece", 0.05),
            ("lesion_flip_rate", 0.5),
            ("random_flip_rate", 0.1),
        ]
    ]
    for arm, budget, acc, small in [
        ("E3:w8", "", 0.79, 0.69),
        ("E3:w4", "", 0.77, 0.60),
        ("E5:taylor", "B2", 0.76, 0.50),
        ("E6", "B2", 0.75, 0.52),
        ("E7", "B2", 0.76, 0.62),
    ]:
        rows += [
            _row(arm, budget, "overall_accuracy", acc),
            _row(arm, budget, "small_finding_recall", small),
            _row(arm, budget, "ece", 0.07),
            _row(arm, budget, "lesion_flip_rate", 0.4),
            _row(arm, budget, "random_flip_rate", 0.1),
        ]
    for budget, mem, small in [("B1", 0.75e9, 0.66), ("B2", 0.6e9, 0.62), ("B3", 0.5e9, 0.55)]:
        rows += [_row("E7", budget, "peak_memory_bytes", mem)]
        if budget != "B2":
            rows += [_row("E7", budget, "small_finding_recall", small)]
    rows += [_row("E3:w4", "", "peak_memory_bytes", 0.4e9)]
    return pd.DataFrame(rows)


def test_family_of() -> None:
    assert family_of("E3:w4") == "quantization"
    assert family_of("E5:taylor") == "generic pruning"
    assert family_of("A4:n64") == "clinical (ours)"
    assert family_of("E1") == "distillation only"


@pytest.mark.parametrize(
    "summary",
    [synthetic_summary(), pd.DataFrame(columns=["experiment", "budget", "metric", "mean"])],
)
def test_figures_are_written_as_pdfs(summary: pd.DataFrame, tmp_path: Path) -> None:
    paths = write_figures(summary, tmp_path)
    assert {p.name for p in paths} == set(FIGURE_WRITERS)
    assert all(p.read_bytes().startswith(b"%PDF") for p in paths)


def test_verdicts_pending_without_evidence() -> None:
    verdicts = all_verdicts(
        pd.DataFrame(columns=["experiment", "budget", "metric", "mean", "ci_low", "ci_high"])
    )
    assert [v.verdict for v in verdicts] == [PENDING] * 4
    assert "no evidence rows yet" in verdicts_markdown(verdicts)


@pytest.mark.parametrize(
    ("small", "mask", "p_small", "p_mask", "expected"),
    [
        ((0.05, 0.01, 0.09), (0.04, 0.01, 0.07), 0.01, 0.02, SUPPORTED),
        ((-0.05, -0.09, -0.01), (0.04, 0.01, 0.07), 0.01, 0.02, NOT_SUPPORTED),
        ((0.02, -0.01, 0.05), (0.04, 0.01, 0.07), 0.20, 0.02, INCONCLUSIVE),
    ],
)
def test_h3_rule(small: tuple, mask: tuple, p_small: float, p_mask: float, expected: str) -> None:
    rows = [
        _row("E7_vs_best", "B2", "diff:small_finding_recall", *small),
        _row("E7_vs_best", "B2", "diff:lesion_mask_effect", *mask),
        _row("E7_vs_best", "B2", "p_holm:small_finding_recall", p_small),
        _row("E7_vs_best", "B2", "p_holm:lesion_mask_effect", p_mask),
    ]
    assert verdict_h3(pd.DataFrame(rows)).verdict == expected


def test_h1_h2_h4_rules() -> None:
    rows = [
        _row("E3:w4_vs_E2", "", "diff:drop_gap", 0.1, 0.05, 0.15),
        _row("E5:taylor_vs_E2", "B2", "diff:drop_gap", 0.2, 0.1, 0.3),
        _row("best_pruning_vs_E3:w8", "", "diff:small_finding_recall", -0.1, -0.15, -0.05),
        _row("accuracy_vs_clinical", "", "tau:small_finding_recall", 0.4, 0.1, 0.7),
    ]
    h1, h2, _, h4 = all_verdicts(pd.DataFrame(rows))
    assert (h1.verdict, h2.verdict, h4.verdict) == (SUPPORTED, SUPPORTED, SUPPORTED)


def _written_metric_names() -> set[str]:
    """Every metric name the evaluation and benchmark stages write, on synthetic outcomes."""
    import numpy as np

    from medcompress.evaluation import probe_suites as ps
    from medcompress.evaluation.efficiency import REPORT_METRICS
    from medcompress.evaluation.image_reliance import MaskAuditResult
    from medcompress.evaluation.runner import metrics_from_summary

    truth = np.array([True, False, True, False, True, False])
    out = ps.ProbeOutcomes(
        sample_ids=("a", "a", "b", "b", "c", "c"),
        findings=("polyp", "bleeding") * 3,
        truth=truth,
        yes=np.array([True, False, False, True, True, False]),
        margin=np.array([1.0, -1.0, -0.5, 0.5, 2.0, -2.0]),
        area=np.array([0.004, np.nan, 0.02, np.nan, 0.01, np.nan]),
        has_mask=truth,
    )
    metrics = ps.clinical_metrics(out, ["polyp", "bleeding"], n_resamples=5)
    metrics += ps.small_finding_metrics(out, [0.006, 0.013, 0.018], ["polyp"], n_resamples=5)
    metrics += ps.rare_finding_metrics(out, {"common": ["polyp"], "rare": ["bleeding"]})
    metrics += ps.calibration_metrics(out, n_bins=5)
    flips = np.array([True, False, True])
    audit = MaskAuditResult(("a", "b", "c"), ("polyp",) * 3, flips, flips, ~flips,
                            np.ones(3), np.zeros(3), np.zeros(3))  # fmt: skip
    metrics += metrics_from_summary(audit.summary(n_resamples=5))
    return {m.name for m in metrics} | set(REPORT_METRICS) | {"low_precision_kernels_ok"}


def test_figures_and_tables_read_metrics_the_stages_write() -> None:
    from medcompress.evaluation import figures, report

    used = {
        report.ACCURACY, report.BALANCED_ACCURACY, report.SMALL_RECALL, report.MASK_EFFECT,
        *report.MAIN_METRICS, *(m for m, _ in report.EFFICIENCY_METRICS),
        figures.RARE_RECALL, figures.ECE, figures.PEAK_MEMORY, figures.FLIP_LESION,
        figures.FLIP_RANDOM,
    }  # fmt: skip
    assert used - _written_metric_names() == set()
