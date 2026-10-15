"""Appendix tables and figures, the calibration points behind them, and compute bookkeeping."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medcompress.evaluation import appendix, calibration
from medcompress.evaluation.probe_suites import ProbeOutcomes, calibration_metrics


def _summary_row(arm: str, budget: str, metric: str, mean: float) -> dict:
    return {"experiment": arm, "budget": budget, "split": "test", "metric": metric, "mean": mean,
            "sd": 0.01, "ci_low": math.nan, "ci_high": math.nan}  # fmt: skip


def _row(experiment: str, metric: str, value: float, budget: str = "", seed: int = 0) -> dict:
    return {"experiment": experiment, "budget": budget, "seed": seed, "metric": metric,
            "value": value, "ci_low": math.nan, "ci_high": math.nan, "split": "val"}  # fmt: skip


def test_reliability_bins_and_risk_at_coverage_follow_their_definitions() -> None:
    conf = np.array([0.55, 0.58, 0.95, 0.97, 0.99, 0.62])
    ok = np.array([True, False, True, True, False, True])
    bins = calibration.reliability_bins(conf, ok, n_bins=5)
    assert [b for b, *_ in bins] == [2, 3, 4]  # [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
    assert bins[0][1:] == pytest.approx((2 / 6, 0.5, 0.565))
    shares = sum(share for _, share, _, _ in bins)
    weighted = sum(s * abs(a - c) for _, s, a, c in bins)
    assert shares == pytest.approx(1.0)
    assert weighted == pytest.approx(calibration.expected_calibration_error(conf, ok, n_bins=5))
    risk = calibration.risk_at_coverage(conf, ok, levels=(50, 100))
    assert risk[100] == pytest.approx(2 / 6)
    assert risk[50] == pytest.approx(1 / 3)  # the 3 most confident: 0.99 wrong, 0.97, 0.95 right


def test_the_calibration_suite_writes_the_plot_points() -> None:
    out = ProbeOutcomes(
        sample_ids=("a", "b", "c", "d"),
        findings=("polyp",) * 4,
        truth=np.array([True, False, True, False]),
        yes=np.array([True, False, False, False]),
        margin=np.array([2.0, -1.0, -0.2, -3.0]),
        area=np.full(4, np.nan),
        has_mask=np.zeros(4, dtype=bool),
    )
    names = {m.name for m in calibration_metrics(out, n_bins=5)}
    assert {"ece", "risk_at_coverage:10", "risk_at_coverage:100"} <= names
    assert any(n.startswith("reliability_accuracy:") for n in names)


def test_the_compute_table_sums_hours_per_experiment() -> None:
    rows = pd.DataFrame(
        [
            _row("E2", "train_seconds", 3600.0, seed=0),
            _row("E2", "train_seconds", 7200.0, seed=1),
            _row("E5:taylor", "scoring_seconds", 1800.0, budget=""),
            _row("E5:taylor", "recovery_seconds", 1800.0, budget="B2"),
            _row("E2", "final:benchmark_seconds", 360.0),
        ]
    )
    lines = appendix.compute_table(rows).splitlines()
    e2 = next(line for line in lines if line.startswith("E2 "))
    assert "3.00" in e2 and "0.10" in e2 and e2.rstrip(" \\").endswith("3.10")
    e5 = next(line for line in lines if line.startswith("E5 "))
    assert e5.rstrip(" \\").endswith("1.00")
    assert next(line for line in lines if line.startswith("Total")).rstrip(" \\").endswith("4.10")


def test_the_selection_table_shows_the_round0_checks() -> None:
    from medcompress.selection.driver import PEAK_MEMORY_METRIC

    rows = pd.DataFrame(
        [
            _row("R0:qwen2_5_vl_3b", "r0_params", 3.75e9),
            _row("R0:qwen2_5_vl_3b", PEAK_MEMORY_METRIC, 8 * 2**30),
            _row("R0:qwen2_5_vl_3b", "advance", 1.0),
            _row("R0:other", "advance", 0.0),
            _row("R0:other", "gate_failed:G1_hardware", 1.0),
        ]
    )
    text = appendix.selection_table(rows)
    assert re.search(r"qwen2\\_5\\_vl\\_3b & pass & 3.75 & -- & -- & 8.00", text)
    assert "fail: G1\\_hardware" in text


def test_per_finding_tables_have_one_column_per_finding() -> None:
    summary = pd.DataFrame(
        [
            _summary_row("E2", "", "sensitivity:polyp", 0.8),
            _summary_row("E2", "", "sensitivity:ulcerative colitis", 0.6),
            _summary_row("E7", "B2", "sensitivity:polyp", 0.7),
        ]
    )
    text = appendix.per_finding_table(summary, "sensitivity")
    assert "Arm & polyp & ulcerative colitis" in text
    assert "E2 & 0.80 & 0.60" in text and "E7 @B2 & 0.70 & --" in text


def test_the_hyperparameter_table_reads_the_configs() -> None:
    text = appendix.hyperparameter_table()
    assert "LoRA rank / alpha / dropout & 16 / 32 / 0.05" in text
    assert "Ranking of A3 (random scores) & none" in text
    assert "Scoring method & --" in text  # still open (D07 section 1)


def test_the_appendix_is_written_with_placeholders_and_with_data(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=["experiment", "budget", "seed", "metric", "value", "split"])
    summary = pd.DataFrame(columns=["experiment", "budget", "split", "metric", "mean", "sd"])
    paths = appendix.write_appendix(empty, summary, tmp_path / "t", tmp_path / "f")
    assert all(p.exists() for p in paths)
    points = [_summary_row("E2", "", f"reliability_accuracy:{b}", 0.6 + b / 50) for b in (8, 12)]
    points += [
        _summary_row("E2", "", f"reliability_confidence:{b}", 0.55 + b / 40) for b in (8, 12)
    ]
    points += [_summary_row("E7", "B2", f"risk_at_coverage:{c}", c / 500) for c in (10, 50, 100)]
    paths = appendix.write_appendix(empty, pd.DataFrame(points), tmp_path / "t", tmp_path / "f")
    assert all(p.read_bytes().startswith(b"%PDF") for p in paths if p.suffix == ".pdf")


def test_the_appendix_reads_metrics_the_stages_write() -> None:
    from medcompress.evaluation.hallucination import ErrorType

    source = "\n".join(
        p.read_text()
        for p in (Path(__file__).resolve().parents[1] / "src" / "medcompress").rglob("*.py")
        if p.name != "appendix.py"
    )
    free_text = {"exact_match", "token_f1", *(f"{k.value}_rate" for k in ErrorType)}
    assert {m for m, _ in appendix.FREE_TEXT_METRICS} <= free_text
    for metric, _ in appendix.TIME_METRICS:
        assert f'"{metric.removeprefix("final:")}"' in source, metric


def test_session_seconds_add_up_over_sessions(tmp_path: Path) -> None:
    from medcompress.config import RunContext
    from medcompress.models.teacher_cache import add_session_seconds

    run = RunContext("E0", seed=0)
    assert add_session_seconds(run, tmp_path, "teacher_cache_seconds", 100.0) == 100.0
    assert add_session_seconds(run, tmp_path, "teacher_cache_seconds", 50.0) == 150.0
    with (tmp_path / "E0/nobudget/seed0/metrics.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["metric"] == "teacher_cache_seconds"]
    assert len(rows) == 1 and float(rows[0]["value"]) == 150.0
