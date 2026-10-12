"""The committed results tree is valid, and the reader catches bad files (synthetic trees).

Run this before committing results from a GPU run: ``pytest tests/test_results.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from medcompress.evaluation.aggregate import (
    classify_path,
    load_results,
    summarise_over_seeds,
    validate_results_tree,
    write_summary,
)
from medcompress.utils.io import append_results

REPO_RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_committed_results_tree_is_valid() -> None:
    assert validate_results_tree(REPO_RESULTS) == []


def _run(root: Path, exp: str, budget: str, seed: int, value: float, metric: str = "acc") -> None:
    base, _, variant = exp.partition(":")
    parts = [base, *([variant] if variant else []), budget or "nobudget", f"seed{seed}"]
    append_results(
        [{"experiment": exp, "budget": budget, "seed": seed, "metric": metric, "value": value}],
        root.joinpath(*parts, "metrics.csv"),
    )


def test_valid_tree_and_seed_summary(tmp_path: Path) -> None:
    for seed, value in enumerate([0.70, 0.72, 0.74]):
        _run(tmp_path, "E7", "B2", seed, value)
        _run(tmp_path, "E5:taylor", "B2", seed, value - 0.1)
    _run(tmp_path, "E0", "", 0, 0.9)
    append_results(
        [{"experiment": "R1:qwen3_vl_2b", "seed": 0, "metric": "score", "value": 0.5}],
        tmp_path / "selection" / "R1" / "qwen3_vl_2b" / "metrics.csv",
    )
    append_results(
        [
            {
                "experiment": "E7_vs_best",
                "budget": "B2",
                "seed": 0,
                "metric": "diff:small_finding_recall",
                "value": 0.05,
                "ci_low": 0.01,
                "ci_high": 0.09,
            }
        ],
        tmp_path / "comparisons" / "E7_vs_best" / "B2" / "comparison.csv",
    )
    assert validate_results_tree(tmp_path) == []
    rows = load_results(tmp_path)
    summary = summarise_over_seeds(rows).set_index(["experiment", "budget", "metric"])
    e7 = summary.loc[("E7", "B2", "acc")]
    assert (
        e7["n_seeds"] == 3 and e7["mean"] == pytest.approx(0.72) and e7["sd"] == pytest.approx(0.02)
    )
    assert math.isnan(e7["ci_low"])
    assert math.isnan(summary.loc[("E0", "", "acc")]["sd"])
    comparison = summary.loc[("E7_vs_best", "B2", "diff:small_finding_recall")]
    assert comparison["ci_low"] == 0.01
    paths = write_summary(rows, tmp_path / "summary")
    assert all(p.is_file() for p in paths)
    assert validate_results_tree(tmp_path) == []  # summary outputs are ignored


def test_reader_catches_layout_and_schema_problems(tmp_path: Path) -> None:
    _run(tmp_path, "E7", "B2", 0, 0.5)
    # row says seed 1 but the folder says seed0
    append_results(
        [{"experiment": "E7", "budget": "B2", "seed": 1, "metric": "x", "value": 1}],
        tmp_path / "E7" / "B2" / "seed0" / "efficiency.csv",
    )
    # unknown experiment and variant
    _run(tmp_path, "E99", "", 0, 0.5)
    _run(tmp_path, "E5:pruneall", "B2", 0, 0.5)
    # value outside its CI
    append_results(
        [
            {
                "experiment": "E2",
                "seed": 0,
                "metric": "acc",
                "value": 0.9,
                "ci_low": 0.1,
                "ci_high": 0.2,
            }
        ],
        tmp_path / "E2" / "nobudget" / "seed0" / "metrics.csv",
    )
    # wrong columns
    bad = tmp_path / "E1" / "nobudget" / "seed0" / "metrics.csv"
    bad.parent.mkdir(parents=True)
    bad.write_text("a,b\n1,2\n")
    problems = "\n".join(validate_results_tree(tmp_path))
    assert "seed 1 != path seed0" in problems
    assert "unknown experiment 'E99'" in problems
    assert "unknown experiment 'E5:pruneall'" in problems
    assert "value outside its CI" in problems
    assert "columns" in problems
    with pytest.raises(ValueError, match="Invalid results"):
        load_results(tmp_path)


def test_classify_path(tmp_path: Path) -> None:
    run = classify_path(tmp_path / "A4" / "n64" / "B2" / "seed2" / "metrics.csv", tmp_path)
    assert run is not None and (run.experiment, run.budget, run.seed) == ("A4:n64", "B2", 2)
    assert classify_path(tmp_path / "summary" / "by_arm.csv", tmp_path) is None
    assert classify_path(tmp_path / "README.md", tmp_path) is None
