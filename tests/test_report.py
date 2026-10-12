"""Generated LaTeX tables and number macros (synthetic results)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from medcompress.evaluation.aggregate import load_results, summarise_over_seeds
from medcompress.evaluation.report import (
    GENERATED_HEADER,
    MISSING,
    ablation_table,
    fmt_mean_sd,
    macro_name,
    main_table,
    numbers_tex,
    primary_table,
)
from medcompress.evaluation.statistics import aggregate_results
from medcompress.utils.io import append_results


def _tree(root: Path) -> None:
    for seed, value in enumerate([0.70, 0.72, 0.74]):
        append_results(
            [
                {
                    "experiment": "E7",
                    "budget": "B2",
                    "seed": seed,
                    "metric": "small_finding_recall",
                    "value": value,
                }
            ],
            root / "E7" / "B2" / f"seed{seed}" / "metrics.csv",
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
            },
            {
                "experiment": "E7_vs_best",
                "budget": "B2",
                "seed": 0,
                "metric": "p_holm:small_finding_recall",
                "value": 0.012,
            },
        ],
        root / "comparisons" / "E7_vs_best" / "B2" / "comparison.csv",
    )


def test_empty_results_give_placeholder_tables() -> None:
    empty = pd.DataFrame(
        columns=["experiment", "budget", "metric", "mean", "sd", "ci_low", "ci_high"]
    )
    text = main_table(empty)
    assert text.startswith(GENERATED_HEADER) and MISSING in text and "\\toprule" in text


def test_tables_contain_generated_numbers(tmp_path: Path) -> None:
    _tree(tmp_path / "results")
    summary = summarise_over_seeds(load_results(tmp_path / "results"))
    assert "0.720 $\\pm$ 0.020" in main_table(summary)
    primary = primary_table(summary)
    assert "0.050 [0.010, 0.090]" in primary and "0.012" in primary
    macros = numbers_tex(summary)
    assert "\\newcommand{\\resESevenBTwoSmallFindingRecall}{0.720}" in macros
    assert "\\newcommand{\\resESevenVsBestBTwoDiffSmallFindingRecallLow}{0.010}" in macros


def test_arms_identical_to_e7_show_e7_only_without_runs_of_their_own(tmp_path: Path) -> None:
    _tree(tmp_path / "results")
    summary = summarise_over_seeds(load_results(tmp_path / "results"))
    lines = ablation_table(summary).splitlines()
    reused = next(line for line in lines if line.startswith("A4:n128"))
    assert "(= E7)" in reused and "0.720 $\\pm$ 0.020" in reused
    assert "(= E7)" not in next(line for line in lines if line.startswith("A4:n64"))
    append_results(
        [{"experiment": "A4:n128", "budget": "B2", "seed": 0,
          "metric": "small_finding_recall", "value": 0.5}],
        tmp_path / "results" / "A4" / "n128" / "B2" / "seed0" / "metrics.csv",
    )  # fmt: skip
    summary = summarise_over_seeds(load_results(tmp_path / "results"))
    own = next(r for r in ablation_table(summary).splitlines() if r.startswith("A4:n128"))
    assert "(= E7)" not in own and "0.500" in own


def test_fmt_and_macro_names() -> None:
    assert fmt_mean_sd({"mean": 0.5, "sd": float("nan")}) == "0.500"
    assert fmt_mean_sd(None) == MISSING
    assert (
        macro_name("E5:taylor", "B2", "small_finding_recall")
        == "resEFiveTaylorBTwoSmallFindingRecall"
    )


def _test_split_copy(root: Path) -> None:
    """Move the E7 rows to ``metrics_test.csv`` and put different validation rows in their place."""
    for seed in range(3):
        run = root / "E7" / "B2" / f"seed{seed}"
        (run / "metrics.csv").rename(run / "metrics_test.csv")
        append_results(
            [
                {
                    "experiment": "E7",
                    "budget": "B2",
                    "seed": seed,
                    "metric": "small_finding_recall",
                    "value": 0.111,
                }
            ],
            run / "metrics.csv",
        )


def test_aggregate_results_end_to_end(tmp_path: Path) -> None:
    _tree(tmp_path / "results")
    _test_split_copy(tmp_path / "results")
    written = aggregate_results(
        tmp_path / "results",
        tmp_path / "summary",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        verdicts_path=tmp_path / "VERDICTS.md",
    )
    names = {p.name for p in written}
    assert {"by_arm.csv", "main.tex", "primary.tex", "ablations.tex", "numbers.tex"} <= names
    main = (tmp_path / "tables" / "main.tex").read_text()
    # the paper reads the test split only; the validation rows never reach it
    assert "0.720" in main and "0.111" not in main
    assert "0.111" not in (tmp_path / "tables" / "numbers.tex").read_text()
    by_arm = pd.read_csv(tmp_path / "summary" / "by_arm.csv", keep_default_na=False)
    e7 = by_arm[(by_arm["experiment"] == "E7") & (by_arm["metric"] == "small_finding_recall")]
    assert sorted(zip(e7["split"], e7["mean"].round(3), strict=True)) == [
        ("test", 0.72),
        ("val", 0.111),
    ]
    assert (tmp_path / "figures" / "budget_curves.pdf").read_bytes().startswith(b"%PDF")
    assert "## H3: pending" in (tmp_path / "VERDICTS.md").read_text()
