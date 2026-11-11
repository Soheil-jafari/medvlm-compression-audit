"""One float format for every results and generated CSV (pandas 2.3.3 on Kaggle, 3.x locally):
reading is correctly rounded and writing is Python's shortest round-trip repr."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from medcompress.evaluation.aggregate import classify_path, read_result_file
from medcompress.utils.io import append_results, format_cell, parse_float, write_frame

HARD = "54602.917348027004"
"""A committed value (E0's teacher_train_seconds) that pandas 3.0.5's default parser read as
54602.91734802701, a different float, so the summary drifted from its source by one ulp."""


def test_cells_are_written_the_same_way_for_every_type() -> None:
    assert format_cell(float(HARD)) == HARD  # shortest round-trip repr
    assert format_cell(np.float64(HARD)) == HARD
    assert format_cell(np.float32(0.1)) == "0.1"  # numpy's shortest text for float32, as before
    assert (
        format_cell(625.0) == "625.0" and format_cell(3) == "3" and format_cell(np.int64(7)) == "7"
    )
    assert format_cell(math.nan) == "" and format_cell(None) == "" and format_cell(np.nan) == ""
    assert format_cell(True) == "True" and format_cell("B2") == "B2"
    assert parse_float(HARD) == float(HARD) and math.isnan(parse_float(""))
    assert math.isnan(parse_float("not a number"))


def test_a_results_file_is_read_exactly_and_the_summary_written_byte_for_byte(
    tmp_path: Path,
) -> None:
    run = tmp_path / "results" / "E0" / "nobudget" / "seed0" / "metrics.csv"
    rows = [{"experiment": "E0", "budget": "", "seed": 0, "metric": "teacher_train_seconds",
             "value": float(HARD)},
            {"experiment": "E0", "budget": "", "seed": 0, "metric": "ece",
             "value": float("0.02577073614121868"), "ci_low": 0.01, "ci_high": 0.05}]  # fmt: skip
    append_results(rows, run)
    frame, problems = read_result_file(classify_path(run, tmp_path / "results"))
    assert problems == []
    assert frame["value"].tolist() == [float(HARD), float("0.02577073614121868")]  # exact
    out = write_frame(frame.drop(columns=["source", "split"]), tmp_path / "all_rows.csv")
    text = out.read_text(encoding="utf-8")
    assert f"E0,,0,teacher_train_seconds,{HARD},," in text
    assert "E0,,0,ece,0.02577073614121868,0.01,0.05" in text
    assert "\r" not in text  # the summaries keep pandas' line ending
    again = write_frame(pd.read_csv(out, dtype=str, keep_default_na=False), tmp_path / "b.csv")
    assert again.read_bytes() == out.read_bytes()  # text in, the same text out
