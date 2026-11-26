"""The native-format readout (note of 2026-11-25): the parser, the rates, the first-token table,
the training-pattern count, and the exploratory analysis's tables."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from medcompress.evaluation import native_readout as nr


def test_the_native_parser_reads_the_datasets_answer_style() -> None:
    assert nr.parse_native("No evidence of polypoid lesion") == "no"
    assert nr.parse_native("no") == "no" and nr.parse_native("  No.") == "no"
    assert nr.parse_native("evidence of ulcerative colitis") == "yes"
    assert nr.parse_native("Evidence  of oesophagitis present") == "yes"
    assert nr.parse_native("Yes, a polyp") == "yes"
    assert nr.parse_native("normal mucosa") is None  # "no" must be a whole word
    assert nr.parse_native("evidently fine") is None and nr.parse_native("") is None
    assert nr.parse_native("! evidence of polyp") == "yes"  # leading non-letters dropped
    from medcompress.evaluation.clinical import parse_yes_no

    assert parse_yes_no("evidence of polypoid lesion") is None  # the pre-registered one: unparsed
    assert nr.leading_pattern("No evidence of polypoid lesion") == "no evidence"


def test_rates_unparsed_answers_are_wrong_both_ways() -> None:
    truth = np.array([True, True, False, False])
    rates = nr.native_rates(truth, ["yes", None, "no", None])
    assert (rates["sensitivity"], rates["specificity"], rates["unparsed"]) == (0.5, 0.5, 2)
    assert rates["yes_rate"] == 0.25


def test_the_first_token_table_and_the_readout_rows() -> None:
    answers = {
        "probe_findings": np.array(["polyp"] * 4),
        "probe_truth": np.array([True, True, False, False]),
        "probe_decoded_text": np.array(["evidence of polypoid lesion", "!", "No evidence", "no"]),
        "probe_first_token": np.array([68, 0, 2753, 2152]),
        "probe_first_token_text": np.array(["e", "!", "No", "no"]),
        "yes_ids": np.array([9454, 9693]),
        "no_ids": np.array([2753, 2152]),
    }
    readout, tokens = nr.readout_rows("E2 seed0", "val", answers, ["polyp"])
    assert readout[0]["sensitivity"] == 0.5 and readout[0]["specificity"] == 1.0
    assert readout[-1]["finding"] == "macro" and readout[-1]["unparsed"] == 1
    positive = next(t for t in tokens if t["label"] == "positive")
    negative = next(t for t in tokens if t["label"] == "negative")
    assert (positive["e"], positive["other"], positive["yes spelling"]) == (1, 1, 0)
    assert negative["no spelling"] == 2


def test_the_training_patterns_are_counted_without_ids() -> None:
    answers = ["evidence of polypoid lesion"] * 3 + ["No evidence of oesophagitis"] * 2 + ["none"]
    rows, coverage = nr.answer_patterns(answers)
    assert rows[0] == {"pattern": "evidence of", "count": 3, "native_reading": "yes"}
    assert coverage == {"answers": 6, "yes": 3, "no": 2, "unparsed": 1}
    assert nr.finding_questions("Is there a polyp?", ["polyp"])
    assert nr.finding_questions("Are there any abnormalities in the image?", ["polyp"])
    assert not nr.finding_questions("What color is it?", ["polyp"])


def test_the_exploratory_analysis_writes_the_native_tables(tmp_path: Path) -> None:
    from medcompress.evaluation import exploratory as ex
    from medcompress.registry import REGISTRY

    folder = tmp_path / "in" / "fbr" / "probe_answers" / "E5" / "magnitude" / "B2" / "seed0"
    folder.mkdir(parents=True)
    np.savez_compressed(
        folder / "probe_answers_val.npz",
        probe_findings=np.array(["polyp", "polyp"]), probe_truth=np.array([True, False]),
        probe_decoded_text=np.array(["evidence of polyp", "No evidence"]),
        probe_first_token=np.array([68, 2753]), probe_first_token_text=np.array(["e", "No"]),
        yes_ids=np.array([9693]), no_ids=np.array([2753]),
    )  # fmt: skip
    ex.run_all([tmp_path / "in"], tmp_path / "out", order=["polyp"], size_thresholds=[0.01],
               mask_findings=["polyp"], n_bins=15, known=list(REGISTRY), draws=5,
               resamples=10)  # fmt: skip
    with (tmp_path / "out" / "native_readout.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["run"] for r in rows} == {"E5:magnitude@B2 seed0"}
    assert float(next(r for r in rows if r["finding"] == "macro")["balanced_accuracy"]) == 1.0
    assert (tmp_path / "out" / "first_tokens.csv").is_file()
