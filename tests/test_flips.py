"""The secondary flip analysis (pre-registered 2026-10-31): items whose answer flips against E0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_compare_stage import _arrays, _config, _write_tree

from medcompress.config import RunContext
from medcompress.evaluation import compare_stage as cs
from medcompress.evaluation import flips


def _items() -> flips.FlipItems:
    # 6 items on 4 images: findings polyp (0) / bleeding (1); strata of the positive polyps
    return flips.FlipItems(
        images=np.array(["a", "a", "b", "c", "c", "d"]),
        findings=("polyp", "bleeding"),
        codes=np.array([0, 1, 0, 0, 1, 0]),
        stratum=np.array([0, -1, 0, 1, -1, 3]),
        n_strata=4,
        reference_yes=np.zeros(6, bool),
    )


def test_flip_values_by_hand() -> None:
    items = _items()
    flipped = np.array([[True, False, True, False, True, False]])
    observed = flips.flip_values(flipped, items, np.arange(6))[0]
    values = dict(zip(flips.metric_names(items), observed, strict=True))
    assert values["flip_rate"] == pytest.approx(3 / 6)
    assert values["flip_rate:polyp"] == pytest.approx(2 / 4)
    assert values["flip_rate:bleeding"] == pytest.approx(1 / 2)
    assert values["flip_ratio:polyp"] == pytest.approx((2 / 4) / (3 / 6))
    assert values["flip_rate:stratum0"] == pytest.approx(1.0)
    assert values["flip_rate:stratum1"] == pytest.approx(0.0)
    assert np.isnan(values["flip_rate:stratum2"])  # no item in that stratum
    assert values["flip_ratio:stratum0"] == pytest.approx(1.0 / 0.5)
    # small (stratum 0: 2 of 2 flip) minus larger (strata 1-3: 0 of 2)
    assert values["diff:flip_rate:small_minus_larger"] == pytest.approx(1.0)


def test_majority_answer_from_majority_correctness() -> None:
    truth = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    assert flips.majority_yes(right, truth).tolist() == [True, False, False, True]


def test_rows_hold_counts_shares_and_intervals() -> None:
    items = _items()
    flipped = np.array([[True, False, True, False, True, False], [False] * 6])
    rows = flips.flip_rows([("E7", "B2"), ("E2", "")], flipped, items, n_resamples=200, alpha=0.05)
    e7 = {r["metric"]: r for r in rows if r["experiment"] == "E7_vs_E0"}
    assert e7["n_flips"]["value"] == 3 and e7["n_items"]["value"] == 6
    assert e7["flip_share:polyp"]["value"] == pytest.approx(2 / 3)
    assert e7["n_items:stratum0"]["value"] == 2 and e7["n_flips:stratum0"]["value"] == 2
    assert e7["flip_rate"]["ci_low"] <= 0.5 <= e7["flip_rate"]["ci_high"]
    assert all(r["budget"] == "B2" for r in e7.values())
    e2 = {r["metric"]: r for r in rows if r["experiment"] == "E2_vs_E0"}
    assert e2["n_flips"]["value"] == 0
    assert "flip_share:polyp" not in e2  # undefined without flips
    assert "flip_ratio:polyp" not in e2


def test_the_stage_writes_flip_rows_against_e0(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_tree(tmp_path, config)
    out = Path(config["paths"]["output_root"])
    reference = _arrays(0.95, seed=999)
    path = RunContext("E0", seed=0).run_dir(out) / cs.OUTCOMES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **reference)
    cs.run_comparisons(config, n_resamples=100)
    results = Path(config["paths"]["results_root"])
    choices = json.loads((results / "comparisons" / cs.CHOICES_FILE).read_text())
    assert choices["flips"]["reference"] == "E0" and "E2_vs_E0" in choices["flips"]["arms"]
    assert "E7_vs_E0" in choices["flips"]["arms"]
    frame = pd.read_csv(results / "comparisons" / "E7_vs_E0" / "B2" / "comparison.csv")
    values = dict(zip(frame["metric"], frame["value"], strict=True))
    # by hand: E7's majority answer over its 3 seeds against E0's answer
    seeds = [_arrays(0.82, seed=100 * 2 + s, flip=0.82) for s in (0, 1, 2)]
    yes = np.mean([s["probe_yes"] for s in seeds], axis=0) > 0.5
    assert values["n_flips"] == int((yes != reference["probe_yes"]).sum())
    assert values["n_items"] == len(yes)
    assert {"flip_rate", "flip_ratio:stratum0", "diff:flip_rate:small_minus_larger"} <= set(
        frame["metric"]
    )


def test_the_stage_says_why_without_e0(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_tree(tmp_path, config)
    cs.run_comparisons(config, n_resamples=50)
    results = Path(config["paths"]["results_root"])
    choices = json.loads((results / "comparisons" / cs.CHOICES_FILE).read_text())
    assert "E0 seed 0" in choices["flips"]["skipped"]
    assert not (results / "comparisons" / "E7_vs_E0").exists()


def test_no_rare_group_gives_no_rare_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    domain = Path(config["components"]["domain"])
    data = yaml.safe_load(domain.read_text())
    data["domain"]["prevalence_groups"] = {"common": ["polyp", "bleeding"], "medium": [],
                                           "rare": []}  # fmt: skip
    domain.write_text(yaml.safe_dump(data))
    _write_tree(tmp_path, config)
    cs.run_comparisons(config, n_resamples=50)
    results = Path(config["paths"]["results_root"])
    choices = json.loads((results / "comparisons" / cs.CHOICES_FILE).read_text())
    assert choices["rare_group"].startswith("none") and choices["n_rare_positives"] == 0
    frame = pd.read_csv(results / "comparisons" / "E7_vs_E2" / "B2" / "comparison.csv")
    assert "diff:drop_gap" in set(frame["metric"])
    assert "diff:drop_gap_rare" not in set(frame["metric"])
    taus = pd.read_csv(
        results / "comparisons" / "accuracy_vs_clinical" / "nobudget" / "comparison.csv"
    )
    assert set(taus["metric"]) == {"tau:small_finding_recall", "tau:ece"}
