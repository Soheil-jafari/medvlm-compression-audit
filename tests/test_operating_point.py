"""The probe operating point (specificity 0.95 on validation), the trend seeds at B1 / B3, the
environment record and the training safeguards (all decided 2026-11-06)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medcompress.compression.distillation.offline_kd import TrainResult
from medcompress.evaluation import probe_suites as ps
from medcompress.registry import seeds_at

REPO = Path(__file__).resolve().parents[1]
FINDINGS = ("polyp", "ulcerative_colitis")


def _outcomes(n_images: int = 200, seed: int = 0, shift: float = -2.0) -> ps.ProbeOutcomes:
    rng = np.random.default_rng(seed)
    ids = tuple(np.repeat([f"img{i:03d}:x" for i in range(n_images)], 2))
    names = tuple(np.tile(FINDINGS, n_images))
    truth = rng.random(2 * n_images) < 0.4
    margin = rng.normal(np.where(truth, 2.0, -2.0), 1.0) + shift  # a conservative model
    masked = truth & (np.array(names) == "polyp")
    area = np.where(masked, rng.uniform(0.01, 0.3, 2 * n_images), np.nan)
    return ps.ProbeOutcomes(sample_ids=ids, findings=names, truth=truth, yes=margin > 0,
                            margin=margin, area=area, has_mask=masked)  # fmt: skip


def test_the_threshold_gives_specificity_095_on_the_validation_negatives() -> None:
    out = _outcomes()
    thresholds = ps.fit_thresholds(out, FINDINGS)
    yes = ps.operating_point_answers(out, thresholds)
    for finding in FINDINGS:
        negatives = (np.array(out.findings) == finding) & ~out.truth
        specificity = float((~yes[negatives]).mean())
        assert 0.95 <= specificity < 0.95 + 1.5 / negatives.sum()  # at least, and just above
    # the conservative plain answers miss positives that the operating point finds
    assert yes[out.truth].mean() > out.yes[out.truth].mean()


def test_metrics_at_the_operating_point_and_thresholds_unchanged_on_a_test() -> None:
    val, test = _outcomes(seed=0), _outcomes(seed=1)
    thresholds = ps.fit_thresholds(val, FINDINGS)
    rows = {m.name: m for m in ps.operating_point_metrics(
        test, FINDINGS, thresholds, size_thresholds=[0.05, 0.1, 0.2], mask_findings=["polyp"],
        n_resamples=100,
    )}  # fmt: skip
    assert rows["op_threshold:polyp"].value == thresholds["polyp"]  # the validation one
    assert rows["op_target_specificity"].value == 0.95
    low, high = rows["op_small_finding_recall"].ci_low, rows["op_small_finding_recall"].ci_high
    assert low <= rows["op_small_finding_recall"].value <= high
    assert {"op_sensitivity:polyp", "op_specificity:polyp", "op_yes_rate:polyp",
            "op_macro_balanced_accuracy", "op_recall_stratum:0"} <= set(rows)  # fmt: skip
    test_yes = ps.operating_point_answers(test, thresholds)
    pos = (np.array(test.findings) == "polyp") & test.truth
    assert rows["op_sensitivity:polyp"].value == pytest.approx(test_yes[pos].mean())


def test_a_finding_without_negatives_keeps_its_plain_answers() -> None:
    out = _outcomes()
    truth = out.truth.copy()
    truth[np.array(out.findings) == "ulcerative_colitis"] = True
    only_pos = ps.ProbeOutcomes(sample_ids=out.sample_ids, findings=out.findings, truth=truth,
                                yes=out.yes, margin=out.margin, area=out.area,
                                has_mask=out.has_mask)  # fmt: skip
    thresholds = ps.fit_thresholds(only_pos, FINDINGS)
    assert math.isnan(thresholds["ulcerative_colitis"])
    yes = ps.operating_point_answers(only_pos, thresholds)
    uc = np.array(out.findings) == "ulcerative_colitis"
    assert np.array_equal(yes[uc], out.yes[uc])


def test_three_seeds_at_b2_and_seed0_only_at_b1_and_b3() -> None:
    from medcompress.cards import build_plan

    assert seeds_at((0, 1, 2), "B2") == (0, 1, 2) and seeds_at((0, 1, 2), None) == (0, 1, 2)
    assert seeds_at((0, 1, 2), "B1") == (0,) and seeds_at((0, 1, 2), "B3") == (0,)
    for card in ("RC-E5", "RC-E6", "RC-E7", "RC-E8"):
        budgets = {
            s.label.split()[-2] for s in build_plan(card, 2).steps if s.label.startswith("run")
        }
        assert budgets == {"B2"}
        assert {s.label.split()[-2] for s in build_plan(card, 0).steps
                if s.label.startswith("run")} == {"B1", "B2", "B3"}  # fmt: skip


def test_trend_rows_are_left_out_of_the_h1_verdict() -> None:
    from medcompress.evaluation.verdicts import SUPPORTED, verdict_h1

    def row(budget: str, low: float, high: float) -> dict:
        return {"experiment": "E7_vs_E2", "budget": budget, "split": "", "metric": "diff:drop_gap",
                "n_seeds": 1, "mean": (low + high) / 2, "sd": np.nan, "min": 0, "max": 0,
                "ci_low": low, "ci_high": high}  # fmt: skip

    summary = pd.DataFrame([row("B2", 0.1, 0.3), row("B1", -0.4, -0.2)])
    assert verdict_h1(summary).verdict == SUPPORTED  # B1 (seed 0 only) is the trend


def test_every_stage_records_its_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medcompress.cli import note_environment
    from medcompress.config import RunContext
    from medcompress.utils.environment import ENVIRONMENT_FILE

    monkeypatch.setenv("MEDCOMPRESS_RECORD_ENVIRONMENT", "1")
    config = {"paths": {"results_root": str(tmp_path / "results")}}
    note_environment(config, RunContext("E1", seed=0), "02_train_student")
    note_environment(config, RunContext("E1", seed=0), "06_evaluate")
    path = tmp_path / "results" / "E1" / "nobudget" / "seed0" / ENVIRONMENT_FILE
    data = json.loads(path.read_text())
    assert set(data) == {"02_train_student", "06_evaluate"}
    record = data["06_evaluate"]
    assert {"python", "packages", "nvidia_driver"} <= set(record)
    assert {"torch", "transformers", "peft"} <= set(record["packages"])
    monkeypatch.setenv("MEDCOMPRESS_RECORD_ENVIRONMENT", "0")
    note_environment(config, RunContext("E2", seed=0), "06_evaluate")
    assert not (tmp_path / "results" / "E2").exists()


def test_the_training_counts_non_finite_steps_and_keeps_its_history() -> None:
    result = TrainResult(
        steps=12, losses=[0.9, 0.8, float("nan"), *[0.5] * 9], nonfinite_steps=[3],
        nonfinite_microbatches=1, skipped_steps=[3],
    )  # fmt: skip
    assert math.isnan(result.final_loss())  # one NaN step among the last 10, as E1 seed 0
    assert result.final_loss_finite() == pytest.approx((0.8 + 9 * 0.5) / 10)
    history = result.history()
    assert history["losses"][2] is None and history["skipped_steps"] == [3]
    json.dumps(history)  # NaN is written as null, so the file is valid JSON


def test_the_inspector_reads_a_loss_history() -> None:
    spec = importlib.util.spec_from_file_location(
        "inspect", REPO / "scripts" / "inspect_adapter.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    line = module.loss_history([0.9, None, 0.7, float("nan"), 0.5])
    assert "5 steps; 2 non-finite at steps [2, 4]" in line and "last 0.5000" in line


def test_the_skipped_step_report_maps_a_step_to_its_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("skipped", REPO / "scripts" / "skipped_steps.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "skipped", module)  # its dataclass needs the module
    spec.loader.exec_module(module)
    order = list(range(100, 140))
    assert module.step_indices(order, 1, 1, 16) == order[:16]
    assert module.step_indices(order, 3, 1, 16) == order[32:] + order[:8]  # wraps like the loop
    row = module.features(3, [(True, 12, 4), (False, 30, 9), (True, 5, 2)])
    assert (row.differs, row.teacher_tokens_max, row.gt_words_max) == (2, 30, 9)
    assert module.mid_rank(2, [1, 2, 2, 3]) == 0.5
