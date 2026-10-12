"""Probe suites, free-text suites and the evaluation stage, with deterministic fake models.

The fake model says yes when the image shows a bright blob at least ``min_pixels``
large, so recall by lesion size, the mask audit and the free-text suites have known
answers. The stage is run against a tiny dataset written to a temporary folder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from eval_fixtures import eval_config, write_dataset
from PIL import Image

from medcompress.config import ConfigError, RunContext
from medcompress.data.base import VQASample
from medcompress.evaluation import free_text, probe_suites
from medcompress.evaluation.aggregate import validate_results_tree
from medcompress.evaluation.clinical import macro_f1
from medcompress.evaluation.probe_suites import Metric, ProbeOutcomes
from medcompress.evaluation.runner import (
    evaluate_model,
    load_run_model,
    resolve_settings,
    run_evaluation,
    to_rows,
)
from medcompress.models.base import GenerationOutput, YesNoScore


def _outcomes(
    findings: list[str],
    truth: list[bool],
    yes: list[bool],
    *,
    margin: list[float] | None = None,
    area: list[float] | None = None,
    has_mask: list[bool] | None = None,
) -> ProbeOutcomes:
    n = len(findings)
    return ProbeOutcomes(
        sample_ids=tuple(f"s{i}" for i in range(n)),
        findings=tuple(findings),
        truth=np.array(truth, dtype=bool),
        yes=np.array(yes, dtype=bool),
        margin=np.array(margin if margin is not None else [1.0 if y else -1.0 for y in yes]),
        area=np.array(area if area is not None else [math.nan] * n, dtype=float),
        has_mask=np.array(has_mask if has_mask is not None else [False] * n, dtype=bool),
    )


# ------------------------------------------------------------------ pure suites


def test_probe_outcomes_check_their_shapes() -> None:
    with pytest.raises(ValueError, match="same length"):
        ProbeOutcomes(
            ("a",), ("f", "g"), np.zeros(1, bool), np.zeros(1, bool), np.zeros(1), np.zeros(1),
            np.zeros(1, bool),
        )  # fmt: skip
    with pytest.raises(ValueError, match="no probes"):
        _outcomes([], [], [])
    out = _outcomes(["f", "f"], [True, False], [True, True], margin=[0.0, 50.0])
    assert out.p_yes[0] == pytest.approx(0.5) and out.p_yes[1] == pytest.approx(1.0)
    assert out.correct.tolist() == [True, False]


def test_clinical_metrics_agree_with_the_reference_implementations() -> None:
    rng = np.random.default_rng(0)
    findings = list(rng.choice(["polyp", "ulcer", "stool"], size=120))
    truth = list(rng.random(120) < 0.4)
    yes = [bool(t) if rng.random() < 0.7 else not t for t in truth]
    out = _outcomes(findings, truth, yes)
    names = ["polyp", "ulcer", "stool"]
    computed = probe_suites.clinical_metrics(out, names, n_resamples=200, seed=1)
    metrics = {m.name: m for m in computed}
    expected_f1 = macro_f1(
        {f: [t for g, t in zip(findings, truth, strict=True) if g == f] for f in set(findings)},
        {f: [y for g, y in zip(findings, yes, strict=True) if g == f] for f in set(findings)},
    )
    assert metrics["macro_f1"].value == pytest.approx(expected_f1)
    accuracy = np.mean([t == y for t, y in zip(truth, yes, strict=True)])
    assert metrics["overall_accuracy"].value == pytest.approx(accuracy)
    for name in ("overall_accuracy", "macro_f1"):
        m = metrics[name]
        assert m.ci_low is not None and m.ci_low <= m.value <= m.ci_high
    assert metrics["n_probes_scored"].value == 120
    sens = [y for f, t, y in zip(findings, truth, yes, strict=True) if f == "polyp" and t]
    assert metrics["sensitivity:polyp"].value == pytest.approx(np.mean(sens))
    assert {"specificity:ulcer", "sensitivity:stool"} <= set(metrics)


def test_small_finding_recall_by_lesion_size() -> None:
    # six mask-annotated positives of "polyp" in areas 1..6 percent, plus noise rows
    areas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    detected = [False, False, True, True, True, True]
    out = _outcomes(
        ["polyp"] * 6 + ["polyp", "ulcer"],
        [True] * 6 + [False, True],
        detected + [True, False],
        area=[*areas, math.nan, 0.01],
        has_mask=[True] * 6 + [False, True],
    )
    metrics = {
        m.name: m
        for m in probe_suites.small_finding_metrics(
            out, [0.025, 0.045, 0.055], ["polyp"], n_resamples=100, seed=0
        )
    }
    # strata: {0.01, 0.02} {0.03, 0.04} {0.05} {0.06}; the "ulcer" row is not a mask finding
    assert metrics["n_mask_positives"].value == 6
    assert metrics["small_finding_recall"].value == 0.0
    assert metrics["recall_stratum:1"].value == 1.0 and metrics["recall_stratum:3"].value == 1.0
    assert metrics["n_stratum:0"].value == 2 and metrics["n_stratum:2"].value == 1
    with pytest.raises(ValueError, match="no positive"):
        probe_suites.small_finding_metrics(
            _outcomes(["polyp"], [False], [False]), [0.1], ["polyp"], n_resamples=10
        )


def test_rare_finding_recall_by_group() -> None:
    out = _outcomes(
        ["polyp", "polyp", "polyp", "ulcer", "ulcer", "stool"],
        [True, True, False, True, True, True],
        [True, False, True, False, False, True],
    )
    groups = {"common": ["polyp"], "medium": [], "rare": ["ulcer", "stool"]}
    rows = {m.name: m.value for m in probe_suites.rare_finding_metrics(out, groups)}
    assert rows["rare_recall:common"] == 0.5 and rows["rare_recall:rare"] == pytest.approx(1 / 3)
    assert "rare_recall:medium" not in rows  # a group without findings has no row
    mean = {
        m.name: m.value
        for m in probe_suites.rare_finding_metrics(out, groups, pooling="mean_per_finding")
    }
    assert mean["rare_recall:rare"] == pytest.approx((0 + 1) / 2)
    with pytest.raises(ValueError, match="two prevalence groups"):
        probe_suites.group_of_finding({"common": ["a"], "rare": ["a"]})


def test_calibration_metrics() -> None:
    sure = _outcomes(["f"] * 4, [True, False, True, False], [True, False, True, False],
                     margin=[40.0, -40.0, 40.0, -40.0])  # fmt: skip
    rows = {m.name: m.value for m in probe_suites.calibration_metrics(sure, n_bins=5)}
    assert rows["ece"] == pytest.approx(0.0, abs=1e-9)
    assert rows["brier"] == pytest.approx(0.0, abs=1e-9) and rows["aurc"] == 0.0
    wrong = _outcomes(["f"] * 2, [True, True], [False, False], margin=[-40.0, -40.0])
    rows = {m.name: m.value for m in probe_suites.calibration_metrics(wrong, n_bins=5)}
    assert rows["ece"] == pytest.approx(1.0) and rows["aurc"] == 1.0


# ------------------------------------------------------------ free-text suites


def _qa(i: int, answer: str, findings: dict[str, bool]) -> VQASample:
    return VQASample(f"q{i}", "d", f"images/{i}.png", "q", answer, findings=findings)


def test_qa_rows_are_sampled_the_same_way_for_every_model() -> None:
    rows = [_qa(i, "a", {}) for i in range(30)]
    first = free_text.sample_qa_rows(rows, 8, seed=0)
    assert first == free_text.sample_qa_rows(rows, 8, seed=0) and len(first) == 8
    assert free_text.sample_qa_rows(rows, 8, seed=1) != first
    assert free_text.sample_qa_rows(rows, None, seed=0) == rows
    assert free_text.sample_qa_rows(rows, 99, seed=0) == rows
    assert [r.sample_id for r in first] == sorted(r.sample_id for r in first)  # keeps dataset order


def test_language_and_hallucination_metrics() -> None:
    rows = [
        _qa(0, "polyp", {"polyp": True}),
        _qa(1, "no polyp", {"polyp": False}),
        _qa(2, "polyp", {"polyp": True}),
    ]
    predictions = ["polyp", "polyp seen", "no polyp"]
    language = {m.name: m.value for m in free_text.language_metrics(predictions, ["polyp"] * 3)}
    assert language["exact_match"] == pytest.approx(1 / 3) and language["n_answers"] == 3
    assert language["answer_words_mean"] == pytest.approx(5 / 3)
    errors = {
        m.name: m.value
        for m in free_text.hallucination_metrics(predictions, rows, ["polyp"], {"polyp": "polyp"})
    }
    assert errors["hallucination_rate"] == pytest.approx(1 / 3)  # row 1 asserts an absent polyp
    assert errors["omission_rate"] == pytest.approx(1 / 3)  # row 2 denies a present polyp
    assert errors["negation_rate"] == pytest.approx(2 / 3)  # rows 1 and 2 flip polarity


def test_to_rows_leaves_out_undefined_metrics_and_repairs_odd_intervals() -> None:
    metrics = [
        Metric("fine", 0.5, 0.4, 0.6),
        Metric("undefined", math.nan),
        Metric("outside", 0.9, 0.5, 0.8),
        Metric("no_interval", 0.1, math.nan, math.nan),
        Metric("plain", 2.0),
    ]
    rows = {r["metric"]: r for r in to_rows(metrics, "E2", "", 3)}
    assert set(rows) == {"fine", "outside", "no_interval", "plain"}
    assert (rows["outside"]["ci_low"], rows["outside"]["ci_high"]) == (0.5, 0.9)
    assert "ci_low" not in rows["no_interval"] and "ci_low" not in rows["plain"]
    assert all(r["experiment"] == "E2" and r["seed"] == 3 for r in rows.values())


# ----------------------------------------------------------------- the stage


class Detector:
    """Says yes iff the image has at least ``min_pixels`` bright pixels (a size-limited reader)."""

    def __init__(self, min_pixels: int) -> None:
        self.min_pixels = min_pixels

    def _yes(self, image: object) -> bool:
        if isinstance(image, str | Path):
            with Image.open(image) as handle:
                image = handle.convert("L")
        else:
            image = image.convert("L")  # type: ignore[attr-defined]
        return int((np.asarray(image) > 200).sum()) >= self.min_pixels

    def score_yes_no(self, images, questions):  # noqa: ANN001, ANN201
        return [YesNoScore(-0.1, -3.0) if self._yes(i) else YesNoScore(-3.0, -0.1) for i in images]

    def generate(self, images, questions, *, max_new_tokens, return_topk=None):  # noqa: ANN001, ANN201
        return [
            GenerationOutput("yes there is a polyp" if self._yes(i) else "no polyp is visible")
            for i in images
        ]


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, list[VQASample]]:
    return tmp_path, write_dataset(tmp_path)


def test_settings_name_every_open_value(tmp_path: Path) -> None:
    config = eval_config(tmp_path, bootstrap_resamples="TODO", calibration_bins="TODO")
    config["evaluation"].pop("generation_samples")
    config["image_reliance_check"]["mask_findings"] = "TODO"
    with pytest.raises(ConfigError) as info:
        resolve_settings(config)
    message = str(info.value)
    for name in (
        "evaluation.bootstrap_resamples",
        "evaluation.calibration_bins",
        "evaluation.generation_samples",
        "image_reliance_check.mask_findings",
    ):
        assert name in message


def test_settings_only_ask_for_what_the_enabled_suites_need(tmp_path: Path) -> None:
    config = eval_config(
        tmp_path, suites=["clinical"], calibration_bins="TODO", max_new_tokens="TODO"
    )
    config["image_reliance_check"]["mask_findings"] = "TODO"
    settings = resolve_settings(config)
    assert settings.suites == ("clinical",) and settings.n_resamples == 60
    assert settings.thresholds is None and settings.n_bins is None and settings.fill is None


def test_settings_reject_bad_requests(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown evaluation suites"):
        resolve_settings(eval_config(tmp_path, suites=["clinical", "vibes"]))
    with pytest.raises(ConfigError, match="split must be"):
        resolve_settings(eval_config(tmp_path, split="holdout"))
    config = eval_config(tmp_path)
    config["image_reliance_check"]["mask_findings"] = ["lesion"]
    with pytest.raises(ConfigError, match="must be in domain.findings"):
        resolve_settings(config)
    unsorted = eval_config(tmp_path, domain={"small_findings": {"thresholds": [0.3, 0.1]}})
    with pytest.raises(ConfigError, match="sorted"):
        resolve_settings(unsorted)
    empty = eval_config(
        tmp_path,
        domain={
            "small_findings": {"thresholds": []},
            "prevalence_groups": {"common": [], "medium": [], "rare": []},
        },
    )
    with pytest.raises(ConfigError) as info:
        resolve_settings(empty)
    assert "small_findings.thresholds" in str(info.value)
    assert "prevalence_groups" in str(info.value)


def test_test_split_is_locked_until_the_preregistration_is_frozen(tmp_path: Path) -> None:
    config = eval_config(tmp_path, split="test")
    with pytest.raises(ConfigError, match="locked"):
        resolve_settings(config)
    (tmp_path / "weights.yaml").write_text(
        yaml.safe_dump({"preregistration": {"frozen_on": "2026-11-01"}})
    )
    assert resolve_settings(config).split == "test"


def test_stage_writes_every_suite_and_the_outcomes(dataset: tuple[Path, list[VQASample]]) -> None:
    root, samples = dataset
    run = RunContext("E2", seed=1)
    run_evaluation(eval_config(root), run, model=Detector(min_pixels=20), samples=samples)
    table = pd.read_csv(root / "results" / "E2" / "nobudget" / "seed1" / "metrics.csv")
    values = dict(zip(table["metric"], table["value"], strict=True))
    assert validate_results_tree(root / "results") == []
    assert (
        values["n_probes_scored"] == 24
    )  # 12 images x 2 findings; two QA rows per image count once
    # blob widths 3, 6, 9, ... give 12, 24, 36, ... bright pixels: the first lesion is missed
    assert values["small_finding_recall"] == 0.0 and values["n_mask_positives"] == 6
    assert values["recall_stratum:3"] == 1.0
    assert values["sensitivity:polyp"] == pytest.approx(5 / 6)
    assert values["specificity:polyp"] == 1.0
    # "bleeding" is positive on images 0, 3, 6, 9 and the reader only sees the blob of image 6
    assert values["rare_recall:rare"] == pytest.approx(0.25)
    assert "rare_recall:medium" not in values
    assert values["lesion_mask_effect"] > 0 and values["blank_accuracy_drop"] >= 0
    assert 0 <= values["ece"] <= 1 and values["brier"] >= 0
    # 24 QA rows; the two rows of image 0 (a missed lesion) are the only wrong answers
    assert values["exact_match"] == pytest.approx(22 / 24)
    assert values["omission_rate"] == pytest.approx(2 / 24)
    assert values["answer_words_mean"] > 0 and "hallucination_rate" in values
    small = table[table["metric"] == "small_finding_recall"].iloc[0]
    assert small["ci_low"] <= small["value"] <= small["ci_high"]
    outcomes = np.load(root / "outputs" / "E2" / "nobudget" / "seed1" / "outcomes.npz")
    assert len(outcomes["probe_yes"]) == 24 and len(outcomes["audit_lesion_flip"]) == 6
    assert outcomes["probe_findings"].dtype.kind == "U"


def test_rerunning_replaces_rows_and_a_better_reader_scores_higher(
    dataset: tuple[Path, list[VQASample]],
) -> None:
    root, samples = dataset
    run = RunContext("E2", seed=0)
    for pixels in (20, 1):
        run_evaluation(eval_config(root), run, model=Detector(min_pixels=pixels), samples=samples)
    table = pd.read_csv(root / "results" / "E2" / "nobudget" / "seed0" / "metrics.csv")
    assert table["metric"].is_unique
    values = dict(zip(table["metric"], table["value"], strict=True))
    assert values["small_finding_recall"] == 1.0  # the last run read every lesion


def test_free_text_suites_use_the_same_rows_for_every_model(
    dataset: tuple[Path, list[VQASample]], capsys: pytest.CaptureFixture[str]
) -> None:
    root, samples = dataset
    config = eval_config(root, suites=["language"], generation_samples=7)
    run_evaluation(config, RunContext("E1", seed=0), model=Detector(1), samples=samples)
    table = pd.read_csv(root / "results" / "E1" / "nobudget" / "seed0" / "metrics.csv")
    assert dict(zip(table["metric"], table["value"], strict=True))["n_answers"] == 7
    assert "split val" in capsys.readouterr().out
    assert not (root / "outputs").exists()  # no probe outcomes without probe suites


def test_stage_needs_a_run_and_a_loadable_model(dataset: tuple[Path, list[VQASample]]) -> None:
    root, samples = dataset
    with pytest.raises(ConfigError, match="--exp"):
        run_evaluation(eval_config(root), None, model=Detector(1), samples=samples)
    with pytest.raises(ConfigError, match="image_longest_side"):
        run_evaluation(eval_config(root), RunContext("E2", seed=0), samples=samples)
    with pytest.raises(ConfigError, match="model.config"):
        load_run_model(eval_config(root), RunContext("E2", seed=0), 448)
    with pytest.raises(NotImplementedError, match="E7"):
        load_run_model(eval_config(root), RunContext("E7", seed=0, budget="B2"), 448)


def test_blank_image_and_mask_audit_suites_run_on_their_own(
    dataset: tuple[Path, list[VQASample]],
) -> None:
    root, samples = dataset
    config = eval_config(root, suites=["clinical", "small_findings", "blank_image", "mask_audit"])
    settings = resolve_settings(config)
    result = evaluate_model(Detector(min_pixels=20), samples, settings, root, seed=0)
    values = {m.name: m.value for m in result.metrics}
    assert values["blank_accuracy_drop"] > 0 and values["lesion_mask_effect"] > 0
    assert result.probes is not None and result.probes.yes_blank is not None
    assert not result.probes.yes_blank.any()  # a blank image shows no blob
    assert len(result.arrays["probe_yes_blank"]) == 24
    assert len(result.arrays["audit_lesion_flip"]) == 6
    assert "blank_balanced_accuracy" in values and "n_probes_scored" in values
    both = eval_config(root, suites=["image_reliance", "blank_image"])
    with pytest.raises(ConfigError, match="already contains"):
        resolve_settings(both)


def test_audit_rows_do_not_depend_on_the_run_seed(
    dataset: tuple[Path, list[VQASample]],
) -> None:
    root, samples = dataset
    config = eval_config(root, suites=["mask_audit"])
    config["image_reliance_check"]["max_mask_probes"] = 4
    settings = resolve_settings(config)
    first = evaluate_model(Detector(20), samples, settings, root, seed=0).arrays
    second = evaluate_model(Detector(20), samples, settings, root, seed=7).arrays
    assert first["audit_sample_ids"].tolist() == second["audit_sample_ids"].tolist()
    assert len(first["audit_sample_ids"]) == 4
