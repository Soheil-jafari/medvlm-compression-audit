"""Stage 09: paired comparisons from per-probe outcomes (synthetic outcomes, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from eval_fixtures import eval_config

from medcompress.config import ConfigError, RunContext
from medcompress.evaluation import compare_stage as cs
from medcompress.evaluation.probe_suites import (
    ProbeOutcomes,
    calibration_metrics,
    clinical_metrics,
    rare_finding_metrics,
    small_finding_metrics,
)
from medcompress.evaluation.statistics import paired_bootstrap_test
from medcompress.utils.io import append_results

FINDINGS = ["polyp", "bleeding"]
N_IMAGES = 80
LABEL_SEED = 5


def _labels() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(LABEL_SEED)
    ids = np.repeat([f"s{i:03d}" for i in range(N_IMAGES)], len(FINDINGS))
    names = np.tile(FINDINGS, N_IMAGES)
    truth = np.where(names == "polyp", rng.random(len(ids)) < 0.5, rng.random(len(ids)) < 0.3)
    masked = truth & (names == "polyp")
    area = np.where(masked, rng.uniform(0.001, 0.03, len(ids)), np.nan)
    return {"ids": ids, "names": names, "truth": truth, "masked": masked, "area": area}


def _arrays(quality: float, seed: int, flip: float = 0.5) -> dict[str, np.ndarray]:
    """One run's outcomes: answers right with probability ``quality``."""
    labels = _labels()
    rng = np.random.default_rng(seed)
    truth = labels["truth"]
    yes = np.where(rng.random(len(truth)) < quality, truth, ~truth)
    margin = np.where(yes, 1.0, -1.0) * rng.uniform(0.1, 3.0, len(truth))
    audited = np.flatnonzero(labels["masked"])
    return {
        "probe_sample_ids": labels["ids"],
        "probe_findings": labels["names"],
        "probe_truth": truth,
        "probe_yes": yes,
        "probe_margin": margin,
        "probe_area": labels["area"],
        "probe_has_mask": labels["masked"],
        "audit_sample_ids": labels["ids"][audited],
        "audit_findings": labels["names"][audited],
        "audit_original_yes": yes[audited],
        "audit_lesion_flip": rng.random(len(audited)) < flip,
        "audit_random_flip": rng.random(len(audited)) < 0.1,
    }


def _config(root: Path) -> dict:
    config = eval_config(root, bootstrap_resamples=200, calibration_bins=5)
    config["evaluation"]["headline_accuracy"] = "macro_balanced_accuracy"
    (root / "weights.yaml").write_text(
        yaml.safe_dump({"preregistration": {"frozen_on": "2026-10-20"}})
    )
    return config


def _outcomes(arrays: dict[str, np.ndarray]) -> ProbeOutcomes:
    return ProbeOutcomes(
        sample_ids=tuple(arrays["probe_sample_ids"]),
        findings=tuple(arrays["probe_findings"]),
        truth=arrays["probe_truth"],
        yes=arrays["probe_yes"],
        margin=arrays["probe_margin"],
        area=arrays["probe_area"],
        has_mask=arrays["probe_has_mask"],
    )


def _table(config: dict, arrays: dict[str, np.ndarray]) -> tuple[cs.ProbeTable, object]:
    settings = cs._settings(config)
    table = cs.build_table(
        arrays, settings.findings, settings.thresholds, settings.mask_findings, settings.groups
    )
    return table, settings


def test_the_metrics_equal_those_of_the_evaluation_stage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    arrays = _arrays(0.8, seed=1)
    table, settings = _table(config, arrays)
    arm = cs.arm_outcomes(cs.Arm("E2"), [0], [arrays], table)
    out = _outcomes(arrays)
    every = np.arange(len(table.keys))
    expected = {}
    for suite in (
        clinical_metrics(out, FINDINGS, n_resamples=10),
        small_finding_metrics(out, settings.thresholds, ["polyp"], n_resamples=10),
        rare_finding_metrics(out, settings.groups),
        calibration_metrics(out, n_bins=5),
    ):
        expected |= {m.name: m.value for m in suite}
    for headline, key in (
        ("macro_balanced_accuracy", "macro_balanced_accuracy"),
        ("overall_accuracy", "overall_accuracy"),
    ):
        scorer = cs.Scorer(table, headline, 5)
        values = scorer.values(arm, scorer.bins(arm), every)
        assert values[cs.HEAD] == pytest.approx(expected[key])
        assert values[cs.SMALL_COL] == pytest.approx(expected["small_finding_recall"])
        assert values[cs.RARE_COL] == pytest.approx(expected["rare_recall:rare"])
        assert values[cs.ECE_COL] == pytest.approx(expected["ece"])


def test_paired_rows_use_the_same_resamples_as_the_paired_bootstrap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    a_arrays, b_arrays = _arrays(0.9, seed=1), _arrays(0.6, seed=2)
    table, _ = _table(config, a_arrays)
    a = cs.arm_outcomes(cs.Arm("E7", "B2"), [0], [a_arrays], table)
    b = cs.arm_outcomes(cs.Arm("E6", "B2"), [0], [b_arrays], table)
    scorer = cs.Scorer(table, "overall_accuracy", 5)
    observed, draws = cs.metric_table(scorer, [a, b], n_resamples=300)
    rows = {r["metric"]: r for r in cs.paired_rows(observed, draws, 0, 1, "x", "B2", 0.05)}

    def recall(data: np.ndarray) -> float:
        hits = data[:, 1] > 0
        return float(data[hits, 0].mean()) if hits.any() else float("nan")

    stack = [np.stack([o.correct, table.small], axis=1) for o in (a, b)]
    diff, low, high, p = paired_bootstrap_test(
        recall, *stack, n_resamples=300, seed=0, clusters=table.images
    )
    got = rows["diff:small_finding_recall"]
    assert got["value"] == pytest.approx(diff)
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((low, high))
    assert rows["p:small_finding_recall"]["value"] == pytest.approx(p)


def test_seeds_are_averaged_per_probe_and_must_ask_the_same_probes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runs = [_arrays(0.7, seed=s) for s in (1, 2, 3)]
    table, _ = _table(config, runs[0])
    arm = cs.arm_outcomes(cs.Arm("E2"), [0, 1, 2], runs, table)
    correct = np.mean([r["probe_yes"] == r["probe_truth"] for r in runs], axis=0)
    assert np.allclose(arm.correct, correct)
    assert np.array_equal(arm.majority, correct > 0.5)
    shuffled = {k: v.copy() for k, v in runs[1].items()}
    order = np.random.default_rng(0).permutation(len(shuffled["probe_truth"]))
    for key in [k for k in shuffled if k.startswith("probe_")]:
        shuffled[key] = shuffled[key][order]
    again = cs.arm_outcomes(cs.Arm("E2"), [1], [shuffled], table)
    assert np.allclose(again.correct, (runs[1]["probe_yes"] == runs[1]["probe_truth"]))
    fewer = {k: (v[:-2] if k.startswith("probe_") else v) for k, v in runs[1].items()}
    with pytest.raises(ValueError, match="not the same probes"):
        cs.arm_outcomes(cs.Arm("E2"), [1], [fewer], table)


def test_the_drop_gap_is_positive_when_recall_falls_more_than_the_headline() -> None:
    observed = np.array([[0.80, 0.60, 0.5, 0.1], [0.78, 0.30, 0.5, 0.1]])  # arm 1 vs reference 0
    draws = np.repeat(observed[None], 5, axis=0)
    rows = {r["metric"]: r for r in cs.drop_gap_rows(observed, draws, 1, 0, "E7_vs_E2", "B2", 0.05)}
    assert rows["diff:drop_gap"]["value"] == pytest.approx(0.5 - 0.025)
    assert rows["diff:drop_gap_rare"]["value"] == pytest.approx(0.0 - 0.025)


# ------------------------------------------------------------------ the whole stage


QUALITY = {"E2": 0.85, "E3:w8": 0.8, "E7": 0.82, "E5:magnitude": 0.7, "E5:activation": 0.72,
           "E5:taylor": 0.75, "E6": 0.65}  # fmt: skip


def _write_tree(root: Path, config: dict, *, drop: tuple[str, int] | None = None) -> None:
    out, res = Path(config["paths"]["output_root"]), Path(config["paths"]["results_root"])
    budgets = {"E7": "B2", "E5": "B2", "E6": "B2"}
    for number, (label, quality) in enumerate(QUALITY.items()):
        experiment, _, variant = label.partition(":")
        for seed in (0, 1, 2):
            run = RunContext(
                experiment, seed=seed, budget=budgets.get(experiment), variant=variant or None
            )
            path = run.run_dir(out) / cs.OUTCOMES_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            if drop != (label, seed):  # the test outcomes only; validation rows stay
                np.savez_compressed(
                    path, **_arrays(quality, seed=100 * number + seed, flip=quality)
                )
            if experiment in ("E5", "E6"):
                rows = [
                    {"experiment": label, "budget": "B2", "seed": seed,
                     "metric": "small_finding_recall", "value": quality},
                    {"experiment": label, "budget": "B2", "seed": seed,
                     "metric": "budget_target", "value": 6.0e9},
                ]  # fmt: skip
                append_results(rows, run.run_dir(res) / "metrics.csv")
            if label == "E3:w8":
                row = {"experiment": label, "budget": "", "seed": seed,
                       "metric": "peak_memory_bytes", "value": 6.2e9}  # fmt: skip
                append_results([row], run.run_dir(res) / "efficiency.csv")
    for budget, target in (("B1", 7.0e9), ("B3", 5.0e9)):
        row = {"experiment": "E6", "budget": budget, "seed": 0, "metric": "budget_target",
               "value": target}  # fmt: skip
        append_results([row], RunContext("E6", seed=0, budget=budget).run_dir(res) / "metrics.csv")


def test_the_stage_writes_the_evidence_of_every_hypothesis(tmp_path: Path) -> None:
    from medcompress.evaluation.aggregate import load_results, paper_rows, summarise_over_seeds
    from medcompress.evaluation.verdicts import all_verdicts

    config = _config(tmp_path)
    _write_tree(tmp_path, config)
    written = cs.run_comparisons(config, n_resamples=100)
    results = Path(config["paths"]["results_root"])
    choices = json.loads((results / "comparisons" / cs.CHOICES_FILE).read_text())
    assert choices["h3"]["comparator"] == "E5:taylor"  # the highest validation recall
    assert choices["h2"]["budget"] == "B2" and choices["h2"]["arm"] == "E5:taylor"
    assert choices["headline_accuracy"] == "macro_balanced_accuracy"
    assert results / "comparisons" / "E5-taylor_vs_E2" / "B2" / "comparison.csv" in written
    summary = summarise_over_seeds(paper_rows(load_results(results)))
    metrics = set(summary[summary["experiment"] == "E7_vs_best"]["metric"])
    assert {"diff:small_finding_recall", "p_holm:lesion_mask_effect", "mcnemar_p_holm:polyp"} <= (
        metrics
    )
    taus = set(summary[summary["experiment"] == "accuracy_vs_clinical"]["metric"])
    assert taus == {"tau:small_finding_recall", "tau:rare_recall", "tau:ece"}
    assert all(v.verdict != "pending" for v in all_verdicts(summary))


def test_an_arm_with_a_missing_seed_is_left_out_and_listed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_tree(tmp_path, config, drop=("E5:taylor", 2))
    cs.run_comparisons(config, n_resamples=50)
    results = Path(config["paths"]["results_root"])
    choices = json.loads((results / "comparisons" / cs.CHOICES_FILE).read_text())
    assert "E5:taylor@B2: 1 of 3 seeds missing" in choices["incomplete"]
    assert "E5:taylor@B2" not in choices["arms"]
    # the comparator is still chosen on validation; its test outcomes are incomplete
    assert "incomplete" in choices["h3"]["skipped"]
    assert not (results / "comparisons" / "E7_vs_best").exists()


def test_the_stage_needs_the_freeze_and_a_known_headline(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["evaluation"]["headline_accuracy"] = "TODO"
    with pytest.raises(ConfigError, match="headline_accuracy"):
        cs.headline_metric(config)
    config["evaluation"]["headline_accuracy"] = "overall_accuracy"
    (tmp_path / "weights.yaml").write_text(
        yaml.safe_dump({"preregistration": {"frozen_on": "TODO"}})
    )
    with pytest.raises(ConfigError, match="locked"):
        cs.run_comparisons(config)
