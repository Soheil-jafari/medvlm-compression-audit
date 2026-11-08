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
        "probe_yes_op": yes,  # the answers at the operating point (2026-11-06)
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
           "E5:taylor": 0.75, "E6": 0.65, "A1": 0.78}  # fmt: skip
MAIN = {"E2", "E7", "E5:taylor", "A1", "E3:w8"}
"""3 seeds: the main tier with the comparator E5:taylor (the highest validation recall below),
and A1 at B2 for the secondary comparison E7 vs A1."""
"""The main tier with the comparator E5:taylor (the highest validation recall below)."""


def _val_rows(label: str, budget: str, small: float, mask: float, margin: float) -> list:
    return [
        {"experiment": label, "budget": budget, "seed": 0, "metric": metric, "value": value}
        for metric, value in (
            ("op_small_finding_recall", small),
            ("op_mask_positive_recall", mask),
            ("op_small_margin_mean", margin),
        )
    ]


def _write_tree(
    root: Path,
    config: dict,
    *,
    drop: tuple[str, int] | None = None,
    seeds: dict[str, tuple[int, ...]] | None = None,
    validation: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Test outcomes by the seed plan (3 seeds for :data:`MAIN`, seed 0 otherwise, or
    ``seeds``), seed 0's validation rows of the candidates and E7 at B2, the budget rows."""
    out, res = Path(config["paths"]["output_root"]), Path(config["paths"]["results_root"])
    budgets = {"E7": "B2", "E5": "B2", "E6": "B2", "A1": "B2"}
    validation = validation or {}
    for number, (label, quality) in enumerate(QUALITY.items()):
        experiment, _, variant = label.partition(":")
        planned = (seeds or {}).get(label, (0, 1, 2) if label in MAIN else (0,))
        for seed in planned:
            run = RunContext(
                experiment, seed=seed, budget=budgets.get(experiment), variant=variant or None
            )
            path = run.run_dir(out) / cs.OUTCOMES_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            if drop != (label, seed):
                np.savez_compressed(
                    path, **_arrays(quality, seed=100 * number + seed, flip=quality)
                )
            if label == "E3:w8":
                row = {"experiment": label, "budget": "", "seed": seed,
                       "metric": "peak_memory_bytes", "value": 6.2e9}  # fmt: skip
                append_results([row], run.run_dir(res) / "efficiency.csv")
        if experiment in ("E5", "E6", "E7"):  # seed 0's validation rows (RC-G2)
            run = RunContext(experiment, seed=0, budget="B2", variant=variant or None)
            small, mask, margin = validation.get(label, (quality, quality, quality))
            rows = _val_rows(label, "B2", small, mask, margin)
            rows.append({"experiment": label, "budget": "B2", "seed": 0,
                         "metric": "budget_target", "value": 6.0e9})  # fmt: skip
            append_results(rows, run.run_dir(res) / "metrics.csv")
    for budget, target in (("B1", 7.0e9), ("B3", 5.0e9)):
        row = {"experiment": "E6", "budget": budget, "seed": 0, "metric": "budget_target",
               "value": target}  # fmt: skip
        append_results([row], RunContext("E6", seed=0, budget=budget).run_dir(res) / "metrics.csv")


def _run(tmp_path: Path, **tree) -> tuple[dict, Path, list[Path]]:
    config = _config(tmp_path)
    _write_tree(tmp_path, config, **tree)
    written = cs.run_comparisons(config, n_resamples=100)
    results = Path(config["paths"]["results_root"])
    choices = json.loads((results / "comparisons" / cs.CHOICES_FILE).read_text())
    return choices, results, written


def test_the_stage_writes_the_evidence_of_every_hypothesis(tmp_path: Path) -> None:
    from medcompress.evaluation.aggregate import load_results, paper_rows, summarise_over_seeds
    from medcompress.evaluation.verdicts import EXPLORATORY, PENDING, all_verdicts

    choices, results, written = _run(tmp_path)
    assert choices["h3"]["comparator"] == "E5:taylor"  # the highest validation recall
    assert choices["h3"]["futility_stop"] is False  # E7 0.82 ranks above 0.75
    assert choices["h2"]["budget"] == "B2" and choices["h2"]["arm"] == "E5:taylor"
    assert choices["headline_accuracy"] == "macro_balanced_accuracy"
    assert choices["tiers"]["E7@B2"] == {"tier": "main", "seeds": [0, 1, 2]}
    assert choices["tiers"]["E5:magnitude@B2"] == {"tier": "exploratory", "seeds": [0]}
    folder = results / "comparisons"
    assert folder / "E5-taylor_vs_E2" / "B2" / "comparison.csv" in written  # confirmatory
    assert folder / "E7_vs_E2" / "B2" / "comparison.csv" in written
    summary = summarise_over_seeds(paper_rows(load_results(results)))
    metrics = set(summary[summary["experiment"] == "E7_vs_best"]["metric"])
    assert {"diff:small_finding_recall", "p_holm:lesion_mask_effect", "mcnemar_p_holm:polyp"} <= (
        metrics
    )
    taus = set(summary[summary["experiment"] == "accuracy_vs_clinical"]["metric"])
    assert taus == {"tau:small_finding_recall", "tau:rare_recall", "tau:ece"}
    h1, h2, h3, h4 = all_verdicts(summary)
    assert PENDING not in (h1.verdict, h2.verdict, h3.verdict)
    assert h2.verdict != EXPLORATORY  # E3 w8 and the comparator at B2 both have 3 seeds
    assert h4.verdict == EXPLORATORY
    assert choices["tiers"]["E3:w8"] == {"tier": "main", "seeds": [0, 1, 2]}
    assert "flag" not in choices["h2"]  # matched at B2: the arm is the comparator
    assert set(summary[summary["experiment"] == "E7_vs_best"]["tier"]) == {"confirmatory"}


def test_e7_against_a1_is_a_secondary_comparison_on_3_seeds(tmp_path: Path) -> None:
    from medcompress.evaluation.aggregate import load_results, paper_rows, summarise_over_seeds
    from medcompress.evaluation.verdicts import all_verdicts

    choices, results, written = _run(tmp_path)
    assert choices["secondary_e7_vs_a1"] == {"name": "E7_vs_A1", "seeds": [0, 1, 2]}
    assert choices["tiers"]["A1@B2"] == {"tier": "secondary", "seeds": [0, 1, 2]}
    assert results / "comparisons" / "E7_vs_A1" / "B2" / "comparison.csv" in written
    assert not (results / "comparisons" / "A1_vs_E2").exists()  # not an H1 arm
    summary = summarise_over_seeds(paper_rows(load_results(results)))
    rows = summary[summary["experiment"] == "E7_vs_A1"]
    assert {"diff:small_finding_recall", "p:small_finding_recall", "diff:lesion_mask_effect",
            "mcnemar_p:polyp"} <= set(rows["metric"])  # fmt: skip
    assert set(rows["tier"]) == {"secondary"}
    for verdict in all_verdicts(summary):  # never read by a verdict
        assert not any(e.startswith("E7_vs_A1") for e in verdict.evidence)
    # A1 with seed 0 only (a futility stop): the rows are exploratory
    again = tmp_path / "again"
    again.mkdir()
    choices, results, _ = _run(again, seeds={"A1": (0,)})
    assert choices["secondary_e7_vs_a1"]["name"] == "E7_vs_A1_seed0"


def test_a_seed0_only_arm_is_reported_and_not_tested(tmp_path: Path) -> None:
    from medcompress.evaluation.aggregate import load_results, paper_rows, summarise_over_seeds
    from medcompress.evaluation.verdicts import verdict_h1

    choices, results, written = _run(tmp_path)
    folder = results / "comparisons"
    for arm in ("E5-magnitude", "E5-activation", "E6"):
        assert folder / f"{arm}_vs_E2_seed0" / "B2" / "comparison.csv" in written  # reported
        assert not (folder / f"{arm}_vs_E2").exists()  # never tested against the 3-seed E2
    summary = summarise_over_seeds(paper_rows(load_results(results)))
    seed0 = summary[summary["experiment"] == "E5:magnitude_vs_E2_seed0"]
    assert not seed0.empty and set(seed0["tier"]) == {"exploratory"}
    used = verdict_h1(summary).evidence
    assert used and all(e.split()[0] in ("E7_vs_E2", "E5:taylor_vs_E2") for e in used)
    assert (folder / "best_pruning_vs_E3-w8").exists()  # H2: confirmatory on 3 seeds each
    w8 = summary[summary["experiment"] == "E3:w8_vs_E2"]
    assert not w8.empty and set(w8["tier"]) == {"secondary"}  # 3 seeds, but not an H1 arm


def test_an_arm_with_two_seeds_is_refused(tmp_path: Path) -> None:
    from medcompress.seed_plan import SeedPlanError

    config = _config(tmp_path)
    _write_tree(tmp_path, config, seeds={"E6": (0, 1)})
    with pytest.raises(SeedPlanError, match="E6@B2 has 2 seeds"):
        cs.run_comparisons(config, n_resamples=20)
    again = tmp_path / "again"
    again.mkdir()
    config = _config(again)
    _write_tree(again, config, drop=("E5:taylor", 2))  # a main arm with 2 seeds left
    with pytest.raises(SeedPlanError, match="2 seeds"):
        cs.run_comparisons(config, n_resamples=20)


def test_a_main_arm_with_seed0_only_is_exploratory_after_a_futility_stop(tmp_path: Path) -> None:
    from medcompress.evaluation.aggregate import load_results, paper_rows, summarise_over_seeds
    from medcompress.evaluation.verdicts import EXPLORATORY, all_verdicts

    # E7 ranks below the comparator on validation: seeds 1 and 2 never ran
    validation = {"E7": (0.70, 0.9, 1.0), "E5:taylor": (0.75, 0.8, 1.0)}
    choices, results, _ = _run(
        tmp_path, validation=validation, seeds={"E7": (0,), "E5:taylor": (0,), "E2": (0,)}
    )
    assert choices["h3"]["futility_stop"] is True
    assert "E7@B2: 2 of 3 seeds missing: seed 0 only, exploratory" in choices["incomplete"]
    assert (results / "comparisons" / "E7_vs_best_seed0").exists()
    assert not (results / "comparisons" / "E7_vs_best").exists()
    summary = summarise_over_seeds(paper_rows(load_results(results)))
    h1, _, h3, _ = all_verdicts(summary)
    assert h3.verdict == EXPLORATORY and h3.evidence
    assert h1.verdict == "pending"  # no 3-seed arm: no confirmatory H1


def test_the_comparator_is_chosen_from_seed0_validation_with_the_tie_breaks(
    tmp_path: Path,
) -> None:
    # all four at 13/13 (1.0): (ii) the mask-positive recall decides between E6 and taylor;
    # their (ii) tie is broken by (iii), the margin
    validation = {
        "E5:magnitude": (1.0, 0.90, 5.0),
        "E5:activation": (1.0, 0.95, 1.0),
        "E5:taylor": (1.0, 0.95, 2.0),
        "E6": (1.0, 0.95, 3.0),
        "E7": (1.0, 0.99, 0.5),
    }
    choices, _, _ = _run(
        tmp_path, validation=validation, seeds={"E5:taylor": (0,), "E6": (0, 1, 2)}
    )
    assert choices["h3"]["comparator"] == "E6"
    assert choices["h3"]["ordering"] == ["E6", "E5:taylor", "E5:activation", "E5:magnitude"]
    assert choices["tiers"]["E6@B2"]["seeds"] == [0, 1, 2]  # only the comparator has 3
    assert choices["h3"]["futility_stop"] is False  # E7 wins at (ii)


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


def test_h2_is_flagged_when_the_matched_budget_is_not_b2(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_tree(tmp_path, config)
    res = Path(config["paths"]["results_root"])
    row = {"experiment": "E3:w8", "budget": "", "seed": 0, "metric": "peak_memory_bytes",
           "value": 7.1e9}  # nearest the B1 target (7.0e9)  # fmt: skip
    for seed in (0, 1, 2):
        path = RunContext("E3", seed=seed, variant="w8").run_dir(res) / "efficiency.csv"
        path.unlink()
        append_results([{**row, "seed": seed}], path)
    for number, label in enumerate(("E5:magnitude", "E5:activation", "E5:taylor", "E6")):
        experiment, _, variant = label.partition(":")
        run = RunContext(experiment, seed=0, budget="B1", variant=variant or None)
        append_results(_val_rows(label, "B1", 0.5 + number / 10, 0.5, 0.5),
                       run.run_dir(res) / "metrics.csv")  # fmt: skip
    out = RunContext("E6", seed=0, budget="B1").run_dir(Path(config["paths"]["output_root"]))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / cs.OUTCOMES_FILE, **_arrays(0.6, seed=777))
    cs.run_comparisons(config, n_resamples=20)
    choices = json.loads((res / "comparisons" / cs.CHOICES_FILE).read_text())
    assert choices["h2"]["budget"] == "B1" and choices["h2"]["arm"] == "E6"
    assert "E6@B1 needs seeds 1 and 2" in choices["h2"]["flag"]
    assert choices["h2"]["exploratory"] == "a side has seed 0 only"  # E6@B1 has seed 0 only


def test_e2_against_e0_is_a_secondary_row_when_e0_is_there(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_tree(tmp_path, config)
    out = Path(config["paths"]["output_root"])
    path = RunContext("E0", seed=0).run_dir(out) / cs.OUTCOMES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_arrays(0.9, seed=999, flip=0.9))
    cs.run_comparisons(config, n_resamples=20)
    res = Path(config["paths"]["results_root"])
    rows = (res / "comparisons" / "E2_vs_E0" / "nobudget" / "comparison.csv").read_text()
    assert "diff:small_finding_recall" in rows and "flip_rate" in rows  # with the flips


def test_the_two_co_primary_tests_combine_by_intersection_union() -> None:
    from medcompress.evaluation import verdicts as v

    assert v.combine_tests(v.SUPPORTED, v.SUPPORTED) == v.SUPPORTED
    assert v.combine_tests(v.SUPPORTED, v.NOT_SUPPORTED) == v.PARTLY_SUPPORTED
    assert v.combine_tests(v.INCONCLUSIVE, v.SUPPORTED) == v.PARTLY_SUPPORTED
    assert v.combine_tests(v.NOT_SUPPORTED, v.NOT_SUPPORTED) == v.NOT_SUPPORTED
    assert v.combine_tests(v.NOT_SUPPORTED, v.INCONCLUSIVE) == v.INCONCLUSIVE
    assert v.combine_tests(v.PENDING, v.NOT_SUPPORTED) == v.PENDING
    assert v.combine_tests(v.EXPLORATORY, v.EXPLORATORY) == v.EXPLORATORY
