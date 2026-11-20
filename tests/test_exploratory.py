"""Exploratory analyses A and B (amendment of 2026-11-20) on synthetic outcomes."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from medcompress.evaluation import exploratory as ex
from medcompress.evaluation import probe_suites
from medcompress.evaluation.probe_suites import ProbeOutcomes

ORDER = ("polyp", "ulcerative_colitis", "oesophagitis")
SIZES = (0.05, 0.1, 0.2)


def _outcomes(n_images: int, seed: int, *, shift: float = 1.5, thresholds=None):  # noqa: ANN001, ANN202
    """Synthetic probe outcomes: three findings per image, margins shifted by the truth."""
    rng = np.random.default_rng(seed)
    ids, findings, truth = [], [], []
    for i in range(n_images):
        present = rng.integers(0, 3)
        for k, finding in enumerate(ORDER):
            ids.append(f"img{i:03d}")
            findings.append(finding)
            truth.append(k == present)
    truth_arr = np.array(truth)
    margin = rng.normal(0.0, 1.0, len(ids)) + shift * truth_arr - 1.0
    area = np.where(np.array(findings) == "polyp", rng.uniform(0.01, 0.3, len(ids)), np.nan)
    has_mask = np.isfinite(area) & truth_arr
    out = ProbeOutcomes(tuple(ids), tuple(findings), truth_arr, margin >= 0, margin,
                        np.where(has_mask, area, np.nan), has_mask)  # fmt: skip
    fitted = thresholds or probe_suites.fit_thresholds(out, ORDER)
    return {
        "probe_sample_ids": np.array(ids),
        "probe_findings": np.array(findings),
        "probe_truth": truth_arr,
        "probe_yes": margin >= 0,
        "probe_margin": margin,
        "probe_area": out.area,
        "probe_has_mask": has_mask,
        "probe_yes_op": probe_suites.operating_point_answers(out, fitted),
        "probe_op_threshold": np.array([fitted[f] for f in findings]),
    }, fitted


def test_recalibration_reproduces_the_operating_point_and_the_plain_answers() -> None:
    val, fitted = _outcomes(120, 0)
    test, _ = _outcomes(90, 1, thresholds=fitted)  # the test split uses validation's thresholds
    rows, summary = ex.recalibration_curve(val, test, ORDER, sizes=(8, 32), draws=50)
    assert summary["thresholds"] == pytest.approx(fitted)  # n = all: the pre-registered rule
    get = {(r["n"], r["finding"], r["metric"]): r for r in rows}
    op = ex.per_finding_rates(
        test["probe_findings"], test["probe_truth"], test["probe_yes_op"], ORDER
    )
    plain = ex.per_finding_rates(
        test["probe_findings"], test["probe_truth"], test["probe_yes"], ORDER
    )
    for finding in (*ORDER, "macro"):
        for metric in ("sensitivity", "specificity", "balanced_accuracy"):
            assert get[("all", finding, metric)]["median"] == pytest.approx(op[finding][metric])
            assert get[(0, finding, metric)]["median"] == pytest.approx(plain[finding][metric])
    assert get[(8, "macro", "balanced_accuracy")]["draws"] == 50
    low, high = (
        get[(8, "macro", "balanced_accuracy")]["low"],
        get[(8, "macro", "balanced_accuracy")]["high"],
    )
    assert low <= get[(8, "macro", "balanced_accuracy")]["median"] <= high
    again, _ = ex.recalibration_curve(val, test, ORDER, sizes=(8, 32), draws=50)
    assert again == rows  # seeded: the same draws every time (and for every arm)
    assert summary["smallest_n_within"] in (8, 32, None)


def test_the_stratified_draw_takes_each_strata_share() -> None:
    strata = {("a",): np.arange(0, 60), ("b",): np.arange(60, 90), (): np.arange(90, 100)}
    chosen = ex.stratified_draw(strata, 10, np.random.default_rng(0))
    assert len(chosen) == len(set(chosen.tolist())) == 10
    assert (chosen < 60).sum() == 6 and ((chosen >= 60) & (chosen < 90)).sum() == 3
    assert (chosen >= 90).sum() == 1


def test_flip_counts_against_the_reference() -> None:
    ref, _ = _outcomes(40, 0)
    arm = {k: v.copy() for k, v in ref.items()}
    truth = ref["probe_truth"]
    right = np.flatnonzero(ref["probe_yes_op"] == truth)[:5]  # make 5 of E2's right probes wrong
    wrong = np.flatnonzero(ref["probe_yes_op"] != truth)[:2]  # and 2 of its wrong probes right
    arm["probe_yes_op"][right] = ~arm["probe_yes_op"][right]
    arm["probe_yes_op"][wrong] = ~arm["probe_yes_op"][wrong]
    order = np.random.default_rng(1).permutation(len(truth))  # another probe order
    arm = {k: v[order] for k, v in arm.items()}
    masks = ex.flip_masks(ref, arm, "op")
    assert masks.forgotten.sum() == 5 and masks.gained.sum() == 2 and masks.present.all()
    assert set(np.flatnonzero(masks.forgotten)) == set(right.tolist())
    plain = ex.flip_masks(ref, arm, "plain")
    assert plain.forgotten.sum() == 0 and plain.gained.sum() == 0  # plain answers unchanged
    groups = ex.probe_groups(ref, ORDER, SIZES, ("polyp",))
    index, weights = ex.bootstrap_weights(ref["probe_sample_ids"], 2000, 0)
    rows = {(r["group"], r["kind"]): r for r in ex.flip_table(masks, groups, index, weights)}
    total = rows[("all", "forgotten")]
    assert total["count"] == 5 and total["denominator"] == (ref["probe_yes_op"] == truth).sum()
    assert total["count_ci_low"] <= 5 <= total["count_ci_high"]
    assert sum(rows[(f"finding:{f}", "forgotten")]["count"] for f in ORDER) == 5
    assert rows[("mask:yes", "gained")]["count"] + rows[("mask:no", "gained")]["count"] == 2


def test_jaccard_overlap_and_its_interval() -> None:
    a = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    b = np.array([0, 1, 1, 1, 0, 0], dtype=bool)
    assert ex.jaccard(a, b) == pytest.approx(2 / 4)
    assert math.isnan(ex.jaccard(np.zeros(3, bool), np.zeros(3, bool)))
    images = np.array(["i0", "i0", "i1", "i1", "i2", "i2"])
    index, weights = ex.bootstrap_weights(images, 500, 0)
    rows = ex.jaccard_table({"A": a, "B": b, "C": a}, index, weights)
    got = {(r["a"], r["b"]): r for r in rows}
    assert got[("A", "B")]["jaccard"] == pytest.approx(0.5)
    assert got[("A", "C")]["jaccard"] == 1.0 and got[("A", "C")]["ci_low"] == 1.0
    assert got[("A", "B")]["ci_low"] <= 0.5 <= got[("A", "B")]["ci_high"]


def test_the_exemplar_rule_needs_at_least_half_of_the_arms() -> None:
    sets = {
        "E3:w4": np.array([1, 1, 0, 0], dtype=bool),
        "E5:magnitude@B2": np.array([1, 0, 1, 0], dtype=bool),
        "E7@B2": np.array([1, 1, 0, 0], dtype=bool),
    }
    mask, needed = ex.exemplar_mask(sets)
    assert needed == 2 and mask.tolist() == [True, True, False, False]
    mask4, needed4 = ex.exemplar_mask({**sets, "E6@B2": np.array([0, 0, 1, 0], dtype=bool)})
    assert needed4 == 2 and mask4.tolist() == [True, True, True, False]  # half of 4 is enough
    runs = [ex.ArmRun("E5", "magnitude", b, 0, {"test": Path("x")}) for b in ("B1", "B2", "B3")]
    runs += [ex.ArmRun("E3", "w4", "nobudget", 0, {"test": Path("x")}),
             ex.ArmRun("E1", None, "nobudget", 0, {"test": Path("x")}),
             ex.ArmRun("E9", None, "nobudget", 0, {"test": Path("x")}),
             ex.ArmRun("E6", None, "B3", 0, {"test": Path("x")}),
             ex.ArmRun("E7", None, "B2", 1, {"test": Path("x")})]  # fmt: skip
    chosen = {r.label for r in ex.exemplar_arms(runs, "test")}
    assert chosen == {"E5:magnitude@B2 seed0", "E3:w4 seed0", "E6@B3 seed0"}  # not E1, E9, seed 1


def test_the_mask_audit_at_the_threshold_shows_the_yes_rate_confound() -> None:
    n = 40
    ids = np.array([f"img{i:02d}" for i in range(n)])
    base = np.full(n, -3.0)  # always "no" on plain answers: no plain flip is possible
    lesion = base - np.r_[np.full(20, 2.0), np.zeros(20)]
    random_ = base.copy()
    audit = {
        "audit_sample_ids": ids, "audit_findings": np.array(["polyp"] * n),
        "audit_original_yes": base >= 0, "audit_lesion_flip": (lesion >= 0) != (base >= 0),
        "audit_random_flip": (random_ >= 0) != (base >= 0),
        "audit_original_margin": base, "audit_lesion_margin": lesion,
        "audit_random_margin": random_,
    }  # fmt: skip
    result = ex.mask_audit_at_threshold(audit, {"polyp": -4.0}, resamples=500)
    assert result["plain_lesion_mask_effect"] == 0.0 and result["plain_original_yes_rate"] == 0.0
    assert result["op_original_yes_rate"] == 1.0  # above the threshold of -4: "yes"
    assert result["op_lesion_mask_effect"] == pytest.approx(0.5)  # 20 of 40 drop below it
    assert result["op_lesion_mask_effect_ci_low"] <= 0.5 <= result["op_lesion_mask_effect_ci_high"]


def test_the_audit_keeps_its_margins() -> None:
    from medcompress.evaluation.image_reliance import MaskAuditResult

    audit = MaskAuditResult(("a", "b"), ("polyp", "polyp"), np.array([True, False]),
                            np.array([True, False]), np.array([False, False]),
                            np.array([2.0, 0.5]), np.array([0.1, -0.2]), np.zeros(2),
                            original_margin=np.array([1.0, -1.0]))  # fmt: skip
    margins = audit.margins()
    assert margins["audit_lesion_margin"].tolist() == [-1.0, -1.5]
    assert margins["audit_random_margin"].tolist() == [0.9, -0.8]
    old = MaskAuditResult(("a",), ("polyp",), *(np.zeros(1),) * 6)
    assert old.margins() == {}


def test_the_driver_finds_the_runs_and_writes_every_table(tmp_path: Path) -> None:
    val, fitted = _outcomes(60, 0)
    roots = tmp_path / "input"

    def save(tag: str, split: str, arrays: dict, extra: dict | None = None) -> None:  # noqa: E306
        folder = roots / "bundle" / "output_root" / tag
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / ex.OUTCOME_FILES[split], **arrays, **(extra or {}))

    tags = (("E2/nobudget/seed0", 2), ("E3/w4/nobudget/seed0", 3), ("E5/magnitude/B2/seed0", 4),
            ("E7/B2/seed0", 5), ("E1/nobudget/seed0", 6))  # fmt: skip
    for tag, seed in tags:
        save(tag, "val", _outcomes(60, 0)[0])
        save(tag, "test", _outcomes(45, seed, thresholds=fitted)[0])
    n = 10
    audit = {"audit_sample_ids": np.array([f"img{i:03d}" for i in range(n)]),
             "audit_findings": np.array(["polyp"] * n), "audit_original_yes": np.zeros(n, bool),
             "audit_lesion_flip": np.zeros(n, bool), "audit_random_flip": np.zeros(n, bool),
             "audit_original_margin": np.full(n, 5.0), "audit_lesion_margin": np.full(n, -5.0),
             "audit_random_margin": np.full(n, 5.0)}  # fmt: skip
    folder = roots / "bundle" / "output_root" / "E7/B2/seed0"
    np.savez_compressed(folder / "audit_test.npz", **audit)
    (roots / "noise" / "seed0").mkdir(parents=True)
    np.savez_compressed(roots / "noise" / "seed0" / "outcomes.npz", x=np.zeros(1))  # not a run
    counts = ex.run_all([roots], tmp_path / "out", order=ORDER, size_thresholds=SIZES,
                        mask_findings=("polyp",), n_bins=10,
                        known=("E1", "E2", "E3", "E5", "E7"), draws=20, resamples=200)  # fmt: skip
    assert counts["inventory"] == 5 and counts["mask_audit_operating"] == 1
    assert counts["flips"] > 0 and counts["jaccard_arms"] == 2 * 6  # 4 arms: 6 pairs, 2 decisions
    assert counts["recalibration_summary"] == 5 and counts["exemplars"] > 0
    families = list(csv.DictReader((tmp_path / "out" / "jaccard_families.csv").open()))
    assert len(families) == 6  # three families present (E3; E5, E7; E1): 3 pairs, 2 decisions
    assert {frozenset((r["a"], r["b"])) for r in families} == {
        frozenset(("quantization", "pruning")),
        frozenset(("quantization", "distillation")),
        frozenset(("pruning", "distillation")),
    }
    audit_row = next(csv.DictReader((tmp_path / "out" / "mask_audit_operating.csv").open()))
    assert audit_row["run"] == "E7@B2 seed0" and float(audit_row["op_lesion_mask_effect"]) == 1.0
    summary = (tmp_path / "out" / "SUMMARY.md").read_text()
    assert "| E7@B2 seed0 | pruning | 1 | 1 | 0 | 0 / 1 / 0 |" in summary
    assert ex.parse_run(Path("x/E3/w4/nobudget/seed0/outcomes.npz"), ("E3",)) == (
        "E3",
        "w4",
        "nobudget",
        0,
    )
    assert ex.parse_run(Path("x/noise/seed0/outcomes.npz"), ("E3",)) is None
