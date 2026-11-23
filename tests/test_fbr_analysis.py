"""FBR's analysis (amendment of 2026-11-22): folds, the cross-fitted threshold rule, the two
recalibrations, the decoded-answer parse, the gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from medcompress.evaluation import fbr_analysis as fa
from medcompress.evaluation import probe_suites
from medcompress.evaluation.probe_suites import ProbeOutcomes

ORDER = ("polyp", "oesophagitis", "ulcerative_colitis")


def _arrays(n_images: int = 120, seed: int = 0, shift: float = 2.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids, findings, truth, margin = [], [], [], []
    for i in range(n_images):
        for j, f in enumerate(ORDER):
            present = bool(rng.random() < (0.3 if j == 0 else 0.2))
            ids.append(f"s{i}")
            findings.append(f)
            truth.append(present)
            margin.append(rng.normal(shift if present else -shift, 1.0) - 1.0)
    return {"probe_sample_ids": np.array(ids), "probe_findings": np.array(findings),
            "probe_truth": np.array(truth), "probe_margin": np.array(margin)}  # fmt: skip


def test_folds_group_images_depend_on_labels_only_and_fall_back_to_two() -> None:
    a = _arrays()
    plan = fa.assign_folds(a["probe_sample_ids"], a["probe_findings"], a["probe_truth"], ORDER)
    assert plan.k == 3 and not plan.short
    folds = plan.folds(a["probe_sample_ids"])
    for image in np.unique(a["probe_sample_ids"]):  # all probes of an image in one fold
        assert len(set(folds[a["probe_sample_ids"] == image])) == 1
    other = dict(a, probe_margin=-a["probe_margin"])  # another model: the same folds
    again = fa.assign_folds(other["probe_sample_ids"], other["probe_findings"],
                            other["probe_truth"], ORDER)  # fmt: skip
    assert again.fold_of == plan.fold_of
    sizes = np.bincount(np.array(list(plan.fold_of.values())))
    assert sizes.max() - sizes.min() <= 3  # balanced
    small = _arrays(n_images=24)
    fallback = fa.assign_folds(small["probe_sample_ids"], small["probe_findings"],
                               small["probe_truth"], ORDER, min_per_class=4)  # fmt: skip
    assert fallback.k == 2


def test_the_crossfitted_threshold_is_fitted_on_the_other_folds_only() -> None:
    a = _arrays()
    plan = fa.assign_folds(a["probe_sample_ids"], a["probe_findings"], a["probe_truth"], ORDER)
    ids, f, t, m = (a[k] for k in ("probe_sample_ids", "probe_findings", "probe_truth",
                                   "probe_margin"))  # fmt: skip
    folds = plan.folds(ids)
    yes = fa.crossfit(f, t, m, folds, ORDER, plan.k)
    for finding in ORDER:
        for fold in range(plan.k):
            fit = (f == finding) & (folds != fold)
            n = int(fit.sum())
            out = ProbeOutcomes(tuple(ids[fit]), tuple(f[fit]), t[fit], m[fit] >= 0, m[fit],
                                np.full(n, math.nan), np.zeros(n, bool))  # fmt: skip
            limit = probe_suites.fit_thresholds(out, [finding])[finding]
            held = (f == finding) & (folds == fold)
            assert np.array_equal(yes[held], m[held] > limit)
    # changing a held-out margin never moves its own fold's threshold
    changed = m.copy()
    target = np.flatnonzero((f == "polyp") & (folds == 0) & ~t)[0]
    changed[target] = 50.0
    moved = fa.crossfit(f, t, changed, folds, ORDER, plan.k)
    others = (f == "polyp") & (folds == 0)
    others[target] = False
    assert np.array_equal(moved[others], yes[others])
    spec = (~yes[~t]).mean()
    assert 0.9 < spec < 1.0  # pooled held-out specificity is reported, near 0.95


def test_weighted_threshold_is_the_pre_registered_rule_with_multiplicities() -> None:
    m = np.array([0.5, -1.0, 2.0, 0.1, -0.3, 1.2, 0.7, -2.0, 0.0, 0.9])
    assert fa.weighted_threshold(m, np.ones(10), 0.95) == np.sort(m)[9]
    assert fa.weighted_threshold(m, np.ones(10), 0.5) == np.sort(m)[4]
    doubled = fa.weighted_threshold(np.repeat(m, 2), np.ones(20), 0.7)
    assert fa.weighted_threshold(m, np.full(10, 2.0), 0.7) == doubled
    assert math.isnan(fa.weighted_threshold(m, np.zeros(10), 0.95))


def test_the_recalibrations_recover_known_settings() -> None:
    rng = np.random.default_rng(1)
    m = rng.normal(0, 2, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-m / 2.5))
    assert fa.fit_temperature(m, y) == pytest.approx(2.5, rel=0.05)
    y2 = rng.random(20000) < 1 / (1 + np.exp(-(m - 1.0)))
    assert fa.fit_intercept(m, y2) == pytest.approx(-1.0, abs=0.06)
    few = fa.fit_intercept(np.array([0.0, 0.0]), np.array([True, True]))
    assert 0 < few < 1.0  # the ridge keeps a two-probe fit small
    a = _arrays()
    plan = fa.assign_folds(a["probe_sample_ids"], a["probe_findings"], a["probe_truth"], ORDER)
    p = fa.crossfit_probabilities(a["probe_findings"], a["probe_truth"], a["probe_margin"],
                                  plan.folds(a["probe_sample_ids"]), plan.k, ORDER)  # fmt: skip
    assert set(p) == {"raw", "intercept", "temperature"}
    assert all(((0 < v) & (v < 1)).all() for v in p.values())


def test_the_decoded_answer_parse() -> None:
    assert fa.parse_answer("Yes, there is a polyp.") == 1
    assert fa.parse_answer("  no.") == 0 and fa.parse_answer("NO") == 0
    assert fa.parse_answer("There is no polyp") == -1 and fa.parse_answer("") == -1
    r = fa.decoded_rates(np.array([True, True, False, False]), np.array([1, -1, 0, -1]))
    assert (r["sensitivity"], r["specificity"], r["parse_failures"]) == (0.5, 0.5, 2)
    with pytest.raises(ValueError, match="validation only"):
        fa.require_validation("test")


def test_the_gate_and_the_paired_bootstrap() -> None:
    rare = ("oesophagitis", "ulcerative_colitis")
    reg = {"R": 0.50, "sens:polyp": 0.90, "sens:oesophagitis": 0.40,
           "sens:ulcerative_colitis": 0.60, "spec:polyp": 0.95, "spec:oesophagitis": 0.95,
           "spec:ulcerative_colitis": 0.95, "macro_brier": 0.10, "plain_macro_ba": 0.5,
           "decoded_macro_ba": 0.50}  # fmt: skip
    good = dict(reg, R=0.56, **{"sens:oesophagitis": 0.50, "sens:ulcerative_colitis": 0.62,
                                "decoded_macro_ba": 0.54, "macro_brier": 0.105})  # fmt: skip
    assert all(fa.gate(good, reg, collapsed=True, rare=rare).values())
    weak = dict(good, decoded_macro_ba=0.52)  # +2 points: enough only on a non-collapsed arm
    assert not fa.gate(weak, reg, collapsed=True, rare=rare)["decoded_macro_ba"]
    assert fa.gate(weak, reg, collapsed=False, rare=rare)["decoded_macro_ba"]
    worse = dict(good, **{"spec:oesophagitis": 0.92, "macro_brier": 0.115})
    result = fa.gate(worse, reg, collapsed=True, rare=rare)
    assert not result["rare_specificity"] and not result["macro_brier"]
    a = _arrays()
    plan = fa.assign_folds(a["probe_sample_ids"], a["probe_findings"], a["probe_truth"], ORDER)
    model = fa.ModelArrays("x", a, None)
    better = fa.ModelArrays("y", dict(a, probe_margin=a["probe_margin"] * 1.5), None)
    intervals = fa.paired_bootstrap(better, model, plan, ORDER, rare, resamples=50)
    low, high = intervals["R"]
    assert low <= high and "macro_brier" in intervals
    same = fa.paired_bootstrap(model, model, plan, ORDER, rare, resamples=20)
    assert same["R"] == (0.0, 0.0)
    rows = fa.report_rows(model, plan, ORDER)
    assert {r["finding"] for r in rows} == set(ORDER)
    assert all(r["native_tp"] + r["native_fn"] == r["operating_tp"] + r["operating_fn"]
               for r in rows)  # fmt: skip


def _save(folder, arrays, answers=None) -> None:  # noqa: ANN001
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "outcomes.npz", **arrays)
    if answers is not None:
        np.savez_compressed(folder / fa.ANSWERS_FILE.format(split="val"), **answers)


def _answers(arrays: dict) -> dict:
    n = len(arrays["probe_margin"])
    return {**{k: arrays[k] for k in ("probe_sample_ids", "probe_findings", "probe_truth",
                                      "probe_margin")},
            "probe_yes_rank": np.ones(n, dtype=np.int64), "probe_no_rank": np.full(n, 2),
            "probe_decoded": np.where(arrays["probe_margin"] >= 0, 1, 0).astype(np.int8),
            "probe_decoded_text": np.array(["yes"] * n)}  # fmt: skip


def test_the_driver_finds_the_runs_and_writes_the_diagnostic_and_the_pilot(tmp_path) -> None:  # noqa: ANN001
    import json

    from medcompress.registry import REGISTRY

    base = _arrays()
    rng = np.random.default_rng(3)
    polyps = np.flatnonzero((base["probe_findings"] == "polyp") & base["probe_truth"])
    base["audit_sample_ids"] = base["probe_sample_ids"][polyps]
    base["audit_findings"] = base["probe_findings"][polyps]
    base["audit_original_yes"] = base["probe_margin"][polyps] >= 0
    base["audit_lesion_flip"] = rng.random(len(polyps)) < 0.3
    base["audit_random_flip"] = rng.random(len(polyps)) < 0.1
    margins = {"audit_original_margin": base["probe_margin"][polyps],
               "audit_lesion_margin": base["probe_margin"][polyps] - 1.0,
               "audit_random_margin": base["probe_margin"][polyps] - 0.2}  # fmt: skip
    inputs = tmp_path / "input"
    _save(inputs / "a" / "E2" / "nobudget" / "seed0", base)
    for variant, scale in (("magnitude", 0.5), ("activation", 0.4)):
        arrays = dict(base, probe_margin=base["probe_margin"] * scale - 2.0)
        _save(inputs / "g2" / "E5" / variant / "B2" / "seed0", arrays)
        answers = inputs / "ans" / "fbr" / "probe_answers" / "E5" / variant / "B2" / "seed0"
        answers.mkdir(parents=True)
        np.savez_compressed(answers / fa.ANSWERS_FILE.format(split="val"), **_answers(arrays))
    rerun = inputs / "rescored" / "E5" / "magnitude" / "B2" / "seed0"  # an audit-only rerun
    rerun.mkdir(parents=True)
    audit = {k: base[k] for k in ("audit_sample_ids", "audit_findings", "audit_original_yes",
                                  "audit_lesion_flip", "audit_random_flip")}  # fmt: skip
    np.savez_compressed(rerun / "audit_val.npz", **audit, **margins)
    rows = fa.run_diagnostic([inputs], tmp_path / "out", ORDER, list(REGISTRY))
    assert rows == 3 * len(ORDER)
    text = (tmp_path / "out" / "DIAGNOSTIC.md").read_text()
    assert "E2" in text and "not available" not in text.split("\n")[0]
    with pytest.raises(ValueError, match="no FBR run"):
        fa.run_pilot([inputs], tmp_path / "out", ORDER, list(REGISTRY))
    for label, arm, control, scale in (("E5_magnitude", "E5:magnitude", False, 0.9),
                                       ("E5_activation", "E5:activation", False, 0.9),
                                       ("E5_activation_sampling_only", "E5:activation", True,
                                        0.6)):  # fmt: skip
        arrays = dict(base, probe_margin=base["probe_margin"] * scale)
        folder = inputs / "fbr_runs" / "fbr" / label
        _save(folder, arrays, _answers(arrays))
        (folder / "fbr_run.json").write_text(json.dumps(
            {"arm": arm, "control": control, "label": label, "margin_coefficient": 0.25,
             "schedule": "x"}))  # fmt: skip
    import medcompress.compression.fbr as fbr_module

    settings = fbr_module.FBRSettings.load()
    small = __import__("dataclasses").replace(settings, bootstrap_resamples=30)
    original = fbr_module.FBRSettings.load
    fbr_module.FBRSettings.load = classmethod(lambda cls, overrides=(): small)  # type: ignore
    try:
        result = fa.run_pilot([inputs], tmp_path / "out", ORDER, list(REGISTRY))
    finally:
        fbr_module.FBRSettings.load = original  # type: ignore
    assert set(result["verdict"]) == {"E5_magnitude", "E5_activation",
                                      "E5_activation_sampling_only"}  # fmt: skip
    assert result["simpler"]["control"] == "E5_activation_sampling_only"
    assert isinstance(result["passed"], bool)
    with (tmp_path / "out" / "pilot_mask_audit.csv").open() as handle:
        audits = {r["model"]: r for r in __import__("csv").DictReader(handle)}
    assert audits["E5:magnitude@B2"]["operating_effect"] != ""  # margins from the rerun
    assert float(audits["E5:magnitude@B2"]["lesion_margin_change"]) == pytest.approx(-1.0)
    assert audits["E5:activation@B2"]["operating_effect"] == ""  # no margins: plain only
    # without the registered arms' probe answers the gate is not decidable, never "failed"
    for variant in ("magnitude", "activation"):
        (inputs / "ans" / "fbr" / "probe_answers" / "E5" / variant / "B2" / "seed0" /
         fa.ANSWERS_FILE.format(split="val")).unlink()  # fmt: skip
    fbr_module.FBRSettings.load = classmethod(lambda cls, overrides=(): small)  # type: ignore
    try:
        undecided = fa.run_pilot([inputs], tmp_path / "out", ORDER, list(REGISTRY))
    finally:
        fbr_module.FBRSettings.load = original  # type: ignore
    assert undecided["passed"] is None
    assert "not decidable yet (probe answers missing" in (tmp_path / "out" / "PILOT.md").read_text()
    # an audit rerun that does not reproduce the saved flips stops the analysis
    np.savez_compressed(rerun / "audit_val.npz", **dict(audit, audit_lesion_flip=~audit[
        "audit_lesion_flip"]), **margins)  # fmt: skip
    with pytest.raises(ValueError, match="does not reproduce the saved audit"):
        fa.run_diagnostic([inputs], tmp_path / "out", ORDER, list(REGISTRY))
    for name in ("pilot_gate.csv", "pilot_verdict.csv", "pilot_report.csv",
                 "pilot_mask_audit.csv", "PILOT.md"):  # fmt: skip
        assert (tmp_path / "out" / name).is_file()
