"""Per-finding AUROC and "yes" rate (pre-registered secondary metrics, 2026-11-02)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from medcompress.config import ConfigError, RunContext, load_experiment
from medcompress.evaluation.clinical import auroc
from medcompress.evaluation.probe_suites import ProbeOutcomes, clinical_metrics
from medcompress.evaluation.runner import to_rows
from medcompress.utils.io import append_results

REPO = Path(__file__).resolve().parents[1]
FINDINGS = ("polyp", "ulcerative_colitis", "oesophagitis")


def test_auroc_is_the_pairwise_order_with_half_ties() -> None:
    assert auroc([True, True, False, False], [3.0, 2.0, 1.0, 0.0]) == 1.0
    assert auroc([True, True, False, False], [0.0, 1.0, 2.0, 3.0]) == 0.0
    assert auroc([True, False], [1.0, 1.0]) == 0.5
    assert math.isnan(auroc([True, True], [1.0, 2.0]))
    rng = np.random.default_rng(3)
    truth = rng.random(60) < 0.4
    score = np.round(rng.normal(truth.astype(float), 1.0), 1)  # with ties
    pairs = [(p > n) + 0.5 * (p == n) for p in score[truth] for n in score[~truth]]
    assert auroc(truth, score) == pytest.approx(float(np.mean(pairs)))


def read_results_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _outcomes(shift: float = 0.0, seed: int = 0) -> ProbeOutcomes:
    rng = np.random.default_rng(seed)
    n = 90
    ids = tuple(np.repeat([f"img{i:03d}:x" for i in range(n // 3)], 3))
    names = tuple(np.tile(FINDINGS, n // 3))
    truth = rng.random(n) < 0.4
    margin = rng.normal(np.where(truth, 1.5, -1.5), 1.0) + shift
    return ProbeOutcomes(
        sample_ids=ids, findings=names, truth=truth, yes=margin > 0, margin=margin,
        area=np.full(n, np.nan), has_mask=np.zeros(n, bool),
    )  # fmt: skip


def _values(out: ProbeOutcomes) -> dict[str, object]:
    return {m.name: m for m in clinical_metrics(out, FINDINGS, n_resamples=200, seed=0)}


def test_a_bias_shift_keeps_the_auroc_and_moves_the_yes_rate() -> None:
    plain, shifted = _values(_outcomes()), _values(_outcomes(shift=-2.0))
    for finding in FINDINGS:
        a, b = plain[f"auroc:{finding}"], shifted[f"auroc:{finding}"]
        assert a.value == pytest.approx(b.value)  # same order of the scores
        assert a.ci_low <= a.value <= a.ci_high
        assert shifted[f"yes_rate:{finding}"].value < plain[f"yes_rate:{finding}"].value
        assert shifted[f"sensitivity:{finding}"].value <= plain[f"sensitivity:{finding}"].value
        counts = plain[f"n_positives:{finding}"].value + plain[f"n_negatives:{finding}"].value
        assert counts == 30


def test_lost_discrimination_lowers_the_auroc() -> None:
    sharp = _outcomes()
    noisy_margin = sharp.margin + np.random.default_rng(9).normal(0.0, 4.0, len(sharp.margin))
    noisy = ProbeOutcomes(
        sample_ids=sharp.sample_ids, findings=sharp.findings, truth=sharp.truth,
        yes=noisy_margin > 0, margin=noisy_margin, area=sharp.area, has_mask=sharp.has_mask,
    )  # fmt: skip
    a, b = _values(sharp), _values(noisy)
    assert all(b[f"auroc:{f}"].value < a[f"auroc:{f}"].value for f in FINDINGS)


def test_one_class_gives_no_auroc_but_the_rates() -> None:
    out = _outcomes()
    truth = out.truth.copy()
    truth[np.array(out.findings) == "oesophagitis"] = False
    values = _values(
        ProbeOutcomes(
            sample_ids=out.sample_ids,
            findings=out.findings,
            truth=truth,
            yes=out.yes,
            margin=out.margin,
            area=out.area,
            has_mask=out.has_mask,
        )  # fmt: skip
    )
    assert "auroc:oesophagitis" not in values and "yes_rate:oesophagitis" in values
    assert values["n_positives:oesophagitis"].value == 0


def _script():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("rescore", REPO / "scripts" / "rescore_probes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rescore_adds_only_the_new_rows_and_checks_the_outcomes(tmp_path: Path) -> None:
    config, _ = load_experiment("E0")
    config["paths"]["results_root"] = str(tmp_path / "results")
    config["evaluation"]["bootstrap_resamples"] = 200
    run = RunContext("E0", seed=0)
    out = _outcomes()
    new_rows = ("auroc:", "yes_rate:", "n_positives:", "n_negatives:")
    old = [
        m for m in clinical_metrics(out, FINDINGS, n_resamples=200, seed=0)
        if not m.name.startswith(new_rows)
    ]  # fmt: skip
    target = run.run_dir(tmp_path / "results") / "metrics.csv"
    append_results(to_rows(old, "E0", "", 0), target)
    before = {r["metric"]: r for r in read_results_rows(target)}
    npz = tmp_path / "outcomes.npz"
    np.savez_compressed(
        npz, probe_sample_ids=np.array(out.sample_ids), probe_findings=np.array(out.findings),
        probe_truth=out.truth, probe_yes=out.yes, probe_margin=out.margin, probe_area=out.area,
        probe_has_mask=out.has_mask,
    )  # fmt: skip
    _script().rescore(config, run, npz)
    after = {r["metric"]: r for r in read_results_rows(target)}
    assert {k: after[k] for k in before} == before  # no committed number changes
    assert {f"auroc:{f}" for f in FINDINGS} | {f"yes_rate:{f}" for f in FINDINGS} <= set(after)
    other = tmp_path / "other.npz"
    flipped = _outcomes(seed=5)
    np.savez_compressed(
        other, probe_sample_ids=np.array(flipped.sample_ids),
        probe_findings=np.array(flipped.findings), probe_truth=flipped.truth,
        probe_yes=flipped.yes, probe_margin=flipped.margin, probe_area=flipped.area,
        probe_has_mask=flipped.has_mask,
    )  # fmt: skip
    with pytest.raises(ConfigError, match="does not belong to the committed run"):
        _script().rescore(config, run, other)
