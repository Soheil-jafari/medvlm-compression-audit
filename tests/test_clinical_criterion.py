"""Arithmetic of the clinically weighted pruning criterion (synthetic scores)."""

from __future__ import annotations

import math

import pytest

from medcompress.compression.pruning.criteria.base import PruningCriterion
from medcompress.compression.pruning.criteria.clinical import (
    ClinicalCriterion,
    FindingWeight,
    aggregate_importance,
    compute_weights,
    delta,
    load_finding_weights,
    load_tier_multipliers,
    rarity_factor,
)
from medcompress.config import config_root, load_config
from medcompress.data.probes import FindingProbe
from medcompress.models.base import YesNoScore

TABLE = {
    "preregistration": {"frozen_on": "2026-11-01"},
    "weighting": {
        "tiers": [{"tier": "high", "multiplier": 2.0}, {"tier": "low", "multiplier": 1.0}]
    },
    "findings": [
        {
            "finding": "polyp",
            "prevalence": 0.5,
            "importance_tier": "high",
            "cap": 10,
            "final_weight": 4.0,
        },
        {
            "finding": "ulcer",
            "prevalence": 0.01,
            "importance_tier": "low",
            "cap": 10,
            "final_weight": 10.0,
        },
    ],
}


def _probe(finding: str, sample: str, answer: str) -> FindingProbe:
    return FindingProbe(sample, f"{sample}.jpg", finding, "q", answer)


def test_load_weights_and_tiers() -> None:
    rows = load_finding_weights(TABLE)
    assert rows[0] == FindingWeight("polyp", 0.5, "high", 10.0, 4.0)
    assert load_tier_multipliers(TABLE) == {"high": 2.0, "low": 1.0}


def test_unfrozen_template_is_refused() -> None:
    template = load_config(config_root() / "preregistration" / "finding_weights.yaml")
    with pytest.raises(ValueError, match="not frozen"):
        load_finding_weights(template)
    with pytest.raises(ValueError, match="Incomplete"):
        load_finding_weights(template, require_frozen=False)


def test_compute_weights_schemes() -> None:
    rows = load_finding_weights(TABLE)
    tiers = load_tier_multipliers(TABLE)
    assert compute_weights(["polyp", "ulcer"], rows, "clinical", tier_multipliers=tiers) == {
        "polyp": 4.0,
        "ulcer": 10.0,
    }
    assert compute_weights(["polyp", "ulcer"], rows, "uniform") == {"polyp": 1.0, "ulcer": 1.0}
    assert compute_weights(["polyp", "ulcer"], rows, "rarity_only") == {"polyp": 2.0, "ulcer": 10.0}
    normalised = compute_weights(["polyp", "ulcer"], rows, "clinical", normalisation="sum_to_n")
    assert sum(normalised.values()) == pytest.approx(2.0)
    assert rarity_factor(0.0, 5.0) == 5.0


def test_compute_weights_detects_drift_from_formula() -> None:
    rows = [FindingWeight("polyp", 0.5, "high", 10.0, 3.0)]
    with pytest.raises(ValueError, match="does not match"):
        compute_weights(["polyp"], rows, tier_multipliers={"high": 2.0})
    with pytest.raises(ValueError, match="missing"):
        compute_weights(["ulcer"], rows)


def test_delta_logprob_drop_uses_the_correct_answer() -> None:
    probes = [_probe("polyp", "x1", "yes"), _probe("polyp", "x2", "no")]
    base = [YesNoScore(math.log(0.9), math.log(0.1)), YesNoScore(math.log(0.2), math.log(0.8))]
    abl = [YesNoScore(math.log(0.5), math.log(0.5)), YesNoScore(math.log(0.2), math.log(0.8))]
    out = delta(base, abl, probes)
    assert out[("polyp", "x1")] == pytest.approx(math.log(0.9) - math.log(0.5))
    assert out[("polyp", "x2")] == pytest.approx(0.0)


def test_delta_kl_is_zero_for_identical_and_positive_otherwise() -> None:
    probes = [_probe("polyp", "x1", "yes")]
    base = [YesNoScore(-0.1, -2.5)]
    assert delta(base, base, probes, "kl")[("polyp", "x1")] == pytest.approx(0.0)
    assert delta(base, [YesNoScore(-2.5, -0.1)], probes, "kl")[("polyp", "x1")] > 0
    with pytest.raises(ValueError):
        delta(base, [], probes)


def test_aggregate_importance_formula_and_clipping() -> None:
    deltas = {("polyp", "x1"): 1.0, ("polyp", "x2"): -0.5, ("ulcer", "x1"): 2.0}
    weights = {"polyp": 4.0, "ulcer": 10.0, "bleeding": 3.0}
    assert aggregate_importance(deltas, weights) == pytest.approx(4.0 * 0.25 + 10.0 * 2.0)
    assert aggregate_importance(deltas, weights, clip_negative=True) == pytest.approx(
        4.0 * 0.5 + 20.0
    )
    with pytest.raises(ValueError):
        aggregate_importance({("x", "s"): 1.0}, weights)


def test_rank_is_ascending_with_deterministic_ties() -> None:
    criterion: PruningCriterion = ClinicalCriterion({})
    assert criterion.rank({"layer.3": 0.5, "layer.1": 0.1, "layer.2": 0.1}) == [
        "layer.1",
        "layer.2",
        "layer.3",
    ]
    with pytest.raises(ValueError):
        criterion.rank({"a": float("nan")})
