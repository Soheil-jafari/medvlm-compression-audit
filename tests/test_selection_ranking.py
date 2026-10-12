"""Score S, scaling, ranks and bootstrap intervals of a selection round (no model needed)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medcompress.evaluation.aggregate import validate_results_tree
from medcompress.evaluation.clinical import macro_f1
from medcompress.evaluation.image_reliance import macro_balanced_accuracy
from medcompress.selection import ranking
from medcompress.selection.ranking import CandidateOutcomes, rank_round

WEIGHTS = (0.4, 0.3, 0.3)
N_IMAGES = 60
FINDINGS = ("polyp", "ulcer")


def _base(seed: int = 0) -> dict[str, np.ndarray]:
    """Two probes (one per finding) for each of 60 images; the first 20 images have masks."""
    rng = np.random.default_rng(seed)
    image_ids = np.repeat([f"img{i:03d}" for i in range(N_IMAGES)], 2)
    finding_idx = np.tile([0, 1], N_IMAGES)
    truth = rng.random(2 * N_IMAGES) < 0.5
    masked = np.arange(2 * N_IMAGES) < 40
    truth = np.where((finding_idx == 0) & masked, True, truth)  # masked images show a polyp
    in_small = (finding_idx == 0) & masked & (np.arange(2 * N_IMAGES) < 20)
    audit_ids = np.array([f"img{i:03d}" for i in range(20)])
    return {
        "image_ids": image_ids,
        "finding_idx": finding_idx,
        "truth": truth,
        "in_small": in_small,
        "audit_image_ids": audit_ids,
    }


def _candidate(skill: float, *, seed: int, base: dict[str, np.ndarray]) -> CandidateOutcomes:
    """A candidate that answers correctly with probability ``skill`` and relies on the image
    in proportion to it: lesion masking flips its answer with probability ``skill``."""
    rng = np.random.default_rng(seed)
    truth = base["truth"]
    right = rng.random(len(truth)) < skill
    yes = np.where(right, truth, ~truth)
    blank = rng.random(len(truth)) < (1 - skill) + 0.1  # a skilled model answers "yes" less blind
    n_audit = len(base["audit_image_ids"])
    return CandidateOutcomes(
        yes_blank=blank,
        yes=yes,
        audit_lesion_flip=rng.random(n_audit) < skill,
        audit_random_flip=rng.random(n_audit) < 0.1,
        **{
            k: base[k] for k in ("image_ids", "finding_idx", "truth", "in_small", "audit_image_ids")
        },
    )


# ------------------------------------------------------------------ components


def test_components_match_the_reference_metrics() -> None:
    base = _base()
    o = _candidate(0.8, seed=1, base=base)
    f1, small, blank_drop = ranking.component_values(o, "blank_image_drop", 2)
    names = [FINDINGS[i] for i in o.finding_idx]
    truth = {f: [bool(t) for n, t in zip(names, o.truth, strict=True) if n == f] for f in FINDINGS}
    pred = {f: [bool(y) for n, y in zip(names, o.yes, strict=True) if n == f] for f in FINDINGS}
    assert f1 == pytest.approx(macro_f1(truth, pred))
    assert small == pytest.approx(o.yes[o.in_small].mean())
    expected_drop = macro_balanced_accuracy(names, list(o.truth), list(o.yes)) - (
        macro_balanced_accuracy(names, list(o.truth), list(o.yes_blank))
    )
    assert blank_drop == pytest.approx(expected_drop)
    effect = ranking.component_values(o, "lesion_mask_effect", 2)[2]
    assert effect == pytest.approx(o.audit_lesion_flip.mean() - o.audit_random_flip.mean())


def test_integer_weights_equal_repeating_rows() -> None:
    base = _base()
    o = _candidate(0.7, seed=2, base=base)
    rng = np.random.default_rng(3)
    probe_w = rng.integers(0, 3, len(o.image_ids))
    audit_w = rng.integers(0, 3, len(o.audit_image_ids))
    repeated = CandidateOutcomes(
        image_ids=np.repeat(o.image_ids, probe_w),
        finding_idx=np.repeat(o.finding_idx, probe_w),
        truth=np.repeat(o.truth, probe_w),
        yes=np.repeat(o.yes, probe_w),
        yes_blank=np.repeat(o.yes_blank, probe_w),
        in_small=np.repeat(o.in_small, probe_w),
        audit_image_ids=np.repeat(o.audit_image_ids, audit_w),
        audit_lesion_flip=np.repeat(o.audit_lesion_flip, audit_w),
        audit_random_flip=np.repeat(o.audit_random_flip, audit_w),
    )
    for metric in ranking.RELIANCE_METRICS:
        weighted = ranking.component_values(
            o, metric, 2, probe_w.astype(float), audit_w.astype(float)
        )
        assert weighted == pytest.approx(ranking.component_values(repeated, metric, 2))


def test_undefined_components_are_nan() -> None:
    base = _base()
    o = _candidate(0.8, seed=1, base=base)
    none_small = ranking.component_values(
        o, "lesion_mask_effect", 2, np.where(o.in_small, 0.0, 1.0)
    )
    assert math.isnan(none_small[1])
    assert math.isnan(ranking.component_values(o, "lesion_mask_effect", 2, None, np.zeros(20))[2])
    with pytest.raises(ValueError, match="reliance metric"):
        ranking.component_values(o, "vibes", 2)


def test_scaling_is_min_max_per_component() -> None:
    values = np.array([[0.2, 10.0, 0.5], [0.6, 30.0, 0.5], [0.4, 20.0, 0.5]])
    score, scaled = ranking.scaled_score(values, WEIGHTS)
    assert scaled[:, 0].tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert scaled[:, 1].tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert scaled[:, 2].tolist() == [0.0, 0.0, 0.0]  # equal for all: cannot separate them
    assert score.tolist() == pytest.approx([0.0, 0.7, 0.35])
    nan_low = ranking.scaled_score(np.array([[0.2, 1.0, 1.0], [math.nan, 2.0, 2.0]]), WEIGHTS)[1]
    assert nan_low[1, 0] == 0.0  # an undefined component counts as the round's minimum
    with pytest.raises(ValueError, match="one column per weight"):
        ranking.scaled_score(np.zeros((2, 2)), WEIGHTS)


# ---------------------------------------------------------------------- rounds


def _round() -> dict[str, CandidateOutcomes]:
    base = _base()
    skills = {"strong": 0.95, "good": 0.8, "okay": 0.65, "weak": 0.5}
    return {name: _candidate(s, seed=i, base=base) for i, (name, s) in enumerate(skills.items())}


def _rank(outcomes: dict[str, CandidateOutcomes], **over: object) -> ranking.RoundRanking:
    kwargs: dict[str, object] = {
        "weights": WEIGHTS, "reliance_metric": "lesion_mask_effect", "keep": 2,
        "n_findings": 2, "n_resamples": 200, "seed": 0,
    }  # fmt: skip
    return rank_round(outcomes, **(kwargs | over))  # type: ignore[arg-type]


def test_the_most_skilled_candidates_advance() -> None:
    result = _rank(_round())
    assert result.candidates == ("strong", "good", "okay", "weak")
    assert result.advance == ("strong", "good")
    assert result.scores[0] > result.scores[-1] and result.scores[-1] == pytest.approx(0.0)
    assert result.scaled.shape == (4, 3) and result.raw.shape == (4, 3)
    assert math.isnan(result.gap[-1]) and (result.gap[:-1] > 0).all()
    for i in range(4):
        low, high = result.score_ci[i]
        assert low <= high
    assert np.nansum(result.gap) == pytest.approx(result.scores[0] - result.scores[-1])


def test_the_blank_image_metric_changes_the_reliance_component() -> None:
    outcomes = _round()
    a = _rank(outcomes, reliance_metric="blank_image_drop")
    b = _rank(outcomes, reliance_metric="lesion_mask_effect")
    assert not np.allclose(a.raw[:, 2], b.raw[:, 2])


def test_gate_failures_leave_the_ranking() -> None:
    result = _rank(_round(), gate_failures={"strong": ["G2_uses_image"], "okay": []})
    assert "strong" not in result.candidates and result.advance == ("good", "okay")
    assert result.failed_gates == {"strong": ("G2_uses_image",)}
    rows = ranking.ranking_rows(result, "R1")
    failed = [r for r in rows if r["experiment"] == "R1:strong"]
    assert {r["metric"] for r in failed} == {"advance", "gate_failed:G2_uses_image"}
    assert all(r["value"] in (0.0, 1.0) for r in failed)
    with pytest.raises(ValueError, match="every candidate failed"):
        _rank(_round(), gate_failures={n: ["G1"] for n in _round()})


def test_fewer_survivors_than_places_advance_everyone_left() -> None:
    result = _rank(_round(), keep=10)
    assert len(result.advance) == 4


def test_bootstrap_is_paired_and_reproducible() -> None:
    outcomes = _round()
    first, second = _rank(outcomes, seed=5), _rank(outcomes, seed=5)
    assert np.array_equal(first.score_ci, second.score_ci)
    assert not np.array_equal(first.score_ci, _rank(outcomes, seed=6).score_ci)
    # two identical candidates have an identical score in every resample: a gap of exactly zero
    twin = dict(outcomes) | {"twin": outcomes["good"]}
    result = _rank(twin)
    i = result.candidates.index("good")
    j = result.candidates.index("twin")
    assert abs(i - j) == 1
    upper = min(i, j)
    assert result.gap[upper] == pytest.approx(0.0, abs=1e-12)
    assert result.gap_ci[upper] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_exact_ties_follow_the_tie_break_values() -> None:
    outcomes = _round()
    twin = dict(outcomes) | {"twin": outcomes["good"]}
    plain = _rank(twin, tie_break={"good": (2.0,), "twin": (1.0,)})
    flipped = _rank(twin, tie_break={"good": (1.0,), "twin": (2.0,)})
    assert plain.candidates.index("twin") < plain.candidates.index("good")
    assert flipped.candidates.index("good") < flipped.candidates.index("twin")


def test_candidates_must_be_scored_on_the_same_probes() -> None:
    outcomes = _round()
    other = _base(seed=9)  # different labels
    bad = dict(outcomes) | {"odd": _candidate(0.7, seed=3, base=other)}
    with pytest.raises(ValueError, match="same probes"):
        _rank(bad)


def test_ranking_rows_follow_the_results_schema(tmp_path: Path) -> None:
    result = _rank(_round())
    rows = ranking.ranking_rows(result, "R1")
    target = tmp_path / "results" / "selection" / "R1" / "ranking.csv"
    target.parent.mkdir(parents=True)
    pd.DataFrame(rows).reindex(
        columns=["experiment", "budget", "seed", "metric", "value", "ci_low", "ci_high"]
    ).to_csv(target, index=False)
    assert validate_results_tree(tmp_path / "results") == []
    table = pd.DataFrame(rows)
    top = table[(table["experiment"] == "R1:strong") & (table["metric"] == "score_S")].iloc[0]
    assert top["ci_low"] <= top["value"] <= top["ci_high"]
    ranks = table[table["metric"] == "rank"].set_index("experiment")["value"]
    assert ranks["R1:strong"] == 1 and ranks["R1:weak"] == 4
    assert table[table["metric"] == "advance"]["value"].sum() == 2


def test_outcomes_round_trip_through_npz(tmp_path: Path) -> None:
    o = _candidate(0.8, seed=1, base=_base())
    path = o.to_npz(tmp_path / "outputs" / "R1" / "cand" / "outcomes.npz")
    back = CandidateOutcomes.from_npz(path)
    for name in CandidateOutcomes.__dataclass_fields__:
        assert np.array_equal(getattr(back, name), getattr(o, name))
    with pytest.raises(ValueError, match="equal length"):
        CandidateOutcomes(
            **{**{n: getattr(o, n) for n in o.__dataclass_fields__}, "yes": o.yes[:5]}
        )
