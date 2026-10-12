"""Score S, ranks and bootstrap confidence intervals for one selection round.

Each candidate of a round is scored on the same fixed validation probes. Three
components make up the score of D05:

    S = 0.4 x probe macro-F1 + 0.3 x small-finding recall + 0.3 x image reliance

where the image-reliance component is the blank-image accuracy drop in round R1 and the
lesion-mask effect in R2 and R3. Each component is scaled to 0-1 across the candidates of
the round (min-max) before weighting. Candidates that fail a gate do not take part in
the scaling or the ranking. The advance list follows the point estimate of S. The
reported intervals come from a paired bootstrap over validation *images*: the same
resampled images are used for every candidate, so the interval of the score gap between
neighbouring ranks reflects the paired design.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COMPONENTS: tuple[str, ...] = ("probe_macro_f1", "small_finding_recall", "image_reliance")
RELIANCE_METRICS: tuple[str, ...] = ("blank_image_drop", "lesion_mask_effect")


@dataclass(frozen=True)
class CandidateOutcomes:
    """Per-probe answers of one candidate on the fixed validation subset.

    Rows ``0..n-1`` are the probes (one per image and finding) in a fixed order shared
    by all candidates. ``yes`` and ``yes_blank`` are the predicted answers on the real
    and on the blanked image; ``in_small`` marks positive, mask-annotated probes of the
    smallest lesion-size stratum. The audit arrays cover the mask-annotated positives:
    whether masking the lesion and masking a control region flipped the answer.
    """

    image_ids: np.ndarray
    finding_idx: np.ndarray
    truth: np.ndarray
    yes: np.ndarray
    yes_blank: np.ndarray
    in_small: np.ndarray
    audit_image_ids: np.ndarray
    audit_lesion_flip: np.ndarray
    audit_random_flip: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.image_ids)
        probe = (self.finding_idx, self.truth, self.yes, self.yes_blank, self.in_small)
        if n == 0 or any(len(a) != n for a in probe):
            raise ValueError("probe arrays must be non-empty and of equal length")
        m = len(self.audit_image_ids)
        if len(self.audit_lesion_flip) != m or len(self.audit_random_flip) != m:
            raise ValueError("audit arrays must be of equal length")

    def to_npz(self, path: Path | str) -> Path:
        """Write the arrays to a compressed ``.npz`` file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **{k: getattr(self, k) for k in self.__dataclass_fields__})
        return target

    @classmethod
    def from_npz(cls, path: Path | str) -> CandidateOutcomes:
        """Read arrays written by :meth:`to_npz`."""
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(**{k: data[k] for k in cls.__dataclass_fields__})


# ------------------------------------------------------------ weighted components


def _weighted_macro_f1(o: CandidateOutcomes, w: np.ndarray, n_findings: int) -> float:
    truth, pred = o.truth.astype(bool), o.yes.astype(bool)
    f = o.finding_idx
    tp = np.bincount(f, weights=w * (truth & pred), minlength=n_findings)
    fp = np.bincount(f, weights=w * (~truth & pred), minlength=n_findings)
    fn = np.bincount(f, weights=w * (truth & ~pred), minlength=n_findings)
    denominator = 2 * tp + fp + fn
    defined = denominator > 0
    return float(np.mean(2 * tp[defined] / denominator[defined])) if defined.any() else math.nan


def _weighted_macro_balanced_accuracy(
    o: CandidateOutcomes, w: np.ndarray, yes: np.ndarray, n_findings: int
) -> float:
    truth, pred, f = o.truth.astype(bool), yes.astype(bool), o.finding_idx
    pos = np.bincount(f, weights=w * truth, minlength=n_findings)
    neg = np.bincount(f, weights=w * ~truth, minlength=n_findings)
    hit_pos = np.bincount(f, weights=w * (truth & pred), minlength=n_findings)
    hit_neg = np.bincount(f, weights=w * (~truth & ~pred), minlength=n_findings)
    both = (pos > 0) & (neg > 0)
    if not both.any():
        return math.nan
    return float(np.mean((hit_pos[both] / pos[both] + hit_neg[both] / neg[both]) / 2))


def component_values(
    o: CandidateOutcomes,
    reliance_metric: str,
    n_findings: int,
    probe_weights: np.ndarray | None = None,
    audit_weights: np.ndarray | None = None,
) -> np.ndarray:
    """The three score components ``[macro-F1, small-finding recall, image reliance]``.

    Weights are per-row multiplicities (all ones for the point estimate). A component that
    is undefined (for example no small positives drawn) is NaN.
    """
    if reliance_metric not in RELIANCE_METRICS:
        raise ValueError(f"reliance metric must be one of {RELIANCE_METRICS}")
    w = np.ones(len(o.image_ids)) if probe_weights is None else probe_weights
    f1 = _weighted_macro_f1(o, w, n_findings)
    small_total = float((w * o.in_small).sum())
    small = float((w * o.in_small * o.yes).sum() / small_total) if small_total > 0 else math.nan
    if reliance_metric == "blank_image_drop":
        reliance = _weighted_macro_balanced_accuracy(
            o, w, o.yes, n_findings
        ) - _weighted_macro_balanced_accuracy(o, w, o.yes_blank, n_findings)
    else:
        a = np.ones(len(o.audit_image_ids)) if audit_weights is None else audit_weights
        total = float(a.sum())
        reliance = (
            float((a * o.audit_lesion_flip).sum() / total - (a * o.audit_random_flip).sum() / total)
            if total > 0
            else math.nan
        )
    return np.array([f1, small, reliance], dtype=float)


# ------------------------------------------------------------------- scaling


def scaled_score(values: np.ndarray, weights: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Min-max scale each component across candidates and weight it.

    ``values`` has shape ``(n_candidates, 3)``. Returns ``(S, scaled)``. A component that
    is equal for all candidates scales to 0 for all of them (it cannot separate them); a
    NaN component counts as the round's minimum.
    """
    v = np.array(values, dtype=float)
    if v.ndim != 2 or v.shape[1] != len(weights):
        raise ValueError("values must have one column per weight")
    scaled = np.zeros_like(v)
    for c in range(v.shape[1]):
        column = v[:, c]
        known = ~np.isnan(column)
        if not known.any():
            continue
        low, high = column[known].min(), column[known].max()
        if high > low:
            scaled[:, c] = np.where(known, (column - low) / (high - low), 0.0)
    return scaled @ np.asarray(weights, dtype=float), scaled


# ------------------------------------------------------------------- the round


@dataclass(frozen=True)
class RoundRanking:
    """Result of ranking one round."""

    candidates: tuple[str, ...]  # rank order (best first); gate failures are not included
    scores: np.ndarray  # S, aligned with ``candidates``
    scaled: np.ndarray  # scaled components, shape (n, 3)
    raw: np.ndarray  # raw components, shape (n, 3)
    score_ci: np.ndarray  # (n, 2) percentile interval of S
    gap: np.ndarray  # S minus S of the next rank (NaN for the last)
    gap_ci: np.ndarray  # (n, 2) interval of the gap
    advance: tuple[str, ...]
    failed_gates: Mapping[str, tuple[str, ...]]


def _check_aligned(outcomes: Mapping[str, Sequence[CandidateOutcomes]]) -> None:
    flat = [(name, o) for name in sorted(outcomes) for o in outcomes[name]]
    first_name, first = flat[0]
    for name, other in flat[1:]:
        if not (
            np.array_equal(first.image_ids, other.image_ids)
            and np.array_equal(first.finding_idx, other.finding_idx)
            and np.array_equal(first.truth, other.truth)
            and np.array_equal(first.audit_image_ids, other.audit_image_ids)
        ):
            raise ValueError(f"{name} was not scored on the same probes as {first_name}")


def rank_round(
    outcomes: Mapping[str, CandidateOutcomes | Sequence[CandidateOutcomes]],
    *,
    weights: Sequence[float],
    reliance_metric: str,
    keep: int,
    n_findings: int,
    n_resamples: int,
    gate_failures: Mapping[str, Sequence[str]] | None = None,
    tie_break: Mapping[str, tuple[float, ...]] | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> RoundRanking:
    """Rank the candidates of a round and decide who advances.

    A candidate may have several outcomes (one per seed); its components are the mean over
    the seeds. ``gate_failures`` lists the gates each candidate failed; those candidates are
    left out of the scaling and the ranking. ``tie_break`` maps a candidate to a tuple that decides
    exact ties on S (smaller sorts first, for example fp16 peak memory). The bootstrap
    resamples the validation images with the same draws for every candidate.

    Raises:
        ValueError: the candidates were not scored on the same probes, or nobody is left.
    """
    failed = {k: tuple(v) for k, v in (gate_failures or {}).items() if v}
    runs = {n: [v] if isinstance(v, CandidateOutcomes) else list(v) for n, v in outcomes.items()}
    if any(not v for v in runs.values()):
        raise ValueError("a candidate has no outcomes")
    names = sorted(n for n in runs if n not in failed)
    if not names:
        raise ValueError("every candidate failed a gate")
    _check_aligned(runs)
    first = runs[names[0]][0]
    images = np.unique(np.concatenate([first.image_ids, first.audit_image_ids]))
    probe_index = np.searchsorted(images, first.image_ids)
    audit_index = np.searchsorted(images, first.audit_image_ids)

    def evaluate(probe_w: np.ndarray | None, audit_w: np.ndarray | None) -> np.ndarray:
        return np.stack(
            [
                np.mean(
                    [
                        component_values(o, reliance_metric, n_findings, probe_w, audit_w)
                        for o in runs[n]
                    ],
                    axis=0,
                )
                for n in names
            ]
        )

    raw = evaluate(None, None)
    score, scaled = scaled_score(raw, weights)
    rng = np.random.default_rng(seed)
    replicates = np.empty((n_resamples, len(names)))
    for b in range(n_resamples):
        counts = np.bincount(rng.integers(0, len(images), len(images)), minlength=len(images))
        replicates[b] = scaled_score(
            evaluate(counts[probe_index].astype(float), counts[audit_index].astype(float)),
            weights,
        )[0]
    ties = {n: tuple((tie_break or {}).get(n, ())) for n in names}
    order = sorted(range(len(names)), key=lambda i: (-round(float(score[i]), 12), ties[names[i]]))
    ranked = tuple(names[i] for i in order)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    score_ci = np.stack([np.percentile(replicates[:, i], [lo, hi]) for i in order])
    gap = np.full(len(order), math.nan)
    gap_ci = np.full((len(order), 2), math.nan)
    for k in range(len(order) - 1):
        diff = replicates[:, order[k]] - replicates[:, order[k + 1]]
        gap[k] = score[order[k]] - score[order[k + 1]]
        gap_ci[k] = np.percentile(diff, [lo, hi])
    return RoundRanking(
        candidates=ranked,
        scores=score[order],
        scaled=scaled[order],
        raw=raw[order],
        score_ci=score_ci,
        gap=gap,
        gap_ci=gap_ci,
        advance=ranked[:keep],
        failed_gates=failed,
    )


# ---------------------------------------------------------------------- rows


def ranking_rows(result: RoundRanking, round_name: str) -> list[dict[str, Any]]:
    """Rows of ``results/selection/<round>/ranking.csv`` in the results schema.

    ``experiment`` is ``<round>:<candidate>``. Besides ``score_S`` (with its interval),
    ``rank``, ``advance`` and ``score_gap_to_next`` (with its interval; an interval that
    contains zero means the point-estimate order is not separated from the next rank),
    the raw and scaled components are written.
    """

    def row(candidate: str, metric: str, value: float, low: float | None = None,
            high: float | None = None) -> dict[str, Any]:  # fmt: skip
        out: dict[str, Any] = {
            "experiment": f"{round_name}:{candidate}",
            "budget": "",
            "seed": 0,
            "metric": metric,
            "value": float(value),
        }
        if low is not None and high is not None and not (math.isnan(low) or math.isnan(high)):
            out["ci_low"], out["ci_high"] = min(low, value), max(high, value)
        return out

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(result.candidates):
        rows.append(row(name, "score_S", result.scores[i], *result.score_ci[i]))
        rows.append(row(name, "rank", i + 1))
        rows.append(row(name, "advance", float(name in result.advance)))
        if not math.isnan(result.gap[i]):
            rows.append(row(name, "score_gap_to_next", result.gap[i], *result.gap_ci[i]))
        for j, component in enumerate(COMPONENTS):
            if not math.isnan(result.raw[i, j]):
                rows.append(row(name, f"raw:{component}", result.raw[i, j]))
            rows.append(row(name, f"scaled:{component}", result.scaled[i, j]))
    for name, gates in result.failed_gates.items():
        rows.append(row(name, "advance", 0.0))
        for gate in gates:
            rows.append(row(name, f"gate_failed:{gate}", 1.0))
    return rows
