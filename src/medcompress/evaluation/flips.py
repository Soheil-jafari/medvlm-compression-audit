"""Which test items flip under compression (pre-registered secondary analysis, 2026-10-31).

In the spirit of Hooker, Courville, Clark, Dauphin and Frome, "What Do Compressed Deep Neural
Networks Forget?" (arXiv:1911.05248): they found that the cost of compression falls on a small
subset of examples, which a headline accuracy hides. This analysis asks where those examples
lie here. It needs no extra GPU time: it reads the per-probe answers that stage 06 already
saved for every arm (``outcomes_test.npz``).

* An **item** is one probe (image, finding). An item **flips** for an arm when the arm's
  majority answer over its seeds (yes or no; 3 seeds, so no ties) differs from E0's answer
  (the teacher, seed 0). The answer is compared, not its correctness.
* For each arm (method, variant and budget) the rows are:
  - ``n_items``, ``n_flips`` and ``flip_rate`` (all items);
  - per finding ``f``: ``flip_rate:<f>``, ``flip_share:<f>`` (the finding's part of the flips)
    and ``flip_ratio:<f>`` = ``flip_rate:<f>`` / ``flip_rate``;
  - per lesion-size stratum ``s`` of the positive, mask-annotated probes of the mask findings
    (the strata of the small-finding recall, from the frozen thresholds): ``n_items:stratum<s>``,
    ``n_flips:stratum<s>``, ``flip_rate:stratum<s>`` and ``flip_ratio:stratum<s>``;
  - ``diff:flip_rate:small_minus_larger``: the flip rate of stratum 0 (the small polyps) minus
    that of strata 1-3 together.
* Reading: small polyps are over-represented among the flips when the interval of
  ``flip_ratio:stratum0`` lies above 1 (equivalently, their share of the flips is larger than
  their share of the items), and smaller polyps flip more often than larger ones when the
  interval of the difference lies above 0. Descriptive only: not part of H1-H4, no verdict,
  no multiplicity correction.
* Intervals: 95 % image-level percentile bootstrap (all probes of an image move together),
  with the comparison step's resamples (seed 0).
* E3-E8 start from E2, which is distilled from E0, so their flips against E0 include those of
  the distillation itself. E2's own row (``E2_vs_E0``) shows how many; it is reported next to
  the compressed arms.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from medcompress.evaluation.statistics import percentile_ci, resample_indices

REFERENCE = "E0"
"""The answers every arm is compared with (the teacher, as pre-registered on 2026-10-31)."""
SMALL_STRATUM = 0


@dataclass(frozen=True)
class FlipItems:
    """What every arm shares: the items, their images, findings and size strata.

    ``stratum`` is the lesion-size stratum of positive, mask-annotated probes of the mask
    findings and -1 for every other item. ``reference_yes`` is E0's answer.
    """

    images: np.ndarray
    findings: tuple[str, ...]
    codes: np.ndarray
    stratum: np.ndarray
    n_strata: int
    reference_yes: np.ndarray


def flip_items(
    reference: Mapping[str, np.ndarray],
    order: np.ndarray,
    findings: Sequence[str],
    thresholds: Sequence[float],
    mask_findings: Sequence[str],
) -> FlipItems:
    """The items from E0's arrays, put in the comparison table's order by ``order``."""
    from medcompress.evaluation.probe_suites import ProbeOutcomes, small_positive_strata

    names = reference["probe_findings"].astype(str)[order]
    area = reference["probe_area"].astype(float)[order]
    has_mask = reference.get("probe_has_mask", np.isfinite(reference["probe_area"]))
    out = ProbeOutcomes(
        sample_ids=tuple(reference["probe_sample_ids"].astype(str)[order]),
        findings=tuple(names),
        truth=reference["probe_truth"].astype(bool)[order],
        yes=reference["probe_yes"].astype(bool)[order],
        margin=reference["probe_margin"].astype(float)[order],
        area=area,
        has_mask=np.asarray(has_mask).astype(bool)[order],
    )
    keep, strata = small_positive_strata(out, thresholds, mask_findings)
    stratum = np.full(len(names), -1, dtype=int)
    stratum[np.flatnonzero(keep)] = strata
    lookup = {f: i for i, f in enumerate(findings)}
    return FlipItems(
        images=np.asarray(out.sample_ids),
        findings=tuple(findings),
        codes=np.array([lookup[f] for f in names], dtype=int),
        stratum=stratum,
        n_strata=len(thresholds) + 1,
        reference_yes=out.yes,
    )


def majority_yes(majority_correct: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The majority answer from the majority correctness (right means the answer is the label)."""
    return np.asarray(majority_correct, dtype=bool) == np.asarray(truth, dtype=bool)


def metric_names(items: FlipItems) -> list[str]:
    """The columns of :func:`flip_values`, in order."""
    names = ["flip_rate"]
    names += [f"flip_rate:{f}" for f in items.findings]
    names += [f"flip_ratio:{f}" for f in items.findings]
    names += [f"flip_rate:stratum{s}" for s in range(items.n_strata)]
    names += [f"flip_ratio:stratum{s}" for s in range(items.n_strata)]
    names.append("diff:flip_rate:small_minus_larger")
    return names


def _divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def flip_values(flips: np.ndarray, items: FlipItems, idx: np.ndarray) -> np.ndarray:
    """The :func:`metric_names` columns of every arm on rows ``idx``: shape (arms, columns).

    ``flips`` has one row per arm and one column per item.
    """
    flips = np.asarray(flips, dtype=float)[:, idx]
    codes, stratum = items.codes[idx], items.stratum[idx]
    n_f, n_s = len(items.findings), items.n_strata
    by_finding = np.eye(n_f)[codes]  # items x findings
    by_stratum = np.zeros((len(idx), n_s))
    inside = stratum >= 0
    by_stratum[np.flatnonzero(inside), stratum[inside]] = 1.0
    overall = flips.mean(axis=1, keepdims=True)
    rate_f = _divide(flips @ by_finding, by_finding.sum(axis=0)[None, :])
    rate_s = _divide(flips @ by_stratum, by_stratum.sum(axis=0)[None, :])
    larger = by_stratum[:, SMALL_STRATUM + 1 :].sum(axis=1)
    rate_larger = _divide((flips @ larger)[:, None], np.array([[larger.sum()]]))
    diff = rate_s[:, [SMALL_STRATUM]] - rate_larger
    return np.hstack(
        [overall, rate_f, _divide(rate_f, overall), rate_s, _divide(rate_s, overall), diff]
    )


def flip_rows(
    labels: Sequence[tuple[str, str]],
    flips: np.ndarray,
    items: FlipItems,
    *,
    n_resamples: int,
    alpha: float,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Rows ``<label>_vs_E0`` for every arm (``labels``: (label, budget) per row of ``flips``)."""
    flips = np.asarray(flips, dtype=bool)
    n = flips.shape[1]
    every = np.arange(n)
    observed = flip_values(flips, items, every)
    draws = np.empty((n_resamples, *observed.shape))
    for r, idx in enumerate(resample_indices(n, n_resamples, seed, items.images)):
        draws[r] = flip_values(flips, items, idx)
    names = metric_names(items)
    rows: list[dict[str, Any] | None] = []
    for a, (label, budget) in enumerate(labels):
        name = f"{label}_vs_{REFERENCE}"
        total = int(flips[a].sum())
        counts = [("n_items", float(n)), ("n_flips", float(total))]
        for code, finding in enumerate(items.findings):
            share = flips[a, items.codes == code].sum() / total if total else math.nan
            counts.append((f"flip_share:{finding}", float(share)))
        for s in range(items.n_strata):
            in_stratum = items.stratum == s
            counts.append((f"n_items:stratum{s}", float(in_stratum.sum())))
            counts.append((f"n_flips:stratum{s}", float(flips[a, in_stratum].sum())))
        rows += [_row(name, budget, metric, value) for metric, value in counts]
        rows += [
            _row(name, budget, metric, float(observed[a, c]), percentile_ci(draws[:, a, c], alpha))
            for c, metric in enumerate(names)
        ]
    return [r for r in rows if r is not None]


def _row(
    name: str, budget: str, metric: str, value: float, ci: tuple[float, float] | None = None
) -> dict[str, Any] | None:
    """One result row (None for an undefined value); the CI is widened to hold the value."""
    if math.isnan(value):
        return None
    row: dict[str, Any] = {"experiment": name, "budget": budget, "seed": 0, "metric": metric,
                           "value": value}  # fmt: skip
    if ci is not None and not any(math.isnan(v) for v in ci):
        row["ci_low"], row["ci_high"] = min(ci[0], value), max(ci[1], value)
    return row
