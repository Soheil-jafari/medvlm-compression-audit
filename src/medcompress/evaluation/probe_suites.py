"""Metric suites computed from scored yes/no probes (no model needed).

A probe is one (image, finding) question. The model's answer to a probe is its
yes/no score from one forward pass (``VLMWrapper.score_yes_no``); the functions here
turn the scored probes into rows of the results schema:

* clinical: overall accuracy, macro-F1 and per-finding sensitivity / specificity,
* small findings: recall of positive, mask-annotated probes by lesion-size stratum,
* rare findings: recall by prevalence group,
* calibration: ECE, Brier score and the area under the risk-coverage curve.

Probe answers are never unparseable (the score always picks yes or no), so the
unparseable-answer rule of ``clinical.py`` only matters for the free-text suites.
Definitions (what counts as the overall accuracy, which findings enter the small
finding recall, the pooling of the rare-finding recall) are proposals for the
pre-registration; ``docs/preregistration.md`` fixes them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from medcompress.data.masks import assign_size_stratum
from medcompress.evaluation import calibration, clinical, rare_findings, small_findings
from medcompress.evaluation.statistics import bootstrap_ci


@dataclass(frozen=True)
class Metric:
    """One result row without its run identity (experiment, budget and seed come later)."""

    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(frozen=True)
class ProbeOutcomes:
    """Scored probes, one entry per (image, finding).

    ``margin`` is ``log P(yes) - log P(no)``; ``area`` is the lesion area fraction
    (NaN without a mask) and ``has_mask`` tells whether a lesion mask exists.
    ``yes_blank`` (optional) is the predicted answer on the blanked image.
    """

    sample_ids: tuple[str, ...]
    findings: tuple[str, ...]
    truth: np.ndarray
    yes: np.ndarray
    margin: np.ndarray
    area: np.ndarray
    has_mask: np.ndarray
    yes_blank: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        sizes = {len(self.findings), len(self.truth), len(self.yes), len(self.margin)}
        sizes |= {len(self.area), len(self.has_mask)}
        if self.yes_blank is not None:
            sizes.add(len(self.yes_blank))
        if sizes != {n}:
            raise ValueError("all probe arrays must have the same length")
        if n == 0:
            raise ValueError("no probes")

    @property
    def p_yes(self) -> np.ndarray:
        """Probability of "yes" renormalised over the two answers."""
        return 1.0 / (1.0 + np.exp(-np.clip(self.margin, -50.0, 50.0)))

    @property
    def correct(self) -> np.ndarray:
        """Whether the predicted answer equals the label."""
        return self.yes == self.truth

    def finding_index(self, order: Sequence[str]) -> np.ndarray:
        """Integer code of each probe's finding in ``order``."""
        lookup = {name: i for i, name in enumerate(order)}
        return np.array([lookup[f] for f in self.findings], dtype=int)


def _ci_kwargs(n_resamples: int, alpha: float, seed: int) -> dict[str, Any]:
    return {"n_resamples": n_resamples, "alpha": alpha, "seed": seed}


# ---------------------------------------------------------------------- clinical


def _macro_f1_from_codes(codes: np.ndarray, n_findings: int) -> float:
    """Macro-F1 from rows ``[finding_index, truth, predicted]`` (undefined F1 left out)."""
    finding, truth, pred = codes[:, 0], codes[:, 1].astype(bool), codes[:, 2].astype(bool)
    tp = np.bincount(finding, weights=(truth & pred), minlength=n_findings)
    fp = np.bincount(finding, weights=(~truth & pred), minlength=n_findings)
    fn = np.bincount(finding, weights=(truth & ~pred), minlength=n_findings)
    denominator = 2 * tp + fp + fn
    defined = denominator > 0
    if not defined.any():
        return math.nan
    return float(np.mean(2 * tp[defined] / denominator[defined]))


def clinical_metrics(
    out: ProbeOutcomes,
    findings: Sequence[str],
    *,
    n_resamples: int,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[Metric]:
    """Overall accuracy and macro-F1 (with bootstrap CIs) and per-finding sensitivity / specificity.

    The overall accuracy is the fraction of probes answered correctly. The CIs resample
    images: all probes of one image (one ``sample_id``) are drawn together. Used by: all.
    """
    kwargs = {**_ci_kwargs(n_resamples, alpha, seed), "clusters": np.asarray(out.sample_ids)}
    accuracy = bootstrap_ci(lambda a: float(np.mean(a)), out.correct.astype(float), **kwargs)
    codes = np.stack(
        [out.finding_index(findings), out.truth.astype(int), out.yes.astype(int)], axis=1
    )
    f1 = bootstrap_ci(lambda a: _macro_f1_from_codes(a, len(findings)), codes, **kwargs)
    metrics = [
        Metric("n_probes_scored", float(len(out.sample_ids))),
        Metric("overall_accuracy", *accuracy),
        Metric("macro_f1", *f1),
    ]
    truth = {f: [] for f in findings}
    pred: dict[str, list[bool]] = {f: [] for f in findings}
    for finding, label, answer in zip(out.findings, out.truth, out.yes, strict=True):
        truth[finding].append(bool(label))
        pred[finding].append(bool(answer))
    per_finding = clinical.per_finding_sensitivity_specificity(truth, pred)
    for finding in findings:
        metrics.append(Metric(f"sensitivity:{finding}", per_finding[finding]["sensitivity"]))
        metrics.append(Metric(f"specificity:{finding}", per_finding[finding]["specificity"]))
    return metrics


# ---------------------------------------------------------------- small findings


def small_positive_strata(
    out: ProbeOutcomes, thresholds: Sequence[float], mask_findings: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Rows that count for the small-finding recall and their lesion-size stratum.

    Returns ``(keep, strata)``: ``keep`` marks positive probes of the findings that the
    lesion masks delineate and that have a mask and an area; ``strata`` gives the stratum
    (0 = smallest) of the rows where ``keep`` is true, in order.
    """
    wanted = np.array([f in set(mask_findings) for f in out.findings])
    keep = out.truth & out.has_mask & wanted & np.isfinite(out.area)
    strata = np.array(
        [assign_size_stratum(float(a), thresholds) for a in out.area[keep]], dtype=int
    )
    return keep, strata


def small_finding_metrics(
    out: ProbeOutcomes,
    thresholds: Sequence[float],
    mask_findings: Sequence[str],
    *,
    n_resamples: int,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[Metric]:
    """Recall of positive, mask-annotated probes by lesion-size stratum.

    Only findings in ``mask_findings`` (the ones the lesion masks delineate) count.
    Stratum 0 is the smallest; ``small_finding_recall`` is its recall with a bootstrap
    CI over the images of the positives. ``thresholds`` are the frozen area cut points.
    Used by: all (primary endpoint at B2).
    """
    n_strata = len(thresholds) + 1
    keep, strata = small_positive_strata(out, thresholds, mask_findings)
    if not keep.any():
        raise ValueError("no positive, mask-annotated probes for the small-finding recall")
    detected = out.yes[keep]
    per_stratum = small_findings.recall_by_size_stratum(detected, strata.tolist(), n_strata)

    def stratum0_recall(codes: np.ndarray) -> float:
        in_small = codes[:, 1] == 0
        return float(codes[in_small, 0].mean()) if in_small.any() else math.nan

    codes = np.stack([detected.astype(int), strata], axis=1)
    images = np.asarray(out.sample_ids)[keep]  # an image with two masked findings moves as one
    value, low, high = bootstrap_ci(
        stratum0_recall, codes, clusters=images, **_ci_kwargs(n_resamples, alpha, seed)
    )
    metrics = [
        Metric("n_mask_positives", float(keep.sum())),
        Metric("small_finding_recall", value, low, high),
    ]
    metrics += [Metric(f"recall_stratum:{s}", per_stratum[s]) for s in range(n_strata)]
    metrics += [Metric(f"n_stratum:{s}", float((strata == s).sum())) for s in range(n_strata)]
    return metrics


def blank_image_metrics(out: ProbeOutcomes) -> list[Metric]:
    """Balanced accuracy on real and on blank images and the drop between them.

    The drop is the image-reliance component of selection round R1: a model that still
    answers as well on a blank image does not use the image. Needs ``yes_blank``.
    """
    from medcompress.evaluation.image_reliance import macro_balanced_accuracy

    if out.yes_blank is None:
        raise ValueError("the probes were not scored on blank images")
    names, truth = list(out.findings), out.truth.tolist()
    original = macro_balanced_accuracy(names, truth, out.yes.tolist())
    blank = macro_balanced_accuracy(names, truth, out.yes_blank.tolist())
    return [
        Metric("original_balanced_accuracy", original),
        Metric("blank_balanced_accuracy", blank),
        Metric("blank_accuracy_drop", original - blank),
        Metric("blank_yes_rate", float(out.yes_blank.mean())),
    ]


# --------------------------------------------------------------- rare findings


def group_of_finding(groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Invert ``{group: [findings]}`` to ``{finding: group}`` (no finding may be in two groups)."""
    mapping: dict[str, str] = {}
    for group, members in groups.items():
        for finding in members:
            if finding in mapping:
                raise ValueError(f"finding {finding!r} is in two prevalence groups")
            mapping[finding] = group
    return mapping


def rare_finding_metrics(
    out: ProbeOutcomes, groups: Mapping[str, Sequence[str]], *, pooling: str = "pooled"
) -> list[Metric]:
    """Recall of positive probes by prevalence group (``rare_recall:<group>``)."""
    membership = group_of_finding(groups)
    detected: dict[str, list[bool]] = {f: [] for f in membership}
    for finding, label, answer in zip(out.findings, out.truth, out.yes, strict=True):
        if label and finding in detected:
            detected[finding].append(bool(answer))
    recall = rare_findings.recall_by_prevalence_group(detected, membership, pooling=pooling)
    return [Metric(f"rare_recall:{group}", value) for group, value in recall.items()]


# ------------------------------------------------------------------ calibration


def calibration_metrics(out: ProbeOutcomes, *, n_bins: int) -> list[Metric]:
    """ECE (on the confidence of the predicted answer), Brier score and AURC.

    The confidence of an answer is the larger of P(yes) and P(no); the Brier score
    compares P(yes) with the label.
    """
    p_yes = out.p_yes
    confidence = np.maximum(p_yes, 1.0 - p_yes)
    coverage, risk = calibration.risk_coverage_curve(confidence, out.correct)
    return [
        Metric(
            "ece", calibration.expected_calibration_error(confidence, out.correct, n_bins=n_bins)
        ),
        Metric("brier", calibration.brier_score(p_yes, out.truth)),
        Metric("aurc", calibration.area_under_risk_coverage(coverage, risk)),
        Metric("mean_confidence", float(confidence.mean())),
    ]
