"""FBR's evaluation side (amendment of 2026-11-22; ``docs/decisions/fbr_study_2026-11-22.md``):
the probe-answer pass (decoded free answers and token ranks, GPU), the cross-fitted operating
points, the two recalibrations, the diagnostic and the pilot's gate (CPU, from saved arrays).
Validation only: every function that reads a split refuses any other.

Arrays of an outcomes file (``outcomes.npz``): ``probe_sample_ids``, ``probe_findings``,
``probe_truth``, ``probe_margin`` and the audit arrays; of a probe-answer file
(:data:`ANSWERS_FILE`): the same keys, ``probe_yes_rank``, ``probe_no_rank``,
``probe_decoded`` (1 yes, 0 no, -1 parse failure) and ``probe_decoded_text``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ANSWERS_FILE = "probe_answers_{split}.npz"
SPLIT = "val"
SPECIFICITY = 0.95
QUANTILES = (5, 25, 50, 75, 95)
N_BINS = 15  # the pre-registered calibration bins (configs/base.yaml)
COLLAPSED = ("E5:magnitude", "E5:activation", "E6")
"""The arms whose plain answers collapsed at Gate 2 (plain macro balanced accuracy 0.500)."""


def require_validation(split: str) -> None:
    """Refuse every split but validation."""
    if split != SPLIT:
        raise ValueError(f"the FBR diagnostic and pilot read validation only, not {split!r}")


# ----------------------------------------------------------------------------- GPU pass


_WORD = re.compile(r"[a-z]+")


def parse_answer(text: str) -> int:
    """1 for "yes", 0 for "no", -1 for anything else: the first word of the lower-cased,
    stripped text, punctuation removed (decision file, section 8)."""
    words = text.strip().lower().split()
    if not words:
        return -1
    match = _WORD.search(words[0])
    word = match.group(0) if match else ""
    return 1 if word == "yes" else 0 if word == "no" else -1


def probe_answers(
    model: Any, samples: Sequence[Any], settings: Any, data_root: Path, max_new_tokens: int
) -> dict[str, np.ndarray]:
    """For every probe of the evaluation (the same probes as ``runner.score_probes``): the
    margin, the best rank of a yes and of a no spelling among all tokens at the answer position
    (1 = top), and the greedy decoded answer (at most ``max_new_tokens``), parsed."""
    import torch

    from medcompress.evaluation.image_reliance import subsample_per_finding, unique_image_probes
    from medcompress.evaluation.runner import _CHUNK, PROBE_SAMPLE_SEED

    require_validation(settings.split)
    probes = subsample_per_finding(
        unique_image_probes(samples, settings.findings, settings.template,
                            display_names=settings.display_names),
        settings.max_per_finding, seed=PROBE_SAMPLE_SEED,
    )  # fmt: skip
    yes_ids, no_ids = list(model.yes_ids), list(model.no_ids)
    margin, yes_rank, no_rank, decoded, texts = [], [], [], [], []
    for start in range(0, len(probes), _CHUNK):
        part = probes[start : start + _CHUNK]
        images = [Path(data_root) / p.image_path for p in part]
        questions = [p.question for p in part]
        for rows in model._batches(len(part)):
            inputs = model.build_inputs([images[i] for i in rows], [questions[i] for i in rows])
            with torch.inference_mode():
                z = model.module(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                z = z.float()
            for b in range(z.shape[0]):
                row = z[b]
                margin.append(float(torch.logsumexp(row[yes_ids], 0)
                                    - torch.logsumexp(row[no_ids], 0)))  # fmt: skip
                yes_rank.append(min(int((row > row[i]).sum()) + 1 for i in yes_ids))
                no_rank.append(min(int((row > row[i]).sum()) + 1 for i in no_ids))
        for output in model.generate(images, questions, max_new_tokens=max_new_tokens):
            texts.append(output.text)
            decoded.append(parse_answer(output.text))
    return {
        "probe_sample_ids": np.array([p.sample_id for p in probes]),
        "probe_findings": np.array([p.finding for p in probes]),
        "probe_truth": np.array([p.present for p in probes], dtype=bool),
        "probe_margin": np.array(margin, dtype=float),
        "probe_yes_rank": np.array(yes_rank, dtype=np.int64),
        "probe_no_rank": np.array(no_rank, dtype=np.int64),
        "probe_decoded": np.array(decoded, dtype=np.int8),
        "probe_decoded_text": np.array(texts),
    }


# ----------------------------------------------------------------------------- folds


@dataclass(frozen=True)
class FoldPlan:
    """Image-grouped, finding-stratified folds of the validation probes."""

    fold_of: Mapping[str, int]  # image (sample id) -> fold
    k: int
    short: tuple[str, ...]  # findings without the minimum in every fold of k (no op point)

    def folds(self, ids: np.ndarray) -> np.ndarray:
        return np.array([self.fold_of[str(i)] for i in ids], dtype=np.int64)


def _assign(images: Sequence[str], strata: Mapping[str, tuple[str, ...]], k: int,
            seed: int) -> dict[str, int]:  # fmt: skip
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    position = 0
    by_stratum: dict[tuple[str, ...], list[str]] = {}
    for image in sorted(images):
        by_stratum.setdefault(strata[image], []).append(image)
    for key in sorted(by_stratum):
        group = by_stratum[key]
        for i in rng.permutation(len(group)):
            fold_of[group[int(i)]] = position % k
            position += 1
    return fold_of


def assign_folds(
    ids: np.ndarray,
    findings: np.ndarray,
    truth: np.ndarray,
    order: Sequence[str],
    *,
    k: int = 3,
    min_per_class: int = 5,
    seed: int = 0,
) -> FoldPlan:
    """Folds from the validation labels alone: images grouped (all probes of an image in one
    fold), stratified by the set of findings present, assigned round robin over a seeded
    shuffle of each stratum. ``k`` folds when every fold holds at least ``min_per_class``
    positive and negative images of every finding, else 2 (decision file, section 6)."""
    images = sorted({str(i) for i in ids})
    present: dict[str, set[str]] = {i: set() for i in images}
    for i, f, t in zip(ids, findings, truth, strict=True):
        if t:
            present[str(i)].add(str(f))
    strata = {i: tuple(sorted(present[i])) for i in images}

    def short_of(fold_of: Mapping[str, int], folds: int) -> list[str]:
        short = []
        for f in order:
            rows = findings == f
            for fold in range(folds):
                in_fold = np.array([fold_of[str(i)] == fold for i in ids[rows]], dtype=bool)
                pos = int((truth[rows] & in_fold).sum())
                neg = int((~truth[rows] & in_fold).sum())
                if pos < min_per_class or neg < min_per_class:
                    short.append(f)
                    break
        return short

    for folds in (k, 2):
        fold_of = _assign(images, strata, folds, seed)
        short = short_of(fold_of, folds)
        if not short or folds == 2:
            return FoldPlan(fold_of, folds, tuple(short))
    raise AssertionError("unreachable")


def weighted_threshold(margins: np.ndarray, weights: np.ndarray, specificity: float) -> float:
    """``probe_suites.fit_thresholds`` with multiplicities: the k-th smallest negative margin,
    k = ceil(specificity x total weight)."""
    keep = weights > 0
    margins, weights = margins[keep], weights[keep]
    if margins.size == 0:
        return math.nan
    order = np.argsort(margins, kind="stable")
    cumulative = np.cumsum(weights[order])
    k = max(1, math.ceil(specificity * cumulative[-1] - 1e-9))
    return float(margins[order][int(np.searchsorted(cumulative, k))])


def crossfit(
    findings: np.ndarray,
    truth: np.ndarray,
    margin: np.ndarray,
    folds: np.ndarray,
    order: Sequence[str],
    k: int,
    *,
    weights: np.ndarray | None = None,
    specificity: float = SPECIFICITY,
) -> np.ndarray:
    """Cross-fitted operating-point decisions: for each fold the threshold for ``specificity``
    is fitted on the other folds' negatives and applied to the fold (margin above it means
    "yes"). ``weights``: probe multiplicities (a bootstrap resample)."""
    w = np.ones(len(margin)) if weights is None else np.asarray(weights, dtype=float)
    yes = np.zeros(len(margin), dtype=bool)
    for f in order:
        rows = findings == f
        for fold in range(k):
            fit = rows & (folds != fold) & ~truth
            limit = weighted_threshold(margin[fit], w[fit], specificity)
            held = rows & (folds == fold)
            yes[held] = margin[held] > limit if not math.isnan(limit) else margin[held] >= 0
    return yes


# ----------------------------------------------------------------------------- recalibration


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def fit_intercept(margin: np.ndarray, truth: np.ndarray) -> float:
    """b minimising the summed NLL of sigmoid(m + b) plus b^2 / 2 (Newton; convex)."""
    b = 0.0
    y = truth.astype(float)
    for _ in range(100):
        p = _sigmoid(margin + b)
        grad = float((p - y).sum()) + b
        hess = float((p * (1 - p)).sum()) + 1.0
        step = grad / hess
        b -= step
        if abs(step) < 1e-10:
            break
    return b


def fit_temperature(margin: np.ndarray, truth: np.ndarray) -> float:
    """T > 0 minimising the summed NLL of sigmoid(m / T) (golden section on a = 1 / T in
    [1e-3, 100]; the NLL is convex in a)."""
    y = truth.astype(float)

    def nll(a: float) -> float:
        z = a * margin
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    lo, hi = 1e-3, 100.0
    g = (math.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(200):
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    return 1.0 / ((lo + hi) / 2)


def crossfit_probabilities(
    findings: np.ndarray, truth: np.ndarray, margin: np.ndarray, folds: np.ndarray, k: int,
    order: Sequence[str],
) -> dict[str, np.ndarray]:  # fmt: skip
    """Held-out p(yes) of the raw margin and of both recalibrations, each fitted on the other
    folds: an intercept per finding (ridge) and one pooled temperature."""
    raw = _sigmoid(margin)
    intercept = np.zeros_like(margin)
    temperature = np.zeros_like(margin)
    for fold in range(k):
        fit, held = folds != fold, folds == fold
        t = fit_temperature(margin[fit], truth[fit])
        temperature[held] = _sigmoid(margin[held] / t)
        for f in order:
            rows_fit, rows_held = fit & (findings == f), held & (findings == f)
            b = fit_intercept(margin[rows_fit], truth[rows_fit])
            intercept[rows_held] = _sigmoid(margin[rows_held] + b)
    return {"raw": raw, "intercept": intercept, "temperature": temperature}


# ----------------------------------------------------------------------------- metrics


def auroc(truth: np.ndarray, score: np.ndarray) -> float:
    """AUROC of ``score`` (NaN when one class is missing)."""
    from medcompress.evaluation.clinical import auroc as _auroc

    if truth.all() or not truth.any():
        return math.nan
    return float(_auroc(list(truth), list(score)))


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    """Average precision (the mean precision at each positive, ranked by score)."""
    if not truth.any():
        return math.nan
    order = np.argsort(-score, kind="stable")
    hits = truth[order].astype(float)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def rates(truth: np.ndarray, yes: np.ndarray) -> dict[str, float]:
    """Sensitivity, specificity, balanced accuracy and the confusion counts."""
    pos, neg = truth, ~truth
    sens = float(yes[pos].mean()) if pos.any() else math.nan
    spec = float((~yes[neg]).mean()) if neg.any() else math.nan
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2,
            "tp": int((yes & pos).sum()), "fn": int((~yes & pos).sum()),
            "fp": int((yes & neg).sum()), "tn": int((~yes & neg).sum())}  # fmt: skip


def decoded_rates(truth: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    """Rates of the decoded answers, a parse failure wrong for positives and negatives."""
    correct = np.where(truth, decoded == 1, decoded == 0)
    pos, neg = truth, ~truth
    sens = float(correct[pos].mean()) if pos.any() else math.nan
    spec = float(correct[neg].mean()) if neg.any() else math.nan
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2,
            "parse_failures": int((decoded == -1).sum())}  # fmt: skip


def calibration(truth: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Brier, NLL and ECE (pre-registered bins) of p(yes)."""
    from medcompress.evaluation.calibration import expected_calibration_error

    y = truth.astype(float)
    eps = 1e-12
    confidence = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == truth
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
        "ece": float(expected_calibration_error(confidence, correct, n_bins=N_BINS)),
    }


# ----------------------------------------------------------------------------- diagnostic


def load(path: Path) -> dict[str, np.ndarray]:
    """Every array of an ``.npz`` file, read into memory."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def aligned(answers: Mapping[str, np.ndarray], outcomes: Mapping[str, np.ndarray]) -> bool:
    """True when both files hold the same probes in the same order."""
    return bool(
        np.array_equal(answers["probe_sample_ids"], outcomes["probe_sample_ids"])
        and np.array_equal(answers["probe_findings"], outcomes["probe_findings"])
    )


def diagnostic_rows(
    label: str,
    outcomes: Mapping[str, np.ndarray],
    answers: Mapping[str, np.ndarray] | None,
    plan: FoldPlan,
    order: Sequence[str],
) -> list[dict[str, Any]]:
    """One row per finding (and recalibration): the diagnostic of section 2."""
    ids, findings = outcomes["probe_sample_ids"], outcomes["probe_findings"]
    truth, margin = outcomes["probe_truth"].astype(bool), outcomes["probe_margin"].astype(float)
    folds = plan.folds(ids)
    probabilities = crossfit_probabilities(findings, truth, margin, folds, plan.k, order)
    op_yes = crossfit(findings, truth, margin, folds, order, plan.k)
    if answers is not None and not aligned(answers, outcomes):
        raise ValueError(f"{label}: the probe-answer file and the outcomes hold other probes")
    rows = []
    for f in order:
        r = findings == f
        t, m = truth[r], margin[r]
        row: dict[str, Any] = {"arm": label, "finding": f, "positives": int(t.sum()),
                               "negatives": int((~t).sum()), "auroc": auroc(t, m),
                               "ap": average_precision(t, m)}  # fmt: skip
        for name, sel in (("pos", t), ("neg", ~t)):
            for q in QUANTILES:
                row[f"margin_{name}_q{q}"] = (
                    float(np.percentile(m[sel], q)) if sel.any() else math.nan
                )
        if answers is not None:
            decoded = answers["probe_decoded"][r]
            parsed = decoded >= 0
            row["parse_failures"] = int((~parsed).sum())
            row["sign_agrees_with_decoded"] = (
                float(((m[parsed] >= 0) == (decoded[parsed] == 1)).mean()) if parsed.any()
                else math.nan
            )  # fmt: skip
            for name, sel in (("pos", t), ("neg", ~t)):
                row[f"yes_rank_median_{name}"] = (
                    float(np.median(answers["probe_yes_rank"][r][sel])) if sel.any() else math.nan
                )
                row[f"no_rank_median_{name}"] = (
                    float(np.median(answers["probe_no_rank"][r][sel])) if sel.any() else math.nan
                )
        else:
            row["parse_failures"] = row["sign_agrees_with_decoded"] = "not available"
        op = rates(t, op_yes[r])
        row["crossfit_sensitivity_at_spec95"] = op["sensitivity"]
        row["crossfit_achieved_specificity"] = op["specificity"]
        for name, p in probabilities.items():
            row[f"{name}_brier"] = float(np.mean((p[r] - t) ** 2))
            row[f"{name}_ba_at_0.5"] = rates(t, p[r] >= 0.5)["balanced_accuracy"]
            row[f"{name}_auroc"] = auroc(t, p[r])
        row["folds"] = plan.k
        row["no_operating_point"] = f in plan.short
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- pilot


@dataclass
class ModelArrays:
    """One model's validation arrays for the pilot."""

    label: str
    outcomes: Mapping[str, np.ndarray]
    answers: Mapping[str, np.ndarray] | None


def gate_quantities(
    model: ModelArrays, plan: FoldPlan, order: Sequence[str], rare: Sequence[str],
    weights: np.ndarray | None = None,
) -> dict[str, float]:  # fmt: skip
    """The gate's quantities of one model (percentages as fractions), optionally under
    bootstrap weights (per probe), with the cross-fitted thresholds refitted under them."""
    o = model.outcomes
    ids, findings = o["probe_sample_ids"], o["probe_findings"]
    truth, margin = o["probe_truth"].astype(bool), o["probe_margin"].astype(float)
    w = np.ones(len(margin)) if weights is None else weights
    yes = crossfit(findings, truth, margin, plan.folds(ids), order, plan.k, weights=w)
    p = _sigmoid(margin)

    def wmean(x: np.ndarray, sel: np.ndarray) -> float:
        total = w[sel].sum()
        return float((x[sel] * w[sel]).sum() / total) if total > 0 else math.nan

    out: dict[str, float] = {}
    briers, decoded_ba, plain_ba = [], [], []
    for f in order:
        r = findings == f
        out[f"sens:{f}"] = wmean(yes.astype(float), r & truth)
        out[f"spec:{f}"] = wmean((~yes).astype(float), r & ~truth)
        briers.append(wmean((p - truth) ** 2, r))
        plain = margin >= 0
        plain_ba.append((wmean(plain.astype(float), r & truth)
                         + wmean((~plain).astype(float), r & ~truth)) / 2)  # fmt: skip
        if model.answers is not None:
            d = model.answers["probe_decoded"]
            decoded_ba.append((wmean((d == 1).astype(float), r & truth)
                               + wmean((d == 0).astype(float), r & ~truth)) / 2)  # fmt: skip
    out["R"] = float(np.mean([out[f"sens:{f}"] for f in rare]))
    out["macro_brier"] = float(np.mean(briers))
    out["plain_macro_ba"] = float(np.mean(plain_ba))
    out["decoded_macro_ba"] = float(np.mean(decoded_ba)) if decoded_ba else math.nan
    return out


GATE_ITEMS = ("delta_R", "rare_sensitivity", "rare_specificity", "decoded_macro_ba",
              "macro_brier", "polyp_recall")  # fmt: skip


def gate(
    fbr: Mapping[str, float], registered: Mapping[str, float], *, collapsed: bool,
    rare: Sequence[str], polyp: str = "polyp",
) -> dict[str, bool]:  # fmt: skip
    """The gate of section 7 (absolute percentage points against the arm's registered
    recovery)."""
    pp = 100.0
    d = {k: (fbr[k] - registered[k]) * pp for k in fbr}
    decoded_ok = d["decoded_macro_ba"] >= 3.0 if collapsed else d["decoded_macro_ba"] >= -2.0
    return {
        "delta_R": d["R"] >= 5.0,
        "rare_sensitivity": all(d[f"sens:{f}"] >= -3.0 for f in rare),
        "rare_specificity": all(d[f"spec:{f}"] >= -2.0 for f in rare),
        "decoded_macro_ba": bool(decoded_ok) and math.isfinite(d["decoded_macro_ba"]),
        "macro_brier": (fbr["macro_brier"] - registered["macro_brier"]) <= 0.01,
        "polyp_recall": d[f"sens:{polyp}"] >= -3.0,
    }


def paired_bootstrap(
    a: ModelArrays, b: ModelArrays, plan: FoldPlan, order: Sequence[str], rare: Sequence[str],
    *, resamples: int, seed: int = 0, alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:  # fmt: skip
    """Image-cluster paired bootstrap intervals of ``a`` minus ``b`` for every gate quantity,
    the cross-fitted thresholds refitted inside each resample."""
    ids = a.outcomes["probe_sample_ids"]
    if not np.array_equal(ids, b.outcomes["probe_sample_ids"]):
        raise ValueError(f"{a.label} and {b.label} hold other probes")
    unique, index = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(resamples):
        counts = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)))
        w = counts[index].astype(float)
        qa = gate_quantities(a, plan, order, rare, w)
        qb = gate_quantities(b, plan, order, rare, w)
        for key in qa:
            draws.setdefault(key, []).append(qa[key] - qb[key])
    out = {}
    for key, values in draws.items():
        v = np.asarray(values, dtype=float)
        if np.isnan(v).all():  # e.g. no decoded answers
            out[key] = (math.nan, math.nan)
            continue
        out[key] = (float(np.nanpercentile(v, 100 * alpha / 2)),
                    float(np.nanpercentile(v, 100 * (1 - alpha / 2))))  # fmt: skip
    return out


def report_rows(model: ModelArrays, plan: FoldPlan, order: Sequence[str]) -> list[dict[str, Any]]:
    """Per finding: counts with denominators, confusion matrices at the native threshold
    (m >= 0) and at the cross-fitted operating point, AUROC, AP, NLL, Brier, ECE."""
    o = model.outcomes
    ids, findings = o["probe_sample_ids"], o["probe_findings"]
    truth, margin = o["probe_truth"].astype(bool), o["probe_margin"].astype(float)
    yes_op = crossfit(findings, truth, margin, plan.folds(ids), order, plan.k)
    p = _sigmoid(margin)
    rows = []
    for f in order:
        r = findings == f
        row: dict[str, Any] = {"model": model.label, "finding": f}
        for name, yes in (("native", margin >= 0), ("operating", yes_op)):
            for key, value in rates(truth[r], yes[r]).items():
                row[f"{name}_{key}"] = value
        if model.answers is not None:
            for key, value in decoded_rates(truth[r], model.answers["probe_decoded"][r]).items():
                row[f"decoded_{key}"] = value
        row["auroc"] = auroc(truth[r], margin[r])
        row["ap"] = average_precision(truth[r], margin[r])
        row.update(calibration(truth[r], p[r]))
        rows.append(row)
    return rows


def mask_audit(model: ModelArrays, plan: FoldPlan, order: Sequence[str]) -> dict[str, float]:
    """The lesion-mask audit in both versions (plain answer; the cross-fitted operating
    threshold of the probe's fold) and the mean margin changes (lesion and random region
    masked, minus the original)."""
    o = model.outcomes
    if "audit_lesion_flip" not in o:
        return {}
    out = {"plain_effect": float(o["audit_lesion_flip"].mean() - o["audit_random_flip"].mean()),
           "audit_probes": int(len(o["audit_lesion_flip"]))}  # fmt: skip
    if "audit_original_margin" not in o:
        return out
    ids, findings = o["probe_sample_ids"], o["probe_findings"]
    truth, margin = o["probe_truth"].astype(bool), o["probe_margin"].astype(float)
    folds = plan.folds(ids)
    limit_of = {}
    for f in order:
        for fold in range(plan.k):
            fit = (findings == f) & (folds != fold) & ~truth
            limit_of[(f, fold)] = weighted_threshold(
                margin[fit], np.ones(int(fit.sum())), SPECIFICITY
            )
    audit_ids, audit_f = o["audit_sample_ids"], o["audit_findings"]
    limits = np.array(
        [limit_of[(str(f), plan.fold_of[str(i)])] for i, f in zip(audit_ids, audit_f, strict=True)]
    )
    original = o["audit_original_margin"] > limits
    lesion = o["audit_lesion_margin"] > limits
    random = o["audit_random_margin"] > limits
    out["operating_effect"] = float((lesion != original).mean() - (random != original).mean())
    out["lesion_margin_change"] = float(
        (o["audit_lesion_margin"] - o["audit_original_margin"]).mean()
    )
    out["random_margin_change"] = float(
        (o["audit_random_margin"] - o["audit_original_margin"]).mean()
    )
    return out


# ----------------------------------------------------------------------------- driver

DIAGNOSTIC_ARMS = ("E2", "E5:magnitude@B2", "E5:activation@B2", "E5:taylor@B2", "E5R@B2",
                   "E6@B2", "E7@B2", "E7U@B2")  # fmt: skip
RARE = ("oesophagitis", "ulcerative_colitis")


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _arm_key(run: Any) -> str:
    return run.arm + ("" if run.budget == "nobudget" else f"@{run.budget}")


def find_registered(roots: Sequence[Path], known: Sequence[str]) -> dict[str, ModelArrays]:
    """Seed-0 validation outcomes of E2 and the B2 pruning arms, with their probe answers when
    found (``fbr/probe_answers/<run dir>/probe_answers_val.npz``)."""
    from medcompress.evaluation.exploratory import discover, parse_run

    runs, _notes = discover(roots, known)
    answers: dict[str, Path] = {}
    for root in roots:
        for path in sorted(Path(root).rglob(ANSWERS_FILE.format(split=SPLIT))):
            parsed = parse_run(path, known)
            if parsed is not None and "probe_answers" in path.parts:
                experiment, variant, budget, seed = parsed
                key = experiment + (f":{variant}" if variant else "")
                key += "" if budget == "nobudget" else f"@{budget}"
                if seed == 0:
                    answers.setdefault(key, path)
    found = {}
    for run in runs.values():
        key = _arm_key(run)
        if run.seed != 0 or key not in DIAGNOSTIC_ARMS or SPLIT not in run.files:
            continue
        outcomes = load(run.files[SPLIT])
        if SPLIT in run.audits:  # a mask-audit-only rerun: the audit with its margins
            audit = load(run.audits[SPLIT])
            outcomes.update({k: v for k, v in audit.items() if k.startswith("audit_")})
        found[key] = ModelArrays(key, outcomes, load(answers[key]) if key in answers else None)
    return found


def find_fbr(roots: Sequence[Path]) -> dict[str, tuple[dict[str, Any], ModelArrays]]:
    """The FBR runs (``fbr/<label>/fbr_run.json`` with their outcomes and probe answers)."""
    import json

    found = {}
    for root in roots:
        for path in sorted(Path(root).rglob("fbr_run.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            answers = path.parent / ANSWERS_FILE.format(split=SPLIT)
            found.setdefault(record["label"], (record, ModelArrays(
                f"FBR:{record['label']}", load(path.parent / "outcomes.npz"),
                load(answers) if answers.is_file() else None)))  # fmt: skip
    return found


def _same_probes(models: Sequence[ModelArrays]) -> None:
    first = models[0].outcomes
    for model in models[1:]:
        o = model.outcomes
        for key in ("probe_sample_ids", "probe_findings", "probe_truth"):
            if not np.array_equal(o[key], first[key]):
                raise ValueError(f"{model.label} holds other validation probes than "
                                 f"{models[0].label}")  # fmt: skip


def fold_plan(models: Sequence[ModelArrays], order: Sequence[str], k: int, minimum: int,
              seed: int) -> tuple[FoldPlan, list[dict[str, Any]]]:  # fmt: skip
    """The folds (from the labels of the shared probes) and their counts per finding."""
    _same_probes(models)
    o = models[0].outcomes
    ids, findings, truth = o["probe_sample_ids"], o["probe_findings"], o["probe_truth"]
    plan = assign_folds(ids, findings, truth.astype(bool), order, k=k, min_per_class=minimum,
                        seed=seed)  # fmt: skip
    folds = plan.folds(ids)
    rows = []
    for fold in range(plan.k):
        for f in order:
            r = (findings == f) & (folds == fold)
            rows.append({"fold": fold, "finding": f, "positive_images": int(truth[r].sum()),
                         "negative_images": int((~truth[r].astype(bool)).sum())})  # fmt: skip
    return plan, rows


def run_diagnostic(roots: Sequence[Path], out: Path, order: Sequence[str],
                   known: Sequence[str]) -> int:  # fmt: skip
    """Section 2 of the decision file: ``diagnostic.csv`` and ``folds.csv``."""
    from medcompress.compression.fbr import FBRSettings

    settings = FBRSettings.load()
    models = find_registered(roots, known)
    if not models:
        raise ValueError(f"no validation outcomes of E2 or a B2 pruning arm under {roots}")
    listed = [models[k] for k in DIAGNOSTIC_ARMS if k in models]
    plan, folds = fold_plan(listed, order, settings.folds, settings.fold_min_per_class,
                            settings.seed)  # fmt: skip
    rows = [r for m in listed for r in diagnostic_rows(m.label, m.outcomes, m.answers, plan, order)]
    _write_csv(rows, out / "diagnostic.csv")
    _write_csv(folds, out / "folds.csv")
    missing = [k for k in DIAGNOSTIC_ARMS if k not in models]
    no_answers = [m.label for m in listed if m.answers is None]
    lines = ["# FBR diagnostic (validation, seed 0)", "",
             "Exploratory (pre-registration section 10, amendment of 2026-11-22).", "",
             f"- arms: {', '.join(m.label for m in listed)}",
             f"- not found: {', '.join(missing) or 'none'}",
             f"- probe answers not available (decoded-answer and rank columns empty): "
             f"{', '.join(no_answers) or 'none'}",
             f"- folds: {plan.k} (image-grouped, finding-stratified, seed {settings.seed}); "
             f"findings without an operating point: {', '.join(plan.short) or 'none'}"]  # fmt: skip
    (out / "DIAGNOSTIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def run_pilot(roots: Sequence[Path], out: Path, order: Sequence[str],
              known: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """Sections 5-8 of the decision file: every FBR run against its arm's registered recovery,
    the gate, the sampling-only comparison."""
    from medcompress.compression.fbr import FBRSettings

    settings = FBRSettings.load()
    registered = find_registered(roots, known)
    runs = find_fbr(roots)
    if not runs:
        raise ValueError(f"no FBR run (fbr_run.json) under {roots}")
    pairs = []
    for label, (record, model) in sorted(runs.items()):
        key = f"{record['arm']}@B2"
        if key not in registered:
            raise ValueError(f"{label}: the registered recovery's validation outcomes of {key} "
                             "are not attached")  # fmt: skip
        pairs.append((label, record, model, registered[key]))
    everything = [m for _l, _r, m, _reg in pairs] + [reg for *_x, reg in pairs]
    plan, folds = fold_plan(everything, order, settings.folds, settings.fold_min_per_class,
                            settings.seed)  # fmt: skip
    gate_rows, report, audits, verdict = [], [], [], {}
    quantities = {}
    for label, record, model, reg in pairs:
        q_fbr, q_reg = (
            gate_quantities(model, plan, order, RARE),
            gate_quantities(reg, plan, order, RARE),
        )
        quantities[label] = q_fbr
        collapsed = record["arm"] in COLLAPSED
        items = gate(q_fbr, q_reg, collapsed=collapsed, rare=RARE)
        intervals = paired_bootstrap(
            model,
            reg,
            plan,
            order,
            RARE,
            resamples=settings.bootstrap_resamples,
            seed=settings.seed,
        )
        for name in q_fbr:
            gate_rows.append(
                {
                    "run": label,
                    "arm": record["arm"],
                    "control": record["control"],
                    "quantity": name,
                    "fbr": q_fbr[name],
                    "registered": q_reg[name],
                    "difference": q_fbr[name] - q_reg[name],
                    "ci_low": intervals[name][0],
                    "ci_high": intervals[name][1],
                }
            )
        verdict[label] = {"arm": record["arm"], "control": record["control"],
                          "collapsed": collapsed, **items, "pass": all(items.values())}  # fmt: skip
        report += report_rows(model, plan, order)
        if not any(r["model"] == reg.label for r in report):
            report += report_rows(reg, plan, order)
        for m in (model, reg):
            audits.append({"model": m.label, **mask_audit(m, plan, order)})
    control = next((lb for lb, (r, _m) in runs.items() if r["control"]), None)
    fbr_of_control = next((lb for lb, (r, _m) in runs.items()
                           if not r["control"] and control
                           and r["arm"] == runs[control][0]["arm"]), None)  # fmt: skip
    simpler = None
    if control and fbr_of_control:
        diff = paired_bootstrap(
            runs[fbr_of_control][1],
            runs[control][1],
            plan,
            order,
            RARE,
            resamples=settings.bootstrap_resamples,
            seed=settings.seed,
        )
        low, high = diff["R"]
        simpler = {"fbr": fbr_of_control, "control": control, "ci_low": low, "ci_high": high,
                   "indistinguishable": bool(low <= 0.0 <= high)}  # fmt: skip
    needed = [lb for lb, v in verdict.items() if not v["control"]
              and v["arm"] in ("E5:magnitude", "E5:activation")]  # fmt: skip
    passed = len(needed) == 2 and all(verdict[lb]["pass"] for lb in needed)
    decided = "yes" if passed else "no" if len(needed) == 2 else "not decidable yet"
    _write_csv(gate_rows, out / "pilot_gate.csv")
    _write_csv([{"run": k, **v} for k, v in verdict.items()], out / "pilot_verdict.csv")
    _write_csv(report, out / "pilot_report.csv")
    _write_csv(audits, out / "pilot_mask_audit.csv")
    _write_csv(folds, out / "folds.csv")
    lines = [
        "# FBR pilot (validation, B2, seed 0)",
        "",
        "Exploratory (pre-registration section 10, amendment of 2026-11-22). Gate in "
        "percentage points against each arm's registered recovery; cross-fitted operating "
        "points; a failed gate is a result.",
        "",
        f"- runs: {', '.join(sorted(runs))}",
        f"- folds: {plan.k}; findings without an operating point: "
        f"{', '.join(plan.short) or 'none'}",
        f"- gate passed (E5 magnitude and E5 activation both pass): {decided}",
    ]
    if simpler is not None:
        reading = ("indistinguishable: credit the control" if simpler["indistinguishable"]
                   else "distinguishable")  # fmt: skip
        lines.append(
            f"- sampling-only against FBR on {runs[control][0]['arm']}: delta R interval "
            f"[{simpler['ci_low'] * 100:.1f}, {simpler['ci_high'] * 100:.1f}] points; "
            f"{reading}"
        )
    (out / "PILOT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"passed": passed, "verdict": verdict, "simpler": simpler}
