"""Exploratory analyses A and B (pre-registration section 10, amendment of 2026-11-20; CPU).

Both read the per-probe outcomes that stage 06 saves (``outcomes.npz``, ``outcomes_test.npz``,
``outcomes_test_unfiltered.npz``; the mask audit's margins in them, or in ``audit_<split>.npz``
from ``scripts/save_outcomes.py --audit-only``). Everything here is exploratory: outside H1-H4,
no verdicts, no multiplicity correction. Specification:
``docs/decisions/exploratory_additions_2026-11-20.md``.

* **A. Silent collapse and re-calibration cost:** plain answers against the operating point
  (:func:`plain_vs_operating`), the re-calibration curve (:func:`recalibration_curve`) and the
  lesion-mask audit at the operating threshold (:func:`mask_audit_at_threshold`).
* **B. What each method forgets:** flips against E2 (:func:`flip_masks`, :func:`flip_table`),
  their overlap (:func:`jaccard_table`) and the compression-identified exemplars
  (:func:`exemplar_mask`).

The *plain answer* is the one-forward-pass decision, "yes" when log p(yes) >= log p(no); the
*operating point* is the pre-registered per-finding threshold (specificity 0.95 on validation).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OUTCOME_FILES: Mapping[str, str] = {
    "val": "outcomes.npz",
    "test": "outcomes_test.npz",
    "test_unfiltered": "outcomes_test_unfiltered.npz",
}
TEST_SPLITS: tuple[str, ...] = ("test", "test_unfiltered")
"""The external test (PolypGen) joins once its evaluation writes outcomes."""
AUDIT_FILE = "audit_{split}.npz"
SUBSET_SIZES: tuple[int, ...] = (8, 16, 32, 64, 128)
DRAWS = 1000
RESAMPLES = 10_000
SEED = 0
SPECIFICITY = 0.95
WITHIN = 0.02
"""The smallest n whose median macro balanced accuracy is within this of the full threshold's."""

FAMILIES: Mapping[str, tuple[str, ...]] = {
    "pruning": ("E5", "E5R", "E6", "E7", "E7U", "A1", "A2", "A3", "A4", "A5", "A6"),
    "quantization": ("E3", "E4"),
    "pruning_quantization": ("E8",),
    "distillation": ("E1",),
    "native_small": ("E10", "E10Q"),
}
"""Families of the overlap analysis; E9 and E0 are compared with E2 but belong to none."""
COMPRESSED_FAMILIES: tuple[str, ...] = ("pruning", "quantization", "pruning_quantization")
"""The compressed student arms that define the exemplars (not E1, E9, E10, E10Q)."""
BUDGET_PREFERENCE: tuple[str, ...] = ("B2", "nobudget", "B1", "B3")
"""Which run of an arm stands for it among the exemplar arms: B2, else no budget (E3, E4),
else B1, then B3 (fixed before any result)."""
_SEED_DIR = re.compile(r"^seed(\d+)$")


def family_of(experiment: str) -> str | None:
    """The family of :data:`FAMILIES` an experiment belongs to (None for E0, E2, E9)."""
    for name, members in FAMILIES.items():
        if experiment in members:
            return name
    return None


# ----------------------------------------------------------------------- discovery


@dataclass
class ArmRun:
    """The outcome files of one run (an arm at one budget and seed)."""

    experiment: str
    variant: str | None
    budget: str  # "nobudget" or B1-B3
    seed: int
    files: dict[str, Path] = field(default_factory=dict)  # split -> outcomes file
    audits: dict[str, Path] = field(default_factory=dict)  # split -> audit_<split>.npz

    @property
    def arm(self) -> str:
        return self.experiment + (f":{self.variant}" if self.variant else "")

    @property
    def label(self) -> str:
        budget = "" if self.budget == "nobudget" else f"@{self.budget}"
        return f"{self.arm}{budget} seed{self.seed}"


def parse_run(path: Path, known: Sequence[str]) -> tuple[str, str | None, str, int] | None:
    """``(experiment, variant, budget, seed)`` from ``.../<ID>[/<variant>]/<budget>/seed<k>/
    <file>``, or None when the path is not a run folder of a known experiment."""
    parts = path.parts
    if len(parts) < 4:
        return None
    match = _SEED_DIR.match(parts[-2])
    if match is None:
        return None
    budget = parts[-3]
    if budget != "nobudget" and not re.fullmatch(r"B\d", budget):
        return None
    for start in range(len(parts) - 4, -1, -1):
        if parts[start] in known:
            variant = "/".join(parts[start + 1 : -3]) or None
            return parts[start], variant, budget, int(match.group(1))
    return None


def discover(
    roots: Sequence[Path | str], known: Sequence[str]
) -> tuple[dict[str, ArmRun], list[str]]:
    """Every run with outcome files under ``roots`` (the attached outcome hand-offs or an
    output root), and notes (duplicates: the first root wins)."""
    runs: dict[str, ArmRun] = {}
    notes: list[str] = []
    names = {name: split for split, name in OUTCOME_FILES.items()}
    audits = {AUDIT_FILE.format(split=split): split for split in OUTCOME_FILES}
    for root in roots:
        for path in sorted(Path(root).rglob("*.npz")):
            split = names.get(path.name) or audits.get(path.name)
            if split is None:
                continue
            parsed = parse_run(path, known)
            if parsed is None:
                continue
            experiment, variant, budget, seed = parsed
            run = ArmRun(experiment, variant, budget, seed)
            run = runs.setdefault(run.label, run)
            target = run.audits if path.name in audits else run.files
            if split in target and target[split] != path:
                notes.append(f"{run.label} {path.name}: also at {path}; kept {target[split]}")
                continue
            target[split] = path
    return runs, notes


def load(path: Path) -> dict[str, np.ndarray]:
    """Every array of an outcomes file, read into memory."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# ------------------------------------------------------------------ A1: plain vs op


def _rates(truth: np.ndarray, yes: np.ndarray) -> dict[str, float]:
    pos, neg = truth, ~truth
    sens = float(yes[pos].mean()) if pos.any() else math.nan
    spec = float((~yes[neg]).mean()) if neg.any() else math.nan
    ba = (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else math.nan
    return {
        "yes_rate": float(yes.mean()) if yes.size else math.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": ba,
    }


def per_finding_rates(
    findings: np.ndarray, truth: np.ndarray, yes: np.ndarray, order: Sequence[str]
) -> dict[str, dict[str, float]]:
    """``{finding or "macro": {yes_rate, sensitivity, specificity, balanced_accuracy}}``; macro
    is the mean over the findings (balanced accuracy over those with both classes)."""
    out = {
        f: _rates(truth[findings == f], yes[findings == f]) for f in order if (findings == f).any()
    }
    macro = {}
    for key in ("yes_rate", "sensitivity", "specificity", "balanced_accuracy"):
        values = [v[key] for v in out.values() if not math.isnan(v[key])]
        macro[key] = float(np.mean(values)) if values else math.nan
    out["macro"] = macro
    return out


def ece(margin: np.ndarray, truth: np.ndarray, yes: np.ndarray, n_bins: int) -> float:
    """The pre-registered ECE: confidence of the plain answer from the two-answer probability."""
    from medcompress.evaluation.calibration import expected_calibration_error

    p_yes = 1.0 / (1.0 + np.exp(-np.clip(margin, -50.0, 50.0)))
    confidence = np.maximum(p_yes, 1.0 - p_yes)
    return float(expected_calibration_error(confidence, yes == truth, n_bins=n_bins))


def plain_vs_operating(
    arrays: Mapping[str, np.ndarray], order: Sequence[str], n_bins: int
) -> list[dict[str, object]]:
    """Rows ``finding, metric, plain, operating_point, gap`` (gap = operating point minus
    plain), plus ``ece`` per finding and macro (the whole split for macro)."""
    findings, truth = arrays["probe_findings"], arrays["probe_truth"].astype(bool)
    plain = per_finding_rates(findings, truth, arrays["probe_yes"].astype(bool), order)
    op = per_finding_rates(findings, truth, arrays["probe_yes_op"].astype(bool), order)
    rows: list[dict[str, object]] = []
    for finding in plain:
        for metric in ("yes_rate", "sensitivity", "specificity", "balanced_accuracy"):
            a, b = plain[finding][metric], op[finding][metric]
            rows.append(
                {
                    "finding": finding,
                    "metric": metric,
                    "plain": a,
                    "operating_point": b,
                    "gap": b - a,
                }
            )
        keep = slice(None) if finding == "macro" else findings == finding
        rows.append(
            {
                "finding": finding,
                "metric": "ece",
                "plain": ece(
                    arrays["probe_margin"][keep],
                    truth[keep],
                    arrays["probe_yes"].astype(bool)[keep],
                    n_bins,
                ),
                "operating_point": math.nan,
                "gap": math.nan,
            }
        )
    return rows


# ------------------------------------------------------------- A2: re-calibration


def fit_thresholds(
    findings: np.ndarray,
    truth: np.ndarray,
    margin: np.ndarray,
    order: Sequence[str],
    specificity: float = SPECIFICITY,
) -> dict[str, float]:
    """The pre-registered rule (``probe_suites.fit_thresholds``) on arrays: the k-th smallest
    negative margin, k = ceil(specificity x negatives); NaN without negatives."""
    out = {}
    for finding in order:
        negatives = np.sort(margin[(findings == finding) & ~truth])
        if negatives.size == 0:
            out[finding] = math.nan
            continue
        k = max(1, math.ceil(specificity * negatives.size - 1e-9))
        out[finding] = float(negatives[k - 1])
    return out


def apply_thresholds(
    findings: np.ndarray, margin: np.ndarray, plain: np.ndarray, thresholds: Mapping[str, float]
) -> np.ndarray:
    """``probe_suites.operating_point_answers`` on arrays: margin above the threshold; a finding
    without a threshold keeps its plain answers."""
    limits = np.array([thresholds.get(f, math.nan) for f in findings], dtype=float)
    return np.where(np.isnan(limits), plain, margin > np.nan_to_num(limits))


def stratified_draw(
    strata: Mapping[tuple[str, ...], np.ndarray], n: int, rng: np.random.Generator
) -> np.ndarray:
    """``n`` image indices without replacement, each stratum (the set of findings present in
    the image) getting its share of ``n`` by largest remainder."""
    keys = sorted(strata)
    sizes = np.array([len(strata[k]) for k in keys], dtype=float)
    exact = n * sizes / sizes.sum()
    take = np.floor(exact).astype(int)
    order = sorted(range(len(keys)), key=lambda i: (-(exact[i] - take[i]), i))
    for i in order[: n - int(take.sum())]:
        take[i] += 1
    take = np.minimum(take, sizes.astype(int))
    chosen = [
        rng.choice(strata[k], size=t, replace=False) for k, t in zip(keys, take, strict=True) if t
    ]
    return np.concatenate(chosen) if chosen else np.array([], dtype=int)


def recalibration_curve(
    val: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
    order: Sequence[str],
    *,
    sizes: Sequence[int] = SUBSET_SIZES,
    draws: int = DRAWS,
    seed: int = SEED,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """The re-calibration curve: per n, the operating threshold refitted on n validation images
    (stratified by finding presence; ``draws`` draws, the same images for every arm at a given
    n) and applied to the test probes. Rows ``n, finding, metric, median, low, high`` (2.5 and
    97.5 %); n = 0 is the plain answer and n = "all" the full validation threshold (one value).
    Also ``{"full_macro_ba", "smallest_n_within"}``."""
    vf, vt, vm = val["probe_findings"], val["probe_truth"].astype(bool), val["probe_margin"]
    vids = val["probe_sample_ids"]
    tf, tt, tm = test["probe_findings"], test["probe_truth"].astype(bool), test["probe_margin"]
    tplain = test["probe_yes"].astype(bool)
    images = np.array(sorted(set(vids.tolist())))
    index = {s: i for i, s in enumerate(images)}
    probe_image = np.array([index[s] for s in vids])
    present: dict[int, set[str]] = {i: set() for i in range(len(images))}
    for i, f, t in zip(probe_image, vf, vt, strict=True):
        if t:
            present[int(i)].add(str(f))
    strata: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(images)):
        strata.setdefault(tuple(sorted(present[i])), []).append(i)
    strata_arr = {k: np.array(v) for k, v in strata.items()}
    metrics = ("sensitivity", "specificity", "balanced_accuracy")

    def scores(yes: np.ndarray) -> dict[str, dict[str, float]]:
        return per_finding_rates(tf, tt, yes, order)

    rows: list[dict[str, object]] = []

    def emit(n: object, values: list[dict[str, dict[str, float]]]) -> None:
        for finding in values[0]:
            for metric in metrics:
                series = np.array([v[finding][metric] for v in values], dtype=float)
                series = series[~np.isnan(series)]
                if series.size == 0:
                    continue
                rows.append(
                    {
                        "n": n,
                        "finding": finding,
                        "metric": metric,
                        "median": float(np.median(series)),
                        "low": float(np.percentile(series, 2.5)),
                        "high": float(np.percentile(series, 97.5)),
                        "draws": int(series.size),
                    }
                )

    emit(0, [scores(tplain)])
    full = fit_thresholds(vf, vt, vm, order)
    full_scores = scores(apply_thresholds(tf, tm, tplain, full))
    medians: dict[int, float] = {}
    for n in sizes:
        if n >= len(images):
            continue
        rng = np.random.default_rng([seed, n])
        values = []
        for _ in range(draws):
            chosen = np.zeros(len(images), dtype=bool)
            chosen[stratified_draw(strata_arr, n, rng)] = True
            keep = chosen[probe_image]
            fitted = fit_thresholds(vf[keep], vt[keep], vm[keep], order)
            values.append(scores(apply_thresholds(tf, tm, tplain, fitted)))
        emit(n, values)
        macro = np.array([v["macro"]["balanced_accuracy"] for v in values], dtype=float)
        medians[n] = float(np.nanmedian(macro))
    emit("all", [full_scores])
    target = full_scores["macro"]["balanced_accuracy"]
    within = [n for n in sorted(medians) if abs(medians[n] - target) <= WITHIN]
    return rows, {
        "full_macro_ba": target,
        "smallest_n_within": within[0] if within else None,
        "thresholds": full,
    }


# ------------------------------------------------- A3: the mask audit at the threshold


def thresholds_from_outcomes(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """The operating thresholds the evaluation used, from ``probe_op_threshold``."""
    out: dict[str, float] = {}
    for finding, value in zip(arrays["probe_findings"], arrays["probe_op_threshold"], strict=True):
        out.setdefault(str(finding), float(value))
    return out


def mask_audit_at_threshold(
    audit: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    """The flip-rate difference of primary endpoint 2 (lesion masked minus random region
    masked) on operating-point decisions, next to the plain-answer version recomputed from the
    saved flips. Needs ``audit_original_margin``, ``audit_lesion_margin``, ``audit_random_margin``.
    Images are resampled together (paired bootstrap)."""
    from medcompress.evaluation.statistics import paired_bootstrap_ci

    findings = audit["audit_findings"]
    limit = np.array([thresholds.get(str(f), math.nan) for f in findings], dtype=float)
    if np.isnan(limit).any():
        raise ValueError("no operating threshold for an audited finding")
    base = audit["audit_original_margin"] > limit
    lesion = (audit["audit_lesion_margin"] > limit) != base
    random_ = (audit["audit_random_margin"] > limit) != base
    images = audit["audit_sample_ids"]
    effect, low, high = paired_bootstrap_ci(
        np.mean,
        lesion.astype(float),
        random_.astype(float),
        n_resamples=resamples,
        seed=seed,
        clusters=images,
    )
    plain_l = audit["audit_lesion_flip"].astype(float)
    plain_r = audit["audit_random_flip"].astype(float)
    p_effect, p_low, p_high = paired_bootstrap_ci(
        np.mean, plain_l, plain_r, n_resamples=resamples, seed=seed, clusters=images
    )
    return {
        "n_probes": float(len(images)),
        "op_original_yes_rate": float(base.mean()),
        "op_lesion_flip_rate": float(lesion.mean()),
        "op_random_flip_rate": float(random_.mean()),
        "op_lesion_mask_effect": effect,
        "op_lesion_mask_effect_ci_low": low,
        "op_lesion_mask_effect_ci_high": high,
        "plain_original_yes_rate": float(audit["audit_original_yes"].astype(bool).mean()),
        "plain_lesion_mask_effect": p_effect,
        "plain_lesion_mask_effect_ci_low": p_low,
        "plain_lesion_mask_effect_ci_high": p_high,
    }


# --------------------------------------------------------------------- B: flips


def probe_keys(arrays: Mapping[str, np.ndarray]) -> list[tuple[str, str]]:
    """``(sample id, finding)`` of every probe, the key that aligns two arms' outcomes."""
    return list(
        zip(arrays["probe_sample_ids"].tolist(), arrays["probe_findings"].tolist(), strict=True)
    )


@dataclass(frozen=True)
class FlipMasks:
    """Per probe of the reference (E2), aligned with an arm: forgotten and gained."""

    forgotten: np.ndarray
    gained: np.ndarray
    ref_correct: np.ndarray
    present: np.ndarray  # the arm scored this probe


def flip_masks(
    ref: Mapping[str, np.ndarray], arm: Mapping[str, np.ndarray], decision: str
) -> FlipMasks:
    """Forgotten (E2 right, the arm wrong) and gained (E2 wrong, the arm right) on the
    reference's probes; ``decision`` is "op" (``probe_yes_op``) or "plain" (``probe_yes``)."""
    key = "probe_yes_op" if decision == "op" else "probe_yes"
    truth = ref["probe_truth"].astype(bool)
    ref_correct = ref[key].astype(bool) == truth
    where = {k: i for i, k in enumerate(probe_keys(arm))}
    rows = np.array([where.get(k, -1) for k in probe_keys(ref)])
    present = rows >= 0
    arm_yes = np.zeros(len(rows), dtype=bool)
    arm_yes[present] = arm[key].astype(bool)[rows[present]]
    arm_correct = (arm_yes == truth) & present
    return FlipMasks(
        ref_correct & ~arm_correct & present, ~ref_correct & arm_correct, ref_correct, present
    )


def probe_groups(
    ref: Mapping[str, np.ndarray],
    order: Sequence[str],
    size_thresholds: Sequence[float],
    mask_findings: Sequence[str],
) -> dict[str, np.ndarray]:
    """The breakdowns of B: all, each finding, small and other polyps (positive, mask-annotated
    probes of the mask findings, the pre-registered size cut points), mask present or not."""
    from medcompress.data.masks import assign_size_stratum

    findings = ref["probe_findings"]
    truth = ref["probe_truth"].astype(bool)
    has_mask = ref["probe_has_mask"].astype(bool)
    area = ref["probe_area"].astype(float)
    groups = {"all": np.ones(len(findings), dtype=bool)}
    for finding in order:
        groups[f"finding:{finding}"] = findings == finding
    lesion = truth & has_mask & np.isfinite(area) & np.isin(findings, list(mask_findings))
    stratum = np.array(
        [
            assign_size_stratum(float(a), size_thresholds) if ok else -1
            for a, ok in zip(area, lesion, strict=True)
        ]
    )
    groups["size:small_polyp"] = lesion & (stratum == 0)
    groups["size:other_polyp"] = lesion & (stratum > 0)
    groups["mask:yes"] = has_mask
    groups["mask:no"] = ~has_mask
    return groups


def bootstrap_weights(
    images: np.ndarray, resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(probe -> image index, weights)``: resampled image counts, ``resamples x images``."""
    unique, index = np.unique(images, return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=resamples)
    return index, weights


def _per_image(values: np.ndarray, index: np.ndarray, n_images: int) -> np.ndarray:
    return np.bincount(index, weights=values.astype(float), minlength=n_images)


def count_and_rate(
    numerator: np.ndarray, denominator: np.ndarray, index: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Count and rate with image-bootstrap 95 % intervals (the same weights for every quantity
    of a split, so they are comparable)."""
    n_images = weights.shape[1]
    num_i = _per_image(numerator, index, n_images)
    den_i = _per_image(denominator, index, n_images)
    counts = weights @ num_i
    dens = weights @ den_i
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(dens > 0, counts / np.maximum(dens, 1), np.nan)
    den = float(denominator.sum())
    return {
        "count": float(numerator.sum()),
        "count_ci_low": float(np.percentile(counts, 2.5)),
        "count_ci_high": float(np.percentile(counts, 97.5)),
        "denominator": den,
        "rate": float(numerator.sum() / den) if den else math.nan,
        "rate_ci_low": float(np.nanpercentile(rates, 2.5))
        if np.isfinite(rates).any()
        else math.nan,
        "rate_ci_high": float(np.nanpercentile(rates, 97.5))
        if np.isfinite(rates).any()
        else math.nan,
    }


def flip_table(
    masks: FlipMasks, groups: Mapping[str, np.ndarray], index: np.ndarray, weights: np.ndarray
) -> list[dict[str, object]]:
    """Rows ``group, kind, count (CI), rate (CI)``: forgotten over the probes E2 got right,
    gained over the probes E2 got wrong, within each group."""
    rows = []
    for name, group in groups.items():
        right = group & masks.ref_correct & masks.present
        wrong = group & ~masks.ref_correct & masks.present
        rows.append(
            {
                "group": name,
                "kind": "forgotten",
                **count_and_rate(masks.forgotten & group, right, index, weights),
            }
        )
        rows.append(
            {
                "group": name,
                "kind": "gained",
                **count_and_rate(masks.gained & group, wrong, index, weights),
            }
        )
    return rows


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """``|a & b| / |a | b|`` of two boolean masks (NaN when both are empty)."""
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else math.nan


def jaccard_table(
    sets: Mapping[str, np.ndarray], index: np.ndarray, weights: np.ndarray
) -> list[dict[str, object]]:
    """The Jaccard overlap of every pair of forgotten sets, with image-bootstrap intervals."""
    names = list(sets)
    if len(names) < 2:
        return []
    matrix = np.stack([sets[n].astype(float) for n in names], axis=1)  # probes x sets
    n_images = weights.shape[1]
    per_image = np.zeros((n_images, len(names), len(names)))
    np.add.at(per_image, index, matrix[:, :, None] * matrix[:, None, :])
    resampled = np.tensordot(weights, per_image, axes=(1, 0))  # resamples x sets x sets
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = resampled[:, i, j]
            union = resampled[:, i, i] + resampled[:, j, j] - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.where(union > 0, inter / np.maximum(union, 1e-12), np.nan)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "jaccard": jaccard(sets[names[i]], sets[names[j]]),
                    "ci_low": float(np.percentile(finite, 2.5)) if finite.size else math.nan,
                    "ci_high": float(np.percentile(finite, 97.5)) if finite.size else math.nan,
                    "n_a": int(sets[names[i]].sum()),
                    "n_b": int(sets[names[j]].sum()),
                }
            )
    return rows


def exemplar_mask(forgotten: Mapping[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """The compression-identified exemplars: probes forgotten by at least half (rounded up) of
    the given compressed arms. Returns ``(mask, arms needed)``."""
    if not forgotten:
        raise ValueError("no compressed arms")
    counts = np.sum(np.stack([v.astype(int) for v in forgotten.values()]), axis=0)
    needed = math.ceil(len(forgotten) / 2)
    return counts >= needed, needed


def exemplar_arms(runs: Sequence[ArmRun], split: str) -> list[ArmRun]:
    """One run per compressed student arm, by :data:`BUDGET_PREFERENCE`, seed 0, among the runs
    with outcomes of ``split``."""
    chosen: dict[str, ArmRun] = {}
    for run in runs:
        if run.seed != 0 or split not in run.files:
            continue
        if family_of(run.experiment) not in COMPRESSED_FAMILIES:
            continue
        old = chosen.get(run.arm)
        if old is None or _budget_rank(run.budget) < _budget_rank(old.budget):
            chosen[run.arm] = run
    return [chosen[k] for k in sorted(chosen)]


def _budget_rank(budget: str) -> int:
    return (
        BUDGET_PREFERENCE.index(budget) if budget in BUDGET_PREFERENCE else len(BUDGET_PREFERENCE)
    )


# ------------------------------------------------------------------------ driver


def audit_arrays(run: ArmRun, split: str) -> dict[str, np.ndarray] | None:
    """The mask audit of a split with its margins: from the outcomes file (evaluations from
    2026-11-20 on) or from ``audit_<split>.npz`` (a rerun); None when neither has them."""
    for path in (run.files.get(split), run.audits.get(split)):
        if path is None:
            continue
        arrays = load(path)
        if "audit_original_margin" in arrays:
            return arrays
    return None


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
            )


def run_all(
    roots: Sequence[Path | str],
    out: Path | str,
    *,
    order: Sequence[str],
    size_thresholds: Sequence[float],
    mask_findings: Sequence[str],
    n_bins: int,
    known: Sequence[str],
    draws: int = DRAWS,
    resamples: int = RESAMPLES,
) -> dict[str, int]:
    """Run A and B on every run found under ``roots`` and write ``out`` (CSV files and
    ``SUMMARY.md``). Returns the number of rows of each file."""
    out = Path(out)
    runs, notes = discover(roots, known)
    ordered = [runs[k] for k in sorted(runs)]
    written: dict[str, list[dict[str, object]]] = {
        k: []
        for k in (
            "inventory",
            "plain_vs_operating",
            "recalibration",
            "recalibration_summary",
            "mask_audit_operating",
            "flips",
            "jaccard_arms",
            "jaccard_families",
            "exemplars",
        )
    }
    for run in ordered:
        written["inventory"].append(
            {
                "run": run.label,
                "family": family_of(run.experiment) or "",
                **{split: int(split in run.files) for split in OUTCOME_FILES},
                **{
                    f"audit_margins_{split}": int(audit_arrays(run, split) is not None)
                    for split in OUTCOME_FILES
                },
            }
        )
    # A1, A2, A3
    for run in ordered:
        val = load(run.files["val"]) if "val" in run.files else None
        for split in TEST_SPLITS:
            if split not in run.files:
                continue
            test = load(run.files[split])
            for row in plain_vs_operating(test, order, n_bins):
                written["plain_vs_operating"].append({"run": run.label, "split": split, **row})
            if val is not None:
                rows, summary = recalibration_curve(val, test, order, draws=draws)
                written["recalibration"] += [{"run": run.label, "split": split, **r} for r in rows]
                written["recalibration_summary"].append(
                    {
                        "run": run.label,
                        "split": split,
                        "full_macro_ba": summary["full_macro_ba"],
                        "smallest_n_within_0.02": summary["smallest_n_within"] or "",
                    }
                )
        for split in OUTCOME_FILES:
            audit = audit_arrays(run, split)
            source = (
                val if val is not None else (load(run.files[split]) if split in run.files else None)
            )
            if audit is None or source is None:
                continue
            result = mask_audit_at_threshold(
                audit, thresholds_from_outcomes(source), resamples=resamples
            )
            written["mask_audit_operating"].append({"run": run.label, "split": split, **result})
    # B
    for split in TEST_SPLITS:
        ref_run = next(
            (
                r
                for r in ordered
                if r.arm == "E2" and r.seed == 0 and r.budget == "nobudget" and split in r.files
            ),
            None,
        )
        if ref_run is None:
            notes.append(f"B on {split}: E2 seed 0 has no {OUTCOME_FILES[split]} (RC-R1): skipped")
            continue
        ref = load(ref_run.files[split])
        groups = probe_groups(ref, order, size_thresholds, mask_findings)
        index, weights = bootstrap_weights(ref["probe_sample_ids"], resamples, SEED)
        forgotten: dict[str, dict[str, np.ndarray]] = {"op": {}, "plain": {}}
        others = [r for r in ordered if r is not ref_run and split in r.files]
        for run in others:
            arm = load(run.files[split])
            for decision in ("op", "plain"):
                masks = flip_masks(ref, arm, decision)
                forgotten[decision][run.label] = masks.forgotten
                for row in flip_table(masks, groups, index, weights):
                    written["flips"].append(
                        {"run": run.label, "split": split, "decision": decision, **row}
                    )
        for decision in ("op", "plain"):
            for row in jaccard_table(forgotten[decision], index, weights):
                written["jaccard_arms"].append({"split": split, "decision": decision, **row})
            families: dict[str, np.ndarray] = {}
            for run in others:
                family = family_of(run.experiment)
                if family is None or run.seed != 0:
                    continue
                current = forgotten[decision][run.label]
                families[family] = families.get(family, np.zeros_like(current)) | current
            for row in jaccard_table(families, index, weights):
                written["jaccard_families"].append({"split": split, "decision": decision, **row})
            arms = exemplar_arms(others, split)
            if not arms:
                continue
            mask, needed = exemplar_mask({r.label: forgotten[decision][r.label] for r in arms})
            base = {
                "split": split,
                "decision": decision,
                "arms": len(arms),
                "needed": needed,
                "arm_list": " ".join(r.label for r in arms),
            }
            for name, group in groups.items():
                written["exemplars"].append(
                    {**base, "group": name, **count_and_rate(mask & group, group, index, weights)}
                )
            lesion = groups["size:small_polyp"] | groups["size:other_polyp"]
            written["exemplars"].append(
                {
                    **base,
                    "group": "small_share_among_polyp_lesions",
                    **count_and_rate(
                        mask & groups["size:small_polyp"], mask & lesion, index, weights
                    ),
                }
            )
            area = ref["probe_area"].astype(float)
            finite = np.isfinite(area)
            written["exemplars"].append(
                {
                    **base,
                    "group": "mask_area_median",
                    "exemplars": float(np.median(area[mask & finite]))
                    if (mask & finite).any()
                    else math.nan,
                    "all_probes": float(np.median(area[finite])) if finite.any() else math.nan,
                }
            )
    for name, rows in written.items():
        _write(out / f"{name}.csv", rows)
    lines = [
        "# Exploratory analyses A and B (amendment of 2026-11-20)",
        "",
        "Written by `scripts/exploratory_analysis.py` from the attached outcome files. "
        "Exploratory: outside H1-H4, no verdicts, no multiplicity correction.",
        "",
        "## Outcomes found",
        "",
        "| run | family | val | test | test_unfiltered | audit margins (val / test / unfiltered) |",
        "|---|---|---|---|---|---|",
    ]
    for row in written["inventory"]:
        lines.append(
            f"| {row['run']} | {row['family']} | {row['val']} | {row['test']} | "
            f"{row['test_unfiltered']} | {row['audit_margins_val']} / "
            f"{row['audit_margins_test']} / {row['audit_margins_test_unfiltered']} |"
        )
    lines += ["", "## Files", ""] + [
        f"- `{name}.csv`: {len(rows)} rows" for name, rows in written.items()
    ]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {name: len(rows) for name, rows in written.items()}
