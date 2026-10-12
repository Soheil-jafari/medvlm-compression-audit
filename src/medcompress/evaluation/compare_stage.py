"""Stage 09: paired comparisons on the test split, the evidence rows of H1-H4.

Stage 06 leaves the per-probe answers of every run in ``<output_root>/<run dir>/``
``outcomes_test.npz`` (never committed). This stage reads them for every arm and seed and
averages each probe's outcome over the arm's seeds (pre-registration section 6). It then draws
one sequence of image-level bootstrap resamples (seed 0) and computes, for every arm and every
resample, the headline accuracy, the small-finding recall, the rare-group recall and the ECE.
Every comparison below is a function of that one table, so all intervals share the resamples:

* H1, ``<arm>_vs_E2``: ``diff:drop_gap`` = relative drop of the small-finding recall minus
  relative drop of the headline accuracy (both against E2); ``diff:drop_gap_rare`` uses the
  rare-group recall instead;
* H2, ``best_pruning_vs_E3:w8`` at the budget whose target is nearest to the measured peak
  memory of E3 ``w8``: ``diff:small_finding_recall`` of the best generic pruning arm (highest
  validation small-finding recall at that budget) minus E3 ``w8``;
* H3, ``E7_vs_best`` at B2: ``diff:``, ``p:`` and ``p_holm:`` rows of the two primary
  endpoints against the comparator (highest validation small-finding recall among E5 and E6
  at B2); the lesion-mask effect is bootstrapped over the audited probes, by image;
* H4, ``accuracy_vs_clinical``: Kendall's tau between the order of the compressed arms by the
  headline accuracy and by small-finding recall, rare-group recall and ECE (negated, so that
  higher is better in both orders).

H2 and H3 also get McNemar's test per finding on each probe's majority answer over the seeds,
Holm-corrected across findings (``mcnemar_p:`` and ``mcnemar_p_holm:``). The rule-based
choices (comparator, matched budget) and the arms left out for missing seeds are written to
``results/comparisons/choices.json``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medcompress.config import ConfigError, RunContext
from medcompress.evaluation.comparisons import add_holm_rows, compare_arms, pick_best_arm
from medcompress.evaluation.statistics import (
    bootstrap_p_value,
    holm_correction,
    kendall_tau,
    mcnemar_test,
    percentile_ci,
    resample_indices,
)
from medcompress.registry import BUDGET_IDS, PRIMARY_BUDGET, REGISTRY

OUTCOMES_FILE = "outcomes_test.npz"
CHOICES_FILE = "choices.json"
COMPARISON_FILE = "comparison.csv"
REFERENCE = "E2"
COMPRESSED: tuple[str, ...] = ("E3", "E4", "E5", "E6", "E7", "E8")
"""The compressed students: every H1 arm and the arms ranked for H4."""
GENERIC_PRUNING: tuple[str, ...] = ("E5", "E6")
QUANT_W8 = "E3:w8"
OURS = "E7"
BOOTSTRAP_SEED = 0
HEADLINES: tuple[str, ...] = ("overall_accuracy", "macro_balanced_accuracy")
SMALL = "small_finding_recall"
RARE = "rare_recall"
ECE = "ece"
MASK_EFFECT = "lesion_mask_effect"
RARE_GROUP = "rare"
COMPARISON_SUITES: tuple[str, ...] = ("clinical", "small_findings", "rare_findings", "calibration")
HEAD, SMALL_COL, RARE_COL, ECE_COL = range(4)
"""Columns of the per-arm metric table."""


@dataclass(frozen=True)
class Arm:
    """One compared arm: an experiment label (``E5:taylor``) at a budget (``""`` if none)."""

    label: str
    budget: str = ""

    @property
    def name(self) -> str:
        """``E5:taylor@B2`` style name used in messages and ``choices.json``."""
        return f"{self.label}@{self.budget}" if self.budget else self.label

    def runs(self, seeds: Sequence[int]) -> list[RunContext]:
        """The run of every seed of this arm."""
        experiment, _, variant = self.label.partition(":")
        return [
            RunContext(experiment, seed=s, budget=self.budget or None, variant=variant or None)
            for s in seeds
        ]


def arms_of(experiment_id: str) -> list[Arm]:
    """Every (variant, budget) arm of a registry experiment."""
    spec = REGISTRY[experiment_id]
    labels = [f"{experiment_id}:{v}" for v in spec.variants] or [experiment_id]
    budgets = list(spec.budgets) or [""]
    return [Arm(label, budget) for label in labels for budget in budgets]


# ------------------------------------------------------------------ outcomes


@dataclass(frozen=True)
class ProbeTable:
    """What every arm shares: the probes, their images and labels, and the subsets.

    ``small`` marks the positive probes of the mask findings in the smallest lesion-size
    stratum (the small-finding recall); ``rare`` the positive probes of rare-group findings
    (the pooled rare-group recall). The audit arrays describe the lesion-mask audit probes.
    """

    keys: np.ndarray
    images: np.ndarray
    findings: tuple[str, ...]
    codes: np.ndarray
    truth: np.ndarray
    small: np.ndarray
    rare: np.ndarray
    audit_keys: np.ndarray
    audit_images: np.ndarray


@dataclass(frozen=True)
class ArmOutcomes:
    """One arm's outcomes, averaged over its seeds, aligned with the :class:`ProbeTable`."""

    arm: Arm
    seeds: tuple[int, ...]
    correct: np.ndarray
    confidence: np.ndarray
    majority: np.ndarray
    audit_effect: np.ndarray


def _keys(sample_ids: np.ndarray, findings: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(sample_ids.astype(str), "\t"), findings.astype(str))


def read_outcomes(path: Path) -> dict[str, np.ndarray]:
    """The arrays of one ``outcomes_test.npz`` (plain arrays only, no pickled objects)."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def build_table(
    arrays: Mapping[str, np.ndarray],
    findings: Sequence[str],
    thresholds: Sequence[float],
    mask_findings: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> ProbeTable:
    """The shared probe table from one run's arrays and the frozen definitions."""
    from medcompress.evaluation.probe_suites import (
        ProbeOutcomes,
        group_of_finding,
        small_positive_strata,
    )

    names = arrays["probe_findings"].astype(str)
    truth = arrays["probe_truth"].astype(bool)
    area = arrays["probe_area"].astype(float)
    has_mask = arrays.get("probe_has_mask", np.isfinite(area)).astype(bool)
    out = ProbeOutcomes(
        sample_ids=tuple(arrays["probe_sample_ids"].astype(str)),
        findings=tuple(names),
        truth=truth,
        yes=arrays["probe_yes"].astype(bool),
        margin=arrays["probe_margin"].astype(float),
        area=area,
        has_mask=has_mask,
    )
    keep, strata = small_positive_strata(out, thresholds, mask_findings)
    small = np.zeros(len(truth), dtype=bool)
    small[np.flatnonzero(keep)[strata == 0]] = True
    group = group_of_finding(groups)
    rare = truth & np.array([group.get(f) == RARE_GROUP for f in names])
    lookup = {f: i for i, f in enumerate(findings)}
    unknown = sorted(set(names) - set(lookup))
    if unknown:
        raise ValueError(f"probes of findings outside domain.findings: {unknown}")
    return ProbeTable(
        keys=_keys(arrays["probe_sample_ids"], names),
        images=arrays["probe_sample_ids"].astype(str),
        findings=tuple(findings),
        codes=np.array([lookup[f] for f in names], dtype=int),
        truth=truth,
        small=small,
        rare=rare,
        audit_keys=_keys(arrays["audit_sample_ids"], arrays["audit_findings"]),
        audit_images=arrays["audit_sample_ids"].astype(str),
    )


def _order(keys: np.ndarray, reference: np.ndarray, what: str) -> np.ndarray:
    """Indices that put ``keys`` in the order of ``reference`` (the same probe set only)."""
    if keys.shape == reference.shape and np.array_equal(keys, reference):
        return np.arange(len(keys))
    if len(set(keys.tolist())) != len(keys) or set(keys.tolist()) != set(reference.tolist()):
        raise ValueError(f"{what}: not the same probes as the reference run")
    position = {k: i for i, k in enumerate(keys.tolist())}
    return np.array([position[k] for k in reference.tolist()], dtype=int)


def arm_outcomes(
    arm: Arm, seeds: Sequence[int], per_seed: Sequence[Mapping[str, np.ndarray]], table: ProbeTable
) -> ArmOutcomes:
    """Average one arm's per-seed arrays over the seeds, aligned with ``table``."""
    correct, confidence, effect = [], [], []
    for seed, arrays in zip(seeds, per_seed, strict=True):
        what = f"{arm.name} seed {seed}"
        order = _order(
            _keys(arrays["probe_sample_ids"], arrays["probe_findings"]), table.keys, what
        )
        truth = arrays["probe_truth"].astype(bool)[order]
        if not np.array_equal(truth, table.truth):
            raise ValueError(f"{what}: probe labels differ from the reference run")
        yes = arrays["probe_yes"].astype(bool)[order]
        margin = arrays["probe_margin"].astype(float)[order]
        p_yes = 1.0 / (1.0 + np.exp(-np.clip(margin, -50.0, 50.0)))
        correct.append((yes == truth).astype(float))
        confidence.append(np.maximum(p_yes, 1.0 - p_yes))
        audit = _order(
            _keys(arrays["audit_sample_ids"], arrays["audit_findings"]), table.audit_keys, what
        )
        lesion = arrays["audit_lesion_flip"].astype(float)[audit]
        effect.append(lesion - arrays["audit_random_flip"].astype(float)[audit])
    hits = np.mean(correct, axis=0)
    return ArmOutcomes(
        arm=arm,
        seeds=tuple(seeds),
        correct=hits,
        confidence=np.mean(confidence, axis=0),
        majority=hits > 0.5,
        audit_effect=np.mean(effect, axis=0),
    )


# ------------------------------------------------------------------- metrics


class Scorer:
    """The four metrics of an arm on a set of probe rows (a resample or all rows)."""

    def __init__(self, table: ProbeTable, headline: str, n_bins: int) -> None:
        if headline not in HEADLINES:
            raise ConfigError(f"evaluation.headline_accuracy must be one of {HEADLINES}")
        self.table = table
        self.headline = headline
        self.n_bins = n_bins
        self.n_findings = len(table.findings)

    def bins(self, outcomes: ArmOutcomes) -> np.ndarray:
        """Equal-width confidence bin of every probe (the ECE bins of ``calibration.py``)."""
        return np.minimum((outcomes.confidence * self.n_bins).astype(int), self.n_bins - 1)

    def values(self, outcomes: ArmOutcomes, bins: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """``[headline, small-finding recall, rare-group recall, ECE]`` on rows ``idx``."""
        t = self.table
        c, truth = outcomes.correct[idx], t.truth[idx]
        if self.headline == "overall_accuracy":
            head = float(c.mean())
        else:
            head = _balanced(t.codes[idx], truth, c, self.n_findings)
        gap = c - outcomes.confidence[idx]
        ece = float(np.abs(np.bincount(bins[idx], weights=gap, minlength=self.n_bins)).sum())
        return np.array(
            [head, _masked_mean(c, t.small[idx]), _masked_mean(c, t.rare[idx]), ece / len(c)]
        )


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else math.nan


def _balanced(codes: np.ndarray, truth: np.ndarray, correct: np.ndarray, n: int) -> float:
    """Macro balanced accuracy from per-probe (possibly seed-averaged) correctness."""
    pos = np.bincount(codes, weights=truth, minlength=n)
    neg = np.bincount(codes, weights=~truth, minlength=n)
    hit_pos = np.bincount(codes, weights=correct * truth, minlength=n)
    hit_neg = np.bincount(codes, weights=correct * ~truth, minlength=n)
    both = (pos > 0) & (neg > 0)
    if not both.any():
        return math.nan
    return float(np.mean((hit_pos[both] / pos[both] + hit_neg[both] / neg[both]) / 2))


def metric_table(
    scorer: Scorer, arms: Sequence[ArmOutcomes], *, n_resamples: int, seed: int = BOOTSTRAP_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """``(observed, draws)``: arrays ``(arms, 4)`` and ``(resamples, arms, 4)`` of the metrics.

    All arms are scored on the same image-level resamples (``statistics.resample_indices``).
    """
    n = len(scorer.table.keys)
    bins = [scorer.bins(a) for a in arms]
    everything = np.arange(n)
    observed = np.array([scorer.values(a, b, everything) for a, b in zip(arms, bins, strict=True)])
    draws = np.empty((n_resamples, len(arms), 4))
    for r, idx in enumerate(resample_indices(n, n_resamples, seed, scorer.table.images)):
        for i, (a, b) in enumerate(zip(arms, bins, strict=True)):
            draws[r, i] = scorer.values(a, b, idx)
    return observed, draws


def _relative_drop(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(reference > 0, (reference - value) / reference, np.nan)


def _row(name: str, budget: str, metric: str, value: float, ci: tuple[float, float] | None = None):
    row: dict[str, Any] = {
        "experiment": name,
        "budget": budget,
        "seed": 0,
        "metric": metric,
        "value": value,
    }
    if ci is not None and not any(math.isnan(v) for v in ci):
        row["ci_low"], row["ci_high"] = min(ci[0], value), max(ci[1], value)
    return row


def drop_gap_rows(
    observed: np.ndarray, draws: np.ndarray, arm: int, reference: int, name: str, budget: str,
    alpha: float,
) -> list[dict[str, Any]]:  # fmt: skip
    """H1 rows of one arm: relative drop of a clinical recall minus that of the headline."""
    rows = []
    for metric, column in (("diff:drop_gap", SMALL_COL), ("diff:drop_gap_rare", RARE_COL)):

        def gap(table: np.ndarray, column: int = column) -> np.ndarray:
            ref, val = table[..., reference, :], table[..., arm, :]
            clinical = _relative_drop(ref[..., column], val[..., column])
            return clinical - _relative_drop(ref[..., HEAD], val[..., HEAD])

        value = float(gap(observed))
        if math.isnan(value):
            continue
        rows.append(_row(name, budget, metric, value, percentile_ci(gap(draws), alpha)))
    return rows


def paired_rows(
    observed: np.ndarray, draws: np.ndarray, a: int, b: int, name: str, budget: str, alpha: float
) -> list[dict[str, Any]]:
    """``diff:`` and ``p:`` rows of the small-finding recall, arm ``a`` minus arm ``b``."""
    value = float(observed[a, SMALL_COL] - observed[b, SMALL_COL])
    diffs = draws[:, a, SMALL_COL] - draws[:, b, SMALL_COL]
    return [
        _row(name, budget, f"diff:{SMALL}", value, percentile_ci(diffs, alpha)),
        _row(name, budget, f"p:{SMALL}", bootstrap_p_value(diffs)),
    ]


def mcnemar_rows(
    table: ProbeTable, a: ArmOutcomes, b: ArmOutcomes, name: str, budget: str
) -> list[dict[str, Any]]:
    """McNemar per finding on the majority answers, Holm-corrected across findings."""
    tested, p_values = [], []
    for code, finding in enumerate(table.findings):
        rows = table.codes == code
        if rows.any():
            tested.append(finding)
            p_values.append(mcnemar_test(a.majority[rows], b.majority[rows])[1])
    out = [_row(name, budget, f"mcnemar_p:{f}", p) for f, p in zip(tested, p_values, strict=True)]
    adjusted = holm_correction(p_values)
    out += [
        _row(name, budget, f"mcnemar_p_holm:{f}", p) for f, p in zip(tested, adjusted, strict=True)
    ]
    return out


def tau_rows(observed: np.ndarray, draws: np.ndarray, arms: Sequence[int], alpha: float) -> list:
    """H4 rows: Kendall's tau of the headline order against each clinical order."""
    rows = []
    for metric, column, sign in (
        (SMALL, SMALL_COL, 1.0),
        (RARE, RARE_COL, 1.0),
        (ECE, ECE_COL, -1.0),
    ):
        x, y = observed[arms, HEAD], sign * observed[arms, column]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        taus = np.fromiter(
            (kendall_tau(d[arms, HEAD], sign * d[arms, column]) for d in draws), float, len(draws)
        )
        value = kendall_tau(x, y)
        rows.append(
            _row("accuracy_vs_clinical", "", f"tau:{metric}", value, percentile_ci(taus, alpha))
        )
    return rows


# ------------------------------------------------------------------- choices


def validation_means(
    rows: pd.DataFrame, labels: Sequence[str], budget: str, metric: str, seeds: Sequence[int]
) -> tuple[dict[str, float] | None, str]:
    """Mean over seeds of a validation metric for each label; None (with why) if incomplete."""
    if rows.empty:
        return None, "no validation rows"
    data = rows[
        (rows["split"] == "val")
        & (rows["metric"] == metric)
        & (rows["budget"].fillna("") == budget)
    ]
    out: dict[str, float] = {}
    for label in labels:
        hit = data[data["experiment"] == label]
        if set(hit["seed"].astype(int)) != set(seeds):
            return (
                None,
                f"{label}@{budget or '-'}: validation {metric} not there for seeds {list(seeds)}",
            )
        out[label] = float(hit["value"].mean())
    return out, ""


def matched_budget(rows: pd.DataFrame) -> tuple[dict[str, Any] | None, str]:
    """The budget whose target is nearest to E3 ``w8``'s measured peak memory (H2)."""
    if rows.empty:
        return None, "no rows"
    memory = rows[
        (rows["experiment"] == QUANT_W8)
        & (rows["metric"] == "peak_memory_bytes")
        & (rows["split"] == "")
    ]
    if memory.empty:
        return None, f"no peak_memory_bytes of {QUANT_W8}"
    peak = float(memory["value"].mean())
    targets = {}
    for budget in BUDGET_IDS:
        hit = rows[(rows["metric"] == "budget_target") & (rows["budget"] == budget)]
        if hit.empty:
            return None, f"no budget_target at {budget}"
        targets[budget] = float(hit["value"].median())
    best = min(targets, key=lambda b: (abs(targets[b] - peak), b))
    return {
        "budget": best,
        "w8_peak_memory_bytes": peak,
        "targets": targets,
        "relative_gap": (peak - targets[best]) / targets[best],
    }, ""


# --------------------------------------------------------------------- stage


def _settings(config: Mapping[str, Any]) -> Any:
    from medcompress.config import apply_overrides
    from medcompress.evaluation.runner import resolve_settings

    restricted = apply_overrides(
        config, [f"evaluation.suites={list(COMPARISON_SUITES)}", "evaluation.split=test"]
    )
    settings = resolve_settings(restricted)
    if settings.pooling != "pooled":
        raise ConfigError("the comparison step implements the pooled rare-group recall only")
    return settings


def headline_metric(config: Mapping[str, Any]) -> str:
    """``evaluation.headline_accuracy``, checked."""
    value = (config.get("evaluation") or {}).get("headline_accuracy")
    if value not in HEADLINES:
        raise ConfigError(f"evaluation.headline_accuracy must be one of {HEADLINES}, got {value!r}")
    return str(value)


def _seeds(experiment_id: str) -> tuple[int, ...]:
    from medcompress.config import load_experiment

    return tuple(load_experiment(experiment_id)[1].seeds)


def load_arms(
    output_root: Path, arms: Sequence[Arm], seeds_of: Mapping[str, Sequence[int]]
) -> tuple[dict[Arm, list[dict[str, np.ndarray]]], list[str]]:
    """Per-seed arrays of every arm with all its seeds; the others are listed as incomplete.

    An arm with any seed missing is left out entirely (never compared on fewer seeds).
    """
    found: dict[Arm, list[dict[str, np.ndarray]]] = {}
    incomplete = []
    for arm in arms:
        seeds = seeds_of[arm.label.partition(":")[0]]
        paths = [run.run_dir(output_root) / OUTCOMES_FILE for run in arm.runs(seeds)]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            incomplete.append(f"{arm.name}: {len(missing)} of {len(paths)} seeds missing")
            continue
        found[arm] = [read_outcomes(p) for p in paths]
    return found, incomplete


def write_comparisons(rows: Sequence[Mapping[str, Any]], results_root: Path) -> list[Path]:
    """Write the rows to ``comparisons/<name>/<budget|nobudget>/comparison.csv`` (replaced)."""
    from medcompress.utils.io import append_results

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["experiment"]), str(row["budget"])), []).append(row)
    written = []
    for (name, budget), group in sorted(grouped.items()):
        path = Path(results_root) / "comparisons" / name.replace(":", "-")
        path = path / (budget or "nobudget") / COMPARISON_FILE
        path.unlink(missing_ok=True)
        append_results(group, path)
        written.append(path)
    return written


def run_comparisons(
    config: Mapping[str, Any],
    *,
    results_root: Path | None = None,
    output_root: Path | None = None,
    n_resamples: int | None = None,
) -> list[Path]:
    """Stage entry point for ``scripts/09_compare_arms.py`` (see the module docstring).

    Needs the frozen pre-registration (the test split), the evaluation settings and the
    committed results (validation rows for the comparator, efficiency and budget rows for H2).

    Used by: S11 (after every arm is evaluated on the test split).
    """
    from medcompress.evaluation.aggregate import load_results
    from medcompress.utils.io import write_json

    settings = _settings(config)
    headline = headline_metric(config)
    results_root = Path(results_root or config["paths"]["results_root"])
    output_root = Path(output_root or config["paths"]["output_root"])
    resamples = int(n_resamples or settings.n_resamples)
    alpha = settings.alpha
    seeds_of = {e: _seeds(e) for e in (REFERENCE, *COMPRESSED)}
    wanted = [Arm(REFERENCE)] + [arm for e in COMPRESSED for arm in arms_of(e)]
    found, incomplete = load_arms(output_root, wanted, seeds_of)
    if Arm(REFERENCE) not in found:
        raise ConfigError(f"the outcomes of {REFERENCE} (all seeds) are needed under {output_root}")
    groups = dict(settings.groups or {})
    table = build_table(
        found[Arm(REFERENCE)][0],
        settings.findings,
        settings.thresholds,
        settings.mask_findings,
        groups,
    )
    arms = list(found)
    outcomes = [arm_outcomes(a, seeds_of[a.label.partition(":")[0]], found[a], table) for a in arms]
    index = {a: i for i, a in enumerate(arms)}
    scorer = Scorer(table, headline, int(settings.n_bins))
    observed, draws = metric_table(scorer, outcomes, n_resamples=resamples)
    ref = index[Arm(REFERENCE)]
    results = load_results(results_root)
    rows: list[dict[str, Any]] = []
    choices: dict[str, Any] = {
        "headline_accuracy": headline,
        "n_resamples": resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_probes": int(len(table.keys)),
        "n_images": int(len(set(table.images.tolist()))),
        "n_small_positives": int(table.small.sum()),
        "n_rare_positives": int(table.rare.sum()),
        "arms": [a.name for a in arms],
        "incomplete": incomplete,
    }

    # H1: every compressed arm against E2
    for arm in arms:
        if arm.label != REFERENCE:
            name = f"{arm.label}_vs_{REFERENCE}"
            rows += drop_gap_rows(observed, draws, index[arm], ref, name, arm.budget, alpha)

    # H3: E7 against the best of E5 / E6 at B2, chosen on validation
    generic = [a.label for e in GENERIC_PRUNING for a in arms_of(e) if a.budget == PRIMARY_BUDGET]
    scores, why = validation_means(results, generic, PRIMARY_BUDGET, SMALL, seeds_of["E5"])
    ours = Arm(OURS, PRIMARY_BUDGET)
    if scores is None:
        choices["h3"] = {"skipped": why}
    else:
        best = Arm(pick_best_arm(scores), PRIMARY_BUDGET)
        choices["h3"] = {"comparator": best.label, "validation_small_finding_recall": scores}
        if ours in index and best in index:
            a, b = outcomes[index[ours]], outcomes[index[best]]
            name = f"{OURS}_vs_best"
            primary = paired_rows(
                observed, draws, index[ours], index[best], name, PRIMARY_BUDGET, alpha
            )
            primary += compare_arms(
                name,
                PRIMARY_BUDGET,
                a.audit_effect,
                b.audit_effect,
                {MASK_EFFECT: np.mean},
                n_resamples=resamples,
                alpha=alpha,
                seed=BOOTSTRAP_SEED,
                clusters=table.audit_images,
            )
            rows += add_holm_rows(primary, [SMALL, MASK_EFFECT])
            rows += mcnemar_rows(table, a, b, name, PRIMARY_BUDGET)
        else:
            choices["h3"]["skipped"] = "test outcomes of E7 or the comparator at B2 are incomplete"

    # H2: the best generic pruning arm against E3 w8 at the matched budget
    matched, why = matched_budget(results)
    if matched is None:
        choices["h2"] = {"skipped": why}
    else:
        budget = matched["budget"]
        labels = [a.label for e in GENERIC_PRUNING for a in arms_of(e) if a.budget == budget]
        scores, why = validation_means(results, labels, budget, SMALL, seeds_of["E5"])
        if scores is None:
            choices["h2"] = {**matched, "skipped": why}
        else:
            best = Arm(pick_best_arm(scores), budget)
            choices["h2"] = {
                **matched,
                "arm": best.label,
                "validation_small_finding_recall": scores,
            }
            w8 = Arm(QUANT_W8)
            if best in index and w8 in index:
                name = f"best_pruning_vs_{QUANT_W8}"
                rows += paired_rows(observed, draws, index[best], index[w8], name, budget, alpha)
                rows += mcnemar_rows(
                    table, outcomes[index[best]], outcomes[index[w8]], name, budget
                )
            else:
                choices["h2"]["skipped"] = "test outcomes of the arm or of E3 w8 are incomplete"

    # H4: the compressed arms ranked by the headline and by the clinical metrics
    ranked = [index[a] for a in arms if a.label != REFERENCE]
    if len(ranked) >= 2:
        rows += tau_rows(observed, draws, ranked, alpha)

    written = write_comparisons(rows, results_root)
    choices_path = results_root / "comparisons" / CHOICES_FILE
    write_json(choices, choices_path)
    print(f"compared {len(arms)} arms on {len(table.keys)} probes; wrote {len(written)} files")
    return [*written, choices_path]
