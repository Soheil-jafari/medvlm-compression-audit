"""Stage 09: paired comparisons on the test split, the evidence rows of H1-H4.

Stage 06 leaves the per-probe answers of every run in ``<output_root>/<run dir>/``
``outcomes_test.npz`` (never committed). This stage reads them for every arm and seed. The seed
plan (``medcompress.seed_plan``, pre-registration section 11, amendment of 2026-11-08) decides
how: the **main-claim tier** (E2, E7 at B2 and the comparator at B2) is averaged per probe over
its 3 seeds; **every other arm is read at seed 0 only** and reported as exploratory; an arm with
2 seeds is refused. It then draws one sequence of image-level bootstrap resamples (seed 0) and
computes, for every arm and every resample, the headline accuracy, the small-finding recall,
the rare-group recall and the ECE. Every comparison below is a function of that one table, so
all intervals share the resamples:

* H1, ``<arm>_vs_E2`` for the 3-seed arms of the main tier (confirmatory): ``diff:drop_gap`` =
  relative drop of the small-finding recall minus relative drop of the headline accuracy (both
  against E2); ``diff:drop_gap_rare`` uses the rare-group recall instead. Every other arm gets
  the same rows against E2 seed 0, named ``<arm>_vs_E2_seed0`` (exploratory);
* H2, ``best_pruning_vs_E3:w8`` at the budget whose target is nearest to the measured peak
  memory of E3 ``w8``: ``diff:small_finding_recall`` of the best generic pruning arm at that
  budget (the ordering of the comparator, validation seed 0) minus E3 ``w8``. With a
  single-seed side (E3 ``w8`` is seed 0 only by the plan) the rows are
  ``best_pruning_vs_E3:w8_seed0`` (exploratory);
* H3, ``E7_vs_best`` at B2: ``diff:``, ``p:`` and ``p_holm:`` rows of the two primary
  endpoints against the comparator (``seed_plan.choose_comparator`` on validation, seed 0);
  the lesion-mask effect is bootstrapped over the audited probes, by image. If E7 or the
  comparator has seed 0 only (the futility stop), the rows are ``E7_vs_best_seed0``
  (exploratory);
* H4, ``accuracy_vs_clinical`` (exploratory, seed 0 of every compressed arm at B2 and without
  a budget): Kendall's tau between the order of the arms by the headline accuracy and by
  small-finding recall, rare-group recall and ECE (negated, so that higher is better in both
  orders).

H2 and H3 also get McNemar's test per finding on each probe's majority answer over the seeds,
Holm-corrected across findings (``mcnemar_p:`` and ``mcnemar_p_holm:``). The rule-based
choices (comparator, futility decision, matched budget), the tier and seeds of every arm and
the arms left out are written to ``results/comparisons/choices.json``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medcompress.config import ConfigError, RunContext
from medcompress.evaluation import flips as flips_mod
from medcompress.evaluation.comparisons import add_holm_rows, compare_arms
from medcompress.evaluation.statistics import (
    bootstrap_p_value,
    holm_correction,
    kendall_tau,
    mcnemar_test,
    percentile_ci,
    resample_indices,
)
from medcompress.registry import BUDGET_IDS, PRIMARY_BUDGET, REGISTRY
from medcompress.seed_plan import (
    COMPARATOR_CANDIDATES,
    SEEDS,
    SeedPlanError,
    check_seed_count,
    choose_comparator,
    futility_stop,
    planned_seeds,
    rank,
    scores_from_rows,
    tier,
)

STRICT_FILE = "strict_test.json"
"""RC-D1 run 4's strict test subset (D02 amendment 3): img_ids, in the private folder."""
STRICT_DIR = "comparisons_strict"
UNFILTERED_DIR = "comparisons_unfiltered"
UNFILTERED_OUTCOMES_FILE = "outcomes_test_unfiltered.npz"
"""The unfiltered test (D02 amendment 6): the same comparisons on its outcomes."""

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
SEED0_SUFFIX = "_seed0"
"""Appended to the name of every exploratory comparison (a seed-0-only side)."""
HEADLINES: tuple[str, ...] = ("overall_accuracy", "macro_balanced_accuracy")
SMALL = "small_finding_recall"
"""Comparison rows keep this name; since 2026-11-06 it is the recall at the operating point."""
OP_SMALL = "op_small_finding_recall"
"""The validation metric the comparators are chosen on (H2, H3): at the operating point."""
OP_ANSWERS = "probe_yes_op"
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
    correct_plain: np.ndarray | None = None
    """Correctness of the plain yes/no answers (for the ECE); ``correct`` and ``majority``
    are at the operating point (2026-11-06)."""


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
    """Average one arm's per-seed arrays over the seeds, aligned with ``table``.

    The answers are those at the operating point (``probe_yes_op``); outcomes saved before
    2026-11-06 have none and are refused.
    """
    correct, plain, confidence, effect = [], [], [], []
    for seed, arrays in zip(seeds, per_seed, strict=True):
        what = f"{arm.name} seed {seed}"
        order = _order(
            _keys(arrays["probe_sample_ids"], arrays["probe_findings"]), table.keys, what
        )
        truth = arrays["probe_truth"].astype(bool)[order]
        if not np.array_equal(truth, table.truth):
            raise ValueError(f"{what}: probe labels differ from the reference run")
        if OP_ANSWERS not in arrays:
            raise ValueError(f"{what}: no operating-point answers ({OP_ANSWERS}): these outcomes "
                             "were saved before 2026-11-06; evaluate the run again")  # fmt: skip
        yes = arrays["probe_yes"].astype(bool)[order]
        yes_op = arrays[OP_ANSWERS].astype(bool)[order]
        margin = arrays["probe_margin"].astype(float)[order]
        p_yes = 1.0 / (1.0 + np.exp(-np.clip(margin, -50.0, 50.0)))
        correct.append((yes_op == truth).astype(float))
        plain.append((yes == truth).astype(float))
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
        correct_plain=np.mean(plain, axis=0),
    )


def image_of(sample_ids: np.ndarray) -> np.ndarray:
    """The img_id of each sample id (``<img_id>:<hash>``)."""
    return np.array([str(s).split(":", 1)[0] for s in sample_ids])


def strict_images(private_root: Path) -> set[str]:
    """The img_ids of the strict test subset written by RC-D1 run 4."""
    import json

    path = Path(private_root) / STRICT_FILE
    if not path.is_file():
        raise ConfigError(
            f"{path} is missing: the strict test subset comes from RC-D1 run 4 (D02 amendment 3)"
        )
    return set(json.loads(path.read_text(encoding="utf-8")))


def restrict(
    table: ProbeTable, outcomes: Sequence[ArmOutcomes], keep_images: set[str]
) -> tuple[ProbeTable, list[ArmOutcomes]]:
    """The probe table and every arm's outcomes on the probes (and lesion-mask audit probes)
    of ``keep_images`` only: the strict-test sensitivity analysis (pre-registered)."""
    from dataclasses import replace

    keep = np.isin(image_of(table.images), list(keep_images))
    audit = np.isin(image_of(table.audit_images), list(keep_images))
    small = replace(
        table, keys=table.keys[keep], images=table.images[keep], codes=table.codes[keep],
        truth=table.truth[keep], small=table.small[keep], rare=table.rare[keep],
        audit_keys=table.audit_keys[audit], audit_images=table.audit_images[audit],
    )  # fmt: skip
    arms = [
        replace(o, correct=o.correct[keep], confidence=o.confidence[keep],
                majority=o.majority[keep], audit_effect=o.audit_effect[audit],
                correct_plain=None if o.correct_plain is None else o.correct_plain[keep])
        for o in outcomes
    ]  # fmt: skip
    return small, arms


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
        plain = outcomes.correct if outcomes.correct_plain is None else outcomes.correct_plain
        gap = plain[idx] - outcomes.confidence[idx]  # calibration: the plain answers
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


def flip_comparison(
    output_root: Path,
    outcomes_file: str,
    table: ProbeTable,
    outcomes: Sequence[ArmOutcomes],
    settings: Any,
    *,
    n_resamples: int,
    alpha: float,
) -> tuple[list[dict[str, Any]], str]:
    """The flip rows of every arm against E0 (``evaluation/flips.py``), or why there are none."""
    run = RunContext(flips_mod.REFERENCE, seed=0)
    path = run.run_dir(output_root) / outcomes_file
    if not path.is_file():
        return [], f"the outcomes of {flips_mod.REFERENCE} seed 0 ({path.name}) are not there"
    reference = read_outcomes(path)
    order = _order(_keys(reference["probe_sample_ids"], reference["probe_findings"]),
                   table.keys, f"{flips_mod.REFERENCE} seed 0")  # fmt: skip
    if not np.array_equal(reference["probe_truth"].astype(bool)[order], table.truth):
        raise ValueError(
            f"{flips_mod.REFERENCE} seed 0: probe labels differ from the reference run"
        )
    items = flips_mod.flip_items(
        reference, order, table.findings, settings.thresholds, settings.mask_findings
    )
    labels = [(o.arm.label, o.arm.budget) for o in outcomes]
    answers = np.array([flips_mod.majority_yes(o.majority, table.truth) for o in outcomes])
    flipped = answers != items.reference_yes[None, :]
    rows = flips_mod.flip_rows(labels, flipped, items, n_resamples=n_resamples, alpha=alpha,
                               seed=BOOTSTRAP_SEED)  # fmt: skip
    return rows, ""


# ------------------------------------------------------------------- choices


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


@dataclass(frozen=True)
class LoadedArm:
    """One arm's per-seed arrays: ``seeds`` used (3 for the main tier, seed 0 otherwise)."""

    arm: Arm
    seeds: tuple[int, ...]
    arrays: list[dict[str, np.ndarray]]

    @property
    def main(self) -> bool:
        return len(self.seeds) == len(SEEDS)


def load_arms(
    output_root: Path,
    arms: Sequence[Arm],
    comparator: str | None,
    outcomes_file: str = OUTCOMES_FILE,
) -> tuple[dict[Arm, LoadedArm], list[str]]:
    """The arrays of every arm by the seed plan, and notes on what was left out or narrowed.

    * An arm without seed 0 is left out.
    * An arm with 2 seeds is refused (:class:`SeedPlanError`): the plan allows 1 or 3.
    * A main-tier arm (``seed_plan.is_main``) with its 3 seeds is read with all of them; with
      seed 0 only it is read at seed 0 and reported as exploratory.
    * Every other arm is read at seed 0 only (extra seeds are listed and not used).
    """
    found: dict[Arm, LoadedArm] = {}
    notes = []
    for arm in arms:
        paths = {
            s: r.run_dir(output_root) / outcomes_file
            for s, r in zip(SEEDS, arm.runs(SEEDS), strict=True)
        }
        present = tuple(s for s, p in paths.items() if p.is_file())
        check_seed_count(arm.name, present)
        if 0 not in present:
            if present:
                notes.append(f"{arm.name}: seed 0 missing (seeds {list(present)} not used)")
            continue
        planned = planned_seeds(arm.label, arm.budget, comparator)
        if planned == SEEDS and present == SEEDS:
            use = SEEDS
        else:
            use = (0,)
            if planned == SEEDS:
                notes.append(f"{arm.name}: {len(SEEDS) - len(present)} of 3 seeds missing: "
                             "seed 0 only, exploratory")  # fmt: skip
            elif present == SEEDS:
                notes.append(f"{arm.name}: seeds 1 and 2 not used (seed 0 only by the plan)")
        found[arm] = LoadedArm(arm, use, [read_outcomes(paths[s]) for s in use])
    return found, notes


def write_comparisons(
    rows: Sequence[Mapping[str, Any]], results_root: Path, folder: str = "comparisons"
) -> list[Path]:
    """Write the rows to ``<folder>/<name>/<budget|nobudget>/comparison.csv`` (replaced)."""
    from medcompress.utils.io import append_results

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["experiment"]), str(row["budget"])), []).append(row)
    written = []
    for (name, budget), group in sorted(grouped.items()):
        path = Path(results_root) / folder / name.replace(":", "-")
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
    strict: bool = False,
    unfiltered: bool = False,
) -> list[Path]:
    """Stage entry point for ``scripts/09_compare_arms.py`` (see the module docstring).

    ``strict``: the same comparisons on the strict test subset only (D02 amendment 3, the
    pre-registered sensitivity analysis), written to ``comparisons_strict/``.
    ``unfiltered``: the same comparisons on the unfiltered test (D02 amendment 6, a
    sensitivity analysis), from ``outcomes_test_unfiltered.npz``, written to
    ``comparisons_unfiltered/``.

    Needs the frozen pre-registration (the test split), the evaluation settings and the
    committed results (validation rows for the comparator, efficiency and budget rows for H2).

    Used by: S11 (after every arm is evaluated on the test split).
    """
    from medcompress.evaluation.aggregate import load_results
    from medcompress.utils.io import write_json

    if strict and unfiltered:
        raise ConfigError("the strict subset is taken within the filtered test: use one of "
                          "--strict and --unfiltered")  # fmt: skip
    settings = _settings(config)
    headline = headline_metric(config)
    results_root = Path(results_root or config["paths"]["results_root"])
    output_root = Path(output_root or config["paths"]["output_root"])
    resamples = int(n_resamples or settings.n_resamples)
    alpha = settings.alpha
    results = load_results(results_root)
    generic_b2 = list(COMPARATOR_CANDIDATES)
    scores_b2 = scores_from_rows(results, [*generic_b2, OURS], budget=PRIMARY_BUDGET)
    comparator, h3_choice = None, {}
    try:
        best_score = choose_comparator(scores_b2)
        comparator = best_score.label
        h3_choice = {
            "comparator": comparator,
            "ordering": [s.label for s in rank(scores_b2[c] for c in generic_b2)],
            "validation_seed0": {c: asdict(scores_b2[c]) for c in generic_b2},
        }
        if OURS in scores_b2:
            h3_choice["futility_stop"] = futility_stop(scores_b2[OURS], best_score)
    except SeedPlanError as exc:
        h3_choice = {"skipped": f"comparator not chosen: {exc}"}
    wanted = [Arm(REFERENCE)] + [arm for e in COMPRESSED for arm in arms_of(e)]
    outcomes_file = UNFILTERED_OUTCOMES_FILE if unfiltered else OUTCOMES_FILE
    found, notes = load_arms(output_root, wanted, comparator, outcomes_file)
    if Arm(REFERENCE) not in found:
        raise ConfigError(f"the outcomes of {REFERENCE} (seed 0 at least, {outcomes_file}) are "
                          f"needed under {output_root}")  # fmt: skip
    groups = dict(settings.groups or {})
    table = build_table(
        found[Arm(REFERENCE)].arrays[0],
        settings.findings,
        settings.thresholds,
        settings.mask_findings,
        groups,
    )
    # every arm at seed 0 (exploratory reads); the 3-seed arms of the main tier also averaged
    keys: list[tuple[Arm, bool]] = []
    outcomes: list[ArmOutcomes] = []
    for arm, loaded in found.items():
        keys.append((arm, False))
        outcomes.append(arm_outcomes(arm, (0,), loaded.arrays[:1], table))
        if loaded.main:
            keys.append((arm, True))
            outcomes.append(arm_outcomes(arm, loaded.seeds, loaded.arrays, table))
    folder = UNFILTERED_DIR if unfiltered else "comparisons"
    if strict:
        keep = strict_images(Path(config["paths"]["private_root"]))
        table, outcomes = restrict(table, outcomes, keep)
        folder = STRICT_DIR
    index = {k: i for i, k in enumerate(keys)}
    scorer = Scorer(table, headline, int(settings.n_bins))
    observed, draws = metric_table(scorer, outcomes, n_resamples=resamples)
    reference = Arm(REFERENCE)
    rows: list[dict[str, Any]] = []
    choices: dict[str, Any] = {
        "headline_accuracy": headline,
        "n_resamples": resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_probes": int(len(table.keys)),
        "n_images": int(len(set(table.images.tolist()))),
        "n_small_positives": int(table.small.sum()),
        "n_rare_positives": int(table.rare.sum()),
        "arms": [a.name for a in found],
        "tiers": {
            a.name: {
                "tier": tier(a.label, a.budget, comparator) if loaded.main else "exploratory",
                "seeds": list(loaded.seeds),
            }
            for a, loaded in found.items()
        },
        "incomplete": notes,
        "subset": (
            "strict test (D02 amendment 3)"
            if strict
            else "unfiltered test (D02 amendment 6)"
            if unfiltered
            else "test"
        ),
    }

    def pair(a: Arm, b: Arm) -> tuple[int, int, str] | None:
        """Indices of ``a`` and ``b``: both 3-seed (confirmatory, suffix ""), else both at seed
        0 (exploratory, suffix ``_seed0``); None if either is missing."""
        if (a, True) in index and (b, True) in index:
            return index[(a, True)], index[(b, True)], ""
        if (a, False) in index and (b, False) in index:
            return index[(a, False)], index[(b, False)], SEED0_SUFFIX
        return None

    # H1: the 3-seed arms of the main tier against E2 (confirmatory); every other arm at seed 0
    # against E2 seed 0 (exploratory)
    for arm in found:
        if arm.label == REFERENCE:
            continue
        if found[arm].main and found[reference].main:
            i, j, suffix = index[(arm, True)], index[(reference, True)], ""
        else:
            i, j, suffix = index[(arm, False)], index[(reference, False)], SEED0_SUFFIX
        name = f"{arm.label}_vs_{REFERENCE}{suffix}"
        rows += drop_gap_rows(observed, draws, i, j, name, arm.budget, alpha)

    # H3: E7 against the comparator at B2, chosen on validation seed 0 (seed_plan)
    choices["h3"] = h3_choice
    if comparator is not None:
        ours, best = Arm(OURS, PRIMARY_BUDGET), Arm(comparator, PRIMARY_BUDGET)
        both = pair(ours, best)
        if both is None:
            choices["h3"]["skipped"] = "test outcomes of E7 or the comparator at B2 are incomplete"
        else:
            i, j, suffix = both
            name = f"{OURS}_vs_best{suffix}"
            if suffix:
                choices["h3"]["exploratory"] = "E7 or the comparator has seed 0 only"
            a, b = outcomes[i], outcomes[j]
            primary = paired_rows(observed, draws, i, j, name, PRIMARY_BUDGET, alpha)
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

    # H2: the best generic pruning arm against E3 w8 at the matched budget
    matched, why = matched_budget(results)
    if matched is None:
        choices["h2"] = {"skipped": why}
    else:
        budget = matched["budget"]
        scores = scores_from_rows(results, generic_b2, budget=budget)
        missing = [c for c in generic_b2 if c not in scores]
        if missing:
            choices["h2"] = {**matched, "skipped": f"no validation rows at seed 0, {budget} "
                                                   f"for {missing}"}  # fmt: skip
        else:
            best = Arm(rank(scores.values())[0].label, budget)
            choices["h2"] = {**matched, "arm": best.label,
                             "ordering": [s.label for s in rank(scores.values())]}  # fmt: skip
            both = pair(best, Arm(QUANT_W8))
            if both is None:
                choices["h2"]["skipped"] = "test outcomes of the arm or of E3 w8 are incomplete"
            else:
                i, j, suffix = both
                if suffix:
                    choices["h2"]["exploratory"] = "a side has seed 0 only (E3 w8 by the plan)"
                name = f"best_pruning_vs_{QUANT_W8}{suffix}"
                rows += paired_rows(observed, draws, i, j, name, budget, alpha)
                rows += mcnemar_rows(table, outcomes[i], outcomes[j], name, budget)

    # H4 (exploratory): every compressed arm at seed 0, at B2 and without a budget
    ranked = [
        index[(a, False)]
        for a in found
        if a.label != REFERENCE and a.budget in ("", PRIMARY_BUDGET)
    ]
    choices["h4"] = "exploratory: seed 0 of every compressed arm at B2 and without a budget"
    if len(ranked) >= 2:
        rows += tau_rows(observed, draws, ranked, alpha)

    # secondary (pre-registered 2026-10-31): the items whose answer flips against E0, each arm
    # as it is read (3 seeds for the main tier, seed 0 otherwise)
    if strict:
        choices["flips"] = {"skipped": "not computed on the strict subset"}
    else:
        compared = [outcomes[index[(a, loaded.main)]] for a, loaded in found.items()]
        flips, why = flip_comparison(output_root, outcomes_file, table, compared, settings,
                                     n_resamples=resamples, alpha=alpha)  # fmt: skip
        rows += flips
        choices["flips"] = (
            {"reference": flips_mod.REFERENCE, "skipped": why}
            if why
            else {
                "reference": flips_mod.REFERENCE,
                "arms": sorted({r["experiment"] for r in flips}),
            }
        )
    choices["rare_group"] = (
        "none: the domain has no rare prevalence group, so no rare-group rows"
        if not table.rare.any() else "pooled positives of the rare group"
    )  # fmt: skip

    written = write_comparisons(rows, results_root, folder)
    choices_path = results_root / folder / CHOICES_FILE
    write_json(choices, choices_path)
    print(f"compared {len(found)} arms on {len(table.keys)} probes; wrote {len(written)} files")
    return [*written, choices_path]
