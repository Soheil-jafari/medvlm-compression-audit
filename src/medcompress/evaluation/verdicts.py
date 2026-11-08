"""Hypothesis verdicts (H1-H4) from comparison rows, written to ``docs/results/VERDICTS.md``.

The decision rules below are the agent's proposal for the pre-registration
(``docs/preregistration.md`` section 6); they change only by a logged decision.
Every verdict is "supported", "not supported", "inconclusive", "pending" (no
evidence rows yet) or "exploratory" (rows from seed-0-only arms: reported, no confirmatory
claim; seed plan of 2026-11-08, ``medcompress.seed_plan``), and lists the rows it used.

Evidence rows (written by the comparison step, ``evaluation/compare_stage``):

* H1: ``<arm>_vs_E2`` with ``diff:drop_gap`` = relative drop in small-finding
  recall minus relative drop in accuracy (positive = hidden damage larger), for the 3-seed arms
  of the main tier (E7 and the comparator at B2). The ``<arm>_vs_E2_seed0`` rows of the other
  arms are exploratory and not read here.
* H2: ``best_pruning_vs_E3:w8`` with ``diff:small_finding_recall`` at matched
  memory (negative = pruning harms small findings more); ``best_pruning_vs_E3:w8_seed0`` when a
  side has seed 0 only (E3 ``w8`` by the plan): exploratory.
* H3: ``E7_vs_best`` at B2 with ``diff:`` and ``p_holm:`` rows for
  ``small_finding_recall`` and ``lesion_mask_effect``; ``E7_vs_best_seed0`` after a futility
  stop: exploratory.
* H4: ``accuracy_vs_clinical`` with ``tau:small_finding_recall`` (and optionally
  ``tau:ece``): Kendall's tau between rankings, with a bootstrap CI. Exploratory since
  2026-11-08 (seed 0 of every compressed arm): reported, never a confirmatory verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from medcompress.registry import PRIMARY_BUDGET

ALPHA = 0.05
SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
INCONCLUSIVE = "inconclusive"
PENDING = "pending"
EXPLORATORY = "exploratory"
"""Rows from seed-0-only arms: reported with their intervals, no confirmatory claim."""
SEED0_SUFFIX = "_seed0"


@dataclass(frozen=True)
class Verdict:
    """One hypothesis decision with its evidence."""

    hypothesis: str
    statement: str
    verdict: str
    evidence: tuple[str, ...]
    rule: str


def _rows(
    summary: pd.DataFrame, experiment_suffix: str | None = None, **match: str
) -> pd.DataFrame:
    if summary.empty:
        return summary
    data = summary
    for column, value in match.items():
        data = data[data[column].fillna("") == value]
    if experiment_suffix is not None:
        data = data[data["experiment"].str.endswith(experiment_suffix)]
    return data


def _describe(row: pd.Series) -> str:
    ci = ""
    if not (math.isnan(row["ci_low"]) or math.isnan(row["ci_high"])):
        ci = f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    budget = f" @{row['budget']}" if row["budget"] else ""
    return f"{row['experiment']}{budget} {row['metric']} = {row['mean']:.3f}{ci}"


def verdict_h1(summary: pd.DataFrame) -> Verdict:
    """H1: hidden damage exceeds headline damage for every compression family."""
    rows = _rows(summary, "_vs_E2", metric="diff:drop_gap")  # the 3-seed tier only
    rule = (
        "supported if every 3-seed arm's CI is above 0; not supported if any CI is below 0 "
        "(seed-0 arms: exploratory, not read)"
    )
    statement = (
        "Damage to small findings exceeds damage to accuracy, for the pruning arms of the "
        "main tier (E7 and the comparator at B2)."
    )
    if rows.empty:
        return Verdict("H1", statement, PENDING, (), rule)
    lows, highs = rows["ci_low"], rows["ci_high"]
    if (lows > 0).all():
        verdict = SUPPORTED
    elif (highs < 0).any():
        verdict = NOT_SUPPORTED
    else:
        verdict = INCONCLUSIVE
    return Verdict("H1", statement, verdict, tuple(_describe(r) for _, r in rows.iterrows()), rule)


def verdict_h2(summary: pd.DataFrame) -> Verdict:
    """H2: structured pruning harms small findings more than 8-bit quantization."""
    rows = _rows(summary, experiment="best_pruning_vs_E3:w8", metric="diff:small_finding_recall")
    rule = (
        "supported if the CI of (pruning - W8) is below 0; not supported if above 0; "
        "exploratory if a side has seed 0 only"
    )
    statement = "Structured pruning harms small-finding recall more than 8-bit quantization."
    if rows.empty:
        seed0 = _rows(summary, experiment="best_pruning_vs_E3:w8" + SEED0_SUFFIX,
                      metric="diff:small_finding_recall")  # fmt: skip
        if not seed0.empty:
            return Verdict("H2", statement, EXPLORATORY, (_describe(seed0.iloc[0]),), rule)
        return Verdict("H2", statement, PENDING, (), rule)
    row = rows.iloc[0]
    if row["ci_high"] < 0:
        verdict = SUPPORTED
    elif row["ci_low"] > 0:
        verdict = NOT_SUPPORTED
    else:
        verdict = INCONCLUSIVE
    return Verdict("H2", statement, verdict, (_describe(row),), rule)


def verdict_h3(summary: pd.DataFrame) -> Verdict:
    """H3 (primary): E7 beats the best of E5/E6 at B2 on both primary endpoints."""
    rule = (
        "supported if both differences are > 0 with Holm-adjusted p < 0.05; not supported "
        "if either CI is entirely <= 0; otherwise inconclusive; exploratory after a futility "
        "stop (seed 0 only)"
    )
    statement = (
        "At B2, the clinical criterion gives higher small-finding recall and a larger "
        "lesion-mask effect than the best generic criterion."
    )
    evidence = []
    diffs, ps = [], []
    for metric in ("small_finding_recall", "lesion_mask_effect"):
        diff = _rows(
            summary, experiment="E7_vs_best", budget=PRIMARY_BUDGET, metric=f"diff:{metric}"
        )
        holm = _rows(
            summary, experiment="E7_vs_best", budget=PRIMARY_BUDGET, metric=f"p_holm:{metric}"
        )
        if diff.empty or holm.empty:
            seed0 = _rows(summary, experiment="E7_vs_best" + SEED0_SUFFIX,
                          budget=PRIMARY_BUDGET)  # fmt: skip
            if not seed0.empty and not evidence:
                described = tuple(_describe(r) for _, r in seed0.iterrows()
                                  if str(r["metric"]).startswith("diff:"))  # fmt: skip
                return Verdict("H3", statement, EXPLORATORY, described, rule)
            return Verdict("H3", statement, PENDING, tuple(evidence), rule)
        diffs.append(diff.iloc[0])
        ps.append(float(holm.iloc[0]["mean"]))
        evidence += [_describe(diff.iloc[0]), _describe(holm.iloc[0])]
    if all(d["mean"] > 0 for d in diffs) and all(p < ALPHA for p in ps):
        verdict = SUPPORTED
    elif any(d["ci_high"] <= 0 for d in diffs):
        verdict = NOT_SUPPORTED
    else:
        verdict = INCONCLUSIVE
    return Verdict("H3", statement, verdict, tuple(evidence), rule)


def verdict_h4(summary: pd.DataFrame) -> Verdict:
    """H4: ranking by accuracy differs from ranking by clinical metrics."""
    rows = _rows(summary, experiment="accuracy_vs_clinical")
    rows = rows[rows["metric"].str.startswith("tau:")] if not rows.empty else rows
    rule = (
        "exploratory since 2026-11-08 (seed 0 of every compressed arm): the taus and their "
        "image-bootstrap CIs are reported, with no confirmatory claim"
    )
    statement = (
        "Ranking compressed arms by accuracy gives a different order than by clinical metrics."
    )
    if rows.empty:
        return Verdict("H4", statement, PENDING, (), rule)
    evidence = tuple(_describe(r) for _, r in rows.iterrows())
    return Verdict("H4", statement, EXPLORATORY, evidence, rule)


def all_verdicts(summary: pd.DataFrame) -> list[Verdict]:
    """Return the verdicts for H1-H4."""
    return [verdict_h1(summary), verdict_h2(summary), verdict_h3(summary), verdict_h4(summary)]


def verdicts_markdown(verdicts: list[Verdict]) -> str:
    """Return ``VERDICTS.md`` text."""
    lines = [
        "# Hypothesis verdicts",
        "",
        "Generated by `scripts/08_aggregate_results.py` from `results/`. Do not edit.",
        "Rules: `src/medcompress/evaluation/verdicts.py` (to be frozen in the pre-registration).",
        "",
        "| Hypothesis | Statement | Verdict | Rule |",
        "|---|---|---|---|",
    ]
    for v in verdicts:
        lines.append(f"| {v.hypothesis} | {v.statement} | **{v.verdict}** | {v.rule} |")
    lines.append("")
    for v in verdicts:
        lines.append(f"## {v.hypothesis}: {v.verdict}")
        lines.append("")
        lines += [f"- {e}" for e in v.evidence] or ["- no evidence rows yet"]
        lines.append("")
    return "\n".join(lines)


def write_verdicts(summary: pd.DataFrame, path: Path) -> Path:
    """Write ``VERDICTS.md`` to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(verdicts_markdown(all_verdicts(summary)), encoding="utf-8")
    return path
