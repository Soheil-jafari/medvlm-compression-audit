"""Appendix tables and figures of the paper, generated like the main ones (never typed by hand).

From the test-split summary: per-finding sensitivity and specificity, calibration (ECE,
Brier, AURC), the free-text answers, E9 next to the teacher, and the reliability and
risk-coverage plots. From every committed row (bookkeeping that belongs to no split): the
compute used and the student checks of Round 0. From the configs: the hyper-parameters.
With no results yet, cells read ``--`` and plots are labelled placeholders.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from medcompress.evaluation.report import (
    ACCURACY,
    BALANCED_ACCURACY,
    MASK_EFFECT,
    MISSING,
    SMALL_RECALL,
    arm_labels,
    fmt_mean_sd,
    fmt_value,
    latex_escape,
    lookup_summary,
    tabular,
)
from medcompress.registry import PRIMARY_BUDGET

TIME_METRICS: tuple[tuple[str, str], ...] = (
    ("teacher_cache_seconds", "Teacher cache"),
    ("train_seconds", "Training"),
    ("scoring_seconds", "Scoring"),
    ("recovery_seconds", "Recovery"),
    ("quantization_seconds", "Quantization"),
    ("evaluation_seconds", "Evaluation"),
    ("benchmark_seconds", "Benchmark"),
    ("final:benchmark_seconds", "Final timing"),
)
"""Stage wall times summed for the compute table (seconds; shown in hours)."""
FREE_TEXT_METRICS: tuple[tuple[str, str], ...] = (
    ("exact_match", "Exact match"),
    ("token_f1", "Token F1"),
    ("hallucination_rate", "Hallucination"),
    ("omission_rate", "Omission"),
    ("negation_rate", "Negation"),
)
CALIBRATION_METRICS: tuple[tuple[str, str], ...] = (
    ("ece", "ECE"),
    ("brier", "Brier"),
    ("aurc", "AURC"),
)
E9_METRICS: tuple[tuple[str, str], ...] = (
    (ACCURACY, "Accuracy"),
    (BALANCED_ACCURACY, "Balanced acc."),
    (SMALL_RECALL, "Small-finding recall"),
    (MASK_EFFECT, "Lesion-mask effect"),
)
PLOT_ARMS: tuple[tuple[str, str], ...] = (("E2", ""), ("E3:w4", ""), ("E7", PRIMARY_BUDGET))
"""Arms of the reliability and risk-coverage plots: the reference, 4-bit, ours at B2."""
GIB = 2**30


def appendix_arms() -> list[tuple[str, str]]:
    """Every arm the appendix tables list: unmatched arms, and B2 for budget-matched ones."""
    arms = [(a, "") for e in ("E0", "E1", "E2", "E3", "E4") for a in arm_labels(e)]
    arms += [(a, PRIMARY_BUDGET) for e in ("E5", "E6", "E7", "E8") for a in arm_labels(e)]
    return [*arms, ("E9", "")]


def _arm_cell(arm: str, budget: str) -> str:
    return latex_escape(f"{arm} @{budget}" if budget else arm)


def _per_metric_table(summary: pd.DataFrame, metrics: Sequence[tuple[str, str]]) -> str:
    look = lookup_summary(summary)
    rows = [
        [_arm_cell(arm, budget), *[fmt_mean_sd(look.get((arm, budget, m))) for m, _ in metrics]]
        for arm, budget in appendix_arms()
    ]
    return tabular(["Arm", *[label for _, label in metrics]], rows)


def per_finding_table(summary: pd.DataFrame, kind: str) -> str:
    """Sensitivity or specificity per finding (columns) and arm (rows), mean over seeds."""
    prefix = f"{kind}:"
    findings = sorted(
        {m[len(prefix) :] for m in summary["metric"] if str(m).startswith(prefix)}
        if not summary.empty
        else set()
    )
    look = lookup_summary(summary)
    if not findings:
        return tabular(["Arm", "Finding"], [[MISSING, MISSING]])
    rows = []
    for arm, budget in appendix_arms():
        cells = []
        for finding in findings:
            hit = look.get((arm, budget, f"{prefix}{finding}"))
            cells.append(fmt_value(hit["mean"], 2) if hit is not None else MISSING)
        rows.append([_arm_cell(arm, budget), *cells])
    return tabular(["Arm", *[latex_escape(f) for f in findings]], rows)


def e9_table(summary: pd.DataFrame) -> str:
    """E9 (the teacher at 4 bit) next to the teacher, the student and the 4-bit student."""
    look = lookup_summary(summary)
    rows = []
    for arm in ("E0", "E9", "E2", "E3:w4"):
        cells = [fmt_mean_sd(look.get((arm, "", m))) for m, _ in E9_METRICS]
        memory = look.get((arm, "", "final:peak_memory_bytes"))
        if memory is not None:
            memory = {"mean": memory["mean"] / GIB, "sd": memory.get("sd", math.nan) / GIB}
        rows.append([latex_escape(arm), *cells, fmt_mean_sd(memory, digits=2)])
    return tabular(["Arm", *[label for _, label in E9_METRICS], "Peak memory (GiB)"], rows)


def compute_table(rows: pd.DataFrame) -> str:
    """Wall-clock hours of every recorded stage, summed per experiment over all its runs.

    Development stages ran on the Kaggle T4s; "Final timing" is the rented GPU (RC-T1).
    """
    table = []
    totals = {m: 0.0 for m, _ in TIME_METRICS}
    for experiment in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
                       "A1", "A2", "A3", "A4", "A5", "A6"):  # fmt: skip
        if rows.empty:
            break
        mine = rows[rows["experiment"].astype(str).str.split(":").str[0] == experiment]
        hours = {
            m: float(mine.loc[mine["metric"] == m, "value"].sum()) / 3600 for m, _ in TIME_METRICS
        }
        if not any(hours.values()):
            continue
        for metric, value in hours.items():
            totals[metric] += value
        table.append([experiment, *[fmt_value(hours[m], 2) for m, _ in TIME_METRICS],
                      fmt_value(sum(hours.values()), 2)])  # fmt: skip
    if table:
        table.append(["Total", *[fmt_value(totals[m], 2) for m, _ in TIME_METRICS],
                      fmt_value(sum(totals.values()), 2)])  # fmt: skip
    else:
        table = [[MISSING] * (len(TIME_METRICS) + 2)]
    return tabular(["Experiment", *[label for _, label in TIME_METRICS], "Total"], table)


def selection_table(rows: pd.DataFrame) -> str:
    """Round 0 student checks (D05 amendment): one row per candidate."""
    from medcompress.selection.driver import PEAK_MEMORY_METRIC

    header = ["Candidate", "G1", "Params (B)", "Visual tokens", "Scoring s/probe",
              "Peak (GiB)", "LoRA step"]  # fmt: skip
    if rows.empty:
        return tabular(header, [[MISSING] * len(header)])
    r0 = rows[rows["experiment"].astype(str).str.startswith("R0:")]
    table = []
    for experiment in sorted(set(r0["experiment"])):
        mine = r0[r0["experiment"] == experiment]
        values = dict(zip(mine["metric"], mine["value"], strict=True))
        if "advance" in values:
            failed = [m.split(":", 1)[1] for m in values if str(m).startswith("gate_failed:")]
            verdict = "pass" if values["advance"] == 1.0 else "fail: " + ", ".join(failed)
        else:
            verdict = MISSING
        peak = values.get(PEAK_MEMORY_METRIC)
        table.append([
            latex_escape(experiment.split(":", 1)[1]),
            latex_escape(verdict),
            fmt_value(values.get("r0_params", math.nan) / 1e9, 2),
            fmt_value(values.get("r0_visual_tokens"), 0),
            fmt_value(values.get("r0_scoring_seconds_per_probe")),
            fmt_value(peak / GIB, 2) if peak is not None else MISSING,
            fmt_value(values.get("r0_lora_step_ok"), 0),
        ])  # fmt: skip
    return tabular(header, table or [[MISSING] * len(header)])


# ------------------------------------------------------------------ hyper-parameters

HYPERPARAMETERS: tuple[tuple[str, str, str, str], ...] = (
    ("Students (E1, E2)", "LoRA rank / alpha / dropout", "E2", "training.lora.rank|alpha|dropout"),
    ("Students (E1, E2)", "LoRA targets", "E2", "training.lora.target_modules"),
    ("Students (E1, E2)", "Learning rate, schedule, warm-up", "E2",
     "training.learning_rate|lr_schedule|warmup_steps"),
    ("Students (E1, E2)", "Micro-batch x accumulation", "E2",
     "training.batch_size|grad_accumulation"),
    ("Students (E1, E2)", "KD weight / temperature", "E2", "training.kd_weight|kd_temperature"),
    ("Students (E1, E2)", "Steps / training pool", "E2", "training.steps|data_subset"),
    ("Recovery (E5-E7, A1-A6)", "Steps / examples", "E7", "recovery.steps|data_subset"),
    ("Recovery (E5-E7, A1-A6)", "Learning rate / KD weight", "E7",
     "recovery.learning_rate|kd_weight"),
    ("Pruning", "Unit types", "E7", "pruning.units.types"),
    ("Pruning", "Channel group / calibration images", "E7",
     "pruning.units.mlp_channel_group_size|pruning.calibration.size"),
    ("Pruning", "Protected layers (first / last)", "E7",
     "pruning.removal.protected_layers.first|last"),
    ("Pruning", "Ranking / schedule", "E7",
     "pruning.removal.score_normalisation|schedule"),
    ("Pruning", "Ranking of A3 (random scores)", "A3", "pruning.removal.score_normalisation"),
    ("Clinical criterion", "Effect / weights / negatives per positive", "E7",
     "pruning.clinical.delta|weighting|negatives_per_positive"),
    ("Clinical criterion", "Scoring method", "E7", "pruning.clinical.ablation_method"),
    ("Quantization", "w4: backend / bits / group", "E3",
     "quantization.recipes.w4.backend|weight_bits|group_size"),
    ("Quantization", "w8: backend / bits / group", "E3",
     "quantization.recipes.w8.backend|weight_bits|group_size"),
    ("Quantization", "w8a8: backend / weight bits / activation bits", "E3",
     "quantization.recipes.w8a8.backend|weight_bits|activation_bits"),
    ("Quantization", "Calibration images", "E3", "quantization.calibration.size"),
    ("E4", "Quantizer / adapter", "E4", "quantization.qat.quantizer|adapter"),
    ("E4", "Distillation loss, alpha / beta / temperature", "E4",
     "quantization.distillation.loss|alpha|beta|temperature"),
    ("Evaluation", "Bootstrap resamples / ECE bins / CI level", "E2",
     "evaluation.bootstrap_resamples|calibration_bins|ci_level"),
    ("Evaluation", "Headline accuracy", "E2", "evaluation.headline_accuracy"),
    ("Evaluation", "Free-text answers / max new tokens", "E2",
     "evaluation.generation_samples|max_new_tokens"),
)  # fmt: skip
"""(group, setting, experiment config, keys). ``a.b.c|d`` reads ``a.b.c`` and ``a.b.d``; a
part with a dot is a full path."""


def _expand(spec: str) -> list[str]:
    first, *rest = spec.split("|")
    parent = first.rsplit(".", 1)[0]
    return [first, *[part if "." in part else f"{parent}.{part}" for part in rest]]


def _get(config: Mapping[str, Any], dotted: str) -> Any:
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _value(config: Mapping[str, Any], dotted: str) -> Any:
    """The setting as a run reads it: a generation limit given by the D04 rule is computed."""
    from medcompress.data.summary import resolve_max_new_tokens

    value = _get(config, dotted)
    return resolve_max_new_tokens(value) if dotted.endswith("max_new_tokens") else value


def _show(value: Any) -> str:
    from medcompress.config import find_todos

    if value is None:
        return "none"
    if find_todos({"v": value}):
        return MISSING
    if isinstance(value, list | tuple):
        return ", ".join(_show(v) for v in value)
    if isinstance(value, Mapping):
        return ", ".join(f"{k}: {_show(v)}" for k, v in value.items())
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def hyperparameter_table() -> str:
    """The settings of every arm, read from the experiment configs (open ones read ``--``)."""
    from medcompress.config import load_experiment

    configs: dict[str, Mapping[str, Any]] = {}
    rows = []
    for group, setting, experiment, spec in HYPERPARAMETERS:
        if experiment not in configs:
            configs[experiment] = load_experiment(experiment)[0]
        values = [_show(_value(configs[experiment], key)) for key in _expand(spec)]
        rows.append([latex_escape(group), latex_escape(setting), latex_escape(" / ".join(values))])
    return tabular(["Block", "Setting", "Value"], rows)


# ------------------------------------------------------------------- figures


def _series(summary: pd.DataFrame, arm: str, budget: str, prefix: str) -> list[tuple[int, float]]:
    if summary.empty:
        return []
    hit = summary[
        (summary["experiment"] == arm)
        & (summary["budget"].fillna("") == budget)
        & (summary["metric"].astype(str).str.startswith(prefix))
    ]
    return sorted(
        (int(str(m)[len(prefix) :]), float(v))
        for m, v in zip(hit["metric"], hit["mean"], strict=True)
    )


def reliability_figure(summary: pd.DataFrame, path: Path) -> Path:
    """Reliability diagram: accuracy against mean confidence per bin (seed means)."""
    from medcompress.evaluation.figures import (
        FAMILY_COLOURS,
        MUTED,
        family_of,
        new_figure,
        placeholder,
        save_figure,
    )

    lines = []
    for arm, budget in PLOT_ARMS:
        accuracy = dict(_series(summary, arm, budget, "reliability_accuracy:"))
        confidence = dict(_series(summary, arm, budget, "reliability_confidence:"))
        bins = sorted(set(accuracy) & set(confidence))
        if bins:
            lines.append((arm, [confidence[b] for b in bins], [accuracy[b] for b in bins]))
    if not lines:
        return placeholder(path, "Reliability diagram")
    fig, ax = new_figure()
    ax.plot([0.5, 1.0], [0.5, 1.0], color=MUTED, linewidth=1, linestyle="--")
    for arm, xs, ys in lines:
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5,
                color=FAMILY_COLOURS[family_of(arm)], label=arm)  # fmt: skip
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=7)
    return save_figure(fig, path)


def risk_coverage_figure(summary: pd.DataFrame, path: Path) -> Path:
    """Risk-coverage curves: error rate among the most confident answers (seed means)."""
    from medcompress.evaluation.figures import (
        FAMILY_COLOURS,
        family_of,
        new_figure,
        placeholder,
        save_figure,
    )

    lines = [(arm, _series(summary, arm, budget, "risk_at_coverage:")) for arm, budget in PLOT_ARMS]
    lines = [(arm, points) for arm, points in lines if points]
    if not lines:
        return placeholder(path, "Risk-coverage curves")
    fig, ax = new_figure()
    for arm, points in lines:
        ax.plot([p / 100 for p, _ in points], [r for _, r in points], marker="o", markersize=3,
                linewidth=1.5, color=FAMILY_COLOURS[family_of(arm)], label=arm)  # fmt: skip
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (error rate)")
    ax.legend(frameon=False, fontsize=7)
    return save_figure(fig, path)


def write_appendix(
    rows: pd.DataFrame, summary: pd.DataFrame, tables_dir: Path, figures_dir: Path
) -> list[Path]:
    """Write the appendix tables and figures; ``rows`` are all committed rows, ``summary``
    the test-split summary. Returns the paths written."""
    tables = {
        "appendix_sensitivity.tex": per_finding_table(summary, "sensitivity"),
        "appendix_specificity.tex": per_finding_table(summary, "specificity"),
        "appendix_calibration.tex": _per_metric_table(summary, CALIBRATION_METRICS),
        "appendix_free_text.tex": _per_metric_table(summary, FREE_TEXT_METRICS),
        "appendix_e9.tex": e9_table(summary),
        "appendix_compute.tex": compute_table(rows),
        "appendix_student_selection.tex": selection_table(rows),
        "appendix_hyperparameters.tex": hyperparameter_table(),
    }
    written = []
    for name, text in tables.items():
        path = Path(tables_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    written.append(reliability_figure(summary, Path(figures_dir) / "appendix_reliability.pdf"))
    written.append(risk_coverage_figure(summary, Path(figures_dir) / "appendix_risk_coverage.pdf"))
    return written
