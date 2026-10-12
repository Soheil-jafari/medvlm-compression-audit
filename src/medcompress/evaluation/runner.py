"""Runs the evaluation suites for one run and writes results rows.

Probe suites (one forward pass per question, no free generation) and free-text suites
(greedy generation) are scored on one split of the prepared data: the validation split
during development, the test split only after the pre-registration is frozen. Per-probe
outcomes are saved under ``paths.output_root`` (never committed) because paired
comparisons between arms need them; the summary rows go to ``metrics.csv``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from medcompress.evaluation import free_text, probe_suites
from medcompress.evaluation.probe_suites import Metric, ProbeOutcomes

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

EVALUATION_SUITES: tuple[str, ...] = (
    "clinical",
    "small_findings",
    "rare_findings",
    "image_reliance",
    "calibration",
    "hallucination",
    "language",
    "blank_image",
    "mask_audit",
)
"""``blank_image`` and ``mask_audit`` are the two halves of ``image_reliance`` (answers on blank
images; lesion-mask audit). Student selection uses them separately; they are not run together
with ``image_reliance``, which already contains both."""
PROBE_SUITES = frozenset(
    {
        "clinical",
        "small_findings",
        "rare_findings",
        "calibration",
        "image_reliance",
        "blank_image",
        "mask_audit",
    }
)
FREE_TEXT_SUITES = frozenset({"hallucination", "language"})
SPLITS = ("val", "test")
RUN_FILES_BY_SPLIT: Mapping[str, tuple[str, str]] = {
    "val": ("metrics.csv", "outcomes.npz"),
    "test": ("metrics_test.csv", "outcomes_test.npz"),
}
"""(results file, per-probe outcomes file) of each split. The test split has its own files, so
scoring it never overwrites the validation rows that Gate 2 and "best of E5/E6" are read from,
and the aggregation can keep the two splits apart (the metric names are the same)."""
PROBE_SAMPLE_SEED = 0
"""Fixed seed for capping probes per finding, so every arm and seed scores the same probes."""
_CHUNK = 32


@dataclass(frozen=True)
class EvalSettings:
    """Everything the suites need, read once from the config (None where a suite is off)."""

    suites: tuple[str, ...]
    split: str
    findings: tuple[str, ...]
    template: str | None
    display_names: Mapping[str, str]
    thresholds: tuple[float, ...] | None
    mask_findings: tuple[str, ...] | None
    groups: Mapping[str, tuple[str, ...]] | None
    pooling: str
    n_bins: int | None
    n_resamples: int | None
    alpha: float
    fill: int | None
    max_per_finding: int | None
    max_mask_probes: int | None
    max_new_tokens: int | None
    generation_samples: int | None


def _is_open(value: Any) -> bool:
    from medcompress.config import find_todos

    return value is None or bool(find_todos({"v": value}))


def resolve_settings(config: Mapping[str, Any]) -> EvalSettings:
    """Read the evaluation settings and check that the enabled suites have what they need.

    Raises:
        ConfigError: a setting an enabled suite needs is missing or still ``TODO``, a
            suite is unknown, or the test split is requested before the freeze.
    """
    from medcompress.config import ConfigError, load_config

    evaluation = config.get("evaluation", {})
    suites = tuple(evaluation.get("suites", EVALUATION_SUITES))
    unknown = [s for s in suites if s not in EVALUATION_SUITES]
    if unknown:
        raise ConfigError(f"unknown evaluation suites: {unknown}; use {list(EVALUATION_SUITES)}")
    split = str(evaluation.get("split", "val"))
    if split not in SPLITS:
        raise ConfigError(f"evaluation.split must be one of {SPLITS}, got {split!r}")
    if split == "test":
        table = load_config(config["components"]["finding_weights"])
        if _is_open(table.get("preregistration", {}).get("frozen_on")):
            raise ConfigError(
                "the test split is locked until the pre-registration is frozen "
                "(preregistration.frozen_on is still open)"
            )
    domain = load_config(config["components"]["domain"])["domain"]
    check = config.get("image_reliance_check", {})
    wanted = set(suites)
    open_keys: set[str] = set()

    def need(name: str, value: Any) -> Any:
        if _is_open(value):
            open_keys.add(name)
        return value

    findings = need("domain.findings", domain.get("findings"))
    template = need("domain.probe.template", domain.get("probe", {}).get("template"))
    if not wanted & PROBE_SUITES:
        template = None
    resamples = None
    if "image_reliance" in wanted and wanted & {"blank_image", "mask_audit"}:
        raise ConfigError("image_reliance already contains blank_image and mask_audit")
    if wanted & {"clinical", "small_findings", "image_reliance", "mask_audit"}:
        resamples = need("evaluation.bootstrap_resamples", evaluation.get("bootstrap_resamples"))
    thresholds = mask_findings = groups = n_bins = fill = max_new = limit = None
    if "small_findings" in wanted:
        thresholds = domain.get("small_findings", {}).get("thresholds")
        if not thresholds:
            thresholds = "TODO"  # an empty list is as open as a placeholder
        need("domain.small_findings.thresholds", thresholds)
    if wanted & {"small_findings", "image_reliance", "mask_audit"}:
        mask_findings = need("image_reliance_check.mask_findings", check.get("mask_findings"))
    if "rare_findings" in wanted:
        members = domain.get("prevalence_groups", {})
        groups = {g: tuple(members.get(g) or ()) for g in ("common", "medium", "rare")}
        if not any(groups.values()):
            open_keys.add("domain.prevalence_groups")
    if "calibration" in wanted:
        n_bins = need("evaluation.calibration_bins", evaluation.get("calibration_bins"))
    if wanted & {"image_reliance", "blank_image", "mask_audit"}:
        fill = need("image_reliance_check.fill", check.get("fill"))
    if wanted & FREE_TEXT_SUITES:
        max_new = need("evaluation.max_new_tokens", evaluation.get("max_new_tokens"))
        if "generation_samples" not in evaluation:
            open_keys.add("evaluation.generation_samples")
        elif evaluation["generation_samples"] is not None:  # null means every row
            limit = need("evaluation.generation_samples", evaluation["generation_samples"])
    if open_keys:
        raise ConfigError(f"evaluation settings still open: {', '.join(sorted(open_keys))}")
    findings = tuple(findings)
    if mask_findings is not None and not set(mask_findings) <= set(findings):
        raise ConfigError("image_reliance_check.mask_findings must be in domain.findings")
    if thresholds is not None and list(thresholds) != sorted(float(t) for t in thresholds):
        raise ConfigError("domain.small_findings.thresholds must be sorted")
    pooling = str(evaluation.get("rare_finding_pooling", "pooled"))
    return EvalSettings(
        suites=suites,
        split=split,
        findings=findings,
        template=template,
        display_names=dict(domain.get("probe", {}).get("display_names") or {}),
        thresholds=tuple(float(t) for t in thresholds) if thresholds is not None else None,
        mask_findings=tuple(mask_findings) if mask_findings is not None else None,
        groups=groups,
        pooling=pooling,
        n_bins=int(n_bins) if n_bins is not None else None,
        n_resamples=int(resamples) if resamples is not None else None,
        alpha=1.0 - float(evaluation.get("ci_level", 0.95)),
        fill=int(fill) if fill is not None else None,
        max_per_finding=check.get("max_probes_per_finding"),
        max_mask_probes=check.get("max_mask_probes"),
        max_new_tokens=int(max_new) if max_new is not None else None,
        generation_samples=int(limit) if limit is not None else None,
    )


# ------------------------------------------------------------------- the model


def load_trained_student(
    config: Mapping[str, Any], tag: str, seed: int, side: int, *, device_map: Any = None
) -> VLMWrapper:
    """The student with the adapter that training run ``tag`` (``E1``, ``E2``) saved, merged.

    The compression arms start from this model, so the weights they prune and evaluate
    are the merged ones.
    """
    from medcompress.config import ConfigError, load_config
    from medcompress.models.loading import ModelSpec, load_adapter, load_model

    student = config.get("model", {}).get("config")
    if _is_open(student):
        raise ConfigError("settings still open: model.config (the student, from S05b)")
    spec = ModelSpec.from_config(load_config(student))
    kwargs = {} if device_map is None else {"device_map": device_map}
    model = load_model(spec, allow_candidate=True, image_longest_side=side, **kwargs)
    adapter = Path(config["paths"]["checkpoint_root"]) / tag.replace(":", "/")
    return load_adapter(model, adapter / f"seed{seed}" / "adapter", merge=True)


def load_run_model(config: Mapping[str, Any], run: RunContext, side: int) -> VLMWrapper:
    """Load the model that ``run`` evaluates.

    E0 is the teacher; E1 and E2 are the student with the adapter their training run
    saved (merged into the weights, as the compression arms start from it). The pruned
    arms (E5-E7, A1-A6) reload E2 of the same seed and replay the removal that stage 05
    saved. The post-training quantized arms (E3, E8, E9) reload their parent and swap in the
    kernel modules of the saved codes (``compression/quantization/ptq.py``).
    """
    from medcompress.config import load_config
    from medcompress.models.loading import ModelSpec, load_model

    if run.experiment_id == "E0":
        spec = ModelSpec.from_config(load_config(config["components"]["teacher"]))
        return load_model(spec, image_longest_side=side)
    if run.experiment_id in ("E1", "E2"):
        return load_trained_student(config, run.experiment_tag, run.seed, side)
    from medcompress.registry import SCORE_UNITS, get_experiment

    if SCORE_UNITS in get_experiment(run.experiment_id).scripts:  # E5-E7, A1-A6
        from medcompress.compression.pruning.checkpoint import load_pruned
        from medcompress.compression.recovery import pruned_dir

        base = load_trained_student(config, "E2", run.seed, side)
        return load_pruned(base, pruned_dir(config, run))
    if run.experiment_id in ("E3", "E4", "E8", "E9"):
        from medcompress.compression.quantization.ptq import load_run_quantized

        return load_run_quantized(config, run, side)
    raise NotImplementedError(f"no model loader for {run.experiment_id}")


# ------------------------------------------------------------------- probes


def score_probes(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    settings: EvalSettings,
    data_root: Path,
    *,
    blank_fill: int | None = None,
) -> ProbeOutcomes:
    """Score one yes/no probe per (image, finding); the probe set does not depend on the seed.

    With ``blank_fill`` every probe is scored a second time on a uniform image of that grey
    level (``yes_blank``).
    """
    from medcompress.evaluation.image_reliance import (
        _open,
        blank_image,
        subsample_per_finding,
        unique_image_probes,
    )

    assert settings.template is not None
    probes = subsample_per_finding(
        unique_image_probes(
            samples, settings.findings, settings.template, display_names=settings.display_names
        ),
        settings.max_per_finding,
        seed=PROBE_SAMPLE_SEED,
    )
    if not probes:
        raise ValueError("No labelled probes on this split.")
    area = {s.sample_id: s.lesion_area for s in samples}
    yes, margin, blank = [], [], []
    for start in range(0, len(probes), _CHUNK):
        part = probes[start : start + _CHUNK]
        questions = [p.question for p in part]
        scores = model.score_yes_no([Path(data_root) / p.image_path for p in part], questions)
        yes += [s.predicted == "yes" for s in scores]
        margin += [s.logp_yes - s.logp_no for s in scores]
        if blank_fill is not None:
            blanks = [blank_image(_open(data_root, p.image_path), blank_fill) for p in part]
            blank += [s.predicted == "yes" for s in model.score_yes_no(blanks, questions)]
    return ProbeOutcomes(
        sample_ids=tuple(p.sample_id for p in probes),
        findings=tuple(p.finding for p in probes),
        truth=np.array([p.present for p in probes], dtype=bool),
        yes=np.array(yes, dtype=bool),
        margin=np.array(margin, dtype=float),
        area=np.array([area.get(p.sample_id) or math.nan for p in probes], dtype=float),
        has_mask=np.array([p.mask_path is not None for p in probes], dtype=bool),
        yes_blank=np.array(blank, dtype=bool) if blank_fill is not None else None,
    )


def metrics_from_summary(summary: Mapping[str, float]) -> list[Metric]:
    """Turn a dict with ``<name>_ci_low`` / ``<name>_ci_high`` entries into metrics."""
    out = []
    for name, value in summary.items():
        if name.endswith(("_ci_low", "_ci_high")):
            continue
        low, high = summary.get(f"{name}_ci_low"), summary.get(f"{name}_ci_high")
        out.append(Metric(name, float(value), low, high))
    return out


def _audit_kwargs(settings: EvalSettings, data_root: Path, seed: int) -> dict[str, Any]:
    assert settings.template is not None and settings.fill is not None
    return {
        "seed": seed,
        "data_root": data_root,
        "template": settings.template,
        "display_names": settings.display_names,
        "fill": settings.fill,
    }


def _mask_audit(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    settings: EvalSettings,
    data_root: Path,
    seed: int,
) -> tuple[list[Metric], Any]:
    from medcompress.evaluation.image_reliance import mask_audit

    assert settings.mask_findings is not None and settings.n_resamples is not None
    audit = mask_audit(
        model,
        samples,
        findings=list(settings.mask_findings),
        max_probes=settings.max_mask_probes,
        subsample_seed=PROBE_SAMPLE_SEED,
        **_audit_kwargs(settings, data_root, seed),
    )
    summary = audit.summary(n_resamples=settings.n_resamples, seed=seed)
    return metrics_from_summary(summary), audit


def _image_reliance(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    settings: EvalSettings,
    data_root: Path,
    seed: int,
) -> tuple[list[Metric], Any]:
    from medcompress.evaluation.image_reliance import reliance_metrics

    reliance = reliance_metrics(
        model,
        samples,
        findings=list(settings.findings),
        max_per_finding=settings.max_per_finding,
        subsample_seed=PROBE_SAMPLE_SEED,
        **_audit_kwargs(settings, data_root, seed),
    )
    audit_metrics, audit = _mask_audit(model, samples, settings, data_root, seed)
    return metrics_from_summary(reliance) + audit_metrics, audit


@dataclass
class EvaluationResult:
    """Metrics of the enabled suites plus the per-probe objects the callers may keep."""

    metrics: list[Metric]
    arrays: dict[str, np.ndarray]
    probes: ProbeOutcomes | None = None
    audit: Any = None


def evaluate_model(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    settings: EvalSettings,
    data_root: Path,
    seed: int,
) -> EvaluationResult:
    """Score the suites in ``settings.suites`` for one model on ``samples``.

    Used by the evaluation stage and by student selection (which keeps the per-probe
    answers for its paired bootstrap).
    """
    wanted = set(settings.suites)
    metrics: list[Metric] = []
    arrays: dict[str, np.ndarray] = {}
    out: ProbeOutcomes | None = None
    audit = None
    if wanted & (PROBE_SUITES - {"image_reliance", "mask_audit"}):
        out = score_probes(
            model,
            samples,
            settings,
            data_root,
            blank_fill=settings.fill if "blank_image" in wanted else None,
        )
        arrays |= {
            "probe_sample_ids": np.array(out.sample_ids),
            "probe_findings": np.array(out.findings),
            "probe_truth": out.truth,
            "probe_yes": out.yes,
            "probe_margin": out.margin,
            "probe_area": out.area,
            "probe_has_mask": out.has_mask,
        }
        if out.yes_blank is not None:
            arrays["probe_yes_blank"] = out.yes_blank
        n, alpha = settings.n_resamples, settings.alpha
        if "clinical" in wanted:
            metrics += probe_suites.clinical_metrics(
                out, settings.findings, n_resamples=n, alpha=alpha, seed=seed
            )
        if "small_findings" in wanted:
            metrics += probe_suites.small_finding_metrics(
                out,
                settings.thresholds,
                settings.mask_findings,
                n_resamples=n,
                alpha=alpha,
                seed=seed,
            )
        if "rare_findings" in wanted:
            metrics += probe_suites.rare_finding_metrics(
                out, settings.groups, pooling=settings.pooling
            )
        if "calibration" in wanted:
            metrics += probe_suites.calibration_metrics(out, n_bins=settings.n_bins)
        if "blank_image" in wanted:
            metrics += probe_suites.blank_image_metrics(out)
    if "image_reliance" in wanted:
        reliance, audit = _image_reliance(model, samples, settings, data_root, seed)
        metrics += reliance
    elif "mask_audit" in wanted:
        reliance, audit = _mask_audit(model, samples, settings, data_root, seed)
        metrics += reliance
    if audit is not None:
        arrays |= {
            "audit_sample_ids": np.array(audit.sample_ids),
            "audit_findings": np.array(audit.findings),
            "audit_original_yes": audit.original_yes,
            "audit_lesion_flip": audit.lesion_flip,
            "audit_random_flip": audit.random_flip,
        }
    if wanted & FREE_TEXT_SUITES:
        rows_qa = free_text.sample_qa_rows(samples, settings.generation_samples, PROBE_SAMPLE_SEED)
        answers = free_text.generate_answers(
            model, rows_qa, data_root, max_new_tokens=int(settings.max_new_tokens)
        )
        if "language" in wanted:
            metrics += free_text.language_metrics(answers, [r.answer for r in rows_qa])
        if "hallucination" in wanted:
            metrics += free_text.hallucination_metrics(
                answers, rows_qa, settings.findings, settings.display_names
            )
    return EvaluationResult(metrics, arrays, out, audit)


# --------------------------------------------------------------------- rows


def to_rows(metrics: Sequence[Metric], tag: str, budget: str, seed: int) -> list[dict[str, Any]]:
    """Results-schema rows. Undefined (NaN) metrics are left out and reported by the caller;
    a CI that does not contain its value (rare, tiny samples) is widened to contain it."""
    names = [m.name for m in metrics]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"metric names written twice: {duplicates}")
    rows = []
    for m in metrics:
        if math.isnan(m.value):
            continue
        row: dict[str, Any] = {
            "experiment": tag,
            "budget": budget,
            "seed": seed,
            "metric": m.name,
            "value": m.value,
        }
        if m.ci_low is not None and m.ci_high is not None:
            if not (math.isnan(m.ci_low) or math.isnan(m.ci_high)):
                row["ci_low"], row["ci_high"] = min(m.ci_low, m.value), max(m.ci_high, m.value)
        rows.append(row)
    return rows


def run_evaluation(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    model: VLMWrapper | None = None,
    samples: Sequence[VQASample] | None = None,
) -> None:
    """Stage entry point for ``scripts/06_evaluate.py``.

    Loads the model for ``run``, scores the enabled suites on ``evaluation.split`` of
    the prepared data and upserts the rows into ``<results_root>/<run dir>/metrics.csv``
    (validation) or ``metrics_test.csv`` (test; :data:`RUN_FILES_BY_SPLIT`). The per-probe
    outcomes (needed for paired comparisons) are written to ``<output_root>/<run dir>/``
    ``outcomes.npz`` or ``outcomes_test.npz``. The external test set (EndoBench) is added in
    the final-evaluation stage.

    Used by: all experiments (E0-E9, A1-A6).
    """
    from medcompress.config import ConfigError, load_config
    from medcompress.utils.io import upsert_results

    settings = resolve_settings(config)
    if run is None:
        raise ConfigError("--exp is required")
    side = config.get("inputs", {}).get("image_longest_side")
    if model is None and _is_open(side):
        raise ConfigError("settings still open: inputs.image_longest_side")
    paths = config["paths"]
    data_root = Path(paths["data_root"])
    if samples is None:
        from medcompress.data.prepared import load_prepared

        dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
        samples = load_prepared(data_root, dataset, Path(paths["private_root"]))[settings.split]
    if model is None:
        model = load_run_model(config, run, int(side))
    seed = run.seed
    started = time.perf_counter()
    result = evaluate_model(model, samples, settings, data_root, seed)
    seconds = time.perf_counter() - started  # compute used (appendix)
    metrics, arrays = [*result.metrics, Metric("evaluation_seconds", seconds)], result.arrays
    budget = run.budget or ""
    rows = to_rows(metrics, run.experiment_tag, budget, seed)
    undefined = sorted(m.name for m in metrics if math.isnan(m.value))
    metrics_name, outcomes_name = RUN_FILES_BY_SPLIT[settings.split]
    target = run.run_dir(paths["results_root"]) / metrics_name
    upsert_results(rows, target)
    outcomes = run.run_dir(paths["output_root"]) / outcomes_name
    if arrays:
        outcomes.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outcomes, **arrays)
    note = f"; undefined and left out: {', '.join(undefined)}" if undefined else ""
    print(f"wrote {len(rows)} rows to {target} (split {settings.split}){note}")
