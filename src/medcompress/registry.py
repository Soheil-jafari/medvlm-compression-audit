"""Experiment registry: the single source of truth for experiment IDs.

Every experiment ID used in configs, scripts, results and the paper must be
registered here. ``tests/test_configs.py`` checks that each registered ID has a
config under ``configs/experiments/`` whose ``experiment`` section agrees with
this registry (parent, variant_of, requires, budgets, variants, scripts).

Field meanings
--------------
parent:     experiment whose model this one starts from (``None`` = a pretrained
            checkpoint).
variant_of: for ablations, the experiment they modify (always E7 here).
requires:   other experiments whose artefacts are needed (e.g. the E0 teacher cache).
variants:   sub-runs reported as ``<ID>:<variant>`` (e.g. ``E3:w4``, ``E5:taylor``).
budgets:    matched budgets this experiment is run at (empty = not budget-matched;
            footprint is still measured and reported).
"""

from __future__ import annotations

from dataclasses import dataclass

BUDGET_IDS: tuple[str, ...] = ("B1", "B2", "B3")
PRIMARY_BUDGET = "B2"
TRAINING_SEEDS: tuple[int, ...] = (0, 1, 2)

S = "scripts/"
CHECK_TEACHER = S + "00_check_teacher_image_reliance.py"
CACHE_TEACHER = S + "01_cache_teacher_outputs.py"
TRAIN_STUDENT = S + "02_train_student.py"
QUANTIZE = S + "03_quantize.py"
SCORE_UNITS = S + "04_score_pruning_units.py"
PRUNE_RECOVER = S + "05_prune_and_recover.py"
EVALUATE = S + "06_evaluate.py"
BENCHMARK = S + "07_benchmark_efficiency.py"
AGGREGATE = S + "08_aggregate_results.py"

ALL_SCRIPTS: tuple[str, ...] = (
    CHECK_TEACHER,
    CACHE_TEACHER,
    TRAIN_STUDENT,
    QUANTIZE,
    SCORE_UNITS,
    PRUNE_RECOVER,
    EVALUATE,
    BENCHMARK,
    AGGREGATE,
)

_PRUNE_PIPELINE = (SCORE_UNITS, PRUNE_RECOVER, EVALUATE, BENCHMARK)
_PRUNE_STAGES = ("score_units", "prune", "recover", "mask_audit", "evaluate", "benchmark")
_RUN_OUTPUTS = (
    "results/{id}/.../metrics.csv",
    "results/{id}/.../efficiency.csv",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """Static description of one experiment ID."""

    id: str
    title: str
    description: str
    kind: str  # "main" or "ablation"
    parent: str | None
    stages: tuple[str, ...]
    scripts: tuple[str, ...]
    outputs: tuple[str, ...]
    variant_of: str | None = None
    requires: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    budgets: tuple[str, ...] = ()
    trains: bool = False

    @property
    def uses_budgets(self) -> bool:
        return bool(self.budgets)


def _outputs(exp_id: str, *extra: str) -> tuple[str, ...]:
    return tuple(o.format(id=exp_id) for o in _RUN_OUTPUTS) + extra


_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        id="E0",
        title="Teacher, full precision",
        description="Reference teacher model. Checks image reliance and caches its outputs.",
        kind="main",
        parent=None,
        stages=("check_image_reliance", "cache_teacher_outputs", "evaluate", "benchmark"),
        scripts=(CHECK_TEACHER, CACHE_TEACHER, EVALUATE, BENCHMARK),
        outputs=_outputs("E0", "cache/teacher/ (answers + top-k probabilities, not committed)"),
    ),
    ExperimentSpec(
        id="E1",
        title="Student, fine-tuned on correct answers",
        description="Student fine-tuned on ground-truth answers only, full precision.",
        kind="main",
        parent=None,
        stages=("train_sft", "evaluate", "benchmark"),
        scripts=(TRAIN_STUDENT, EVALUATE, BENCHMARK),
        outputs=_outputs("E1", "checkpoints/E1/ (LoRA adapters, not committed)"),
        trains=True,
    ),
    ExperimentSpec(
        id="E2",
        title="Student + offline distillation",
        description="Student trained with offline distillation from cached E0 outputs. "
        "Starting point of every compression arm.",
        kind="main",
        parent=None,
        requires=("E0",),
        stages=("train_offline_kd", "evaluate", "benchmark"),
        scripts=(TRAIN_STUDENT, EVALUATE, BENCHMARK),
        outputs=_outputs("E2", "checkpoints/E2/ (not committed)"),
        trains=True,
    ),
    ExperimentSpec(
        id="E3",
        title="E2 + post-training quantization",
        description="Weight-only 8-bit and 4-bit (GPTQ/AWQ); W8A8 where supported.",
        kind="main",
        parent="E2",
        variants=("w8", "w4", "w8a8"),
        stages=("quantize", "evaluate", "benchmark"),
        scripts=(QUANTIZE, EVALUATE, BENCHMARK),
        outputs=_outputs("E3", "checkpoints/E3/ (not committed)"),
    ),
    ExperimentSpec(
        id="E4",
        title="Student + quantization-aware distillation",
        description="Student trained with quantization-aware distillation (GRACE-style).",
        kind="main",
        parent=None,
        requires=("E0",),
        stages=("train_qakd", "evaluate", "benchmark"),
        scripts=(TRAIN_STUDENT, EVALUATE, BENCHMARK),
        outputs=_outputs("E4", "checkpoints/E4/ (not committed)"),
        trains=True,
    ),
    ExperimentSpec(
        id="E5",
        title="E2 + generic structured pruning + recovery",
        description="Magnitude, activation (Wanda-style) and gradient (Taylor) criteria.",
        kind="main",
        parent="E2",
        requires=("E0",),
        variants=("magnitude", "activation", "taylor"),
        budgets=BUDGET_IDS,
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("E5", "results/E5/.../unit_scores.csv"),
        trains=True,
    ),
    ExperimentSpec(
        id="E6",
        title="E2 + output-similarity layer pruning + recovery",
        description="Layer pruning by output similarity (Khan et al. style).",
        kind="main",
        parent="E2",
        requires=("E0",),
        budgets=BUDGET_IDS,
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("E6", "results/E6/.../unit_scores.csv"),
        trains=True,
    ),
    ExperimentSpec(
        id="E7",
        title="E2 + clinical pruning criterion (ours) + recovery",
        description="Clinically weighted structured pruning from finding probes.",
        kind="main",
        parent="E2",
        requires=("E0",),
        budgets=BUDGET_IDS,
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("E7", "results/E7/.../unit_scores.csv"),
        trains=True,
    ),
    ExperimentSpec(
        id="E8",
        title="E7 + 4-bit post-training quantization",
        description="Clinical pruning followed by 4-bit PTQ.",
        kind="main",
        parent="E7",
        budgets=BUDGET_IDS,
        stages=("quantize", "evaluate", "benchmark"),
        scripts=(QUANTIZE, EVALUATE, BENCHMARK),
        outputs=_outputs("E8"),
    ),
    ExperimentSpec(
        id="E9",
        title="Teacher quantized to 4-bit (supplementary)",
        description="Teacher at 4-bit, compared at matched memory.",
        kind="main",
        parent="E0",
        stages=("quantize", "evaluate", "benchmark"),
        scripts=(QUANTIZE, EVALUATE, BENCHMARK),
        outputs=_outputs("E9"),
    ),
    ExperimentSpec(
        id="A1",
        title="Ablation: unweighted criterion (w_c = 1)",
        description="Clinical criterion with uniform finding weights.",
        kind="ablation",
        parent="E2",
        variant_of="E7",
        requires=("E0",),
        budgets=(PRIMARY_BUDGET,),
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("A1"),
        trains=True,
    ),
    ExperimentSpec(
        id="A2",
        title="Ablation: rarity-only weights",
        description="Clinical criterion with capped inverse prevalence only (no tier).",
        kind="ablation",
        parent="E2",
        variant_of="E7",
        requires=("E0",),
        budgets=(PRIMARY_BUDGET,),
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("A2"),
        trains=True,
    ),
    ExperimentSpec(
        id="A3",
        title="Ablation: random removal",
        description="Random unit removal at the same budget (lower bound).",
        kind="ablation",
        parent="E2",
        variant_of="E7",
        requires=("E0",),
        budgets=(PRIMARY_BUDGET,),
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("A3"),
        trains=True,
    ),
    ExperimentSpec(
        id="A4",
        title="Ablation: calibration set size",
        description="Clinical criterion with 32 / 64 / 128 / 256 calibration images.",
        kind="ablation",
        parent="E2",
        variant_of="E7",
        requires=("E0",),
        variants=("n32", "n64", "n128", "n256"),
        budgets=(PRIMARY_BUDGET,),
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("A4"),
        trains=True,
    ),
    ExperimentSpec(
        id="A5",
        title="Ablation: Delta definition",
        description="Correct-answer log-probability drop vs prediction change (KL).",
        kind="ablation",
        parent="E2",
        variant_of="E7",
        requires=("E0",),
        variants=("logprob_drop", "kl"),
        budgets=(PRIMARY_BUDGET,),
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("A5"),
        trains=True,
    ),
    ExperimentSpec(
        id="A6",
        title="Ablation: unit granularity",
        description="Decoder layers only vs heads + MLP channel groups.",
        kind="ablation",
        parent="E2",
        variant_of="E7",
        requires=("E0",),
        variants=("layers_only", "heads_channels"),
        budgets=(PRIMARY_BUDGET,),
        stages=_PRUNE_STAGES,
        scripts=_PRUNE_PIPELINE,
        outputs=_outputs("A6"),
        trains=True,
    ),
)

REGISTRY: dict[str, ExperimentSpec] = {spec.id: spec for spec in _SPECS}

# Pre-registered primary endpoints (see docs/preregistration.md).
PRIMARY_ENDPOINTS: tuple[str, ...] = ("small_finding_recall", "lesion_mask_effect")
PRIMARY_COMPARISON = {"ours": "E7", "against_best_of": ("E5", "E6"), "budget": PRIMARY_BUDGET}


def list_experiments(kind: str | None = None) -> list[ExperimentSpec]:
    """Return registered experiments in ID order, optionally filtered by kind."""
    return [spec for spec in _SPECS if kind is None or spec.kind == kind]


def experiment_ids(kind: str | None = None) -> tuple[str, ...]:
    """Return registered experiment IDs, optionally filtered by kind."""
    return tuple(spec.id for spec in list_experiments(kind))


def get_experiment(experiment_id: str) -> ExperimentSpec:
    """Return the spec for ``experiment_id`` or raise ``KeyError`` with valid IDs."""
    try:
        return REGISTRY[experiment_id]
    except KeyError:
        valid = ", ".join(REGISTRY)
        raise KeyError(f"Unknown experiment ID {experiment_id!r}. Valid IDs: {valid}") from None


def validate_run(
    experiment_id: str,
    budget: str | None = None,
    variant: str | None = None,
    *,
    require_budget: bool = True,
    require_variant: bool = False,
) -> ExperimentSpec:
    """Check that an (experiment, budget, variant) combination is allowed.

    Args:
        require_budget: If False, a budget-matched experiment may omit the budget
            (e.g. unit scoring, which is shared across budgets).
        require_variant: If True, experiments with variants must name one.

    Raises:
        KeyError: unknown experiment ID.
        ValueError: budget or variant not valid for this experiment.
    """
    spec = get_experiment(experiment_id)
    if budget is not None and budget not in BUDGET_IDS:
        raise ValueError(f"Unknown budget {budget!r}. Valid budgets: {', '.join(BUDGET_IDS)}")
    if require_budget and spec.uses_budgets and budget is None:
        raise ValueError(f"{experiment_id} requires --budget (one of {', '.join(spec.budgets)})")
    if budget is not None and budget not in spec.budgets:
        allowed = ", ".join(spec.budgets) or "none (not budget-matched)"
        raise ValueError(f"{experiment_id} does not run at budget {budget}. Allowed: {allowed}")
    if require_variant and spec.variants and variant is None:
        raise ValueError(f"{experiment_id} requires --variant (one of {', '.join(spec.variants)})")
    if variant is not None and variant not in spec.variants:
        allowed = ", ".join(spec.variants) or "none"
        raise ValueError(f"{experiment_id} has no variant {variant!r}. Allowed: {allowed}")
    return spec
