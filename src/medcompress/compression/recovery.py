"""Shared recovery training after pruning, IDENTICAL for every pruning arm.

Fairness rule: E5 (all criteria), E6, E7 and A1-A6 use exactly the same recovery
recipe from ``configs/recovery.yaml``: LoRA fine-tuning plus offline distillation
from the cached E0 teacher outputs, same data subset, same number of steps, same
hyper-parameters. Only the pruned model differs. :func:`recovery_fingerprint` is
logged with every run so reviewers can verify this.

The stage (:func:`run_prune_and_recover`) reads the unit scores of stage 04, searches the
removal set whose measured footprint meets the budget, removes it physically, recovers
the model, and saves the result as a pruned checkpoint that stages 06 and 07 reload.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.config import ConfigError, RunContext, collect_open_settings, find_todos

if TYPE_CHECKING:
    from medcompress.compression.distillation.batching import TrainExample
    from medcompress.compression.distillation.offline_kd import TrainResult
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper
    from medcompress.models.hf import HFVLMWrapper

SCHEDULES: tuple[str, ...] = ("one_shot",)
NO_CHANNEL_GROUPS = 1 << 20
"""Group size stored in the manifest of an arm that never removes channel groups."""
PRUNED_SUBDIR = "pruned"


def recovery_fingerprint(recovery_config: Mapping[str, Any]) -> str:
    """Return a short stable hash of the recovery config (logged with every pruning run)."""
    payload = json.dumps(recovery_config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class RemovalSettings:
    """How the scores become a removal order (``pruning.removal`` of ``configs/pruning.yaml``)."""

    schedule: str
    score_normalisation: str
    protected_first: int
    protected_last: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RemovalSettings:
        """Read ``pruning.removal``; every value that is still ``TODO`` is listed.

        Raises:
            ConfigError: a value is open or invalid.
        """
        from medcompress.compression.pruning.criteria.common import SCORE_NORMALISATIONS

        removal = (config.get("pruning") or {}).get("removal") or {}
        protected = removal.get("protected_layers") or {}
        wanted = {
            "pruning.removal.schedule": removal.get("schedule"),
            "pruning.removal.score_normalisation": removal.get("score_normalisation"),
            "pruning.removal.protected_layers.first": protected.get("first"),
            "pruning.removal.protected_layers.last": protected.get("last"),
        }
        open_keys = [k for k, v in wanted.items() if v is None or find_todos({"v": v})]
        if open_keys:
            raise ConfigError(f"settings still open: {', '.join(open_keys)}")
        if removal["schedule"] not in SCHEDULES:
            raise ConfigError(
                f"pruning.removal.schedule must be one of {SCHEDULES} "
                f"(other schedules are not implemented), got {removal['schedule']!r}"
            )
        if removal["score_normalisation"] not in SCORE_NORMALISATIONS:
            raise ConfigError(f"pruning.removal.score_normalisation must be {SCORE_NORMALISATIONS}")
        first, last = protected["first"], protected["last"]
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (first, last)):
            raise ConfigError("pruning.removal.protected_layers values must be integers >= 0")
        return cls(str(removal["schedule"]), str(removal["score_normalisation"]), first, last)


def plan_removal(
    units: Sequence[PrunableUnit], scores: Mapping[str, float], settings: RemovalSettings
) -> tuple[list[PrunableUnit], list[int]]:
    """The removal order of the candidates and the protected layers.

    Returns ``(ordered, protected)``: the units from least to most important (whole layers
    that are protected are left out) and the indices of the protected layers.
    """
    from medcompress.compression.pruning.criteria.common import removal_order
    from medcompress.compression.pruning.criteria.output_similarity import protected_layer_indices
    from medcompress.compression.pruning.units import UnitType

    n_layers = len({u.layer_index for u in units if u.unit_type is UnitType.DECODER_LAYER})
    protected = (
        protected_layer_indices(n_layers, settings.protected_first, settings.protected_last)
        if n_layers
        else []
    )
    ordered = removal_order(
        scores, units, normalisation=settings.score_normalisation, protected_layers=protected
    )
    return ordered, protected


def pruned_dir(config: Mapping[str, Any], run: RunContext) -> Path:
    """Where the pruned and recovered model of ``run`` is saved (not committed)."""
    if run.budget is None:
        raise ConfigError("a pruned model belongs to a budget (--budget)")
    tag = run.experiment_tag.replace(":", "/")
    base = Path(config["paths"]["checkpoint_root"]) / tag / run.budget
    return base / f"seed{run.seed}" / PRUNED_SUBDIR


def run_recovery(
    pruned: HFVLMWrapper,
    examples: Sequence[TrainExample],
    recovery_config: Mapping[str, Any],
    *,
    seed: int,
) -> TrainResult:
    """Run the shared LoRA + offline-KD recovery on a pruned model and merge the adapter.

    ``examples`` are the training examples with their cached teacher records; the data
    subset of the recipe is drawn here, from the seed alone, so it is the same for every arm.

    Used by: E5, E6, E7, A1-A6 (3 seeds each).
    """
    from medcompress.compression.distillation.offline_kd import (
        TrainConfig,
        select_subset,
        train_offline_kd,
    )
    from medcompress.compression.lora import merge_adapter

    if recovery_config.get("method") not in (None, "lora_plus_offline_kd"):
        raise ConfigError(f"unknown recovery method {recovery_config.get('method')!r}")
    cfg = TrainConfig.from_config({**recovery_config, "mode": "offline_kd"})
    n = None if cfg.data_subset == "all" else int(cfg.data_subset)
    chosen = [examples[i] for i in select_subset([e.sample_id for e in examples], n, seed)]
    result = train_offline_kd(pruned, chosen, cfg, seed=seed)
    merge_adapter(pruned)
    return result


@dataclass(frozen=True)
class PruneSettings:
    """Everything the stage needs from the config, read together so open values show at once."""

    removal: RemovalSettings
    scoring: Any
    spec: Any
    protocol: Any


def read_prune_settings(
    config: Mapping[str, Any], run: RunContext, *, need_protocol: bool = True
) -> PruneSettings:
    """Read the pruning, budget, recovery and measurement settings for ``run``.

    Raises:
        ConfigError: one error that lists every value that is still open.
    """
    from medcompress.budgets import budget_spec_for_run
    from medcompress.compression.distillation.offline_kd import TrainConfig
    from medcompress.compression.pruning.stage import ScoringSettings
    from medcompress.config import load_config
    from medcompress.evaluation.efficiency import EfficiencyProtocol

    held: dict[str, Any] = {"protocol": None}
    budgets = load_config(config["components"]["budgets"])["budgets"]

    def protocol() -> None:
        if need_protocol:
            held["protocol"] = EfficiencyProtocol.from_config(budgets.get("measurement") or {})

    collect_open_settings(
        lambda: held.update(removal=RemovalSettings.from_config(config)),
        lambda: held.update(scoring=ScoringSettings.from_config(config, run.variant)),
        lambda: held.update(spec=budget_spec_for_run(config, run.budget or "")),
        lambda: TrainConfig.from_config(
            {**dict(config.get("recovery") or {}), "mode": "offline_kd"}
        ),
        protocol,
    )
    return PruneSettings(held["removal"], held["scoring"], held["spec"], held["protocol"])


def run_prune_and_recover(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    student_factory: Callable[[], VLMWrapper] | None = None,
    train: Sequence[VQASample] | None = None,
    records: Mapping[str, Any] | None = None,
    measure: Callable[[Sequence[PrunableUnit]], float] | None = None,
    reference_value: float | None = None,
) -> None:
    """Stage entry point for ``scripts/05_prune_and_recover.py``.

    Reads ``unit_scores.csv`` from stage 04, finds the shortest prefix of the removal order
    whose measured footprint meets the budget
    (:func:`medcompress.budgets.search_matched_budget`), removes the units physically,
    runs :func:`run_recovery` and saves the model. Writes the removal set
    (``removal.json``), the search trace (``budget_search.csv``) and the numbers of the run
    (``metrics.csv``) to the run directory; the mask audit and the other suites are part
    of stage 06.

    The keyword arguments replace what the stage loads itself (tests, other front ends):
    ``student_factory`` gives a fresh unpruned E2 student, ``train`` the training split,
    ``records`` the cached teacher outputs by sample id, ``measure`` the measured metric of
    a removal set and ``reference_value`` the metric of E2.

    Used by: E5, E6, E7, A1-A6.
    """
    from medcompress.budgets import report_for_metric, search_matched_budget, target_value
    from medcompress.compression.distillation.offline_kd import examples_from_samples
    from medcompress.compression.pruning.apply import remove_units, verify_physical_removal
    from medcompress.compression.pruning.checkpoint import save_pruned
    from medcompress.compression.pruning.criteria.common import read_unit_scores
    from medcompress.compression.pruning.stage import scores_path
    from medcompress.compression.pruning.units import UnitType, enumerate_units
    from medcompress.evaluation.efficiency import count_parameters

    if run is None or run.budget is None:
        raise ConfigError("--exp and --budget are required")
    settings = read_prune_settings(config, run, need_protocol=measure is None)
    scoring, spec = settings.scoring, settings.spec
    paths = config["paths"]
    scores_file = scores_path(config, run)
    if not scores_file.is_file():
        raise ConfigError(f"no unit scores at {scores_file}; run stage 04 first")
    scores = read_unit_scores(scores_file)
    factory = student_factory or _student_factory(config, run)
    data_root = Path(paths["data_root"])

    base = factory()
    units = enumerate_units(
        base,
        [UnitType(t) for t in scoring.unit_types],
        mlp_channel_group_size=scoring.mlp_channel_group_size,
    )
    parameters_before = count_parameters(base)
    del base
    unknown = sorted({u.unit_id for u in units} - set(scores))
    if unknown:
        raise ValueError(f"unit scores lack {len(unknown)} of the units, e.g. {unknown[:3]}")
    ordered, protected = plan_removal(units, scores, settings.removal)

    if reference_value is None:
        from medcompress.budgets import reference_report

        reference = reference_report(paths["results_root"], spec.reference_experiment)
    else:
        reference = report_for_metric(spec.metric, reference_value)
    target = target_value(spec, reference)
    if measure is None:
        measure = _default_measure(config, factory, settings, data_root)
    search = search_matched_budget(
        units, ordered, spec, target, measure, protected_layers=protected
    )
    if search.count == 0:
        raise ValueError(f"{run.budget} asks for no compression: E2 already meets the target")

    model = factory()
    remove_units(model, list(search.removal))
    check = verify_physical_removal(parameters_before, model, search.removal)
    if not check["ok"]:
        raise ValueError(f"the removal did not delete the weights it should: {check}")
    if train is None:
        from medcompress.compression.pruning.stage import load_train_samples

        train = load_train_samples(config)
    if records is None:
        from medcompress.models.teacher_cache import load_teacher_cache

        cache = Path(paths["cache_root"]) / "teacher"
        records = {r.sample_id: r for r in load_teacher_cache(cache)}
    examples = examples_from_samples(train, data_root, records=records)
    recovery = dict(config.get("recovery") or {})
    result = run_recovery(model, examples, recovery, seed=run.seed)  # type: ignore[arg-type]
    directory = save_pruned(
        model,
        pruned_dir(config, run),
        [list(search.removal)],
        mlp_channel_group_size=scoring.mlp_channel_group_size or NO_CHANNEL_GROUPS,
    )
    _write_outputs(config, run, settings, search, check, result, recovery)
    print(
        f"removed {len(search.removal)} units for {run.budget}: footprint {search.footprint:.4g} "
        f"vs target {search.target:.4g} "
        f"({'within' if search.within_tolerance else 'OUTSIDE'} tolerance); saved {directory}"
    )


def _student_factory(config: Mapping[str, Any], run: RunContext) -> Callable[[], VLMWrapper]:
    """Loads a fresh unpruned E2 student of the run's seed (adapter merged)."""
    from medcompress.evaluation.runner import load_trained_student

    side = config.get("inputs", {}).get("image_longest_side")
    if side is None or find_todos({"v": side}):
        raise ConfigError("settings still open: inputs.image_longest_side")
    return lambda: load_trained_student(config, "E2", run.seed, int(side))


def _default_measure(
    config: Mapping[str, Any],
    factory: Callable[[], VLMWrapper],
    settings: PruneSettings,
    data_root: Path,
) -> Callable[[Sequence[PrunableUnit]], float]:
    """Measures candidate removal sets with the protocol on the shared timing samples."""
    from medcompress.budgets import make_removal_measurer
    from medcompress.config import load_config
    from medcompress.data.prepared import load_prepared
    from medcompress.evaluation.efficiency import timing_samples

    paths = config["paths"]
    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    pool = load_prepared(data_root, dataset, Path(paths["private_root"]))["val"]
    return make_removal_measurer(
        factory,
        timing_samples(pool, settings.protocol),
        settings.protocol,
        data_root,
        settings.spec.metric,
    )


def _write_outputs(
    config: Mapping[str, Any],
    run: RunContext,
    settings: PruneSettings,
    search: Any,
    check: Mapping[str, Any],
    result: TrainResult,
    recovery: Mapping[str, Any],
) -> None:
    """Write ``removal.json``, ``budget_search.csv`` and the metrics rows of the run."""
    import csv

    from medcompress.compression.pruning.units import UnitType
    from medcompress.utils.io import upsert_results, write_json

    run_dir = run.run_dir(config["paths"]["results_root"])
    write_json(
        {
            "units": [u.unit_id for u in search.removal],
            "recovery_fingerprint": recovery_fingerprint(recovery),
            "criterion": settings.scoring.criterion,
            "removal_settings": {
                "schedule": settings.removal.schedule,
                "score_normalisation": settings.removal.score_normalisation,
                "protected_first": settings.removal.protected_first,
                "protected_last": settings.removal.protected_last,
            },
        },
        run_dir / "removal.json",
    )
    with (run_dir / "budget_search.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accepted_units_removed", "measured_footprint"])
        writer.writerows(search.trace)
    kinds = {t: sum(1 for u in search.removal if u.unit_type is t) for t in UnitType}
    numbers: list[tuple[str, float]] = [
        ("budget_target", search.target),
        ("budget_footprint", search.footprint),
        ("budget_relative_gap", search.shortfall),
        ("budget_within_tolerance", float(search.within_tolerance)),
        ("budget_measurements", len(search.trace)),
        ("removed_units", len(search.removal)),
        ("removed_layers", kinds[UnitType.DECODER_LAYER]),
        ("removed_kv_groups", kinds[UnitType.ATTENTION_HEAD]),
        ("removed_channel_groups", kinds[UnitType.MLP_CHANNEL_GROUP]),
        ("parameters_before", check["before"]),
        ("parameters_after", check["after"]),
        ("recovery_steps", result.steps),
        ("recovery_loss_final", result.final_loss()),
        ("recovery_seconds", result.seconds),
    ]
    if result.peak_gpu_bytes is not None:
        numbers.append(("recovery_peak_gpu_bytes", result.peak_gpu_bytes))
    upsert_results(
        [
            {
                "experiment": run.experiment_tag,
                "budget": run.budget,
                "seed": run.seed,
                "metric": name,
                "value": float(value),
            }
            for name, value in numbers
        ],
        run_dir / "metrics.csv",
    )
