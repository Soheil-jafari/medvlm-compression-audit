"""Matched-budget specification, measurement and search.

All compression arms are compared at the same MEASURED budget (peak memory or
latency, see ``configs/budgets.yaml``), not at a nominal parameter count. The
reference footprint is that of E2 measured with the protocol in the budgets
config. B1-B3 are placeholders until the team freezes them in
``docs/preregistration.md``.

Used by: E5, E6, E7, E8 (all budgets), A1-A6 (primary budget), and for reporting
where E3, E4 and E9 land relative to B1-B3.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.config import ConfigError, find_todos
from medcompress.evaluation.efficiency import EfficiencyReport
from medcompress.registry import BUDGET_IDS

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.evaluation.efficiency import EfficiencyProtocol
    from medcompress.models.base import VLMWrapper

BUDGET_METRICS: tuple[str, ...] = ("peak_memory", "latency")
_REPORT_FIELD = {"peak_memory": "peak_memory_bytes", "latency": "warm_latency_s"}


@dataclass(frozen=True)
class BudgetSpec:
    """One budget level (B1, B2 or B3)."""

    id: str
    metric: str
    target_fraction: float | None
    reference_experiment: str
    tolerance: float | None
    description: str = ""
    decoder_bytes_removed: float | None = None
    """The pre-registered rule (2026-11-08): remove this fraction of E2's decoder-layer bytes;
    the target is E2's measured peak minus it (RC-B0 measures both)."""

    @property
    def is_frozen(self) -> bool:
        """True once the level has a rule or a number (no longer a placeholder)."""
        return self.target_fraction is not None or self.decoder_bytes_removed is not None


def load_budget_specs(config: Mapping[str, Any]) -> dict[str, BudgetSpec]:
    """Parse the ``budgets`` section of ``configs/budgets.yaml`` into BudgetSpecs.

    Placeholder (``null``) target fractions are allowed; callers that need a
    number should check :attr:`BudgetSpec.is_frozen`.
    """
    section = config.get("budgets")
    if not isinstance(section, Mapping):
        raise ConfigError("Config has no 'budgets' mapping")
    metric = section.get("metric")
    reference = section.get("reference_experiment")
    tolerance = section.get("tolerance")
    levels = section.get("levels", {})
    specs: dict[str, BudgetSpec] = {}
    for budget_id in BUDGET_IDS:
        if budget_id not in levels:
            raise ConfigError(f"budgets.levels is missing {budget_id}")
        level = levels[budget_id] or {}
        fraction = level.get("target_fraction")
        removed = level.get("decoder_bytes_removed")
        specs[budget_id] = BudgetSpec(
            id=budget_id,
            metric=str(metric),
            target_fraction=None if fraction is None else float(fraction),
            reference_experiment=str(reference),
            tolerance=float(tolerance) if isinstance(tolerance, int | float) else None,
            description=str(level.get("description", "")),
            decoder_bytes_removed=None if removed is None else float(removed),
        )
    return specs


def measure_reference_footprint(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    protocol: EfficiencyProtocol,
    data_root: Path,
    **options: Any,
) -> EfficiencyReport:
    """Measure the E2 reference footprint that budgets are fractions of.

    Uses exactly the measurement protocol of ``configs/budgets.yaml`` (the same samples,
    batch size, warm-up and repeats as every other arm); the measurement itself is
    :func:`medcompress.evaluation.efficiency.measure_footprint`.

    Used by: E2 (reference), then E5-E8 and A1-A6 via :func:`search_matched_budget`.
    """
    from medcompress.evaluation.efficiency import measure_footprint

    return measure_footprint(model, samples, protocol, data_root, **options)


def read_efficiency_report(path: Path | str) -> EfficiencyReport:
    """Rebuild a report from the ``efficiency.csv`` of the benchmark stage."""
    values: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["value"])
    return EfficiencyReport(
        num_parameters=int(values["num_parameters"]) if "num_parameters" in values else None,
        peak_memory_bytes=(
            int(values["peak_memory_bytes"]) if "peak_memory_bytes" in values else None
        ),
        cold_latency_s=values.get("cold_latency_s"),
        warm_latency_s=values.get("warm_latency_s"),
        throughput_samples_per_s=values.get("throughput_samples_per_s"),
        energy_joules_per_sample=values.get("energy_joules_per_sample"),
    )


def reference_report(
    results_root: Path | str, reference_experiment: str, *, seed: int = 0
) -> EfficiencyReport:
    """The measured footprint of the reference arm (E2) that the budgets are fractions of.

    Every seed of E2 has the same architecture, so one seed (0) fixes the reference for all.
    """
    path = Path(results_root) / reference_experiment / "nobudget" / f"seed{seed}" / "efficiency.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"no reference footprint at {path}; run scripts/07_benchmark_efficiency.py "
            f"--exp {reference_experiment} --seed {seed} first"
        )
    return read_efficiency_report(path)


def metric_value(report: EfficiencyReport, metric: str) -> float:
    """Return the budget metric from a report: peak memory (bytes) or warm latency (s)."""
    if metric not in _REPORT_FIELD:
        raise ValueError(f"Budget metric must be one of {BUDGET_METRICS}, got {metric!r}")
    value = getattr(report, _REPORT_FIELD[metric])
    if value is None:
        raise ValueError(f"The report has no {_REPORT_FIELD[metric]}")
    return float(value)


LEVEL_TOLERANCE = 0.01
"""The unpruned model's first footprint in a budget search must match the reference within 1 %
(2026-11-15: a 0.59 GiB leftover lifted every footprint of the search by 0.63e9 bytes)."""


class BudgetLevelError(ValueError):
    """The search's measurement of the unpruned model is not on the reference's footing."""


def reference_protocol(results_root: Path | str, reference_experiment: str,
                       seed: int = 0) -> dict[str, Any] | None:  # fmt: skip
    """The measurement protocol recorded with the reference's footprint (None if not recorded)."""
    import json

    path = (Path(results_root) / reference_experiment / "nobudget" / f"seed{seed}"
            / "efficiency_setup.json")  # fmt: skip
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")).get("protocol") or {})


def same_protocol(protocol: Mapping[str, Any], recorded: Mapping[str, Any] | None) -> bool:
    """True when a measurement uses the reference's protocol (or the reference recorded none)."""
    if recorded is None:
        return True
    keys = ("batch_size", "max_new_tokens", "warmup_runs", "repeats", "samples")
    return all(str(protocol.get(k)) == str(recorded.get(k)) for k in keys if k in recorded)


def check_level(unpruned: float, reference: float, *, tolerance: float = LEVEL_TOLERANCE) -> float:
    """The relative gap of the unpruned model's measured footprint from the reference; raise
    :class:`BudgetLevelError` above ``tolerance`` (the search would aim at a shifted target)."""
    gap = (unpruned - reference) / reference
    if abs(gap) > tolerance:
        raise BudgetLevelError(
            f"the unpruned model measures {unpruned:.6g} in this process, the reference "
            f"{reference:.6g} ({gap:+.2%}, more than {tolerance:.0%}): every footprint of the "
            "search would be shifted by the same amount, so the search stops before its first "
            "step (memory left by an earlier load, or another protocol)"
        )
    return gap


def report_for_metric(metric: str, value: float) -> EfficiencyReport:
    """A report that holds only ``value`` as the budget metric (for a stored reference number)."""
    if metric not in _REPORT_FIELD:
        raise ValueError(f"Budget metric must be one of {BUDGET_METRICS}, got {metric!r}")
    return EfficiencyReport(
        **{_REPORT_FIELD[metric]: int(value) if metric == "peak_memory" else float(value)}
    )


def target_value(
    spec: BudgetSpec, reference: EfficiencyReport, *, decoder_bytes: float | None = None
) -> float:
    """Return the absolute target (bytes or seconds) for a budget level.

    With ``decoder_bytes_removed`` (the pre-registered rule): target = the reference peak
    memory minus that fraction of the reference's decoder-layer bytes (``decoder_bytes``, from
    RC-B0's ``decoder_layers_bytes`` row). Otherwise target = target_fraction * reference
    metric. Placeholder budgets are refused.

    Used by: E5-E8, A1-A6.
    """
    if spec.decoder_bytes_removed is not None:
        if spec.metric != "peak_memory":
            raise ValueError("the decoder-bytes rule needs the peak-memory metric")
        if decoder_bytes is None:
            raise ValueError(f"budget {spec.id}: the reference's decoder_layers_bytes is needed")
        return metric_value(reference, spec.metric) - spec.decoder_bytes_removed * decoder_bytes
    if spec.target_fraction is None:
        raise ValueError(f"Budget {spec.id} is a placeholder; freeze it first.")
    return spec.target_fraction * metric_value(reference, spec.metric)


def reference_decoder_bytes(
    results_root: Path | str, reference_experiment: str, *, seed: int = 0
) -> float:
    """The reference arm's ``decoder_layers_bytes`` (RC-B0's ``efficiency.csv``)."""
    path = Path(results_root) / reference_experiment / "nobudget" / f"seed{seed}" / "efficiency.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "decoder_layers_bytes":
                return float(row["value"])
    raise ValueError(f"{path} has no decoder_layers_bytes row: run RC-B0 (stage 07) first")


def within_budget(measured: EfficiencyReport, spec: BudgetSpec, target: float) -> bool:
    """Return True if a measured footprint meets the budget within tolerance.

    Matched means close to the target from either side:
    ``|measured - target| <= tolerance * target``. An arm far below the target is
    not "matched" (it was compressed more than its comparators).

    Used by: E5-E8, A1-A6; reporting for E3, E4, E9.
    """
    if spec.tolerance is None:
        raise ValueError(f"Budget {spec.id} has no tolerance; freeze it first.")
    value = metric_value(measured, spec.metric)
    return abs(value - target) <= spec.tolerance * target


def smallest_prefix_meeting_target(
    n_units: int,
    measure_prefix: Callable[[int], float],
    target: float,
    tolerance: float,
    *,
    max_steps: int | None = None,
) -> tuple[int, float]:
    """Return (k, footprint) for the smallest k whose footprint is <= target * (1 + tolerance).

    ``measure_prefix(k)`` removes the k lowest-ranked units and returns the
    **measured** footprint. Assumes the footprint does not increase with k
    (removing more never costs memory) and bisects over k in [0, n_units], so
    about log2(n_units) + 2 measurements are needed. Raises if even removing every
    unit does not reach the target. The caller checks the lower side of the
    tolerance band with :func:`within_budget` (units are discrete, so the best
    prefix can overshoot).
    """
    if n_units < 0 or tolerance < 0 or target <= 0:
        raise ValueError("Need n_units >= 0, tolerance >= 0 and target > 0.")
    ceiling = target * (1 + tolerance)
    cache: dict[int, float] = {}

    def measured(k: int) -> float:
        if k not in cache:
            cache[k] = float(measure_prefix(k))
        return cache[k]

    if measured(0) <= ceiling:
        return 0, measured(0)
    if measured(n_units) > ceiling:
        raise ValueError("Removing every candidate unit does not reach the budget.")
    low, high = 0, n_units  # measured(low) > ceiling >= measured(high)
    steps = 0
    while high - low > 1 and (max_steps is None or steps < max_steps):
        steps += 1  # max_steps: RC-SMOKE only; a real search runs to the end
        mid = (low + high) // 2
        if measured(mid) <= ceiling:
            high = mid
        else:
            low = mid
    return high, measured(high)


@dataclass(frozen=True)
class BudgetSearch:
    """The outcome of the search for one budget."""

    budget: str
    target: float
    tolerance: float
    count: int
    removal: tuple[PrunableUnit, ...]
    footprint: float
    within_tolerance: bool
    trace: tuple[tuple[int, float], ...]

    @property
    def shortfall(self) -> float:
        """Footprint relative to the target minus one (negative: compressed more than asked)."""
        return self.footprint / self.target - 1.0


def search_matched_budget(
    units: Sequence[PrunableUnit],
    ordered: Sequence[PrunableUnit],
    spec: BudgetSpec,
    target: float,
    measure_removal: Callable[[Sequence[PrunableUnit]], float],
    *,
    protected_layers: Sequence[int] = (),
    max_steps: int | None = None,
) -> BudgetSearch:
    """Find the smallest removal set (lowest-ranked first) that meets the budget.

    ``ordered`` is the removal order of the criterion over ``units`` (all candidates). The
    units that can be removed one after another without breaking the model are taken in that
    order (:func:`~medcompress.compression.pruning.criteria.common.feasible_sequence`), and
    the prefix length is bisected with the **measured** footprint of the model after the
    physical removal (``measure_removal``), not with a parameter count. The result is the
    shortest prefix at or below ``target * (1 + tolerance)``; units are discrete, so it can
    land below the band, which ``within_tolerance`` reports (the footprint is then smaller
    than the budget asks and the arm counts as outside the tolerance).

    Used by: E5, E6, E7 (every budget), A1-A6 (primary budget).
    """
    from medcompress.compression.pruning.criteria.common import feasible_sequence, removal_set

    if spec.tolerance is None:
        raise ConfigError(f"budget {spec.id} has no tolerance yet")
    accepted = feasible_sequence(units, ordered, protected_layers=protected_layers)
    trace: dict[int, float] = {}

    def measure_prefix(count: int) -> float:
        value = float(measure_removal(removal_set(accepted, count)))
        trace[count] = value
        return value

    count, footprint = smallest_prefix_meeting_target(
        len(accepted), measure_prefix, target, spec.tolerance, max_steps=max_steps
    )
    return BudgetSearch(
        budget=spec.id,
        target=target,
        tolerance=spec.tolerance,
        count=count,
        removal=tuple(removal_set(accepted, count)),
        footprint=footprint,
        within_tolerance=abs(footprint - target) <= spec.tolerance * target,
        trace=tuple(sorted(trace.items())),
    )


def uniform_groups(
    units: Sequence[PrunableUnit],
    scores: Mapping[str, float],
    protected_layers: Sequence[int],
    normalisation: str = "none",
) -> dict[int, list[PrunableUnit]]:
    """``{layer: its MLP channel groups, lowest score first}`` for every unprotected layer
    (``uniform_mlp`` allocation, amendment of 2026-11-17). Ties are broken by unit id. A layer's
    groups all have the same size, so dividing by the parameter count changes no order."""
    from medcompress.compression.pruning.criteria.common import normalised_scores
    from medcompress.compression.pruning.units import UnitType

    groups = [u for u in units if u.unit_type is UnitType.MLP_CHANNEL_GROUP
              and u.layer_index not in set(protected_layers)]  # fmt: skip
    if not groups:
        raise ValueError(
            "uniform_mlp allocation: no MLP channel group outside the protected layers"
        )
    missing = sorted(u.unit_id for u in groups if u.unit_id not in scores)
    if missing:
        raise ValueError(f"no score for {len(missing)} channel groups, e.g. {missing[:3]}")
    values = normalised_scores({u.unit_id: scores[u.unit_id] for u in groups}, groups,
                               normalisation)  # fmt: skip
    by_layer: dict[int, list[PrunableUnit]] = {}
    for unit in groups:
        by_layer.setdefault(unit.layer_index, []).append(unit)
    return {
        layer: sorted(members, key=lambda u: (values[u.unit_id], u.unit_id))
        for layer, members in sorted(by_layer.items())
    }


def uniform_removal(groups: Mapping[int, Sequence[PrunableUnit]], k: int) -> list[PrunableUnit]:
    """The ``k`` lowest-scored channel groups of every unprotected layer."""
    return [unit for layer in sorted(groups) for unit in groups[layer][:k]]


def uniform_max_k(groups: Mapping[int, Sequence[PrunableUnit]]) -> int:
    """The largest k that leaves every layer at least one channel group."""
    return min(len(members) for members in groups.values()) - 1


def search_uniform_budget(
    units: Sequence[PrunableUnit],
    scores: Mapping[str, float],
    spec: BudgetSpec,
    target: float,
    measure_removal: Callable[[Sequence[PrunableUnit]], float],
    *,
    protected_layers: Sequence[int],
    normalisation: str = "none",
    max_steps: int | None = None,
) -> BudgetSearch:
    """The budget search of the ``uniform_mlp`` allocation: the smallest k whose measured
    footprint is at or below ``target * (1 + tolerance)`` (the same rule as the global search,
    over k = 0 .. 85 instead of a prefix of one ranking). ``count`` is k."""
    if spec.tolerance is None:
        raise ConfigError(f"budget {spec.id} has no tolerance yet")
    groups = uniform_groups(units, scores, protected_layers, normalisation)
    trace: dict[int, float] = {}

    def measure_k(k: int) -> float:
        value = float(measure_removal(uniform_removal(groups, k)))
        trace[k] = value
        return value

    k, footprint = smallest_prefix_meeting_target(
        uniform_max_k(groups), measure_k, target, spec.tolerance, max_steps=max_steps
    )
    return BudgetSearch(
        budget=spec.id,
        target=target,
        tolerance=spec.tolerance,
        count=k,
        removal=tuple(uniform_removal(groups, k)),
        footprint=footprint,
        within_tolerance=abs(footprint - target) <= spec.tolerance * target,
        trace=tuple(sorted(trace.items())),
    )


def budget_spec_for_run(config: Mapping[str, Any], budget: str) -> BudgetSpec:
    """The frozen spec of ``budget`` (metric, target fraction and tolerance all set).

    Raises:
        ConfigError: a value is still ``TODO`` or ``null`` (all missing values are listed).
    """
    from medcompress.config import load_config

    section = load_config(config["components"]["budgets"])
    specs = load_budget_specs(section)
    if budget not in specs:
        raise ConfigError(f"unknown budget {budget!r}; known: {sorted(specs)}")
    spec, block = specs[budget], section["budgets"]
    open_keys = [
        f"budgets.{key}"
        for key in ("metric", "tolerance")
        if block.get(key) is None or find_todos({"v": block.get(key)})
    ]
    if spec.target_fraction is None and spec.decoder_bytes_removed is None:
        open_keys.append(f"budgets.levels.{budget}.decoder_bytes_removed")
    if open_keys:
        raise ConfigError(f"settings still open: {', '.join(open_keys)}")
    if spec.metric not in BUDGET_METRICS:
        raise ConfigError(f"budgets.metric must be one of {BUDGET_METRICS}, got {spec.metric!r}")
    value = spec.decoder_bytes_removed
    name = "decoder_bytes_removed"
    if value is None:
        value, name = spec.target_fraction, "target_fraction"
    if not 0.0 < float(value) < 1.0:
        raise ConfigError(f"budgets.levels.{budget}.{name} must be in (0, 1)")
    return spec


def release_memory() -> None:
    """Collect garbage and empty the CUDA cache (after the last reference to a model is gone)."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement of the stage
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_removal_measurer(
    fresh_model: Callable[[], VLMWrapper],
    samples: Sequence[VQASample],
    protocol: EfficiencyProtocol,
    data_root: Path,
    metric: str,
    **options: Any,
) -> Callable[[Sequence[PrunableUnit]], float]:
    """A function that returns the measured budget metric of the model after removing a set.

    Each call loads a fresh copy of the unpruned model (``fresh_model``), removes the units
    physically, measures the footprint under ``protocol`` and frees the copy
    (:func:`~medcompress.models.loading.release_model`) before the next call can load one:
    never two copies on the GPU.
    """
    from medcompress.compression.pruning.apply import remove_units
    from medcompress.evaluation.efficiency import measure_footprint
    from medcompress.models.loading import release_model

    def measure(removal: Sequence[PrunableUnit]) -> float:
        model = fresh_model()
        try:
            remove_units(model, list(removal))
            report = measure_footprint(model, samples, protocol, data_root, **options)
        finally:
            release_model(model)
            del model
            release_memory()
            from medcompress.utils.memory import report_leftover_gpu

            report_leftover_gpu(f"after the measurement of {len(removal)} removed units")
        return metric_value(report, metric)

    return measure
