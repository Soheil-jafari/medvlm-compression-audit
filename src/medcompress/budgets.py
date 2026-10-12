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

    @property
    def is_frozen(self) -> bool:
        """True once a numeric target fraction has been set (no longer a placeholder)."""
        return self.target_fraction is not None


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
        specs[budget_id] = BudgetSpec(
            id=budget_id,
            metric=str(metric),
            target_fraction=None if fraction is None else float(fraction),
            reference_experiment=str(reference),
            tolerance=float(tolerance) if isinstance(tolerance, int | float) else None,
            description=str(level.get("description", "")),
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


def report_for_metric(metric: str, value: float) -> EfficiencyReport:
    """A report that holds only ``value`` as the budget metric (for a stored reference number)."""
    if metric not in _REPORT_FIELD:
        raise ValueError(f"Budget metric must be one of {BUDGET_METRICS}, got {metric!r}")
    return EfficiencyReport(
        **{_REPORT_FIELD[metric]: int(value) if metric == "peak_memory" else float(value)}
    )


def target_value(spec: BudgetSpec, reference: EfficiencyReport) -> float:
    """Return the absolute target (bytes or seconds) for a budget level.

    target = target_fraction * reference metric. Placeholder budgets
    (``spec.is_frozen`` is False) are refused.

    Used by: E5-E8, A1-A6.
    """
    if spec.target_fraction is None:
        raise ValueError(f"Budget {spec.id} is a placeholder; freeze it first.")
    return spec.target_fraction * metric_value(reference, spec.metric)


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
    while high - low > 1:
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
        len(accepted), measure_prefix, target, spec.tolerance
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
    if spec.target_fraction is None:
        open_keys.append(f"budgets.levels.{budget}.target_fraction")
    if open_keys:
        raise ConfigError(f"settings still open: {', '.join(open_keys)}")
    if spec.metric not in BUDGET_METRICS:
        raise ConfigError(f"budgets.metric must be one of {BUDGET_METRICS}, got {spec.metric!r}")
    if not 0.0 < spec.target_fraction < 1.0:
        raise ConfigError(f"budgets.levels.{budget}.target_fraction must be in (0, 1)")
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
    physically, measures the footprint under ``protocol`` and frees the copy.
    """
    from medcompress.compression.pruning.apply import remove_units
    from medcompress.evaluation.efficiency import measure_footprint

    def measure(removal: Sequence[PrunableUnit]) -> float:
        model = fresh_model()
        try:
            remove_units(model, list(removal))
            report = measure_footprint(model, samples, protocol, data_root, **options)
        finally:
            del model
            release_memory()
        return metric_value(report, metric)

    return measure
