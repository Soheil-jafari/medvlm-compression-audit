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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.config import ConfigError
from medcompress.registry import BUDGET_IDS

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.evaluation.efficiency import EfficiencyReport
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


def measure_reference_footprint(model: VLMWrapper, protocol: Mapping[str, Any]) -> EfficiencyReport:
    """Measure the E2 reference footprint that budgets are fractions of.

    Must use exactly the measurement protocol in ``configs/budgets.yaml``
    (hardware, batch size, image resolution, prompt length, max new tokens,
    warm-up and repeat counts) and delegate the measurement itself to
    :mod:`medcompress.evaluation.efficiency`.

    Used by: E2 (reference), then E5-E8 and A1-A6 via :func:`search_matched_budget`.
    """
    raise NotImplementedError(
        "TODO: measure E2 peak memory / latency with the frozen protocol (E2 reference)."
    )


def metric_value(report: EfficiencyReport, metric: str) -> float:
    """Return the budget metric from a report: peak memory (bytes) or warm latency (s)."""
    if metric not in _REPORT_FIELD:
        raise ValueError(f"Budget metric must be one of {BUDGET_METRICS}, got {metric!r}")
    value = getattr(report, _REPORT_FIELD[metric])
    if value is None:
        raise ValueError(f"The report has no {_REPORT_FIELD[metric]}")
    return float(value)


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


def search_matched_budget(
    model: VLMWrapper,
    ranked_units: list[PrunableUnit],
    spec: BudgetSpec,
    target: float,
    protocol: Mapping[str, Any],
    *,
    workdir: Path | None = None,
) -> list[PrunableUnit]:
    """Find the smallest removal set (lowest-ranked first) that meets the budget.

    Removes units in ranked order (respecting GQA / structural constraints from
    :func:`medcompress.compression.pruning.units.validate_removal_set`), measures
    the real footprint after physical removal, and stops at the first set that is
    within budget. Measurement-based, not parameter-count-based. A coarse
    estimate followed by measured refinement (e.g. bisection over the prefix
    length) is acceptable; document the choice in ``docs/decisions_log.md``.

    Used by: E5, E6, E7 (every budget), A1-A6 (primary budget).
    """
    raise NotImplementedError(
        "TODO: remove lowest-ranked units until the MEASURED budget is met (E5-E7, A1-A6)."
    )
