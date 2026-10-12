"""Budget targets, tolerance and the measured prefix search (no GPU: footprints are mocked)."""

from __future__ import annotations

import pytest

from medcompress.budgets import (
    BudgetSpec,
    metric_value,
    smallest_prefix_meeting_target,
    target_value,
    within_budget,
)
from medcompress.evaluation.efficiency import EfficiencyReport

SPEC = BudgetSpec("B2", "peak_memory", 0.6, "E2", 0.02)


def test_target_and_tolerance() -> None:
    reference = EfficiencyReport(peak_memory_bytes=10_000)
    target = target_value(SPEC, reference)
    assert target == pytest.approx(6_000)
    assert within_budget(EfficiencyReport(peak_memory_bytes=6_100), SPEC, target)
    assert not within_budget(EfficiencyReport(peak_memory_bytes=6_200), SPEC, target)
    assert not within_budget(EfficiencyReport(peak_memory_bytes=5_000), SPEC, target)


def test_placeholders_and_missing_fields_are_refused() -> None:
    placeholder = BudgetSpec("B1", "peak_memory", None, "E2", None)
    with pytest.raises(ValueError, match="placeholder"):
        target_value(placeholder, EfficiencyReport(peak_memory_bytes=1))
    with pytest.raises(ValueError, match="tolerance"):
        within_budget(EfficiencyReport(peak_memory_bytes=1), placeholder, 1.0)
    with pytest.raises(ValueError):
        metric_value(EfficiencyReport(), "peak_memory")
    assert metric_value(EfficiencyReport(warm_latency_s=0.5), "latency") == 0.5


def test_prefix_search_finds_the_smallest_prefix_with_few_measurements() -> None:
    calls: list[int] = []

    def footprint(k: int) -> float:
        calls.append(k)
        return 1000.0 - 7.0 * k  # each removed unit saves 7 units of memory

    k, value = smallest_prefix_meeting_target(100, footprint, target=600.0, tolerance=0.0)
    assert k == 58 and value == pytest.approx(594.0)
    assert len(set(calls)) <= 10


def test_prefix_search_edge_cases() -> None:
    assert smallest_prefix_meeting_target(10, lambda k: 100.0, 200.0, 0.0) == (0, 100.0)
    with pytest.raises(ValueError, match="does not reach"):
        smallest_prefix_meeting_target(10, lambda k: 1000.0 - k, 10.0, 0.0)
