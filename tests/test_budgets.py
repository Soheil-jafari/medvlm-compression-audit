"""Budget targets, tolerance and the measured prefix search (no GPU: footprints are mocked)."""

from __future__ import annotations

from pathlib import Path

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


# ------------------------------------------------------------------ the matched search


def _units() -> list:
    from medcompress.compression.pruning.units import PrunableUnit, UnitType

    out = []
    for layer in range(4):
        out.append(PrunableUnit(f"lm.layer{layer}", UnitType.DECODER_LAYER, layer, None, 100))
        for g in range(2):
            out.append(
                PrunableUnit(f"lm.layer{layer}.kv_group{g}", UnitType.ATTENTION_HEAD, layer, g, 10)
            )
    return out


def _search(target: float, tolerance: float = 0.02) -> tuple:
    from medcompress.budgets import search_matched_budget

    units = _units()
    ordered = sorted(units, key=lambda u: (u.num_params, u.unit_id))  # head groups, then layers
    measured: list[int] = []

    def footprint(removal: object) -> float:
        units_removed = list(removal)  # type: ignore[call-overload]
        measured.append(len(units_removed))
        return 1000.0 - sum(u.num_params for u in units_removed)

    spec = BudgetSpec("B2", "peak_memory", 0.85, "E2", tolerance)
    return search_matched_budget(units, ordered, spec, target, footprint), measured


def test_the_search_bisects_the_measured_footprint_over_the_feasible_prefix() -> None:
    # accepted order: one head group per layer (the second would empty the layer), then
    # layers 0-2 (a layer absorbs its head group; the last layer cannot go)
    search, measured = _search(875.0)
    assert search.count == 5 and search.footprint == 870.0 and search.within_tolerance
    assert {u.unit_id for u in search.removal} == {
        "lm.layer1.kv_group0",
        "lm.layer2.kv_group0",
        "lm.layer3.kv_group0",
        "lm.layer0",
    }
    assert [c for c, _ in search.trace] == sorted({c for c, _ in search.trace})
    assert dict(search.trace)[5] == 870.0 and len(search.trace) <= 5
    assert search.budget == "B2" and search.tolerance == 0.02 and search.target == 875.0
    assert search.shortfall == pytest.approx(870.0 / 875.0 - 1.0)


def test_a_unit_that_overshoots_the_band_is_reported_as_outside_the_tolerance() -> None:
    search, _ = _search(850.0)  # one more layer takes the footprint from 870 to 780
    assert search.count == 6 and search.footprint == 780.0
    assert not search.within_tolerance and search.shortfall < -0.05


def test_the_search_needs_a_tolerance_and_a_reachable_target() -> None:
    from medcompress.budgets import search_matched_budget
    from medcompress.config import ConfigError

    units = _units()
    with pytest.raises(ConfigError, match="tolerance"):
        search_matched_budget(
            units, units, BudgetSpec("B2", "peak_memory", 0.5, "E2", None), 1.0, lambda r: 1.0
        )
    with pytest.raises(ValueError, match="does not reach"):
        _search(100.0)


def test_budget_specs_are_read_frozen_or_refused(tmp_path: Path) -> None:
    import yaml

    from medcompress.budgets import budget_spec_for_run
    from medcompress.config import ConfigError

    levels = {b: {"target_fraction": None} for b in ("B1", "B2", "B3")}
    block = {"reference_experiment": "E2", "metric": "TODO", "tolerance": "TODO", "levels": levels}
    path = tmp_path / "budgets.yaml"
    path.write_text(yaml.safe_dump({"budgets": block}))
    config = {"components": {"budgets": str(path)}}
    with pytest.raises(ConfigError) as caught:
        budget_spec_for_run(config, "B2")
    for key in ("budgets.metric", "budgets.tolerance", "budgets.levels.B2.target_fraction"):
        assert key in str(caught.value)
    block.update(metric="peak_memory", tolerance=0.03)
    levels["B2"]["target_fraction"] = 0.6
    path.write_text(yaml.safe_dump({"budgets": block}))
    spec = budget_spec_for_run(config, "B2")
    assert (spec.metric, spec.target_fraction, spec.tolerance) == ("peak_memory", 0.6, 0.03)
    with pytest.raises(ConfigError, match="unknown budget"):
        budget_spec_for_run(config, "B9")
    levels["B2"]["target_fraction"] = 1.2
    path.write_text(yaml.safe_dump({"budgets": block}))
    with pytest.raises(ConfigError, match=r"\(0, 1\)"):
        budget_spec_for_run(config, "B2")
    block["metric"] = "energy"
    levels["B2"]["target_fraction"] = 0.5
    path.write_text(yaml.safe_dump({"budgets": block}))
    with pytest.raises(ConfigError, match="metric must be one of"):
        budget_spec_for_run(config, "B2")


def test_the_reference_is_read_back_from_the_efficiency_file_of_e2(tmp_path: Path) -> None:
    from medcompress.budgets import reference_report, report_for_metric
    from medcompress.utils.io import upsert_results

    rows = [
        {"experiment": "E2", "budget": "", "seed": 0, "metric": m, "value": v}
        for m, v in (
            ("peak_memory_bytes", 8.0e9),
            ("warm_latency_s", 0.42),
            ("num_parameters", 3e9),
        )
    ]
    upsert_results(rows, tmp_path / "E2/nobudget/seed0/efficiency.csv")
    report = reference_report(tmp_path, "E2")
    assert report.peak_memory_bytes == 8_000_000_000 and report.warm_latency_s == 0.42
    assert report.num_parameters == 3_000_000_000 and report.cold_latency_s is None
    with pytest.raises(FileNotFoundError, match="07_benchmark_efficiency"):
        reference_report(tmp_path, "E2", seed=5)
    assert report_for_metric("peak_memory", 2.5e9).peak_memory_bytes == 2_500_000_000
    assert report_for_metric("latency", 0.5).warm_latency_s == 0.5
    with pytest.raises(ValueError, match="metric"):
        report_for_metric("energy", 1.0)
