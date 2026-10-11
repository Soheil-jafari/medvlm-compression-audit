"""Experiment registry consistency and run validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from medcompress.cli import stage_functions
from medcompress.config import RunContext
from medcompress.registry import (
    AGGREGATE,
    ALL_SCRIPTS,
    BUDGET_IDS,
    PRIMARY_BUDGET,
    PRIMARY_COMPARISON,
    REGISTRY,
    experiment_ids,
    get_experiment,
    validate_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_ids_are_exactly_the_planned_ladder() -> None:
    assert experiment_ids("main") == tuple(f"E{i}" for i in range(10))
    assert experiment_ids("ablation") == tuple(f"A{i}" for i in range(1, 7))


def test_references_point_to_registered_experiments() -> None:
    for spec in REGISTRY.values():
        for ref in (spec.parent, spec.variant_of, *spec.requires):
            assert ref is None or ref in REGISTRY, f"{spec.id} references unknown {ref}"


def test_parent_chains_have_no_cycles() -> None:
    for spec in REGISTRY.values():
        seen = {spec.id}
        parent = spec.parent
        while parent is not None:
            assert parent not in seen, f"cycle through {spec.id}"
            seen.add(parent)
            parent = REGISTRY[parent].parent


def test_compression_arms_start_from_e2() -> None:
    for exp_id in ("E3", "E5", "E6", "E7"):
        assert REGISTRY[exp_id].parent == "E2"
    assert REGISTRY["E8"].parent == "E7"
    assert REGISTRY["E9"].parent == "E0"


def test_budget_matched_arms() -> None:
    for exp_id in ("E5", "E6", "E7", "E8"):
        assert REGISTRY[exp_id].budgets == BUDGET_IDS
    for spec in REGISTRY.values():
        if spec.kind == "ablation":
            assert spec.variant_of == "E7"
            assert spec.budgets == (PRIMARY_BUDGET,)


def test_scripts_exist_and_are_used() -> None:
    for script in ALL_SCRIPTS:
        assert (REPO_ROOT / script).is_file(), f"{script} missing"
    used = {script for spec in REGISTRY.values() for script in spec.scripts}
    assert set(ALL_SCRIPTS) - used == {AGGREGATE}


def test_every_stage_script_maps_to_a_function() -> None:
    used = {script for spec in REGISTRY.values() for script in spec.scripts}
    assert used == set(stage_functions())


def test_primary_comparison() -> None:
    assert PRIMARY_COMPARISON["ours"] == "E7"
    assert set(PRIMARY_COMPARISON["against_best_of"]) == {"E5", "E6"}
    assert PRIMARY_COMPARISON["budget"] == "B2"


def test_get_experiment_unknown_id() -> None:
    with pytest.raises(KeyError, match="Valid IDs"):
        get_experiment("E42")


@pytest.mark.parametrize(
    ("exp", "budget", "variant", "ok"),
    [
        ("E7", "B2", None, True),
        ("E7", None, None, False),  # budget required
        ("E0", "B1", None, False),  # E0 is not budget-matched
        ("A1", "B1", None, False),  # ablations run at B2 only
        ("E3", None, "w4", True),
        ("E3", None, "w2", False),
        ("E5", "B3", "taylor", True),
        ("E7", "B4", None, False),
    ],
)
def test_validate_run(exp: str, budget: str | None, variant: str | None, ok: bool) -> None:
    if ok:
        assert validate_run(exp, budget, variant).id == exp
    else:
        with pytest.raises(ValueError):
            validate_run(exp, budget, variant)


def test_validate_run_optional_budget_and_required_variant() -> None:
    assert validate_run("E7", None, require_budget=False).id == "E7"
    with pytest.raises(ValueError, match="variant"):
        validate_run("E5", "B2", None, require_variant=True)


def test_run_context_paths_and_tags() -> None:
    run = RunContext("E3", seed=1, variant="w4")
    assert run.experiment_tag == "E3:w4"
    assert run.run_dir("results") == Path("results/E3/w4/nobudget/seed1")
    run = RunContext("E7", seed=0, budget="B2")
    assert run.experiment_tag == "E7"
    assert run.run_dir("results") == Path("results/E7/B2/seed0")
