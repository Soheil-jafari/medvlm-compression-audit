"""The pruning design of the amendment of 2026-11-17: protection of layers 0, 1 and 35 for every
unit type, the ``uniform_mlp`` allocation (E5, E5R, E7U), its search over k, E7U's reuse of
E7's scores, the per-layer shares, the offline replay and the E7 check additions.

The 36-layer units come from the committed score files (the tiny test model has 3 layers, so
only one of them could be unprotected)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import pytest

from medcompress.budgets import (
    BudgetSpec,
    search_uniform_budget,
    uniform_groups,
    uniform_max_k,
    uniform_removal,
)
from medcompress.compression.pruning.criteria.common import normalised_scores
from medcompress.compression.pruning.replay import (
    layer_sizes_from_units,
    measured_drop_per_parameter,
    parameter_measurer,
    units_from_scores,
)
from medcompress.compression.pruning.units import UnitType, validate_removal_set
from medcompress.compression.recovery import (
    RemovalSettings,
    claim_resume_dir,
    removal_by_layer,
    removal_digest,
    search_budget,
)
from medcompress.config import ConfigError, RunContext

REPO = Path(__file__).resolve().parents[1]
E5_MAGNITUDE = REPO / "results" / "E5" / "magnitude" / "nobudget" / "seed0" / "unit_scores.csv"
E7_SCORES = REPO / "results" / "E7" / "nobudget" / "seed0" / "unit_scores.csv"
PROTECTED = (0, 1, 35)
PEAK = 7618471936.0  # E2's committed peak_memory_bytes (RC-B0)
SPEC = BudgetSpec("B2", "peak_memory", None, "E2", 0.03, decoder_bytes_removed=0.30)
TARGET = PEAK - 0.30 * 5549543424.0  # E2's committed decoder_layers_bytes


def _config(exp: str):  # noqa: ANN202
    from medcompress.cli import resolve_config

    return resolve_config(argparse.Namespace(config=None, exp=exp, override=[], data_root=None,
                                             handoffs=[]))  # fmt: skip


@pytest.fixture(scope="module")
def e5() -> tuple[list, dict]:
    return units_from_scores(E5_MAGNITUDE)


def test_a_unit_of_a_protected_layer_is_refused(e5: tuple[list, dict]) -> None:
    units, scores = e5
    for layer in PROTECTED:
        for kind in UnitType:
            unit = next((u for u in units if u.layer_index == layer and u.unit_type is kind), None)
            if unit is None:
                continue
            problems = validate_removal_set(units, [unit], protected_layers=PROTECTED)
            assert any("protected" in p for p in problems), (unit.unit_id, problems)
    groups = uniform_groups(units, scores, PROTECTED)
    assert sorted(groups) == list(range(2, 35))  # layers 2-34 only


def test_uniform_allocation_takes_the_same_k_lowest_groups_from_every_unprotected_layer(
    e5: tuple[list, dict],
) -> None:
    units, scores = e5
    groups = uniform_groups(units, scores, PROTECTED, "per_parameter")
    assert uniform_max_k(groups) == 85  # 86 groups of 128 channels, one always kept
    chosen = uniform_removal(groups, 29)
    per_layer = {i: 0 for i in range(36)}
    for unit in chosen:
        assert unit.unit_type is UnitType.MLP_CHANNEL_GROUP
        per_layer[unit.layer_index] += 1
    assert all(per_layer[i] == 0 for i in PROTECTED)
    assert all(per_layer[i] == 29 for i in range(2, 35))
    assert not validate_removal_set(units, chosen, protected_layers=PROTECTED)
    for layer in (2, 17, 34):  # within a layer: the 29 lowest scores
        mine = [u for u in units if u.layer_index == layer
                and u.unit_type is UnitType.MLP_CHANNEL_GROUP]  # fmt: skip
        lowest = sorted(mine, key=lambda u: (scores[u.unit_id], u.unit_id))[:29]
        assert {u.unit_id for u in chosen if u.layer_index == layer} == {u.unit_id for u in lowest}


def test_the_k_search_is_monotone_and_lands_in_the_band(e5: tuple[list, dict]) -> None:
    units, scores = e5
    calls: list[int] = []

    def fake(removal):  # noqa: ANN001, ANN202 - a measurer with a fixed overhead per group
        calls.append(len(removal))
        return parameter_measurer(PEAK)(removal) - 1e5 * len(removal)

    search = search_uniform_budget(units, scores, SPEC, TARGET, fake, protected_layers=PROTECTED)
    trace = dict(search.trace)
    ks = sorted(trace)
    # less memory with every k
    assert all(trace[a] > trace[b] for a, b in zip(ks, ks[1:], strict=False))
    assert search.within_tolerance and abs(search.shortfall) <= 0.03
    assert search.footprint <= TARGET * 1.03
    groups = uniform_groups(units, scores, PROTECTED)
    assert fake(uniform_removal(groups, search.count - 1)) > TARGET * 1.03  # the smallest k
    assert len(calls) <= 12  # bisection over 0..85, not a scan
    with pytest.raises(ValueError):  # a target no k reaches
        search_uniform_budget(units, scores, SPEC, 1e9, fake, protected_layers=PROTECTED)


def test_e7u_reads_e7s_scores_and_removes_e5s_units() -> None:
    from medcompress.compression.pruning.stage import scores_path

    config = _config("E7U")
    run = RunContext("E7U", seed=0, budget="B2")
    assert scores_path(config, run) == scores_path(_config("E7"), RunContext("E7", seed=0))
    removal = RemovalSettings.from_config(config)
    assert removal.allocation == "uniform_mlp"
    assert RemovalSettings.from_config(_config("E7")).allocation == "global"
    assert RemovalSettings.from_config(_config("E5")).allocation == "uniform_mlp"
    assert RemovalSettings.from_config(_config("E5R")).allocation == "uniform_mlp"
    assert RemovalSettings.from_config(_config("A3")).allocation == "global"
    units, scores = units_from_scores(E7_SCORES)
    search, protected = search_budget(units, scores, removal, SPEC, TARGET,
                                      parameter_measurer(PEAK))  # fmt: skip
    assert protected == list(PROTECTED)
    values = normalised_scores(scores, units, "per_parameter")
    for layer in range(2, 35):
        mine = sorted((u for u in units if u.layer_index == layer
                       and u.unit_type is UnitType.MLP_CHANNEL_GROUP),
                      key=lambda u: (values[u.unit_id], u.unit_id))  # fmt: skip
        got = {u.unit_id for u in search.removal if u.layer_index == layer}
        assert got == {u.unit_id for u in mine[: search.count]}
    assert {u.unit_type for u in search.removal} == {UnitType.MLP_CHANNEL_GROUP}


def test_e7u_stage_04_never_scores(tmp_path: Path) -> None:
    from medcompress.compression.pruning.stage import score_units_for_experiment

    config = _config("E7U")
    config["paths"] = {**config["paths"], "results_root": str(tmp_path)}  # no E7 scores there
    with pytest.raises(ConfigError, match="reuses the scores of E7"):
        score_units_for_experiment(config, RunContext("E7U", seed=0), model=None)


def test_per_layer_shares_sum_to_the_total(e5: tuple[list, dict]) -> None:
    units, scores = e5
    for removal in (uniform_removal(uniform_groups(units, scores, PROTECTED), 29),
                    [u for u in units if u.layer_index == 5][:1]
                    + [u for u in units if u.unit_type is UnitType.ATTENTION_HEAD
                       and u.layer_index == 9][:1]):  # fmt: skip
        rows = removal_by_layer(layer_sizes_from_units(units), removal)
        assert len(rows) == 36
        assert sum(r[2] for r in rows) == sum(u.num_params for u in removal)
        assert sum(r[3] for r in rows) == pytest.approx(1.0)


def test_the_resume_state_of_another_removal_is_refused(tmp_path: Path) -> None:
    from medcompress.compression.distillation.resume import PROGRESS_FILE

    settings = RemovalSettings("one_shot", "per_parameter", 2, 1, "uniform_mlp")
    units, _ = units_from_scores(E5_MAGNITUDE)
    a, b = removal_digest(units[40:42], settings), removal_digest(units[40:43], settings)
    folder = tmp_path / "resume"
    claim_resume_dir(folder, a)
    claim_resume_dir(folder, a)  # the same removal resumes
    with pytest.raises(ValueError, match="another removal set"):
        claim_resume_dir(folder, b)
    old = tmp_path / "old_resume"
    (old / "current").mkdir(parents=True)
    (old / "current" / PROGRESS_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no removal digest"):
        claim_resume_dir(old, a)
    assert a != removal_digest(units[40:42], RemovalSettings("one_shot", "per_parameter", 2, 1))


def test_an_unknown_allocation_is_refused() -> None:
    config = _config("E5")
    config["pruning"]["removal"] = {**config["pruning"]["removal"], "allocation": "per_layer"}
    with pytest.raises(ConfigError, match="allocation"):
        RemovalSettings.from_config(config)


def test_the_fit_of_the_exact_effect_on_the_estimate() -> None:
    from medcompress.compression.pruning.stage import exact_on_estimate_fit

    pairs = [(f"u{i}", float(i), 2.0 * i + 1.0 + (0.1 if i % 2 else -0.1)) for i in range(30)]
    fit = exact_on_estimate_fit(pairs)
    assert fit["slope"] == pytest.approx(2.0, abs=0.01)
    assert fit["intercept"] == pytest.approx(1.0, abs=0.1)
    assert fit["slope_ci_low"] < 2.0 < fit["slope_ci_high"]
    assert fit["intercept_ci_low"] < fit["intercept"] < fit["intercept_ci_high"]


def test_the_substitution_rerun_from_the_committed_files(tmp_path: Path) -> None:
    from medcompress.compression.pruning.stage import CHECK_B2_FILE, first_order_followup

    results = tmp_path / "results"
    shutil.copytree(REPO / "results" / "E2" / "nobudget", results / "E2" / "nobudget")
    folder = results / "E7" / "nobudget" / "seed0"
    folder.mkdir(parents=True)
    shutil.copy(E7_SCORES, folder / "unit_scores.csv")
    units, scores = units_from_scores(E7_SCORES)
    candidates = [u for u in units if u.unit_type is UnitType.MLP_CHANNEL_GROUP
                  and u.layer_index not in PROTECTED]  # fmt: skip
    order = sorted(candidates, key=lambda u: scores[u.unit_id])
    lowest, highest = order[0], order[-1]  # swapped: one switches each way
    with (folder / "first_order_check.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit_id", "first_order", "exact"])
        writer.writerow([lowest.unit_id, scores[lowest.unit_id], scores[highest.unit_id] * 10])
        writer.writerow([highest.unit_id, scores[highest.unit_id], scores[lowest.unit_id] - 1])
        writer.writerow([order[1].unit_id, scores[order[1].unit_id], scores[order[1].unit_id]])
    config = _config("E7")
    config["paths"] = {**config["paths"], "results_root": str(results)}
    values = first_order_followup(config, RunContext("E7", seed=0))
    assert values["first_order_check:b2_switched"] == 2.0
    assert values["first_order_check:b2_switched_to_kept"] == 1.0
    assert values["first_order_check:b2_switched_to_removed"] == 1.0
    rows = list(csv.DictReader((folder / CHECK_B2_FILE).open(encoding="utf-8")))
    assert {r["unit_id"]: (r["removed_with_estimate"], r["removed_with_exact"]) for r in rows}[
        lowest.unit_id
    ] == ("1", "0")
    metrics = (folder / "metrics.csv").read_text(encoding="utf-8")
    assert (
        "first_order_check:slope" in metrics
        and "first_order_check:b2_kv_or_layer_changes" in metrics
    )


def test_the_estimate_matches_session_1s_measured_drop() -> None:
    root = REPO / "results" / "superseded" / "inner_units_removable" / "E5"
    for variant in ("magnitude", "activation"):
        found = measured_drop_per_parameter(root / variant / "B2" / "seed0")
        assert found is not None
        drop, estimate, _ = found
        assert abs(drop / estimate - 1) < 0.01  # the replay's 2 bytes a parameter, within 1 %


def test_the_committed_replay_is_current(tmp_path: Path) -> None:
    import subprocess
    import sys

    done = subprocess.run([sys.executable, str(REPO / "scripts" / "replay_removal.py"), "--out",
                           str(tmp_path)], capture_output=True, text=True, check=False,
                          timeout=600)  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr  # every arm reaches every budget
    for name in ("removal_replay.csv", "removal_replay_by_layer.csv"):
        assert (tmp_path / name).read_text() == (REPO / "results" / "replay" / name).read_text()
    rows = list(csv.DictReader((tmp_path / "removal_replay.csv").open(encoding="utf-8")))
    assert {(r["arm"], r["budget"]) for r in rows} == {
        (a, b) for a in ("E5:magnitude", "E5:activation", "E5R", "E7", "E7U")
        for b in ("B1", "B2", "B3")
    }  # fmt: skip
    assert all(r["removed_units_in_protected_layers"] == "0" for r in rows)
    assert json.dumps(sorted(r["within_tolerance"] for r in rows)) == json.dumps(["1"] * 15)
