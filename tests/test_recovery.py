"""Matched-budget pruning and the shared recovery, end to end on the tiny model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

import numpy as np  # noqa: E402
import tiny_vlm  # noqa: E402
import yaml  # noqa: E402
from eval_fixtures import eval_config, write_dataset  # noqa: E402
from kd_fixtures import teacher_records  # noqa: E402

from medcompress.compression import recovery  # noqa: E402
from medcompress.compression.pruning import stage  # noqa: E402
from medcompress.compression.pruning.apply import remove_units  # noqa: E402
from medcompress.compression.pruning.checkpoint import load_pruned  # noqa: E402
from medcompress.compression.pruning.units import UnitType, enumerate_units  # noqa: E402
from medcompress.config import ConfigError, RunContext  # noqa: E402
from medcompress.evaluation.efficiency import count_parameters  # noqa: E402

TYPES = ["decoder_layer", "attention_head", "mlp_channel_group"]
TINY_TEMPLATE = "Is there a {finding} in this image ? Answer yes or no ."
RECOVERY = {
    "method": "lora_plus_offline_kd",
    "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["decoder_mlp"]},
    "steps": 2,
    "learning_rate": 1e-2,
    "batch_size": 2,
    "grad_accumulation": 1,
    "kd_weight": 0.5,
    "kd_temperature": 1.0,
    "data_subset": 6,
}


def _config(root: Path, *, fraction: float = 0.9, criterion: str = "magnitude") -> dict[str, Any]:
    config = eval_config(
        root, domain={"probe": {"template": TINY_TEMPLATE, "display_names": {}}}, max_new_tokens=4
    )
    levels = {b: {"target_fraction": None, "description": ""} for b in ("B1", "B2", "B3")}
    levels["B2"]["target_fraction"] = fraction
    (root / "budgets.yaml").write_text(
        yaml.safe_dump(
            {
                "budgets": {
                    "reference_experiment": "E2",
                    "metric": "peak_memory",
                    "tolerance": 0.05,
                    "levels": levels,
                }
            }
        )
    )
    config["components"]["budgets"] = str(root / "budgets.yaml")
    config["paths"]["cache_root"] = str(root / "cache")
    config["pruning"] = {
        "criterion": criterion,
        "units": {"types": TYPES, "mlp_channel_group_size": 16, "include_vision_blocks": False},
        "calibration": {"size": 4},
        "removal": {
            "schedule": "one_shot",
            "score_normalisation": "none",
            "protected_layers": {"first": 1, "last": 1},
        },
    }
    config["recovery"] = dict(RECOVERY)
    return config


def _factory() -> Any:
    return tiny_vlm.tiny_wrapper(seed=0)


def _footprint(removal: Any) -> float:
    model = _factory()
    remove_units(model, list(removal))
    return float(count_parameters(model))


def _prepare(root: Path, **kwargs: Any) -> tuple[dict[str, Any], list[Any], float]:
    train = write_dataset(root)
    config = _config(root, **kwargs)
    stage.run_scoring(config, RunContext("E5", seed=0, variant=None), _factory(), train)
    return config, train, float(count_parameters(_factory()))


# ------------------------------------------------------------------------- settings


def test_removal_settings_list_what_is_open_and_reject_what_is_not_implemented() -> None:
    config: dict[str, Any] = {
        "pruning": {
            "removal": {
                "schedule": "TODO",
                "score_normalisation": "TODO",
                "protected_layers": {"first": "TODO", "last": "TODO"},
            }
        }
    }
    with pytest.raises(ConfigError) as caught:
        recovery.RemovalSettings.from_config(config)
    for key in (
        "schedule",
        "score_normalisation",
        "protected_layers.first",
        "protected_layers.last",
    ):
        assert f"pruning.removal.{key}" in str(caught.value)
    removal = config["pruning"]["removal"]
    removal.update(schedule="one_shot", score_normalisation="per_parameter")
    removal["protected_layers"] = {"first": 2, "last": 1}
    settings = recovery.RemovalSettings.from_config(config)
    assert (settings.protected_first, settings.protected_last) == (2, 1)
    for change, match in (
        ({"schedule": "iterative"}, "not implemented"),
        ({"score_normalisation": "z"}, "score_normalisation"),
        ({"protected_layers": {"first": -1, "last": 0}}, ">= 0"),
    ):
        with pytest.raises(ConfigError, match=match):
            recovery.RemovalSettings.from_config({"pruning": {"removal": {**removal, **change}}})


def test_the_removal_plan_follows_the_scores_and_spares_the_protected_layers() -> None:
    w = _factory()
    units = enumerate_units(w, [UnitType(t) for t in TYPES], mlp_channel_group_size=16)
    scores = {u.unit_id: float(i) for i, u in enumerate(units)}
    settings = recovery.RemovalSettings("one_shot", "none", 1, 1)
    ordered, protected = recovery.plan_removal(units, scores, settings)
    assert protected == [0, 2]
    assert [u.unit_id for u in ordered if u.unit_type is UnitType.DECODER_LAYER] == ["lm.layer1"]
    assert [scores[u.unit_id] for u in ordered] == sorted(scores[u.unit_id] for u in ordered)
    heads_only = [u for u in units if u.unit_type is not UnitType.DECODER_LAYER]
    plan, none_protected = recovery.plan_removal(heads_only, scores, settings)
    assert none_protected == [] and len(plan) == len(heads_only)


def test_the_prune_stage_lists_every_open_setting_at_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["pruning"]["removal"]["schedule"] = "TODO"
    config["recovery"]["steps"] = "TODO"
    config["pruning"]["units"]["mlp_channel_group_size"] = "TODO"
    with pytest.raises(ConfigError) as caught:
        recovery.run_prune_and_recover(config, RunContext("E5", seed=0, budget="B1"))
    message = str(caught.value)
    for key in (
        "pruning.removal.schedule",
        "training.steps",
        "mlp_channel_group_size",
        "budgets.levels.B1.target_fraction",
        "budgets.measurement.hardware",
    ):
        assert key in message, key
    with pytest.raises(ConfigError, match="--budget"):
        recovery.run_prune_and_recover(config, RunContext("E5", seed=0))


# --------------------------------------------------------------------- the recovery


def test_recovery_trains_an_adapter_merges_it_and_leaves_plain_weights(tmp_path: Path) -> None:
    from medcompress.compression.distillation.offline_kd import examples_from_samples

    train = write_dataset(tmp_path)
    records = teacher_records(train, tmp_path)
    examples = examples_from_samples(train, tmp_path, records=records)
    model = _factory()
    before = {k: v.clone() for k, v in model.module.state_dict().items()}
    result = recovery.run_recovery(model, examples, RECOVERY, seed=0)
    assert result.steps == 2 and result.examples_seen == 4
    assert not any("lora" in name for name in model.module.state_dict())  # merged and dropped
    after = model.module.state_dict()
    assert set(after) == set(before)
    moved = [k for k in before if not torch.equal(before[k], after[k])]
    assert moved and all("mlp" in k for k in moved)  # only the targeted projections changed
    with pytest.raises(ConfigError, match="unknown recovery method"):
        recovery.run_recovery(_factory(), examples, {**RECOVERY, "method": "x"}, seed=0)


def test_the_data_subset_depends_on_the_seed_only(tmp_path: Path) -> None:
    from medcompress.compression.distillation.offline_kd import (
        examples_from_samples,
        select_subset,
    )

    train = write_dataset(tmp_path)
    examples = examples_from_samples(train, tmp_path, records=teacher_records(train, tmp_path))
    ids = [e.sample_id for e in examples]
    assert select_subset(ids, 6, 0) == select_subset(ids, 6, 0)
    assert select_subset(ids, 6, 0) != select_subset(ids, 6, 1)


# ------------------------------------------------------------------------ the stage


def test_the_stage_meets_the_budget_recovers_and_saves_a_reloadable_model(tmp_path: Path) -> None:
    config, train, total = _prepare(tmp_path)
    run = RunContext("E5", seed=0, budget="B2", variant=None)
    records = teacher_records(train, tmp_path)
    recovery.run_prune_and_recover(
        config,
        run,
        student_factory=_factory,
        train=train,
        records=records,
        measure=_footprint,
        reference_value=total,
    )
    run_dir = tmp_path / "results/E5/B2/seed0"
    with (run_dir / "metrics.csv").open() as handle:
        metrics = {r["metric"]: float(r["value"]) for r in csv.DictReader(handle)}
    target = 0.9 * total
    assert metrics["budget_target"] == pytest.approx(target)
    assert metrics["budget_footprint"] <= target * 1.05
    assert metrics["budget_within_tolerance"] in (0.0, 1.0)
    assert metrics["parameters_before"] == total
    assert metrics["parameters_after"] == metrics["budget_footprint"] < total
    assert metrics["removed_units"] >= 1 and metrics["recovery_steps"] == 2
    kinds = ("layers", "kv_groups", "channel_groups")
    assert sum(metrics[f"removed_{k}"] for k in kinds) == metrics["removed_units"]
    removal = json.loads((run_dir / "removal.json").read_text())
    assert len(removal["units"]) == metrics["removed_units"]
    assert removal["recovery_fingerprint"] == recovery.recovery_fingerprint(RECOVERY)
    trace = list(csv.DictReader((run_dir / "budget_search.csv").open()))
    assert len(trace) == metrics["budget_measurements"] and len(trace) <= 8
    # the saved model reloads on E2 and has the pruned shapes
    directory = recovery.pruned_dir(config, run)
    assert directory == tmp_path / "checkpoints/E5/B2/seed0/pruned"
    reloaded = load_pruned(_factory(), directory)
    assert count_parameters(reloaded) == metrics["parameters_after"]
    ids = {u.unit_id for u in enumerate_units(_factory(), [UnitType.DECODER_LAYER])}
    assert removal["units"] and set(removal["units"]) & ids <= {"lm.layer1"}  # layers 0 and 2 kept


def test_the_runner_reloads_the_pruned_arm_through_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medcompress.evaluation import runner

    config, train, total = _prepare(tmp_path)
    run = RunContext("E5", seed=0, budget="B2", variant="magnitude")
    config["pruning"]["criterion"] = "by_variant"
    config["pruning"]["criterion_by_variant"] = {"magnitude": "magnitude"}
    stage.run_scoring(config, run, _factory(), train)
    recovery.run_prune_and_recover(
        config,
        run,
        student_factory=_factory,
        train=train,
        records=teacher_records(train, tmp_path),
        measure=_footprint,
        reference_value=total,
    )
    assert recovery.pruned_dir(config, run) == tmp_path / "checkpoints/E5/magnitude/B2/seed0/pruned"
    monkeypatch.setattr(runner, "load_trained_student", lambda *a, **k: _factory())
    model = runner.load_run_model(config, run, 56)
    assert count_parameters(model) < total
    scores = model.score_yes_no([tiny_vlm.tiny_image(0)], [tiny_vlm.PROBE])
    assert np.isfinite([scores[0].logp_yes, scores[0].logp_no]).all()
    with pytest.raises(ConfigError, match="belongs to a budget"):
        runner.load_run_model(config, RunContext("E5", seed=0, variant="magnitude"), 56)


def test_the_stage_stops_without_scores_or_when_nothing_needs_removing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = RunContext("E5", seed=0, budget="B2")
    kwargs: dict[str, Any] = {
        "student_factory": _factory,
        "train": [],
        "records": {},
        "measure": _footprint,
        "reference_value": 1000.0,
    }
    with pytest.raises(ConfigError, match="run stage 04 first"):
        recovery.run_prune_and_recover(config, run, **kwargs)
    train = write_dataset(tmp_path)
    stage.run_scoring(config, RunContext("E5", seed=0), _factory(), train)
    with pytest.raises(ValueError, match="asks for no compression"):
        recovery.run_prune_and_recover(
            config,
            run,
            student_factory=_factory,
            train=train,
            records={},
            measure=lambda removal: 1.0,
            reference_value=10.0,
        )


def test_a_budget_the_units_cannot_reach_is_an_error(tmp_path: Path) -> None:
    config, train, total = _prepare(tmp_path, fraction=0.2)
    with pytest.raises(ValueError, match="does not reach the budget"):
        recovery.run_prune_and_recover(
            config,
            RunContext("E5", seed=0, budget="B2"),
            student_factory=_factory,
            train=train,
            records={},
            measure=_footprint,
            reference_value=total,
        )


def test_every_arm_gets_the_same_recovery_fingerprint() -> None:
    assert recovery.recovery_fingerprint(RECOVERY) == recovery.recovery_fingerprint(dict(RECOVERY))
    assert recovery.recovery_fingerprint(RECOVERY) != recovery.recovery_fingerprint(
        {**RECOVERY, "steps": 3}
    )
