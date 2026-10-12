"""Stage 04: settings, variants and the scores file, on the tiny model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")

import tiny_vlm  # noqa: E402
from eval_fixtures import eval_config, write_dataset  # noqa: E402

from medcompress.compression.pruning import stage  # noqa: E402
from medcompress.compression.pruning.criteria import CRITERIA, common  # noqa: E402
from medcompress.compression.pruning.criteria.base import PruningCriterion  # noqa: E402
from medcompress.config import ConfigError, RunContext, load_experiment  # noqa: E402
from medcompress.utils.io import read_json  # noqa: E402

TYPES = ["decoder_layer", "attention_head", "mlp_channel_group"]
TINY_TEMPLATE = "Is there a {finding} in this image ? Answer yes or no ."


def _config(root: Path, **pruning: Any) -> dict[str, Any]:
    config = eval_config(
        root, domain={"probe": {"template": TINY_TEMPLATE, "display_names": {}}}, max_new_tokens=4
    )
    config["pruning"] = {
        "criterion": "magnitude",
        "units": {"types": TYPES, "mlp_channel_group_size": 16, "include_vision_blocks": False},
        "calibration": {"size": 4},
        **pruning,
    }
    return config


def _run(variant: str | None = None, seed: int = 0, budget: str | None = None) -> RunContext:
    return RunContext("E5", seed=seed, budget=budget, variant=variant)


# ------------------------------------------------------------------------ settings


def test_the_real_configs_name_every_open_setting() -> None:
    config, _ = load_experiment("E7")
    with pytest.raises(ConfigError) as caught:
        stage.ScoringSettings.from_config(config)
    message = str(caught.value)
    assert "pruning.units.mlp_channel_group_size" in message
    assert "pruning.calibration.size" in message
    assert "pruning.criterion" not in message  # E7 sets it


def test_variants_resolve_the_criterion_the_sizes_and_the_unit_types() -> None:
    e5, _ = load_experiment("E5")
    e5["pruning"]["units"]["mlp_channel_group_size"] = 64
    e5["pruning"]["calibration"]["size"] = 32
    assert stage.ScoringSettings.from_config(e5, "taylor").criterion == "taylor"
    assert stage.ScoringSettings.from_config(e5, "magnitude").calibration_size is None
    with pytest.raises(ConfigError, match="needs a variant"):
        stage.ScoringSettings.from_config(e5, None)
    a4, _ = load_experiment("A4")
    a4["pruning"]["units"]["mlp_channel_group_size"] = 64
    assert stage.ScoringSettings.from_config(a4, "n128").calibration_size == 128
    a6, _ = load_experiment("A6")
    a6["pruning"]["calibration"]["size"] = 32
    layers = stage.ScoringSettings.from_config(a6, "layers_only")
    assert layers.unit_types == ("decoder_layer",)
    assert layers.mlp_channel_group_size is None  # not needed without channel groups
    e6, _ = load_experiment("E6")
    e6["pruning"]["calibration"]["size"] = 32
    assert stage.ScoringSettings.from_config(e6).unit_types == ("decoder_layer",)


def test_invalid_settings_are_rejected() -> None:
    base = {
        "criterion": "magnitude",
        "units": {"types": TYPES, "mlp_channel_group_size": 16},
        "calibration": {"size": 4},
    }
    ok = {"pruning": base}
    assert stage.ScoringSettings.from_config(ok).criterion == "magnitude"
    for change, match in (
        ({"criterion": "wanda"}, "is not one of"),
        ({"units": {"types": ["head"], "mlp_channel_group_size": 16}}, "non-empty list"),
        ({"units": {**base["units"], "include_vision_blocks": True}}, "vision blocks"),
        ({"units": {**base["units"], "mlp_channel_group_size": 0}}, "positive integer"),
    ):
        with pytest.raises(ConfigError, match=match):
            stage.ScoringSettings.from_config({"pruning": {**base, **change}})
    taylor = {"pruning": {**base, "criterion": "taylor", "calibration": {"size": 0}}}
    with pytest.raises(ConfigError, match="positive integer"):
        stage.ScoringSettings.from_config(taylor)


# --------------------------------------------------------------------------- scoring


def test_scores_are_written_once_per_seed_and_shared_by_all_budgets(tmp_path: Path) -> None:
    train = write_dataset(tmp_path)
    config = _config(tmp_path, criterion="by_variant")
    config["pruning"]["criterion_by_variant"] = {"magnitude": "magnitude", "taylor": "taylor"}
    run = _run("magnitude", budget="B2")
    path = stage.run_scoring(config, run, tiny_vlm.tiny_wrapper(seed=0), train)
    assert path == tmp_path / "results/E5/magnitude/nobudget/seed0/unit_scores.csv"
    scores = common.read_unit_scores(path)
    assert len(scores) == 18 and all(v > 0 for v in scores.values())
    metrics = (path.parent / "metrics.csv").read_text()
    assert "E5:magnitude" in metrics and "scoring_units" in metrics
    assert stage.scores_path(config, _run("magnitude", budget="B1")) == path
    assert read_json(path.with_name(stage.SETTINGS_FILE))["criterion"] == "magnitude"


def test_the_same_settings_reuse_the_file_and_other_settings_score_again(tmp_path: Path) -> None:
    train = write_dataset(tmp_path)
    config = _config(tmp_path)
    run = _run()
    path = stage.run_scoring(config, run, tiny_vlm.tiny_wrapper(seed=0), train)
    stamp = path.stat().st_mtime_ns

    class Boom(PruningCriterion):
        def score(self, model: Any, units: Any, calibration: Any) -> dict[str, float]:
            raise AssertionError("scored again")

    boom = Boom(config)
    assert stage.run_scoring(config, run, tiny_vlm.tiny_wrapper(seed=0), train, criterion=boom)
    assert path.stat().st_mtime_ns == stamp
    # the entry point does not even load a model for a file that exists
    stage.score_units_for_experiment(config, run)
    other = _config(tmp_path, units={"types": ["decoder_layer"], "include_vision_blocks": False})
    stage.run_scoring(other, run, tiny_vlm.tiny_wrapper(seed=0), train)
    assert len(common.read_unit_scores(path)) == 3
    with pytest.raises(AssertionError, match="scored again"):
        stage.run_scoring(
            other, run, tiny_vlm.tiny_wrapper(seed=0), train, criterion=boom, force=True
        )


def test_data_criteria_use_a_nested_calibration_subset_of_the_training_images(
    tmp_path: Path,
) -> None:
    train = write_dataset(tmp_path)
    seen: list[list[str]] = []

    class Spy(CRITERIA["magnitude"]):  # type: ignore[misc]
        needs_calibration = True

        def score(self, model: Any, units: Any, calibration: Any) -> dict[str, float]:
            seen.append([s.image_path for s in calibration])
            return super().score(model, units, calibration)

    for size in (3, 5):
        config = _config(tmp_path, criterion="activation", calibration={"size": size})
        stage.run_scoring(
            config,
            _run(seed=1),
            tiny_vlm.tiny_wrapper(seed=0),
            train,
            criterion=Spy(config, seed=1),
            force=True,
        )
    assert [len(s) for s in seen] == [3, 5]
    assert len(set(seen[0])) == 3 and set(seen[0]) <= set(seen[1])
    # the data-free criteria are given no calibration images at all
    config = _config(tmp_path)
    seen.clear()
    stage.run_scoring(
        config, _run(), tiny_vlm.tiny_wrapper(seed=0), train, criterion=Spy(config), force=True
    )
    assert seen == [[]]


def test_taylor_and_clinical_run_through_the_stage(tmp_path: Path) -> None:
    train = write_dataset(tmp_path)
    taylor = _config(tmp_path, criterion="taylor", taylor={})
    path = stage.run_scoring(taylor, _run(seed=2), tiny_vlm.tiny_wrapper(seed=0), train)
    scores = common.read_unit_scores(path)
    assert len(scores) == 18 and all(v >= 0 for v in scores.values())
    clinical = _config(
        tmp_path,
        criterion="clinical",
        clinical={
            "delta": "logprob_drop",
            "weighting": "uniform",
            "negatives_per_positive": None,
            "negative_delta": "keep",
            "ablation_method": "exact",
            "batch_size": 4,
        },
    )
    run = RunContext("E7", seed=0)
    path = stage.run_scoring(clinical, run, tiny_vlm.tiny_wrapper(seed=0), train)
    assert path == tmp_path / "results/E7/nobudget/seed0/unit_scores.csv"
    assert len(common.read_unit_scores(path)) == 18


def test_bad_scores_are_not_written(tmp_path: Path) -> None:
    train = write_dataset(tmp_path)
    config = _config(tmp_path)

    class Partial(PruningCriterion):
        def score(self, model: Any, units: Any, calibration: Any) -> dict[str, float]:
            return {u.unit_id: float("nan") for u in units}

    class Missing(PruningCriterion):
        def score(self, model: Any, units: Any, calibration: Any) -> dict[str, float]:
            return {units[0].unit_id: 1.0}

    for criterion, match in ((Partial(config), "non-finite"), (Missing(config), "no score")):
        with pytest.raises(ValueError, match=match):
            stage.run_scoring(
                config, _run(), tiny_vlm.tiny_wrapper(seed=0), train, criterion=criterion
            )
    assert not stage.scores_path(config, _run()).exists()


def test_open_settings_stop_the_stage_before_a_model_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("loaded a model")

    monkeypatch.setattr(stage, "load_scoring_model", boom)
    monkeypatch.setattr(stage, "load_train_samples", boom)
    config = _config(tmp_path)
    config["pruning"]["units"]["mlp_channel_group_size"] = "TODO"
    with pytest.raises(ConfigError, match="mlp_channel_group_size"):
        stage.score_units_for_experiment(config, _run())
    with pytest.raises(ConfigError, match="--exp"):
        stage.score_units_for_experiment(config, None)
    config["pruning"]["units"]["mlp_channel_group_size"] = 16
    config["inputs"] = {"image_longest_side": "TODO"}
    monkeypatch.undo()
    with pytest.raises(ConfigError, match="image_longest_side"):
        stage.score_units_for_experiment(config, _run())
