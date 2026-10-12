"""Training settings, example building and schedules (light: no torch needed).

The heavy parts of the training core (losses, batching, adapters, loops) are tested
in ``test_training.py`` on a tiny offline model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medcompress.compression.distillation.batching import TrainExample
from medcompress.compression.distillation.offline_kd import (
    TrainConfig,
    _learning_rate,
    examples_from_samples,
    run_student_training,
    select_subset,
)
from medcompress.compression.lora import TARGET_GROUPS, LoRAConfig, target_regex
from medcompress.config import ConfigError, RunContext
from medcompress.data.base import VQASample


def _section(**over: object) -> dict:
    section: dict = {
        "mode": "offline_kd",
        "method": "lora",
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["decoder_attention", "decoder_mlp"],
        },
        "steps": 100,
        "learning_rate": 2.0e-4,
        "batch_size": 4,
        "grad_accumulation": 2,
        "kd_weight": 0.5,
        "kd_temperature": 1.0,
        "data_subset": "all",
        "teacher_cache_from": "E0",
    }
    section.update(over)
    return section


def test_from_config_reads_every_setting() -> None:
    cfg = TrainConfig.from_config(_section(warmup_steps=5))
    assert cfg.mode == "offline_kd" and cfg.steps == 100 and cfg.batch_size == 4
    assert cfg.learning_rate == pytest.approx(2e-4) and cfg.kd_weight == 0.5
    assert cfg.lora == LoRAConfig(8, 16.0, 0.05, ("decoder_attention", "decoder_mlp"))
    assert cfg.data_subset == "all" and cfg.extra == {"warmup_steps": 5}


@pytest.mark.parametrize("key", ["steps", "learning_rate", "batch_size", "kd_weight"])
def test_from_config_names_open_settings(key: str) -> None:
    with pytest.raises(ConfigError, match=f"training.{key}"):
        TrainConfig.from_config(_section(**{key: "TODO"}))


def test_from_config_names_open_lora_and_subset_settings() -> None:
    section = _section()
    section["lora"] = {"rank": "TODO", "alpha": 16, "dropout": 0.0}
    with pytest.raises(ConfigError) as info:
        TrainConfig.from_config(section)
    assert "training.lora.rank" in str(info.value)
    assert "training.lora.target_modules" in str(info.value)
    with pytest.raises(ConfigError, match="data_subset"):
        TrainConfig.from_config(_section(data_subset="TODO"))


def test_from_config_rejects_unknown_mode_and_needs_temperature_only_for_distillation() -> None:
    with pytest.raises(ConfigError, match="mode"):
        TrainConfig.from_config(_section(mode="rl"))
    with pytest.raises(ConfigError, match="kd_temperature"):
        TrainConfig.from_config(_section(kd_temperature="TODO"))
    sft = TrainConfig.from_config(_section(mode="sft", kd_weight=0.0, kd_temperature="TODO"))
    assert sft.mode == "sft" and sft.kd_temperature == 1.0


def test_lora_config_expands_target_groups() -> None:
    both = LoRAConfig(4, 8, 0.0, ("decoder_attention", "decoder_mlp")).projections()
    assert both == TARGET_GROUPS["decoder_attention"] + TARGET_GROUPS["decoder_mlp"]
    assert len(both) == 7
    assert LoRAConfig(4, 8, 0.0, ("decoder_mlp",)).projections() == TARGET_GROUPS["decoder_mlp"]
    with pytest.raises(ValueError, match="unknown LoRA target"):
        LoRAConfig(4, 8, 0.0, ("vision",)).projections()


def test_target_regex_matches_only_decoder_projections() -> None:
    import re

    pattern = re.compile(target_regex("model.language_model.layers", ["self_attn.q_proj"]))
    assert pattern.fullmatch("model.language_model.layers.3.self_attn.q_proj")
    assert not pattern.fullmatch("model.visual.blocks.0.self_attn.q_proj")
    assert not pattern.fullmatch("model.language_model.layers.3.self_attn.k_proj")
    assert not pattern.fullmatch("model.language_model.layers.3.self_attn.q_proj.lora_A")


def test_select_subset_is_seeded_and_bounded() -> None:
    ids = [f"s{i}" for i in range(20)]
    assert select_subset(ids, 5, seed=1) == select_subset(ids, 5, seed=1)
    assert select_subset(ids, 5, seed=1) != select_subset(ids, 5, seed=2)
    assert sorted(select_subset(ids, None, seed=3)) == list(range(20))
    assert len(select_subset(ids, 5, seed=1)) == 5
    assert len(set(select_subset(ids, 5, seed=1))) == 5


def test_learning_rate_schedules() -> None:
    base = TrainConfig.from_config(_section(steps=10, learning_rate=1.0))
    assert [_learning_rate(base, s) for s in (0, 5, 9)] == [1.0, 1.0, 1.0]
    warm = TrainConfig.from_config(_section(steps=10, learning_rate=1.0, warmup_steps=4))
    assert [_learning_rate(warm, s) for s in range(5)] == [0.25, 0.5, 0.75, 1.0, 1.0]
    linear = TrainConfig.from_config(_section(steps=10, learning_rate=1.0, lr_schedule="linear"))
    assert _learning_rate(linear, 0) == 1.0
    assert _learning_rate(linear, 5) == pytest.approx(0.5)
    assert _learning_rate(linear, 9) == pytest.approx(0.1)


def _sample(i: int) -> VQASample:
    return VQASample(f"s{i}", "d", f"img/{i}.jpg", f"question {i}", f"answer {i}")


def test_examples_from_samples_joins_images_and_teacher_records() -> None:
    samples = [_sample(i) for i in range(3)]
    plain = examples_from_samples(samples, Path("/data"))
    assert [e.sample_id for e in plain] == ["s0", "s1", "s2"]
    assert plain[0].image == Path("/data/img/0.jpg") and plain[0].teacher is None
    records = {"s0": "rec0", "s2": "rec2"}
    with_teacher = examples_from_samples(samples, Path("/data"), records=records)
    assert [e.sample_id for e in with_teacher] == ["s0", "s2"]  # no record, no example
    assert [e.teacher for e in with_teacher] == ["rec0", "rec2"]
    assert isinstance(with_teacher[0], TrainExample)


def test_stage_entry_stops_on_open_settings(tmp_path: Path) -> None:
    run = RunContext("E2", seed=0)
    with pytest.raises(ConfigError, match="training.steps"):
        run_student_training({"training": _section(steps="TODO")}, run)
    config = {
        "training": _section(),
        "model": {"config": "TODO"},
        "inputs": {"image_longest_side": "TODO"},
        "paths": {"results_root": str(tmp_path)},
    }
    with pytest.raises(ConfigError) as info:
        run_student_training(config, run)
    assert "model.config" in str(info.value) and "inputs.image_longest_side" in str(info.value)


def test_qakd_runs_the_e4_stage_which_names_what_is_open() -> None:
    from medcompress.config import ConfigError, RunContext

    with pytest.raises(ConfigError, match="--exp"):
        run_student_training({"training": _section(mode="qakd")}, None)
    with pytest.raises(ConfigError, match="quantization.recipe"):
        run_student_training({"training": _section(mode="qakd")}, RunContext("E4", seed=0))
