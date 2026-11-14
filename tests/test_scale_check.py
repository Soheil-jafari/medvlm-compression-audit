"""The scale check's pieces on the tiny model: the verdict, the real removal set, stopping a
production training loop after a few updates (2026-11-14)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medcompress import scale_check
from medcompress.utils.memory import GrowthCheck


def test_the_verdict_fails_on_growth_or_a_peak_without_headroom() -> None:
    mib = 1 << 20
    flat = GrowthCheck("gpu", warmup=1)
    for value in (9000, 8000, 8010, 8005):
        flat.add(value * mib)
    ok = scale_check.CheckResult("e7_first_order")
    scale_check.judge(ok, [flat], 9.2)
    assert ok.status == "PASS" and ok.numbers["gpu_peak_gib"] == 9.2
    high = scale_check.CheckResult("e7_first_order")
    scale_check.judge(high, [flat], 14.18)  # RC-G2 session 3's figure
    assert high.status == "FAIL" and "headroom" not in high.lines[-1]
    assert "leaves less than" in high.lines[-1]
    climb = GrowthCheck("gpu", warmup=1)
    for step in range(6):
        climb.add((8000 + 400 * step) * mib)
    grows = scale_check.CheckResult("recovery")
    scale_check.judge(grows, [climb], 10.0)
    assert grows.status == "FAIL" and any("grows step over step" in x for x in grows.lines)
    two = scale_check.CheckResult("e9_eval")
    scale_check.judge(two, [], 20.0, gpus=2)
    assert two.status == "PASS"


def test_the_report_has_one_row_per_check(tmp_path: Path) -> None:
    result = scale_check.CheckResult("gptq", lines=["4 layers"], numbers={"gpu_peak_gib": 9.5})
    path = result.write(tmp_path)
    assert path.name == "gptq.json"
    table = scale_check.report_table([{**result.__dict__, "host": "3.1 / 20.0", "minutes": 6.0}])
    assert "| gptq | PASS | 9.50 | 3.1 / 20.0 | 6.0 | 4 layers |" in table
    assert set(scale_check.EXPERIMENT_OF) == set(scale_check.CHECKS)


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

import tiny_vlm  # noqa: E402
from eval_fixtures import write_dataset  # noqa: E402
from kd_fixtures import teacher_records  # noqa: E402
from test_recovery import RECOVERY, _config  # noqa: E402


def test_the_removal_set_follows_the_magnitude_order_to_the_level(tmp_path: Path) -> None:
    from medcompress.compression.pruning.units import UnitType, enumerate_units

    config = _config(tmp_path)
    model = tiny_vlm.tiny_wrapper(seed=0)
    units = enumerate_units(model, [UnitType(t) for t in config["pruning"]["units"]["types"]],
                            mlp_channel_group_size=16)  # fmt: skip
    decoder = sum(u.num_params for u in units if u.unit_type is UnitType.DECODER_LAYER)
    removal = scale_check.magnitude_removal(model, config, 0.3)
    assert sum(u.num_params for u in removal) >= 0.3 * decoder
    shorter = scale_check.magnitude_removal(model, config, 0.1)
    assert len(shorter) <= len(removal)
    assert all(u.layer_index not in (0, 2) or u.unit_type is not UnitType.DECODER_LAYER
               for u in removal)  # the protected first and last layers stay  # fmt: skip


def test_a_production_training_loop_stops_after_the_updates_the_check_needs(
    tmp_path: Path,
) -> None:
    from medcompress.compression import recovery
    from medcompress.compression.distillation import offline_kd
    from medcompress.compression.distillation.offline_kd import examples_from_samples

    train = write_dataset(tmp_path)
    examples = examples_from_samples(train, tmp_path, records=teacher_records(train, tmp_path))
    model = tiny_vlm.tiny_wrapper(seed=0)
    gpu, host = GrowthCheck("gpu"), GrowthCheck("host", warmup=0)
    progress: Any = scale_check._run_training(
        lambda: recovery.run_recovery(model, examples, {**RECOVERY, "steps": 5}, seed=0),
        3, gpu, host,
    )  # fmt: skip
    assert progress.step == 3 and progress.skipped_steps == []
    assert len(host.readings) == 3 and offline_kd.AFTER_STEP_HOOKS == []
    with pytest.raises(RuntimeError, match="ended before"):
        scale_check._run_training(
            lambda: recovery.run_recovery(tiny_vlm.tiny_wrapper(seed=0), examples,
                                          {**RECOVERY, "steps": 2}, seed=0),
            3, gpu, host,
        )  # fmt: skip
    assert offline_kd.AFTER_STEP_HOOKS == []


def test_the_save_of_a_pruned_model_reports_its_host_memory(tmp_path: Path) -> None:
    from medcompress.compression.lora import LoRAConfig, attach_lora
    from medcompress.compression.pruning.apply import remove_units

    config = _config(tmp_path)
    config["paths"]["checkpoint_root"] = str(tmp_path / "checkpoints")
    model = tiny_vlm.tiny_wrapper(seed=0)
    removal = scale_check.magnitude_removal(model, config, 0.3)
    remove_units(model, removal)
    attach_lora(model, LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=("decoder_mlp",)))
    result = scale_check.CheckResult("recovery")
    scale_check.save_pruned_host_memory(model, removal, config, result)
    assert not (tmp_path / "checkpoints" / "scale_check_pruned").exists()  # removed again
    if Path("/proc/self/status").exists():
        assert "process_peak_rss_gib" in result.numbers


def test_e9s_check_waits_without_its_hand_offs(tmp_path: Path) -> None:
    from eval_fixtures import eval_config

    result = scale_check.CheckResult("e9_eval")
    scale_check.check_e9_eval(eval_config(tmp_path), result, batches=1)
    assert result.status == "WAIT" and "not attached" in result.lines[-1]


def test_each_check_reads_only_the_hand_offs_of_its_card() -> None:
    import importlib.util

    from medcompress.cards import build_plan

    path = Path(__file__).resolve().parents[1] / "scripts" / "scale_check.py"
    spec = importlib.util.spec_from_file_location("scale_check_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert set(script.PLAN_OF) == set(scale_check.CHECKS)
    plans = {name: build_plan(card, seed, variants=list(v) or None)
             for name, (card, seed, v) in script.PLAN_OF.items()}  # fmt: skip
    e9 = plans["e9_eval"]
    assert {n.kind for n in e9.needs} == {"teacher_adapter"}  # never a student adapter
    assert {n.kind for n in plans["recovery"].needs} == {"adapter", "teacher_cache"}


def test_a_release_that_leaves_nothing_on_the_gpu_passes() -> None:
    model = tiny_vlm.tiny_wrapper(seed=0)
    scale_check.release_and_verify(model, "tiny")  # no CUDA here: only the release
    assert all(p.device.type == "meta" for p in model.module.parameters())
