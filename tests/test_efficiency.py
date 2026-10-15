"""The efficiency protocol, the sweep and the benchmark stage (fake clocks, tiny model on CPU)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import pytest

from medcompress.config import ConfigError, RunContext
from medcompress.data.base import VQASample
from medcompress.evaluation import efficiency as eff
from medcompress.evaluation.efficiency import EfficiencyProtocol

MEASUREMENT: dict[str, Any] = {
    "hardware": "T4",
    "batch_size": 2,
    "max_new_tokens": 4,
    "warmup_runs": 1,
    "repeats": 3,
    "samples": 4,
}


def _protocol(**over: Any) -> EfficiencyProtocol:
    return EfficiencyProtocol.from_config({**MEASUREMENT, **over})


def _pool() -> list[VQASample]:
    return [
        VQASample(f"s{i}-{k}", "d", f"images/{i}.png", f"q{k}", "a") for i in range(8) for k in "ab"
    ]


# -------------------------------------------------------------------------- protocol


def test_the_protocol_lists_every_open_value_and_checks_the_rest() -> None:
    with pytest.raises(ConfigError) as caught:
        EfficiencyProtocol.from_config({"hardware": "TODO", "batch_size": "TODO"})
    for key in ("hardware", "batch_size", "max_new_tokens", "warmup_runs", "repeats", "samples"):
        assert f"budgets.measurement.{key}" in str(caught.value)
    for change, match in (
        ({"batch_size": 0}, "batch_size must be a positive"),
        ({"repeats": 0}, "repeats must be a positive"),
        ({"warmup_runs": -1}, "warmup_runs must be a non-negative"),
        ({"samples": 0}, "at least one"),
        ({"samples": []}, "at least one"),
        ({"samples": "all"}, "list of ids or a count"),
    ):
        with pytest.raises(ConfigError, match=match):
            _protocol(**change)
    ids = _protocol(samples=["s1-a", "s0-b"])
    assert ids.samples == ("s1-a", "s0-b") and ids.to_dict()["samples"] == ["s1-a", "s0-b"]
    assert _protocol().to_dict()["samples"] == 4


def test_timing_samples_are_the_named_ids_or_a_seeded_draw_of_images() -> None:
    pool = _pool()
    named = eff.timing_samples(pool, _protocol(samples=["s3-b", "s0-a"]))
    assert [s.sample_id for s in named] == ["s3-b", "s0-a"]
    drawn = eff.timing_samples(pool, _protocol(samples=5))
    assert len({s.image_path for s in drawn}) == 5  # one question per image
    assert [s.sample_id for s in drawn] == [
        s.sample_id for s in eff.timing_samples(pool, _protocol(samples=5))
    ]
    shuffled = list(reversed(pool))
    assert [s.sample_id for s in eff.timing_samples(shuffled, _protocol(samples=5))] == [
        s.sample_id for s in drawn
    ]  # the draw does not depend on the order of the pool
    assert [s.image_path for s in eff.timing_samples(pool, _protocol(samples=3))] == [
        s.image_path for s in drawn[:3]
    ]
    with pytest.raises(ValueError, match="not in the pool"):
        eff.timing_samples(pool, _protocol(samples=["nope"]))
    with pytest.raises(ValueError, match="pool has 8"):
        eff.timing_samples(pool, _protocol(samples=9))


# ----------------------------------------------------------------------------- sweep


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeModel:
    """Takes 1 s for its first call and 0.1 s per sample afterwards."""

    def __init__(self, clock: _Clock) -> None:
        self.clock, self.batch_size, self.calls, self.sizes = clock, 8, 0, []

    def generate(self, images: Any, questions: Any, *, max_new_tokens: int) -> list[Any]:
        assert len(images) == len(questions) <= 2 and max_new_tokens == 4
        self.sizes.append(self.batch_size)
        self.clock.now += 1.0 if self.calls == 0 else 0.1 * len(questions)
        self.calls += 1
        return []


class _Meter:
    def __init__(self) -> None:
        self.events: list[str] = []

    def reset(self) -> None:
        self.events.append("reset")

    def peak(self) -> int:
        self.events.append("peak")
        return 123


def test_a_sweep_times_cold_warm_throughput_and_reads_the_peak_after_the_runs() -> None:
    clock, meter = _Clock(), _Meter()
    model = _FakeModel(clock)
    samples = _pool()[:4]
    sweep = eff.run_sweep(
        model,  # type: ignore[arg-type]
        samples,
        _protocol(),
        Path("/data"),
        clock=clock,
        synchronize=lambda: None,
        memory=meter,  # type: ignore[arg-type]
    )
    assert sweep.cold_latency_s == pytest.approx(1.0 / 2)  # first call, per sample
    assert sweep.warm_latency_s == pytest.approx(0.1)
    # 3 timed passes x 4 samples in 3 x 4 x 0.1 s of model time
    assert sweep.throughput_samples_per_s == pytest.approx(12 / 1.2)
    assert sweep.peak_memory_bytes == 123 and meter.events == ["reset", "peak"]
    assert sweep.n_samples == 4 and sweep.energy_joules_per_sample is None
    assert model.calls == 2 * (1 + 1 + 3)  # two chunks per pass: cold, warm-up, three timed
    assert set(model.sizes) == {2} and model.batch_size == 8  # protocol size inside, restored after
    with pytest.raises(ValueError, match="no samples"):
        eff.run_sweep(model, [], _protocol(), Path("/data"))  # type: ignore[arg-type]


def test_the_power_sampler_turns_readings_into_joules() -> None:
    with eff.PowerSampler(lambda: 100.0, interval=0.005) as sampler:
        time.sleep(0.05)
    assert sampler.joules() == pytest.approx(100.0 * sampler.seconds)
    with eff.PowerSampler(lambda: None, interval=0.005) as silent:
        time.sleep(0.02)
    assert silent.joules() is None


def test_the_sweep_adds_energy_per_sample_when_power_can_be_read() -> None:
    clock = _Clock()
    model = _FakeModel(clock)

    def slow_clock() -> float:
        time.sleep(0.001)  # lets the sampling thread run between the fake model calls
        return clock()

    sweep = eff.run_sweep(
        model,  # type: ignore[arg-type]
        _pool()[:4],
        _protocol(),
        Path("/data"),
        clock=slow_clock,
        synchronize=lambda: None,
        memory=_Meter(),  # type: ignore[arg-type]
        power_read=lambda: 50.0,
    )
    assert sweep.energy_joules_per_sample is not None and sweep.energy_joules_per_sample > 0


# --------------------------------------------------------------- sizes and the report


def test_sizes_and_rows(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a/w.bin").write_bytes(b"x" * 100)
    (tmp_path / "m.json").write_bytes(b"y" * 7)
    assert eff.file_size_bytes(tmp_path) == 107
    assert eff.file_size_bytes(tmp_path / "m.json") == 7
    with pytest.raises(FileNotFoundError):
        eff.file_size_bytes(tmp_path / "missing")
    report = eff.EfficiencyReport(num_parameters=10, peak_memory_bytes=5, warm_latency_s=0.5)
    rows = eff.report_rows(report, RunContext("E7", seed=1, budget="B2"), state=40)
    assert {r["metric"]: r["value"] for r in rows} == {
        "num_parameters": 10.0,
        "state_bytes": 40.0,
        "peak_memory_bytes": 5.0,
        "warm_latency_s": 0.5,
    }
    assert {(r["experiment"], r["budget"], r["seed"]) for r in rows} == {("E7", "B2", 1)}


def test_the_kernel_check_rejects_16_bit_and_reference_modules() -> None:
    from medcompress.compression.quantization.kernels import KernelReport, classify_module

    class Tensor:
        def __init__(self, dtype: str) -> None:
            self.dtype = dtype

    Linear = type("Linear", (), {"__module__": "torch.nn.modules.linear"})
    plain = Linear()
    plain.weight = Tensor("torch.float16")
    assert classify_module(plain)[1] == "plain 16-bit Linear"

    Reference = type("ReferenceQuantLinear", (), {"REFERENCE_ONLY": True})
    assert "reference" in str(classify_module(Reference())[1])

    Kernel = type("TorchLinear", (), {"__module__": "gptqmodel.nn_modules.qlinear.torch"})
    kernel = Kernel()
    kernel.qweight = Tensor("torch.int32")
    path, problem = classify_module(kernel)
    assert problem is None and path == "gptqmodel.nn_modules.qlinear.torch.TorchLinear"

    Int8 = type("Linear8bitLt", (), {"__module__": "bitsandbytes.nn.modules"})
    int8 = Int8()
    int8.weight = Tensor("torch.int8")
    assert classify_module(int8)[1] is None

    Odd = type("Odd", (), {})
    odd = Odd()
    odd.weight = Tensor("torch.float16")
    assert "no integer weight" in str(classify_module(odd)[1])

    assert KernelReport(3, {"x": 3}).ok and not KernelReport(3, {}, ("bad",)).ok
    assert not KernelReport(0).ok


# ------------------------------------------------------------------ the tiny model


@pytest.fixture
def tiny() -> Any:
    """The tiny offline model module (skips the test when the heavy dependencies are missing)."""
    for name in ("torch", "transformers", "torchvision", "tokenizers", "PIL.Image"):
        pytest.importorskip(name)
    import tiny_vlm

    return tiny_vlm


def test_parameters_and_state_bytes_follow_the_model(tiny: Any) -> None:
    wrapper = tiny.tiny_wrapper(seed=0)
    n = eff.count_parameters(wrapper)
    assert n == sum(p.numel() for p in wrapper.module.parameters())
    expected = sum(p.numel() * p.element_size() for p in wrapper.module.parameters())
    expected += sum(b.numel() * b.element_size() for b in wrapper.module.buffers())
    assert eff.state_bytes(wrapper) == expected


def test_the_bytes_split_into_decoder_vision_and_the_rest(tiny: Any) -> None:
    from medcompress.compression.pruning import surgery

    wrapper = tiny.tiny_wrapper(seed=0)
    parts = eff.state_bytes_by_part(wrapper)
    assert set(parts) == {"decoder_layers_bytes", "vision_bytes", "other_bytes"}
    assert sum(parts.values()) == eff.state_bytes(wrapper)
    assert all(v > 0 for v in parts.values())
    stack = surgery.find_decoder_stack(wrapper.module)
    layer = stack.layers[1]
    one = sum(p.numel() * p.element_size() for p in layer.parameters())
    surgery.remove_layers(wrapper.module, [1])
    after = eff.state_bytes_by_part(wrapper)
    assert parts["decoder_layers_bytes"] - after["decoder_layers_bytes"] == one
    assert after["vision_bytes"] == parts["vision_bytes"]  # only the decoder changes
    assert after["other_bytes"] == parts["other_bytes"]


def test_the_benchmark_stage_writes_the_efficiency_files(tiny: Any, tmp_path: Path) -> None:
    import yaml
    from eval_fixtures import eval_config, write_dataset

    pool = write_dataset(tmp_path)
    config = eval_config(tmp_path)
    block = {"budgets": {"measurement": {**MEASUREMENT, "samples": 3, "repeats": 2}}}
    (tmp_path / "budgets.yaml").write_text(yaml.safe_dump(block))
    config["components"]["budgets"] = str(tmp_path / "budgets.yaml")
    run = RunContext("E2", seed=0)
    eff.run_efficiency_benchmark(config, run, model=tiny.tiny_wrapper(seed=0), pool=pool)
    target = tmp_path / "results/E2/nobudget/seed0/efficiency.csv"
    with target.open() as handle:
        rows = {r["metric"]: float(r["value"]) for r in csv.DictReader(handle)}
    assert rows["num_parameters"] == eff.count_parameters(tiny.tiny_wrapper(seed=0))
    assert rows["warm_latency_s"] > 0 and rows["cold_latency_s"] > 0
    assert rows["throughput_samples_per_s"] > 0 and rows["state_bytes"] > 0
    parts = ("decoder_layers_bytes", "vision_bytes", "other_bytes")
    assert sum(rows[k] for k in parts) == rows["state_bytes"]
    import torch

    if torch.cuda.is_available():  # GPU session: the peak comes from the CUDA allocator
        assert rows["peak_memory_bytes"] >= 0
    else:
        assert "peak_memory_bytes" not in rows  # no CUDA device: nothing is invented
    setup = json.loads((target.parent / "efficiency_setup.json").read_text())
    assert setup["protocol"]["samples"] == 3 and setup["protocol"]["n_samples"] == 3
    assert "platform" in setup["hardware"]
    with pytest.raises(ConfigError, match="--exp"):
        eff.run_efficiency_benchmark(config, None)
    block["budgets"]["measurement"]["repeats"] = "TODO"
    (tmp_path / "budgets.yaml").write_text(yaml.safe_dump(block))
    with pytest.raises(ConfigError, match="budgets.measurement.repeats"):
        eff.run_efficiency_benchmark(config, run, model=tiny.tiny_wrapper(seed=0), pool=pool)


def test_the_removal_measurer_removes_the_units_measures_and_frees_the_copy(
    tiny: Any, tmp_path: Path
) -> None:
    from eval_fixtures import write_dataset

    from medcompress.budgets import make_removal_measurer
    from medcompress.compression.pruning.units import UnitType, enumerate_units

    pool = write_dataset(tmp_path)
    protocol = _protocol(samples=2, repeats=1, warmup_runs=0)
    samples = eff.timing_samples(pool, protocol)
    built: list[Any] = []

    def factory() -> Any:
        built.append(tiny.tiny_wrapper(seed=0))
        return built[-1]

    measure = make_removal_measurer(factory, samples, protocol, tmp_path, "latency")
    units = enumerate_units(
        tiny.tiny_wrapper(seed=0), [UnitType.DECODER_LAYER], mlp_channel_group_size=16
    )
    full = measure([])
    smaller = measure([units[1]])
    assert full > 0 and smaller > 0 and len(built) == 2
    assert eff.count_parameters(built[0]) > eff.count_parameters(built[1])  # copies are separate
    import torch

    by_memory = make_removal_measurer(factory, samples, protocol, tmp_path, "peak_memory")
    if torch.cuda.is_available():  # GPU session: peak memory is measured
        assert by_memory([]) >= 0
    else:
        with pytest.raises(ValueError, match="no peak_memory_bytes"):
            by_memory([])  # no CUDA device: no peak memory to match a budget on


def test_the_final_timing_has_its_own_file_names_and_gpu(tmp_path: Path) -> None:
    import yaml

    (tmp_path / "budgets.yaml").write_text(
        yaml.safe_dump({"budgets": {"measurement": MEASUREMENT}})
    )
    config: dict[str, Any] = {"components": {"budgets": str(tmp_path / "budgets.yaml")}}
    target, measurement = eff.timing_target(config)  # development is the default
    assert (target.rows_file, target.setup_file, target.prefix) == (
        "efficiency.csv",
        "efficiency_setup.json",
        "",
    )
    assert measurement["hardware"] == "T4"
    config["hardware"] = {"timing_role": "final", "final_timing": "TODO"}
    with pytest.raises(ConfigError, match="hardware.final_timing"):
        eff.timing_target(config)
    config["hardware"]["final_timing"] = "rented GPU"
    target, measurement = eff.timing_target(config)
    assert (target.rows_file, target.prefix) == ("efficiency_final.csv", "final:")
    assert measurement["hardware"] == "rented GPU"
    config["hardware"]["timing_role"] = "both"
    with pytest.raises(ConfigError, match="timing_role"):
        eff.timing_target(config)


def test_the_efficiency_table_reads_only_the_final_timing() -> None:
    import pandas as pd

    from medcompress.evaluation.report import efficiency_table

    def row(metric: str, mean: float) -> dict[str, Any]:
        return {"experiment": "E2", "budget": "", "split": "", "metric": metric, "mean": mean,
                "sd": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}  # fmt: skip

    development = pd.DataFrame([row("peak_memory_bytes", 111.0)])
    assert "111" not in efficiency_table(development)
    final = pd.DataFrame([row("peak_memory_bytes", 111.0), row("final:peak_memory_bytes", 222.0)])
    line = next(r for r in efficiency_table(final).splitlines() if r.startswith("E2 "))
    assert "222.000" in line and "111" not in line
