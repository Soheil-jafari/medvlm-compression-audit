"""Efficiency on real hardware: parameters, bytes, peak memory, latency, throughput, energy.

All numbers are measured, not estimated, on the hardware recorded by
``medcompress.utils.hardware.describe_hardware``. Final timing runs on a rented
GPU (type TODO); development numbers from Kaggle T4s are labelled as such.

One *sweep* answers the model with the same questions several times: the first call is
the cold latency, a few warm-up passes follow, and the timed passes give the warm latency,
the throughput and (while they run) the peak memory and, where the GPU reports power, the
energy. Every arm is swept with the same protocol and the same samples, so the numbers
are comparable between arms.
"""

from __future__ import annotations

import random
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.config import ConfigError, find_todos

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

TIMING_SEED = 0
"""Seed of the draw of the timing samples (fixed, so every arm is timed on the same ones)."""
POWER_INTERVAL_S = 0.05


@dataclass(frozen=True)
class EfficiencyReport:
    """Measured footprint of one model under one protocol."""

    num_parameters: int | None = None
    file_size_bytes: int | None = None
    peak_memory_bytes: int | None = None
    cold_latency_s: float | None = None
    warm_latency_s: float | None = None
    throughput_samples_per_s: float | None = None
    energy_joules_per_sample: float | None = None
    low_precision_kernels_ok: bool | None = None
    hardware: Mapping[str, Any] = field(default_factory=dict)
    protocol: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EfficiencyProtocol:
    """The measurement protocol (``budgets.measurement`` of ``configs/budgets.yaml``).

    ``samples`` is either a list of sample ids (used in that order) or a number of
    validation images drawn with :data:`TIMING_SEED`. The questions are the real ones, and
    the image side is the global ``inputs.image_longest_side``, so neither is set here.
    """

    hardware: str
    batch_size: int
    max_new_tokens: int
    warmup_runs: int
    repeats: int
    samples: tuple[str, ...] | int

    @classmethod
    def from_config(cls, measurement: Mapping[str, Any]) -> EfficiencyProtocol:
        """Read the ``measurement`` block; every value that is still ``TODO`` is listed."""
        keys = ("hardware", "batch_size", "max_new_tokens", "warmup_runs", "repeats", "samples")
        open_keys = [
            f"budgets.measurement.{k}"
            for k in keys
            if measurement.get(k) is None or find_todos({"v": measurement.get(k)})
        ]
        if open_keys:
            raise ConfigError(f"settings still open: {', '.join(open_keys)}")
        samples = measurement["samples"]
        if isinstance(samples, list):
            chosen: tuple[str, ...] | int = tuple(str(s) for s in samples)
        elif isinstance(samples, int) and not isinstance(samples, bool):
            chosen = samples
        else:
            raise ConfigError("budgets.measurement.samples must be a list of ids or a count")
        for key in ("batch_size", "max_new_tokens", "repeats"):
            if not isinstance(measurement[key], int) or measurement[key] < 1:
                raise ConfigError(f"budgets.measurement.{key} must be a positive integer")
        if not isinstance(measurement["warmup_runs"], int) or measurement["warmup_runs"] < 0:
            raise ConfigError("budgets.measurement.warmup_runs must be a non-negative integer")
        if not chosen or (isinstance(chosen, int) and chosen < 1):
            raise ConfigError("budgets.measurement.samples must name at least one sample")
        return cls(
            hardware=str(measurement["hardware"]),
            batch_size=int(measurement["batch_size"]),
            max_new_tokens=int(measurement["max_new_tokens"]),
            warmup_runs=int(measurement["warmup_runs"]),
            repeats=int(measurement["repeats"]),
            samples=chosen,
        )

    def to_dict(self) -> dict[str, Any]:
        """The protocol as recorded next to the numbers."""
        return {
            "hardware": self.hardware,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "warmup_runs": self.warmup_runs,
            "repeats": self.repeats,
            "samples": list(self.samples) if isinstance(self.samples, tuple) else self.samples,
        }


def timing_samples(pool: Sequence[VQASample], protocol: EfficiencyProtocol) -> list[VQASample]:
    """The samples every arm is timed on: the named ids, or a seeded draw of images.

    A count draws one question per image (the first by sample id) from a seeded shuffle of the
    images, so the draw does not depend on the order of ``pool``.
    """
    if isinstance(protocol.samples, tuple):
        by_id = {s.sample_id: s for s in pool}
        missing = [i for i in protocol.samples if i not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} timing samples are not in the pool: {missing[:3]}")
        return [by_id[i] for i in protocol.samples]
    first: dict[str, VQASample] = {}
    for sample in sorted(pool, key=lambda s: (s.image_path, s.sample_id)):
        first.setdefault(sample.image_path, sample)
    images = sorted(first)
    if protocol.samples > len(images):
        raise ValueError(f"asked for {protocol.samples} timing images, the pool has {len(images)}")
    random.Random(TIMING_SEED).shuffle(images)
    return [first[image] for image in images[: protocol.samples]]


def count_parameters(model: VLMWrapper | Any) -> int:
    """Total parameters after physical pruning (shared tensors once). Used by: all experiments."""
    from medcompress.compression.pruning import surgery

    return surgery.count_parameters(getattr(model, "module", model))


def state_bytes(model: VLMWrapper | Any) -> int:
    """Bytes of the parameters and buffers, which is what a saved copy of the weights holds."""
    module = getattr(model, "module", model)
    seen: set[int] = set()
    total = 0
    for tensor in [*module.parameters(), *module.buffers()]:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def state_bytes_by_part(model: VLMWrapper | Any) -> dict[str, int]:
    """Split :func:`state_bytes` into the decoder layers, the vision side and the rest.

    ``decoder_layers_bytes`` is what pruning of layers, attention groups and MLP channels can
    shrink; ``vision_bytes`` is every tensor whose module path names a vision tower;
    ``other_bytes`` holds the token embeddings, the output head, the norms and the connector.
    The parts add up to :func:`state_bytes`. RC-B0 reports them so that B1 to B3 can be set as
    fractions of the part that pruning can reach.

    Used by: E2 (RC-B0), the budget memo.
    """
    from medcompress.compression.pruning import surgery

    module = getattr(model, "module", model)
    stack = surgery.find_decoder_stack(module)
    prefix = f"{stack.path}."
    parts = {"decoder_layers_bytes": 0, "vision_bytes": 0, "other_bytes": 0}
    seen: set[int] = set()
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        size = tensor.numel() * tensor.element_size()
        if name.startswith(prefix):
            parts["decoder_layers_bytes"] += size
        elif any(marker in name.lower() for marker in surgery.VISION_MARKERS):
            parts["vision_bytes"] += size
        else:
            parts["other_bytes"] += size
    return parts


def file_size_bytes(checkpoint_dir: Path) -> int:
    """Size on disk of the saved model (weights, manifest and adapters). Used by: all."""
    root = Path(checkpoint_dir)
    if root.is_file():
        return root.stat().st_size
    if not root.is_dir():
        raise FileNotFoundError(f"no checkpoint at {root}")
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


class MemoryMeter:
    """Peak memory of the model's GPUs; reads nothing (None) when there is no CUDA device."""

    def reset(self) -> None:
        """Start a new peak."""
        from medcompress.models.smoke import reset_peak

        reset_peak()

    def peak(self) -> int | None:
        """Peak allocated bytes since :meth:`reset`."""
        from medcompress.models.smoke import peak_gpu_bytes

        return peak_gpu_bytes()


class PowerSampler:
    """Samples a power reading (watts) in a thread while a block runs; energy = mean W x s."""

    def __init__(
        self, read: Callable[[], float | None], interval: float = POWER_INTERVAL_S
    ) -> None:
        self._read, self._interval = read, interval
        self._values: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self.seconds = 0.0

    def __enter__(self) -> PowerSampler:
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.seconds = time.perf_counter() - self._started

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._read()
            if value is not None:
                self._values.append(float(value))
            self._stop.wait(self._interval)

    def joules(self) -> float | None:
        """Mean power times the sampled duration; None when no reading was possible."""
        return statistics.fmean(self._values) * self.seconds if self._values else None


def gpu_power_reader() -> Callable[[], float | None] | None:
    """A function that reads the power draw of GPU 0 in watts, or None if it cannot be read."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.power_draw(0)  # needs the NVML bindings; raises if they are missing
    except Exception:  # noqa: BLE001 - any failure means "not measurable here"
        return None
    return lambda: float(torch.cuda.power_draw(0)) / 1000.0  # mW -> W


@dataclass(frozen=True)
class Sweep:
    """What one sweep measured (the numbers of an :class:`EfficiencyReport` that need a run)."""

    cold_latency_s: float
    warm_latency_s: float
    throughput_samples_per_s: float
    peak_memory_bytes: int | None
    energy_joules_per_sample: float | None
    n_samples: int
    mean_prompt_chars: float


def run_sweep(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    protocol: EfficiencyProtocol,
    data_root: Path,
    *,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
    memory: MemoryMeter | None = None,
    power_read: Callable[[], float | None] | None = None,
) -> Sweep:
    """Answer ``samples`` under ``protocol`` and time it (see the module docstring).

    The wrapper's own batch size is set to the protocol's for the sweep and restored after.
    Latency is seconds per sample, throughput samples per second of the timed passes.
    """
    if not samples:
        raise ValueError("no samples to time")
    if synchronize is None:
        from medcompress.models.smoke import synchronize as sync

        synchronize = sync
    meter = memory or MemoryMeter()
    images = [Path(data_root) / s.image_path for s in samples]
    questions = [s.question for s in samples]
    size = protocol.batch_size
    chunks = [range(i, min(i + size, len(samples))) for i in range(0, len(samples), size)]

    def one_pass() -> list[float]:
        times = []
        for rows in chunks:
            synchronize()
            started = clock()
            model.generate(
                [images[i] for i in rows],
                [questions[i] for i in rows],
                max_new_tokens=protocol.max_new_tokens,
            )
            synchronize()
            times.append(clock() - started)
        return times

    previous = getattr(model, "batch_size", None)
    if previous is not None:
        model.batch_size = size  # type: ignore[attr-defined]
    try:
        meter.reset()
        first = one_pass()
        cold = first[0] / len(chunks[0])
        for _ in range(protocol.warmup_runs):
            one_pass()
        sampler = PowerSampler(power_read) if power_read is not None else None
        timed: list[list[float]] = []
        started = clock()
        if sampler is not None:
            with sampler:
                timed = [one_pass() for _ in range(protocol.repeats)]
        else:
            timed = [one_pass() for _ in range(protocol.repeats)]
        wall = clock() - started
        peak = meter.peak()
    finally:
        if previous is not None:
            model.batch_size = previous  # type: ignore[attr-defined]
    per_sample = [
        t / len(rows) for pass_times in timed for t, rows in zip(pass_times, chunks, strict=True)
    ]
    energy = sampler.joules() if sampler is not None else None
    runs = protocol.repeats * len(samples)
    return Sweep(
        cold_latency_s=cold,
        warm_latency_s=statistics.median(per_sample),
        throughput_samples_per_s=runs / wall if wall > 0 else float("inf"),
        peak_memory_bytes=peak,
        energy_joules_per_sample=None if energy is None else energy / runs,
        n_samples=len(samples),
        mean_prompt_chars=statistics.fmean(len(q) for q in questions),
    )


def measure_footprint(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    protocol: EfficiencyProtocol,
    data_root: Path,
    *,
    hardware: Mapping[str, Any] | None = None,
    **sweep_options: Any,
) -> EfficiencyReport:
    """The footprint of ``model`` under ``protocol`` (a sweep plus parameters).

    Used by: all experiments; the budget search of E5-E8 and A1-A6 measures candidate
    removal sets with it.
    """
    from medcompress.utils.hardware import describe_hardware

    sweep = run_sweep(model, samples, protocol, data_root, **sweep_options)
    return EfficiencyReport(
        num_parameters=count_parameters(model),
        peak_memory_bytes=sweep.peak_memory_bytes,
        cold_latency_s=sweep.cold_latency_s,
        warm_latency_s=sweep.warm_latency_s,
        throughput_samples_per_s=sweep.throughput_samples_per_s,
        energy_joules_per_sample=sweep.energy_joules_per_sample,
        hardware=dict(hardware) if hardware is not None else describe_hardware().to_dict(),
        protocol={**protocol.to_dict(), "n_samples": sweep.n_samples},
    )


QUANTIZED_ARMS: tuple[str, ...] = ("E3", "E4", "E8", "E9")


def check_low_precision_kernels(model: VLMWrapper) -> bool:
    """True if every quantized decoder layer is a low-precision library module.

    Fails (False) for a plain 16-bit ``Linear``, for the dequantizing reference module and
    for a module without integer weights, so a 4-bit or 8-bit model that silently computes in
    16 bit is never reported as quantized. Details:
    :func:`medcompress.compression.quantization.kernels.kernel_report`.

    Used by: E3, E4, E8, E9.
    """
    from medcompress.compression.quantization.kernels import kernel_report

    return kernel_report(model).ok


REPORT_METRICS: tuple[str, ...] = (
    "num_parameters",
    "state_bytes",
    "file_size_bytes",
    "peak_memory_bytes",
    "cold_latency_s",
    "warm_latency_s",
    "throughput_samples_per_s",
    "energy_joules_per_sample",
)


def report_rows(
    report: EfficiencyReport, run: RunContext, *, state: int | None = None
) -> list[dict[str, Any]]:
    """Rows of the results schema for the numbers a report has (missing numbers are left out)."""
    values = {**{m: getattr(report, m, None) for m in REPORT_METRICS}, "state_bytes": state}
    return [
        {
            "experiment": run.experiment_tag,
            "budget": run.budget or "",
            "seed": run.seed,
            "metric": metric,
            "value": float(value),
        }
        for metric in REPORT_METRICS
        if (value := values[metric]) is not None
    ]


def run_efficiency_benchmark(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    model: VLMWrapper | None = None,
    pool: Sequence[VQASample] | None = None,
) -> None:
    """Stage entry point for ``scripts/07_benchmark_efficiency.py``.

    Measures the run's model under the protocol of ``configs/budgets.yaml`` on the same
    validation samples for every arm and writes ``efficiency.csv`` (results schema) and
    ``efficiency_setup.json`` (hardware and protocol) to the run directory.

    Used by: all experiments.
    """
    from medcompress.config import load_config
    from medcompress.utils.io import upsert_results, write_json

    if run is None:
        raise ConfigError("--exp is required")
    protocol = EfficiencyProtocol.from_config(
        load_config(config["components"]["budgets"])["budgets"].get("measurement") or {}
    )
    paths = config["paths"]
    side = config.get("inputs", {}).get("image_longest_side")
    if model is None:
        if side is None or find_todos({"v": side}):
            raise ConfigError("settings still open: inputs.image_longest_side")
        from medcompress.evaluation.runner import load_run_model

        model = load_run_model(config, run, int(side))
    if pool is None:
        from medcompress.data.prepared import load_prepared

        dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
        pool = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))["val"]
    samples = timing_samples(pool, protocol)
    report = measure_footprint(
        model,
        samples,
        protocol,
        Path(paths["data_root"]),
        power_read=gpu_power_reader(),
    )
    target = run.run_dir(paths["results_root"]) / "efficiency.csv"
    rows = report_rows(report, run, state=state_bytes(model))
    rows += [
        {
            "experiment": run.experiment_tag,
            "budget": run.budget or "",
            "seed": run.seed,
            "metric": metric,
            "value": float(value),
        }
        for metric, value in state_bytes_by_part(model).items()
    ]
    setup: dict[str, Any] = {"hardware": report.hardware, "protocol": report.protocol}
    if run.experiment_id in QUANTIZED_ARMS:
        from medcompress.compression.quantization.kernels import kernel_report

        kernels = kernel_report(model)
        rows.append(
            {
                "experiment": run.experiment_tag,
                "budget": run.budget or "",
                "seed": run.seed,
                "metric": "low_precision_kernels_ok",
                "value": float(kernels.ok),
            }
        )
        setup["kernels"] = {"classes": kernels.classes, "problems": list(kernels.problems)}
    upsert_results(rows, target)
    write_json(setup, target.with_name("efficiency_setup.json"))
    print(f"wrote {len(rows)} rows to {target}")
