"""Efficiency on real hardware: parameters, file size, memory, latency, throughput, energy.

All numbers are measured, not estimated, on the hardware recorded by
``medcompress.utils.hardware.describe_hardware``. Final timing runs on a rented
GPU (type TODO); development numbers from Kaggle T4s are labelled as such.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


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


def count_parameters(model: VLMWrapper) -> int:
    """Total parameters after physical pruning. Used by: all experiments."""
    raise NotImplementedError("TODO: count parameters.")


def file_size_bytes(checkpoint_dir: Path) -> int:
    """Size on disk of the saved model (weights + adapters). Used by: all."""
    raise NotImplementedError("TODO: sum file sizes of a checkpoint directory.")


def measure_peak_memory(
    model: VLMWrapper, samples: Sequence[VQASample], protocol: Mapping[str, Any]
) -> int:
    """Peak GPU memory (bytes) during inference under the frozen protocol. Used by: all."""
    raise NotImplementedError("TODO: peak memory during inference.")


def measure_latency(
    model: VLMWrapper, samples: Sequence[VQASample], protocol: Mapping[str, Any]
) -> tuple[float, float]:
    """Return (cold, warm) latency in seconds per sample (warm-up/repeats from protocol)."""
    raise NotImplementedError("TODO: cold and warm latency.")


def measure_throughput(
    model: VLMWrapper, samples: Sequence[VQASample], protocol: Mapping[str, Any]
) -> float:
    """Samples per second at the protocol batch size."""
    raise NotImplementedError("TODO: throughput.")


def measure_energy(
    model: VLMWrapper, samples: Sequence[VQASample], protocol: Mapping[str, Any]
) -> float | None:
    """Energy per sample in joules if the GPU exposes power readings, else None."""
    raise NotImplementedError("TODO: energy via GPU power sampling, if measurable.")


def check_low_precision_kernels(model: VLMWrapper) -> bool:
    """Verify that quantized layers actually run low-precision kernels.

    Must fail if a 4-bit/8-bit model silently dequantizes to fp16 for every
    matmul, which would make memory/latency claims misleading.

    Used by: E3, E4, E8, E9.
    """
    raise NotImplementedError("TODO: confirm low-precision kernels are used.")


def run_efficiency_benchmark(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/07_benchmark_efficiency.py``.

    Builds an :class:`EfficiencyReport` for the run's model and appends rows to
    ``<run_dir>/efficiency.csv``.

    Used by: all experiments.
    """
    raise NotImplementedError("TODO: measure and write efficiency.csv.")
