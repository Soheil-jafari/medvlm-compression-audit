"""Quantization-aware distillation (QAKD), GRACE-style.

E4 trains the student with simulated low-precision weights while distilling from
the cached E0 teacher outputs. The exact objective follows GRACE (Chen et al.,
arXiv:2601.22709); TODO: re-read the paper and write the objective here before
implementing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.models.base import VLMWrapper


@dataclass(frozen=True)
class QAKDConfig:
    """Settings for quantization-aware distillation (values from configs/experiments/E4.yaml)."""

    weight_bits: int
    group_size: int | None
    kd_temperature: float
    kd_weight: float
    steps: int
    learning_rate: float
    extra: Mapping[str, Any]


def train_qakd(
    student: VLMWrapper, teacher_cache_dir: Path, cfg: QAKDConfig, *, seed: int
) -> VLMWrapper:
    """Train the student with fake-quantized weights and distillation from the cache.

    Used by: E4 (3 seeds).
    """
    raise NotImplementedError("TODO: GRACE-style quantization-aware distillation (E4).")
