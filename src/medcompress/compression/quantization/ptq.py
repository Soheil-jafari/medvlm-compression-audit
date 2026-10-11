"""Post-training quantization backends: GPTQ, AWQ (weight-only) and W8A8.

Variants of E3: ``w8`` (weight-only 8-bit), ``w4`` (weight-only 4-bit), ``w8a8``
(weights and activations 8-bit, only where the backend and hardware support it).
E8 applies 4-bit PTQ to E7; E9 applies 4-bit PTQ to the teacher.

Which backend (GPTQ vs AWQ) serves which bit width is undecided: it depends on
what runs on Kaggle T4s and on the final timing GPU. Every quantized model must
pass :func:`medcompress.evaluation.efficiency.check_low_precision_kernels` so we
never report memory/latency of a model that silently falls back to fp16 kernels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

PTQ_BACKENDS: tuple[str, ...] = ("gptq", "awq", "w8a8")


@dataclass(frozen=True)
class PTQConfig:
    """Settings for one PTQ run (values come from the experiment config)."""

    backend: str
    weight_bits: int
    activation_bits: int | None
    group_size: int | None
    calibration_size: int
    quantize_vision_encoder: bool
    extra: Mapping[str, Any]


def quantize_gptq(
    model: VLMWrapper, calibration: Sequence[VQASample], cfg: PTQConfig
) -> VLMWrapper:
    """Weight-only GPTQ quantization. Used by: E3 (w8/w4), E8, E9."""
    raise NotImplementedError("TODO: GPTQ via the chosen backend (E3, E8, E9).")


def quantize_awq(model: VLMWrapper, calibration: Sequence[VQASample], cfg: PTQConfig) -> VLMWrapper:
    """Weight-only AWQ quantization. Used by: E3 (w4), E8, E9."""
    raise NotImplementedError("TODO: AWQ via the chosen backend (E3, E8, E9).")


def quantize_w8a8(
    model: VLMWrapper, calibration: Sequence[VQASample], cfg: PTQConfig
) -> VLMWrapper:
    """8-bit weights and activations, where supported. Used by: E3 (w8a8)."""
    raise NotImplementedError("TODO: W8A8 where hardware/backend support it (E3).")


def quantize(model: VLMWrapper, calibration: Sequence[VQASample], cfg: PTQConfig) -> VLMWrapper:
    """Dispatch to the backend named in ``cfg.backend``. Used by: E3, E8, E9."""
    raise NotImplementedError("TODO: dispatch to gptq / awq / w8a8.")


def save_quantized(model: VLMWrapper, out_dir: Path) -> Path:
    """Save a quantized model (under paths.checkpoint_root, never committed)."""
    raise NotImplementedError("TODO: save quantized checkpoint.")


def run_quantization(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/03_quantize.py``.

    Loads the parent model (E2 for E3, E7 for E8, E0 for E9), draws the
    calibration set, quantizes with the variant's settings and saves the result.

    Used by: E3, E8, E9.
    """
    raise NotImplementedError("TODO: wire config -> parent model -> PTQ -> save (E3, E8, E9).")
