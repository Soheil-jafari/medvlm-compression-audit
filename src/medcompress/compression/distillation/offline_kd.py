"""Student training loops: answer-only fine-tuning (E1) and offline distillation (E2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.models.base import VLMWrapper

TRAINING_MODES: tuple[str, ...] = ("sft", "offline_kd", "qakd")


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA settings (all values TODO in configs until decided)."""

    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainConfig:
    """Training settings shared by E1, E2 (and E4 via qakd)."""

    mode: str
    lora: LoRAConfig
    steps: int
    learning_rate: float
    batch_size: int
    grad_accumulation: int
    kd_weight: float
    kd_temperature: float
    data_subset: str
    extra: Mapping[str, Any]


def train_student_sft(student: VLMWrapper, cfg: TrainConfig, *, seed: int) -> VLMWrapper:
    """Fine-tune the student on ground-truth answers only (kd_weight = 0).

    Used by: E1 (3 seeds).
    """
    raise NotImplementedError("TODO: LoRA fine-tuning on correct answers (E1).")


def train_offline_kd(
    student: VLMWrapper, teacher_cache_dir: Path, cfg: TrainConfig, *, seed: int
) -> VLMWrapper:
    """Train the student on cached teacher outputs (answers + top-k probabilities).

    Used by: E2 (3 seeds). E2 is the starting point of E3 and E5-E7.
    """
    raise NotImplementedError("TODO: LoRA + offline KD from the E0 cache (E2).")


def run_student_training(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/02_train_student.py``.

    Dispatches on ``training.mode``: ``sft`` (E1), ``offline_kd`` (E2),
    ``qakd`` (E4, via :mod:`medcompress.compression.quantization.qakd`).
    """
    raise NotImplementedError("TODO: wire config -> student -> training mode (E1, E2, E4).")
