"""Model loading from candidate configs (configs/models/*.yaml).

Weights are loaded from a local path or a hub ID given in the config. No IDs are
filled in yet: every candidate config has ``TODO`` fields until the team decides.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.models.base import VLMWrapper


@dataclass(frozen=True)
class ModelSpec:
    """Parsed ``model`` section of a model config."""

    name: str
    role: str  # "teacher" or "student"
    family: str
    source: str  # local path or hub ID (TODO in candidate configs)
    adapter: str | None
    dtype: str
    status: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ModelSpec:
        """Build a ModelSpec from a loaded model config."""
        raise NotImplementedError("TODO: parse configs/models/*.yaml into ModelSpec.")


def load_model(
    spec: ModelSpec,
    *,
    device_map: str | Mapping[str, Any] = "auto",
    quantization: Mapping[str, Any] | None = None,
) -> VLMWrapper:
    """Load a teacher or student as a :class:`VLMWrapper`.

    Imports torch / transformers lazily. Must refuse configs whose ``status`` is
    still a candidate placeholder with unresolved ``TODO`` fields.

    Used by: every experiment.
    """
    raise NotImplementedError("TODO: lazy-import transformers and load the VLM.")


def load_adapter(model: VLMWrapper, adapter_path: Path) -> VLMWrapper:
    """Attach a LoRA adapter (e.g. a published domain adapter for the teacher, or E1/E2)."""
    raise NotImplementedError("TODO: attach LoRA adapter via peft.")


def check_shared_tokenizer(teacher: VLMWrapper, student: VLMWrapper) -> None:
    """Raise ValueError if teacher and student tokenizers differ.

    Used by: E2, E4 (distillation), recovery for E5-E7 and A1-A6.
    """
    raise NotImplementedError("TODO: compare tokenizer fingerprints.")
