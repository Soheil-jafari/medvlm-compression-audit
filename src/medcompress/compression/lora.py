"""LoRA on the language model's decoder (attention and MLP projections) for E1, E2 and recovery.

The vision encoder and the vision-language projector stay frozen. Adapters are
attached with peft to the decoder layers found by
``compression/pruning/surgery.find_decoder_stack``, so the same code serves every
architecture that surgery supports. An adapter can be merged back into the weights,
which is how E2 becomes the plain model that pruning and quantization start from.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DECODER_PROJECTIONS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
TARGET_GROUPS: Mapping[str, tuple[str, ...]] = {
    "decoder_attention": DECODER_PROJECTIONS[:4],
    "decoder_mlp": DECODER_PROJECTIONS[4:],
}


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA settings. ``target_modules`` are group names from :data:`TARGET_GROUPS`."""

    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]

    def projections(self) -> tuple[str, ...]:
        """The decoder projections the target groups stand for."""
        unknown = [t for t in self.target_modules if t not in TARGET_GROUPS]
        if unknown:
            raise ValueError(f"unknown LoRA target groups: {unknown}; use {sorted(TARGET_GROUPS)}")
        return tuple(p for t in self.target_modules for p in TARGET_GROUPS[t])


def target_regex(decoder_path: str, projections: Sequence[str]) -> str:
    """Regular expression matching ``<decoder_path>.<layer>.<projection>`` and nothing else."""
    alternatives = "|".join(re.escape(p) for p in projections)
    return rf"^{re.escape(decoder_path)}\.\d+\.({alternatives})$"


def attach_lora(wrapper: Any, config: LoRAConfig) -> Any:
    """Freeze the model and attach trainable LoRA adapters to the decoder; returns the wrapper.

    ``wrapper.module`` becomes a ``PeftModel``. Adapter weights are kept in float32
    even when the base model is in float16, which is what makes fp16 training stable.
    """
    import peft

    from medcompress.compression.pruning.surgery import find_decoder_stack

    module = wrapper.module
    if isinstance(module, peft.PeftModel):
        raise ValueError("an adapter is already attached")
    stack = find_decoder_stack(module)
    for param in module.parameters():
        param.requires_grad_(False)
    lora = peft.LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=target_regex(stack.path, config.projections()),
        bias="none",
    )
    wrapper.module = peft.get_peft_model(module, lora)
    return wrapper


def trainable_parameters(module: Any) -> list[Any]:
    """The parameters that require gradients (the adapter weights after :func:`attach_lora`)."""
    return [p for p in module.parameters() if p.requires_grad]


def save_adapter(wrapper: Any, directory: Path | str) -> Path:
    """Write the adapter (not the base weights) to ``directory``."""
    import peft

    if not isinstance(wrapper.module, peft.PeftModel):
        raise ValueError("no adapter is attached")
    wrapper.module.save_pretrained(str(directory))
    return Path(directory)


def merge_adapter(wrapper: Any) -> Any:
    """Fold the attached adapter into the base weights and drop the adapter layers."""
    import peft

    if not isinstance(wrapper.module, peft.PeftModel):
        raise ValueError("no adapter is attached")
    wrapper.module = wrapper.module.merge_and_unload()
    return wrapper
