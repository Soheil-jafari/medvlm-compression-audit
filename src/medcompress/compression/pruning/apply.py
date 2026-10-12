"""Physical removal of pruned units (weights are deleted, not masked).

Masking keeps the memory footprint and latency unchanged, so it cannot be used
for matched-budget comparisons. Removal must produce a smaller model whose
measured peak memory and latency actually drop. The tensor work is in
``compression/pruning/surgery.py``; these functions accept either a wrapper with a
``module`` attribute or a torch model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from medcompress.compression.pruning import surgery
from medcompress.compression.pruning.units import UnitType

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.models.base import VLMWrapper


def _module(model: Any) -> Any:
    return getattr(model, "module", model)


def remove_units(model: VLMWrapper | Any, units: Sequence[PrunableUnit]) -> VLMWrapper | Any:
    """Physically remove ``units`` and update the model config (layer count, heads, dims).

    Head and channel groups are removed first (their layer indices refer to the
    original model), then whole layers. Returns the same (modified) model.

    Used by: E5, E6, E7, A1-A6.
    """
    heads: dict[int, list[int]] = {}
    channels: dict[int, list[int]] = {}
    layers: list[int] = []
    for unit in units:
        if unit.unit_type is UnitType.ATTENTION_HEAD:
            if unit.index is None:
                raise ValueError(f"{unit.unit_id} has no group index")
            heads.setdefault(unit.layer_index, []).append(unit.index)
        elif unit.unit_type is UnitType.MLP_CHANNEL_GROUP:
            if unit.channel_start is None or unit.channel_count is None:
                raise ValueError(f"{unit.unit_id} has no channel range")
            channels.setdefault(unit.layer_index, []).extend(
                range(unit.channel_start, unit.channel_start + unit.channel_count)
            )
        elif unit.unit_type is UnitType.DECODER_LAYER:
            layers.append(unit.layer_index)
        else:
            raise NotImplementedError("TODO: vision-block removal is not wired (off by default).")
    remove_attention_heads(model, heads)
    remove_mlp_channels(model, channels)
    remove_decoder_layers(model, layers)
    return model


def remove_decoder_layers(
    model: VLMWrapper | Any, layer_indices: Sequence[int]
) -> VLMWrapper | Any:
    """Delete whole decoder layers and re-index the remaining ones. Used by: E5-E7, A6."""
    surgery.remove_layers(_module(model), layer_indices)
    return model


def remove_attention_heads(
    model: VLMWrapper | Any, heads_by_layer: dict[int, Sequence[int]]
) -> VLMWrapper | Any:
    """Slice q/k/v/o projections to drop GQA groups (indices are KV-group indices).

    Used by: E5, E7, A1-A6.
    """
    surgery.remove_kv_groups(_module(model), heads_by_layer)
    return model


def remove_mlp_channels(
    model: VLMWrapper | Any, channels_by_layer: dict[int, Sequence[int]]
) -> VLMWrapper | Any:
    """Slice gate/up/down projections to drop MLP channels. Used by: E5, E7, A1-A6."""
    surgery.remove_mlp_channels(_module(model), channels_by_layer)
    return model


def verify_physical_removal(
    before_params: int, after: VLMWrapper | Any, removed: Sequence[PrunableUnit]
) -> dict[str, Any]:
    """Check that the parameter count dropped by the removed units' size (no masking).

    Returns ``{"before", "after", "expected_drop", "actual_drop", "ok"}``. Units
    inside removed layers are not double counted.
    """
    after_params = surgery.count_parameters(_module(after))
    removed_layers = {u.layer_index for u in removed if u.unit_type is UnitType.DECODER_LAYER}
    expected = sum(
        u.num_params
        for u in removed
        if u.unit_type is UnitType.DECODER_LAYER or u.layer_index not in removed_layers
    )
    actual = before_params - after_params
    return {
        "before": before_params,
        "after": after_params,
        "expected_drop": expected,
        "actual_drop": actual,
        "ok": actual == expected and after_params < before_params,
    }
