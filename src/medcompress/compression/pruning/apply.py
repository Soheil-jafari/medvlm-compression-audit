"""Physical removal of pruned units (weights are deleted, not masked).

Masking keeps the memory footprint and latency unchanged, so it cannot be used
for matched-budget comparisons. Removal must produce a smaller model whose
measured peak memory and latency actually drop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.models.base import VLMWrapper


def remove_units(model: VLMWrapper, units: Sequence[PrunableUnit]) -> VLMWrapper:
    """Physically remove ``units`` and update the model config (layer count, heads, dims).

    Used by: E5, E6, E7, A1-A6.
    """
    raise NotImplementedError("TODO: dispatch removal per unit type and update config.")


def remove_decoder_layers(model: VLMWrapper, layer_indices: Sequence[int]) -> VLMWrapper:
    """Delete whole decoder layers and re-index the remaining ones. Used by: E5-E7, A6."""
    raise NotImplementedError("TODO: delete decoder layers.")


def remove_attention_heads(
    model: VLMWrapper, heads_by_layer: dict[int, Sequence[int]]
) -> VLMWrapper:
    """Slice q/k/v/o projections to drop head groups (GQA-consistent). Used by: E5, E7, A1-A6."""
    raise NotImplementedError("TODO: slice attention projections per GQA group.")


def remove_mlp_channels(
    model: VLMWrapper, channels_by_layer: dict[int, Sequence[int]]
) -> VLMWrapper:
    """Slice gate/up/down projections to drop MLP channel groups. Used by: E5, E7, A1-A6."""
    raise NotImplementedError("TODO: slice MLP projections.")


def verify_physical_removal(
    before: VLMWrapper, after: VLMWrapper, removed: Sequence[PrunableUnit]
) -> dict[str, Any]:
    """Check parameter count and saved file size dropped as expected; return a report."""
    raise NotImplementedError("TODO: confirm removal shrank the model (no masking).")
