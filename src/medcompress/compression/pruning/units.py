"""Prunable units and structural constraints.

Candidate units j:
* decoder layers (whole transformer blocks of the language model),
* attention heads, removed in groups that respect grouped key-value heads (GQA):
  query heads sharing a KV head are pruned together, or a KV head is only removed
  together with all of its query heads,
* MLP channel groups (contiguous groups of intermediate channels; group size TBD),
* vision encoder blocks (optional, off by default).
The vision-language projector is never pruned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medcompress.models.base import VLMWrapper


class UnitType(str, Enum):
    """Kinds of prunable structure."""

    DECODER_LAYER = "decoder_layer"
    ATTENTION_HEAD = "attention_head"
    MLP_CHANNEL_GROUP = "mlp_channel_group"
    VISION_BLOCK = "vision_block"


@dataclass(frozen=True)
class PrunableUnit:
    """One prunable unit.

    Attributes:
        unit_id: Stable string ID, e.g. ``"lm.layer12.head_group3"``.
        unit_type: Kind of unit.
        layer_index: Index of the decoder (or vision) layer containing the unit.
        index: Index within the layer (head group / channel group), ``None`` for layers.
        num_params: Parameters removed if this unit is removed (for logging only;
            budgets are matched on measured memory/latency).
        component: ``"language"`` or ``"vision"``.
    """

    unit_id: str
    unit_type: UnitType
    layer_index: int
    index: int | None
    num_params: int
    component: str = "language"


def enumerate_units(
    model: VLMWrapper,
    unit_types: Sequence[UnitType],
    *,
    include_vision: bool = False,
    mlp_channel_group_size: int | None = None,
) -> list[PrunableUnit]:
    """List all candidate units of the requested types.

    Used by: E5, E6 (layers only), E7, A1-A6 (A6 switches between layers only and
    heads + channel groups).
    """
    raise NotImplementedError("TODO: walk the architecture and list candidate units.")


def gqa_head_groups(num_query_heads: int, num_kv_heads: int) -> list[list[int]]:
    """Return query-head index groups that share one KV head.

    Used by: :func:`enumerate_units` and :func:`validate_removal_set`.
    """
    raise NotImplementedError("TODO: group query heads by shared KV head.")


def validate_removal_set(
    units: Sequence[PrunableUnit], removed: Sequence[PrunableUnit]
) -> list[str]:
    """Return a list of constraint violations (empty if the removal set is valid).

    Constraints: GQA groups intact; at least one head group and one channel
    group per remaining layer; never the projector, embeddings or LM head;
    no unit removed twice; a head/channel inside a removed layer is not counted
    separately.

    Used by: E5, E6, E7, A1-A6 and the budget search.
    """
    raise NotImplementedError("TODO: structural validity checks for a removal set.")
