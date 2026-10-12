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
from typing import TYPE_CHECKING, Any

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
        channel_start / channel_count: channel range of an MLP channel group.
    """

    unit_id: str
    unit_type: UnitType
    layer_index: int
    index: int | None
    num_params: int
    component: str = "language"
    channel_start: int | None = None
    channel_count: int | None = None


def _module_of(model: Any) -> Any:
    return getattr(model, "module", model)


def unit_id(layer_index: int, unit_type: UnitType, index: int | None = None) -> str:
    """Return the stable ID of a unit, e.g. ``lm.layer12.kv_group3``."""
    base = f"lm.layer{layer_index}"
    if unit_type is UnitType.DECODER_LAYER:
        return base
    if unit_type is UnitType.ATTENTION_HEAD:
        return f"{base}.kv_group{index}"
    if unit_type is UnitType.MLP_CHANNEL_GROUP:
        return f"{base}.mlp_group{index}"
    raise ValueError(f"Unsupported unit type for the language model: {unit_type}")


def enumerate_units(
    model: VLMWrapper | Any,
    unit_types: Sequence[UnitType],
    *,
    include_vision: bool = False,
    mlp_channel_group_size: int | None = None,
) -> list[PrunableUnit]:
    """List all candidate units of the requested types (language model only).

    Accepts a wrapper with a ``module`` attribute or a torch model. Attention units
    are GQA groups (one KV head plus its query heads); MLP units are contiguous
    channel groups of ``mlp_channel_group_size`` (the last group may be shorter).

    Used by: E5, E6 (layers only), E7, A1-A6 (A6 switches between layers only and
    heads + channel groups).
    """
    if include_vision:
        raise NotImplementedError("TODO: vision-block units are off by default and not wired yet.")
    from medcompress.compression.pruning.surgery import find_decoder_stack, layer_shape

    module = _module_of(model)
    stack = find_decoder_stack(module)
    units: list[PrunableUnit] = []
    for i, layer in enumerate(stack.layers):
        shape = layer_shape(layer, stack.config)
        hidden = layer.self_attn.q_proj.in_features
        if UnitType.DECODER_LAYER in unit_types:
            n = sum(p.numel() for p in layer.parameters())
            units.append(
                PrunableUnit(unit_id(i, UnitType.DECODER_LAYER), UnitType.DECODER_LAYER, i, None, n)
            )
        if UnitType.ATTENTION_HEAD in unit_types:
            q_rows = shape["group_ratio"] * shape["head_dim"]
            per_group = hidden * (2 * q_rows + 2 * shape["head_dim"])
            attn = layer.self_attn
            per_group += sum(
                rows
                for proj, rows in (
                    (attn.q_proj, q_rows),
                    (attn.k_proj, shape["head_dim"]),
                    (attn.v_proj, shape["head_dim"]),
                )
                if proj.bias is not None
            )
            for g in range(shape["num_kv_heads"]):
                units.append(
                    PrunableUnit(
                        unit_id(i, UnitType.ATTENTION_HEAD, g),
                        UnitType.ATTENTION_HEAD,
                        i,
                        g,
                        per_group,
                    )
                )
        if UnitType.MLP_CHANNEL_GROUP in unit_types:
            if not mlp_channel_group_size:
                raise ValueError("mlp_channel_group_size is required for MLP channel groups.")
            width = shape["intermediate"]
            for g, start in enumerate(range(0, width, mlp_channel_group_size)):
                size = min(mlp_channel_group_size, width - start)
                units.append(
                    PrunableUnit(
                        unit_id(i, UnitType.MLP_CHANNEL_GROUP, g),
                        UnitType.MLP_CHANNEL_GROUP,
                        i,
                        g,
                        3 * hidden * size,
                        channel_start=start,
                        channel_count=size,
                    )
                )
    return units


def gqa_head_groups(num_query_heads: int, num_kv_heads: int) -> list[list[int]]:
    """Return query-head index groups that share one KV head.

    Query heads ``g * r .. (g + 1) * r - 1`` share KV head ``g``, with
    ``r = num_query_heads / num_kv_heads`` (the layout used by ``repeat_kv``).

    Used by: :func:`enumerate_units` and :func:`validate_removal_set`.
    """
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError("num_query_heads must be a positive multiple of num_kv_heads.")
    ratio = num_query_heads // num_kv_heads
    return [list(range(g * ratio, (g + 1) * ratio)) for g in range(num_kv_heads)]


def validate_removal_set(
    units: Sequence[PrunableUnit],
    removed: Sequence[PrunableUnit],
    *,
    protected_layers: Sequence[int] = (),
) -> list[str]:
    """Return a list of constraint violations (empty if the removal set is valid).

    Constraints: every removed unit is a candidate; no unit removed twice; a head or
    channel group inside a removed layer is not removed separately; at least one
    layer remains; every remaining layer keeps at least one head group and one
    channel group; protected layers (e.g. Qwen3-VL DeepStack injection layers) are
    never removed. Embeddings, LM head and projector are never candidates.

    Used by: E5, E6, E7, A1-A6 and the budget search.
    """
    problems: list[str] = []
    known = {u.unit_id for u in units}
    ids = [u.unit_id for u in removed]
    problems += [f"{u} is not a candidate unit" for u in ids if u not in known]
    problems += [f"{u} removed twice" for u in sorted({u for u in ids if ids.count(u) > 1})]
    layers = {u.layer_index for u in units if u.unit_type is UnitType.DECODER_LAYER}
    removed_layers = {u.layer_index for u in removed if u.unit_type is UnitType.DECODER_LAYER}
    problems += [f"layer {i} is protected" for i in sorted(removed_layers & set(protected_layers))]
    if layers and removed_layers >= layers:
        problems.append("every decoder layer would be removed")
    for unit in removed:
        if unit.unit_type is not UnitType.DECODER_LAYER and unit.layer_index in removed_layers:
            problems.append(f"{unit.unit_id} lies inside removed layer {unit.layer_index}")
    for kind, label in (
        (UnitType.ATTENTION_HEAD, "head group"),
        (UnitType.MLP_CHANNEL_GROUP, "channel group"),
    ):
        per_layer: dict[int, set[str]] = {}
        for u in units:
            if u.unit_type is kind:
                per_layer.setdefault(u.layer_index, set()).add(u.unit_id)
        gone = {u.unit_id for u in removed if u.unit_type is kind}
        for layer_index, members in sorted(per_layer.items()):
            if layer_index not in removed_layers and members <= gone:
                problems.append(f"layer {layer_index} would lose every {label}")
    return problems
