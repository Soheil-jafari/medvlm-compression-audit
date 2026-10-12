"""Structural surgery on the language model of a Hugging Face VLM (torch imported lazily).

Works on the decoder of Qwen-family language models (Qwen2, Qwen2.5, Qwen3 and the
Qwen-based VLMs in ``configs/student_selection.yaml``): decoder layers with
``self_attn.{q,k,v,o}_proj`` and ``mlp.{gate,up,down}_proj``.

* **Scoring** uses reversible ablation (context managers that skip a layer or zero
  a head group / channel group through forward hooks).
* **Pruning** removes weights physically: layers are deleted, attention
  projections and MLP projections are sliced. A pruned model is saved as the base
  revision plus the removal spec plus its weights, and re-created by applying the
  same surgery, because layers can then differ in width.

Head groups follow grouped-query attention: group ``g`` is KV head ``g`` together
with the ``num_heads / num_kv_heads`` query heads that share it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

VISION_MARKERS: tuple[str, ...] = ("visual", "vision", "vit", "image_encoder")


@dataclass(frozen=True)
class DecoderStack:
    """Where the language-model layers live inside a model."""

    path: str
    parent: Any
    attr: str
    config: Any

    @property
    def layers(self) -> Any:
        """The ``nn.ModuleList`` of decoder layers."""
        return getattr(self.parent, self.attr)


def _text_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    for name in ("text_config", "llm_config"):
        sub = getattr(config, name, None)
        if sub is not None:
            return sub
    return config


def find_decoder_stack(model: Any) -> DecoderStack:
    """Return the language model's decoder ``ModuleList`` (vision towers are skipped)."""
    import torch

    found = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if any(marker in name.lower() for marker in VISION_MARKERS):
            continue
        if type(module[0]).__name__.endswith("DecoderLayer"):
            found.append(name)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one language decoder stack, found {found}")
    path = found[0]
    parent_path, _, attr = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return DecoderStack(path=path, parent=parent, attr=attr, config=_text_config(model))


def head_dim(layer: Any, config: Any) -> int:
    """Return the attention head dimension of a decoder layer."""
    value = getattr(layer.self_attn, "head_dim", None) or getattr(config, "head_dim", None)
    if value:
        return int(value)
    return int(config.hidden_size // config.num_attention_heads)


def layer_shape(layer: Any, config: Any) -> dict[str, int]:
    """Return current query heads, KV heads, group ratio and MLP width of a layer."""
    attn, mlp = layer.self_attn, layer.mlp
    dim = head_dim(layer, config)
    n_q = attn.q_proj.out_features // dim
    n_kv = attn.k_proj.out_features // dim
    return {
        "head_dim": dim,
        "num_heads": n_q,
        "num_kv_heads": n_kv,
        "group_ratio": n_q // n_kv,
        "intermediate": mlp.gate_proj.out_features,
    }


def _slice_linear(linear: Any, keep: Any, dim: int) -> Any:
    """Return a new Linear keeping rows (dim=0, outputs) or columns (dim=1, inputs)."""
    import torch

    weight = linear.weight.data.index_select(dim, keep)
    out_f, in_f = weight.shape
    new = torch.nn.Linear(
        in_f, out_f, bias=linear.bias is not None, device=weight.device, dtype=weight.dtype
    )
    new.weight.data.copy_(weight)
    if linear.bias is not None:
        bias = linear.bias.data if dim == 1 else linear.bias.data.index_select(0, keep)
        new.bias.data.copy_(bias)
    return new


def remove_layers(model: Any, indices: Sequence[int]) -> None:
    """Delete decoder layers in place, re-index the rest and update the text config."""
    import torch

    stack = find_decoder_stack(model)
    layers = stack.layers
    drop = set(indices)
    if not drop:
        return
    if not drop <= set(range(len(layers))):
        raise ValueError(f"Layer indices out of range: {sorted(drop)}")
    if len(drop) >= len(layers):
        raise ValueError("Cannot remove every decoder layer.")
    kept = [layer for i, layer in enumerate(layers) if i not in drop]
    setattr(stack.parent, stack.attr, torch.nn.ModuleList(kept))
    for new_index, layer in enumerate(kept):
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_index
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_index
    config = stack.config
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        config.layer_types = [t for i, t in enumerate(layer_types) if i not in drop]
    config.num_hidden_layers = len(kept)


def remove_kv_groups(model: Any, groups_by_layer: Mapping[int, Sequence[int]]) -> None:
    """Slice q/k/v/o projections to drop whole GQA groups (KV head + its query heads)."""
    import torch

    stack = find_decoder_stack(model)
    for layer_index, groups in groups_by_layer.items():
        drop = set(groups)
        if not drop:
            continue
        layer = stack.layers[layer_index]
        shape = layer_shape(layer, stack.config)
        dim, ratio, n_kv = shape["head_dim"], shape["group_ratio"], shape["num_kv_heads"]
        if not drop <= set(range(n_kv)) or len(drop) >= n_kv:
            raise ValueError(f"Layer {layer_index}: invalid KV groups {sorted(drop)} of {n_kv}")
        keep_kv = [g for g in range(n_kv) if g not in drop]
        keep_q_heads = [g * ratio + r for g in keep_kv for r in range(ratio)]
        device = layer.self_attn.q_proj.weight.device
        q_idx = torch.tensor([h * dim + d for h in keep_q_heads for d in range(dim)], device=device)
        kv_idx = torch.tensor([g * dim + d for g in keep_kv for d in range(dim)], device=device)
        attn = layer.self_attn
        attn.q_proj = _slice_linear(attn.q_proj, q_idx, 0)
        attn.k_proj = _slice_linear(attn.k_proj, kv_idx, 0)
        attn.v_proj = _slice_linear(attn.v_proj, kv_idx, 0)
        attn.o_proj = _slice_linear(attn.o_proj, q_idx, 1)
        for name, value in (
            ("num_heads", len(keep_q_heads)),
            ("num_key_value_heads", len(keep_kv)),
        ):
            if hasattr(attn, name):
                setattr(attn, name, value)


def remove_mlp_channels(model: Any, channels_by_layer: Mapping[int, Sequence[int]]) -> None:
    """Slice gate/up (outputs) and down (inputs) projections to drop MLP channels."""
    import torch

    stack = find_decoder_stack(model)
    for layer_index, channels in channels_by_layer.items():
        drop = set(channels)
        if not drop:
            continue
        mlp = stack.layers[layer_index].mlp
        width = mlp.gate_proj.out_features
        if not drop <= set(range(width)) or len(drop) >= width:
            raise ValueError(f"Layer {layer_index}: invalid MLP channels")
        keep = torch.tensor(
            [c for c in range(width) if c not in drop], device=mlp.gate_proj.weight.device
        )
        mlp.gate_proj = _slice_linear(mlp.gate_proj, keep, 0)
        mlp.up_proj = _slice_linear(mlp.up_proj, keep, 0)
        mlp.down_proj = _slice_linear(mlp.down_proj, keep, 1)


def count_parameters(model: Any) -> int:
    """Return the number of parameters (shared tensors counted once)."""
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


@contextlib.contextmanager
def skip_layer(model: Any, layer_index: int) -> Iterator[None]:
    """Temporarily make one decoder layer the identity (output = input hidden states)."""
    layer = find_decoder_stack(model).layers[layer_index]

    def hook(_module: Any, args: tuple, kwargs: dict, output: Any) -> Any:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = layer.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def zero_kv_group(model: Any, layer_index: int, group: int) -> Iterator[None]:
    """Temporarily zero the query heads of one GQA group at the o_proj input."""
    stack = find_decoder_stack(model)
    layer = stack.layers[layer_index]
    shape = layer_shape(layer, stack.config)
    start = group * shape["group_ratio"] * shape["head_dim"]
    stop = start + shape["group_ratio"] * shape["head_dim"]

    def pre_hook(_module: Any, args: tuple) -> tuple:
        x = args[0].clone()
        x[..., start:stop] = 0
        return (x, *args[1:])

    handle = layer.self_attn.o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def zero_mlp_channels(model: Any, layer_index: int, channels: Sequence[int]) -> Iterator[None]:
    """Temporarily zero MLP channels at the down_proj input."""
    import torch

    mlp = find_decoder_stack(model).layers[layer_index].mlp
    index = torch.tensor(list(channels), device=mlp.down_proj.weight.device)

    def pre_hook(_module: Any, args: tuple) -> tuple:
        x = args[0].clone()
        x.index_fill_(-1, index, 0)
        return (x, *args[1:])

    handle = mlp.down_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()
