"""Where a split model's weights and the GPTQ math go (E9 on two T4s; 2026-11-13).

RC-E9 ran out of GPU memory twice. The second time (commit 5a25147) the automatic "balanced"
placement of ``from_pretrained`` put 8.40 GiB of the teacher's weights on GPU 1 (layers 11-27,
the final norm and ``lm_head``; reproduced offline from the teacher's architecture), not the
7.7 GiB the caps were planned for, and the GPTQ buffers of a ``down_proj`` no longer fitted.
This module replaces the guess by an explicit plan and a run-time choice:

* :func:`placement_units` lists **every** module that holds tensors, in the order the model
  runs (each decoder layer or vision block whole; every other module by its own parameters and
  buffers: the vision patch embedding and merger, the embeddings, the final norm, the rotary
  buffers, ``lm_head``), with its size in bytes;
* :func:`plan_device_map` splits them over the GPUs so the bytes are balanced and no GPU
  exceeds its cap, and returns the explicit ``device_map`` that ``from_pretrained`` then
  follows exactly; :func:`device_map_lines` prints it;
* :func:`pick_gptq_device` chooses, for each matrix, where GPTQ's math runs: the layer's own
  GPU if the allocator can still hand out GPTQ's working memory there, otherwise the GPU with
  the most room, otherwise the CPU (float32 there too; slower);
* :func:`memory_report` gives the per-GPU allocated, reserved and peak memory for the log.

Moving the math between two GPUs of the same model (both T4s) gives the same numbers. On the
CPU the float32 results can differ from the GPU's in the last bits (other Cholesky and matrix
product routines), which can flip a rounding code that sits on a tie; the log says when that
happens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain
from typing import Any

GIB = 2**30
GPTQ_DEVICE_MARGIN = 2**30
"""Room kept beyond GPTQ's working memory when choosing a device (1 GiB: the solvers'
workspaces, the allocator's rounding, and the about 0.9 GiB that attempt 2's numbers on GPU 1
do not account for)."""


def gptq_work_bytes(in_features: int, out_features: int) -> int:
    """GPTQ's working memory where its math runs (bytes): two float32 (in x in) matrices at a
    time (the Hessian's copy, the Cholesky factor, the inverse and its factor, each freed when
    the next exists), the float32 weights and the uint8 codes."""
    return 2 * 4 * in_features * in_features + 5 * out_features * in_features


def device_room(index: int) -> int:
    """Bytes the allocator can still hand out on GPU ``index``: the driver's free memory plus
    what PyTorch has reserved but not allocated."""
    import torch

    free, _total = torch.cuda.mem_get_info(index)
    return int(free + torch.cuda.memory_reserved(index) - torch.cuda.memory_allocated(index))


def pick_gptq_device(
    layer_device: Any,
    in_features: int,
    out_features: int,
    *,
    room: Callable[[int], int] | None = None,
    gpus: Iterable[int] | None = None,
) -> Any:
    """Where to run GPTQ's math for one matrix: the layer's GPU if it has room for
    :func:`gptq_work_bytes` plus :data:`GPTQ_DEVICE_MARGIN`, else the GPU with the most room
    if that one has it, else the CPU. A layer on the CPU stays on the CPU."""
    import torch

    device = torch.device(layer_device)
    if device.type != "cuda":
        return device
    room = room or device_room
    indices = list(range(torch.cuda.device_count()) if gpus is None else gpus)
    need = gptq_work_bytes(in_features, out_features) + GPTQ_DEVICE_MARGIN
    own = device.index if device.index is not None else 0
    others = sorted((g for g in indices if g != own), key=lambda g: -room(g))
    for index in [own, *others]:
        if room(index) >= need:
            return torch.device("cuda", index)
    return torch.device("cpu")


def memory_report() -> str:
    """``GPU0 alloc 7.91 / reserved 8.02 / peak 10.95 GiB; GPU1 ...`` ("" without CUDA)."""
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available():
        return ""
    parts = []
    for i in range(torch.cuda.device_count()):
        parts.append(
            f"GPU{i} alloc {torch.cuda.memory_allocated(i) / GIB:.2f} / reserved "
            f"{torch.cuda.memory_reserved(i) / GIB:.2f} / peak "
            f"{torch.cuda.max_memory_allocated(i) / GIB:.2f} GiB"
        )
    return "; ".join(parts)


def reset_peaks() -> None:
    """Reset every GPU's peak counter (the log then shows each layer's own peak)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def _bytes(tensors: Iterable[Any]) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def placement_units(model: Any) -> list[tuple[str, int]]:
    """``[(module name, bytes)]`` for every module that holds tensors, in registration order
    (the order the model runs for the Qwen VL models: vision, embeddings, layers, norm, head).

    A module of a class in ``model._no_split_modules`` (a decoder layer, a vision block) is one
    unit with everything below it; any other module is a unit of its own parameters and
    buffers, if it has any. Every tensor of the model is in exactly one unit.
    """
    no_split = set(getattr(model, "_no_split_modules", None) or ())
    units: list[tuple[str, int]] = []
    inside: str | None = None
    for name, module in model.named_modules():
        if inside is not None and name.startswith(inside + "."):
            continue
        inside = None
        if type(module).__name__ in no_split:
            units.append((name, _bytes(chain(module.parameters(), module.buffers()))))
            inside = name
            continue
        own = list(chain(module.parameters(recurse=False), module.buffers(recurse=False)))
        if own:
            units.append((name, _bytes(own)))
    return units


def plan_device_map(
    units: Sequence[tuple[str, int]], caps: Mapping[int, int]
) -> tuple[dict[str, int], dict[int, int]]:
    """``(device_map, bytes per GPU)``: the units in order, split over the GPUs of ``caps`` so
    each GPU holds about the same bytes (a unit goes to the next GPU when that brings the
    current one closer to its share), none above its cap.

    Raises:
        ValueError: the units cannot be placed within the caps.
    """
    gpus = sorted(caps)
    if not gpus:
        raise ValueError("no GPU to place the model on")
    total = sum(size for _name, size in units)
    share = total / len(gpus)
    device_map: dict[str, int] = {}
    used = {g: 0 for g in gpus}
    position = 0
    for name, size in units:
        gpu = gpus[position]
        last = position == len(gpus) - 1
        if not last and used[gpu] > 0 and abs(used[gpu] + size - share) > abs(used[gpu] - share):
            position += 1
            gpu = gpus[position]
        device_map[name] = gpu
        used[gpu] += size
    over = {g: used[g] for g in gpus if used[g] > caps[g]}
    if over:
        raise ValueError(
            "the plan exceeds the caps: "
            + ", ".join(f"GPU{g} {b / GIB:.2f} > {caps[g] / GIB:.2f} GiB" for g, b in over.items())
        )
    return device_map, used


def device_map_lines(units: Sequence[tuple[str, int]], device_map: Mapping[str, Any]) -> list[str]:
    """The plan as text: consecutive units on the same device as one line (first .. last)."""
    lines: list[str] = []
    group: list[tuple[str, int]] = []

    def flush() -> None:
        if group:
            device = device_map[group[0][0]]
            size = sum(s for _n, s in group)
            span = group[0][0] if len(group) == 1 else f"{group[0][0]} .. {group[-1][0]}"
            lines.append(f"device {device}: {span} ({len(group)} modules, {size / GIB:.2f} GiB)")

    for unit in units:
        if group and device_map[unit[0]] != device_map[group[0][0]]:
            flush()
            group = []
        group.append(unit)
    flush()
    return lines


def placed_bytes(model: Any) -> dict[str, int]:
    """``{device: bytes}`` of the parameters and buffers of a loaded model (what is there)."""
    out: dict[str, int] = {}
    for tensor in chain(model.parameters(), model.buffers()):
        key = str(tensor.device)
        out[key] = out.get(key, 0) + tensor.numel() * tensor.element_size()
    return out
