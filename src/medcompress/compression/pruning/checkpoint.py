"""Save and reload a physically pruned model.

Removing attention groups or MLP channels gives every layer its own shapes, which a
single model config cannot describe, so a pruned model cannot be reloaded with
``from_pretrained``. A pruned checkpoint is therefore two files: the weights
(``model.safetensors``) and a manifest listing the removal steps. Loading takes the
*unpruned* base model, replays the steps on it, and then loads the weights, so the
structure is rebuilt by the same code that created it.

Each step lists unit identifiers relative to the model as it was before that step,
so iterative pruning is recorded as several steps. A step may be empty.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from medcompress.compression.pruning import surgery
from medcompress.compression.pruning.apply import remove_units
from medcompress.compression.pruning.units import PrunableUnit, UnitType, enumerate_units

MANIFEST_NAME = "pruning_manifest.json"
WEIGHTS_NAME = "model.safetensors"
FORMAT_VERSION = 1
_ALL_TYPES = [UnitType.DECODER_LAYER, UnitType.ATTENTION_HEAD, UnitType.MLP_CHANNEL_GROUP]


def _module(model: Any) -> Any:
    return getattr(model, "module", model)


def stream_save(module: Any, path: Path | str) -> Path:
    """Write ``module``'s weights as a safetensors file one tensor at a time (2026-11-14).

    ``safetensors.torch.save_model`` first builds every tensor in host memory (RC-SCALE: host
    RSS 27.5 GiB at the save of a pruned 3B). Here the header is computed from the shapes, then
    each tensor is copied from the GPU, written and freed, so the host holds one tensor at a
    time. The file is the format ``save_model`` writes: shared tensors are stored once, the
    dropped names are listed in the metadata, and ``safetensors`` reads it back unchanged
    (``tests/test_surgery.py``). Written to a temporary name and renamed at the end.
    """
    import os
    import struct

    import torch
    from safetensors.torch import _TYPES, _remove_duplicate_names

    names = {dtype: name for name, dtype in _TYPES.items()}
    state = module.state_dict()
    metadata: dict[str, str] = {}
    for kept, dropped in _remove_duplicate_names(state).items():
        for name in dropped:
            metadata.setdefault(name, kept)
            del state[name]
    header: dict[str, Any] = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name in sorted(state):
        tensor = state[name]
        size = tensor.numel() * tensor.element_size()
        header[name] = {"dtype": names[tensor.dtype], "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + size]}  # fmt: skip
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)  # the data starts on an 8-byte boundary
    target = Path(path)
    partial = target.with_name(target.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for name in sorted(state):
            with torch.no_grad():
                host = state[name].detach().contiguous().to("cpu").reshape(-1)
            handle.write(memoryview(host.view(torch.uint8).numpy()))
            del host
    os.replace(partial, target)
    return target


def save_pruned(
    model: Any,
    directory: Path | str,
    steps: Sequence[Sequence[PrunableUnit]],
    *,
    mlp_channel_group_size: int,
) -> Path:
    """Write the weights and the manifest of a pruned model to ``directory``.

    ``steps`` are the units removed so far, one list per removal step, each relative
    to the model before that step; ``mlp_channel_group_size`` is the group size used
    to enumerate the MLP units.
    """
    from medcompress.utils.disk import require_free_disk

    module = _module(model)
    out = Path(directory)
    seen: set[int] = set()
    need = 0
    for tensor in [*module.parameters(), *module.buffers()]:
        if tensor.data_ptr() not in seen:
            seen.add(tensor.data_ptr())
            need += tensor.numel() * tensor.element_size()
    require_free_disk(out, need, "the pruned model")  # before anything is written
    out.mkdir(parents=True, exist_ok=True)
    stream_save(module, out / WEIGHTS_NAME)  # one tensor at a time in host memory (2026-11-14)
    manifest = {
        "format_version": FORMAT_VERSION,
        "mlp_channel_group_size": mlp_channel_group_size,
        "steps": [[unit.unit_id for unit in step] for step in steps],
        "parameters_after": surgery.count_parameters(module),
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def read_manifest(directory: Path | str) -> dict[str, Any]:
    """Read and check the manifest of a pruned checkpoint."""
    path = Path(directory) / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format_version {manifest.get('format_version')!r}")
    return manifest


def stream_load(module: Any, path: Path | str) -> int:
    """Copy the weights of a safetensors file into ``module`` one tensor at a time (2026-11-18).

    ``safetensors.torch.load_model(..., device="cuda")`` first builds the whole state dict on the
    GPU and then copies it into the parameters, so a second full copy of the model sits next to
    it (RC-G2 session 5: stage 06 out of memory at 14.01 GiB allocated on a 14.56 GiB T4). Here
    each tensor is read into host memory (the file is memory-mapped), checked and copied into
    its parameter or buffer on the model's device, then dropped: the GPU holds the model plus at
    most one tensor in flight.

    Strict, like ``load_model(strict=True)``: the file holds exactly the module's state-dict keys
    (a tied tensor stored once is accepted when the module really shares it), with the same
    shapes and dtypes. Returns the number of tensors copied.
    """
    import torch
    from safetensors import safe_open

    state = module.state_dict(keep_vars=True)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        aliases = dict(handle.metadata() or {})  # dropped name -> kept name (stream_save)
        unexpected = sorted(keys - set(state))
        missing = sorted(set(state) - keys - set(aliases))
        if missing or unexpected:
            raise ValueError(
                f"{path}: the weights do not match the model: missing {missing[:5]} "
                f"({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})"
            )
        for dropped, kept in aliases.items():
            if dropped in state and (
                kept not in state
                or kept not in keys
                or state[dropped].data_ptr() != state[kept].data_ptr()
            ):
                raise ValueError(f"{path}: {dropped} is stored as {kept}, but the model does not "
                                 "share the two tensors")  # fmt: skip
        for name in sorted(keys):
            target = state[name]
            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != tuple(target.shape) or tensor.dtype != target.dtype:
                raise ValueError(
                    f"{path}: {name} is {tuple(tensor.shape)} {tensor.dtype} in the "
                    f"file, {tuple(target.shape)} {target.dtype} in the model"
                )
            with torch.no_grad():
                target.copy_(tensor)
            del tensor
    return len(keys)


def _gpu_note(label: str, module: Any) -> None:
    """``MEDCOMPRESS_MEM`` line plus the model's own parameter bytes, so the log shows the margin;
    above 0.5 GiB of other allocated memory, the largest live CUDA tensors are named."""
    from medcompress.utils.memory import gpu_bytes, live_cuda_tensors, memory_line

    params = sum(p.numel() * p.element_size() for p in module.parameters())
    line = f"{memory_line(label)}; model parameters {params / 2**30:.2f} GiB"
    reading = gpu_bytes()
    if reading is not None and reading[0] - params > 2**29:
        line += f"; other allocated {(reading[0] - params) / 2**30:.2f} GiB: {live_cuda_tensors()}"
    print(line, flush=True)


def load_pruned(base_model: Any, directory: Path | str) -> Any:
    """Replay the recorded removal steps on the unpruned ``base_model`` and load the weights.

    ``base_model`` must be the same architecture and weights source as the model
    that was pruned (a wrapper or a torch model). Raises ``ValueError`` if a recorded
    unit does not exist in it, the weights file does not match the rebuilt model (keys, shapes,
    dtypes) or the parameter count differs from the manifest.

    Memory (2026-11-18): the removed and replaced tensors are freed during the surgery, the GPU
    cache is released before the weights are read, and the weights are streamed one tensor at a
    time (:func:`stream_load`); the allocated and reserved memory are logged before and after.
    """
    from medcompress.utils.memory import release_gpu_cache

    manifest = read_manifest(directory)
    module = _module(base_model)
    for number, step in enumerate(manifest["steps"]):
        units = enumerate_units(
            module, _ALL_TYPES, mlp_channel_group_size=manifest["mlp_channel_group_size"]
        )
        by_id = {u.unit_id: u for u in units}
        missing = [unit_id for unit_id in step if unit_id not in by_id]
        if missing:
            raise ValueError(f"step {number}: units not in the base model: {missing[:3]}")
        remove_units(module, [by_id[unit_id] for unit_id in step])
    release_gpu_cache()
    _gpu_note("load_pruned: rebuilt, before the weights", module)
    stream_load(module, Path(directory) / WEIGHTS_NAME)
    _gpu_note("load_pruned: weights loaded", module)
    count = surgery.count_parameters(module)
    if count != manifest["parameters_after"]:
        raise ValueError(f"parameter count {count} != manifest {manifest['parameters_after']}")
    return base_model
