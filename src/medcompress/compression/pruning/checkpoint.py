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


def load_pruned(base_model: Any, directory: Path | str) -> Any:
    """Replay the recorded removal steps on the unpruned ``base_model`` and load the weights.

    ``base_model`` must be the same architecture and weights source as the model
    that was pruned (a wrapper or a torch model). Raises ``ValueError`` if a recorded
    unit does not exist in it or the parameter count differs from the manifest.
    """
    from safetensors.torch import load_model

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
    device = str(next(module.parameters()).device)  # straight to the GPU: no host copy
    load_model(module, str(Path(directory) / WEIGHTS_NAME), strict=True, device=device)
    count = surgery.count_parameters(module)
    if count != manifest["parameters_after"]:
        raise ValueError(f"parameter count {count} != manifest {manifest['parameters_after']}")
    return base_model
