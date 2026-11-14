"""The vision tower of a model alone, in fp16 on one GPU (the fp16 scan of 2026-11-13).

The teacher (16.6 GB in fp16) does not fit one T4, but its vision tower (about 0.7 B
parameters) does. :func:`load_vision_tower` builds the tower's own module from the model's
config (the same class the full model uses), loads only its weights from the checkpoint's
shards (those the safetensors index names for ``visual``), and returns it with the model's
processor. E1 and E2 train LoRA on the decoder only, so the students' vision tower is the
pretrained one; the teacher's adapter is likewise on the language model only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PREFIXES: tuple[str, ...] = ("model.visual.", "visual.")


def vision_state(files: list[Path]) -> dict[str, Any]:
    """The ``visual`` tensors of safetensors files, keyed relative to the tower."""
    from safetensors import safe_open

    state: dict[str, Any] = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                for prefix in PREFIXES:
                    if key.startswith(prefix):
                        state[key[len(prefix) :]] = handle.get_tensor(key)
                        break
    return state


def _vision_files(source: str, revision: str | None) -> list[Path]:
    """The checkpoint files that hold the vision tower (a local folder or the hub)."""
    import json

    local = Path(source).expanduser()
    if local.is_dir():
        index = local / "model.safetensors.index.json"
        if index.is_file():
            names = {f for k, f in json.loads(index.read_text())["weight_map"].items()
                     if k.startswith(PREFIXES)}  # fmt: skip
            return [local / n for n in sorted(names)]
        return sorted(local.glob("*.safetensors"))
    from huggingface_hub import hf_hub_download

    try:
        index = Path(hf_hub_download(source, "model.safetensors.index.json", revision=revision))
    except Exception:  # noqa: BLE001 - a single-file checkpoint has no index
        return [Path(hf_hub_download(source, "model.safetensors", revision=revision))]
    names = {f for k, f in json.loads(index.read_text())["weight_map"].items()
             if k.startswith(PREFIXES)}  # fmt: skip
    return [Path(hf_hub_download(source, n, revision=revision)) for n in sorted(names)]


def load_vision_tower(spec: Any, *, device: str = "cuda:0") -> tuple[Any, Any]:
    """``(vision tower in fp16 on device, processor)`` of a model spec.

    Raises:
        ValueError: the checkpoint's vision tensors do not match the tower exactly.
    """
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    revision = spec.resolved_revision
    kwargs = {"revision": revision} if revision and not Path(spec.source).exists() else {}
    hub = AutoConfig.from_pretrained(spec.source, **kwargs)
    with init_empty_weights():
        skeleton = AutoModelForImageTextToText.from_config(hub, dtype=torch.float16)
    inner = getattr(skeleton, "model", skeleton)
    template = inner.visual
    with torch.device(device):
        tower = type(template)._from_config(template.config, dtype=torch.float16)
    state = vision_state(_vision_files(spec.source, revision))
    missing, unexpected = tower.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"vision tensors do not match the tower: missing {missing[:3]}, "
                         f"unexpected {unexpected[:3]}")  # fmt: skip
    processor = AutoProcessor.from_pretrained(spec.source, **kwargs)
    return tower.eval(), processor


def output_tensor(value: Any) -> Any:
    """The tensor of a module's output: itself, the first item of a tuple, or a model output's
    ``pooler_output`` (else ``last_hidden_state``); None when there is none."""
    import torch

    if isinstance(value, tuple):
        value = value[0]
    if torch.is_tensor(value):
        return value
    pooled = getattr(value, "pooler_output", None)
    return pooled if pooled is not None else getattr(value, "last_hidden_state", None)


def tower_is_finite(tower: Any, pixel_values: Any, grid_thw: Any) -> tuple[bool, str | None]:
    """``(finite, first non-finite block)`` of the tower's fp16 forward on one image."""
    import torch

    first: list[str] = []
    handles = []
    for index, block in enumerate(getattr(tower, "blocks", [])):

        def hook(_m: Any, _a: Any, out: Any, index: int = index) -> None:
            value = output_tensor(out)
            if not first and value is not None and not bool(torch.isfinite(value).all()):
                first.append(f"block {index}")

        handles.append(block.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            device = next(tower.parameters()).device
            out = tower(pixel_values.to(device), grid_thw=grid_thw.to(device))
    finally:
        for handle in handles:
            handle.remove()
    finite = bool(torch.isfinite(output_tensor(out)).all())
    return finite, (first[0] if first else (None if finite else "merger"))
