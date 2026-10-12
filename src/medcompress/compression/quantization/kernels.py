"""Low-precision modules for the quantized decoder layers, and the check that they are used.

A quantized arm only counts if its decoder's linear layers really hold low-precision weights
and run a library kernel; otherwise its memory and speed would be those of a 16-bit model
(D08 section 5). Three kinds of module can replace a decoder ``nn.Linear``:

* a **GPTQModel kernel module** (``gptq`` recipes, E3 ``w4``/``w8``, E4, E8, E9): built from
  our codes and scales (``grid.py``) by GPTQModel's own packing, then checked against our
  dequantized weights on random inputs (:func:`verify_linear`), so a format mismatch between
  library versions cannot pass silently;
* a **bitsandbytes ``Linear8bitLt``** (``llm_int8``, E3 ``w8a8``): LLM.int8 with 16-bit
  outliers;
* :class:`ReferenceQuantLinear`: our packed storage with a plain dequantize-and-multiply
  forward. It is correct and small in memory but is **not** a low-precision kernel; the tests
  and CPU runs use it, and :func:`kernel_report` never counts it as one.

GPTQModel's internal class layout differs between versions; :func:`build_gptq_linear` takes the
class to use as a dotted path (``quantization.kernel_class``), which the smoke test RC-Q0 picks
and records.

torch, gptqmodel and bitsandbytes are imported only inside the functions that need them.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from medcompress.compression.quantization import grid

if TYPE_CHECKING:
    import torch

DECODER_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
"""The linear layers of a Qwen decoder layer that the quantized arms replace (D08)."""
PARENT_OF = {name: "self_attn" for name in DECODER_TARGETS[:4]} | {
    name: "mlp" for name in DECODER_TARGETS[4:]
}
"""The sub-module of a decoder layer that holds each target (``self_attn`` or ``mlp``)."""
TORCH_KERNEL_CLASS = "gptqmodel.nn_modules.qlinear.torch.TorchLinear"
"""GPTQModel's portable kernel module; it also packs the GPTQ buffers for faster classes."""
LLM_INT8_THRESHOLD = 6.0
"""Outlier threshold of LLM.int8 (the value of the paper; TODO: verify against the bitsandbytes
default printed by RC-Q0)."""


def target_path(layer_index: int, name: str) -> str:
    """``layer<i>.<self_attn|mlp>.<name>``: the key of one quantized matrix."""
    if name not in PARENT_OF:
        raise ValueError(f"unknown decoder target {name!r}; known: {DECODER_TARGETS}")
    return f"layer{layer_index}.{PARENT_OF[name]}.{name}"


def decoder_targets(
    module: Any, names: Sequence[str] = DECODER_TARGETS
) -> list[tuple[str, Any, str]]:
    """``(key, parent module, attribute)`` of every target linear of the decoder stack."""
    from medcompress.compression.pruning.surgery import find_decoder_stack

    stack = find_decoder_stack(getattr(module, "module", module))
    out = []
    for i, layer in enumerate(stack.layers):
        for name in names:
            parent = getattr(layer, PARENT_OF[name])
            out.append((target_path(i, name), parent, name))
    return out


# ------------------------------------------------------------------ the kernel check


@dataclass(frozen=True)
class KernelReport:
    """What the decoder's target layers are after quantization."""

    n_targets: int
    classes: dict[str, int] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every target is a low-precision library module."""
        return self.n_targets > 0 and not self.problems


def classify_module(module: Any) -> tuple[str, str | None]:
    """``(class path, problem or None)`` for one target layer (duck-typed, no torch needed).

    A plain ``torch.nn.Linear`` or :class:`ReferenceQuantLinear` is a problem; so is a module
    that holds no integer-typed weight tensor (``qweight`` or ``weight``).
    """
    cls = type(module)
    path = f"{cls.__module__}.{cls.__qualname__}"
    if getattr(cls, "REFERENCE_ONLY", False):
        return path, "reference module (dequantizes in 16 bit, no low-precision kernel)"
    if cls.__qualname__ == "Linear" and cls.__module__.startswith("torch.nn"):
        return path, "plain 16-bit Linear"
    tensors = [getattr(module, name, None) for name in ("qweight", "weight")]
    dtypes = [str(getattr(t, "dtype", "")) for t in tensors if t is not None]
    if not any("int" in d for d in dtypes):
        return path, f"no integer weight tensor (dtypes {dtypes})"
    return path, None


def kernel_report(model: Any, names: Sequence[str] = DECODER_TARGETS) -> KernelReport:
    """Classify every target layer of the decoder (see :func:`classify_module`)."""
    classes: dict[str, int] = {}
    problems = []
    targets = decoder_targets(model, names)
    for key, parent, attr in targets:
        path, problem = classify_module(getattr(parent, attr))
        classes[path] = classes.get(path, 0) + 1
        if problem:
            problems.append(f"{key}: {problem}")
    return KernelReport(len(targets), classes, tuple(problems))


# ------------------------------------------------------------------- module builders


def _reference_class() -> type:
    import torch

    class ReferenceQuantLinear(torch.nn.Module):
        """Packed codes and 16-bit scales; forward dequantizes and multiplies (no kernel).

        Holds exactly the storage of ``grid.pack_codes`` (``qweight`` int32, ``scales``), so
        its memory is that of the quantized weights; it exists for tests and CPU runs.
        """

        REFERENCE_ONLY = True

        def __init__(
            self,
            qweight: Any,
            scales: Any,
            bias: Any | None,
            *,
            bits: int,
            group_size: int,
            in_features: int,
            dtype: Any,
        ) -> None:
            super().__init__()
            self.bits, self.group_size, self.in_features = bits, group_size, in_features
            self.out_features = int(scales.shape[0])
            self.register_buffer("qweight", qweight)
            self.register_buffer("scales", scales.to(dtype))
            self.register_buffer("bias", None if bias is None else bias.to(dtype))

        def dequantized(self) -> Any:
            codes = grid.unpack_codes(self.qweight.cpu().numpy(), self.bits, self.in_features)
            weight = grid.dequantize(
                codes, self.scales.float().cpu().numpy(), self.bits, self.group_size
            )
            return torch.as_tensor(weight, dtype=self.scales.dtype, device=self.scales.device)

        def forward(self, x: Any) -> Any:
            return torch.nn.functional.linear(x, self.dequantized().to(x.dtype), self.bias)

    return ReferenceQuantLinear


def build_reference_linear(
    codes: np.ndarray,
    scales: np.ndarray,
    bias: Any | None,
    *,
    bits: int,
    group_size: int,
    dtype: Any,
    device: Any,
) -> Any:
    """A :class:`ReferenceQuantLinear` from codes ``(out, in)`` and scales ``(out, groups)``."""
    import torch

    cls = _reference_class()
    qweight = torch.as_tensor(grid.pack_codes(codes, bits), device=device)
    return cls(
        qweight,
        torch.as_tensor(scales, device=device),
        None if bias is None else bias.detach().to(device),
        bits=bits,
        group_size=group_size,
        in_features=codes.shape[1],
        dtype=dtype,
    )


def _import_class(dotted: str) -> type:
    module_name, _, name = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"not a dotted class path: {dotted!r}")
    return getattr(importlib.import_module(module_name), name)


def build_gptq_linear(
    codes: np.ndarray,
    scales: np.ndarray,
    bias: Any | None,
    *,
    bits: int,
    group_size: int,
    kernel_class: str,
    dtype: Any,
    device: Any,
) -> Any:
    """A GPTQModel kernel module holding exactly our codes and scales.

    GPTQModel packs the buffers itself (``TorchLinear.pack`` with our dequantized weights,
    our scales, the zero point ``2**(bits - 1)`` and ``g_idx = column // group_size``), so the
    library's own format conventions are applied. For another ``kernel_class`` the packed
    buffers are copied into it before ``post_init`` (where kernels such as Marlin repack).
    The result is checked with :func:`verify_linear`; a mismatch raises.
    """
    import torch

    out_features, in_features = codes.shape
    groups = grid.check_shape(in_features, bits, group_size)
    weight = grid.dequantize(codes, scales, bits, group_size)
    linear = torch.nn.Linear(in_features, out_features, bias=bias is not None)
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
        if bias is not None:
            linear.bias.copy_(bias.detach().float().cpu())
    scale_t = torch.as_tensor(scales, dtype=torch.float32)
    zeros = torch.full((out_features, groups), grid.zero_point(bits), dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    kwargs = {
        "bits": bits,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias is not None,
    }
    packer = _import_class(TORCH_KERNEL_CLASS)(**kwargs)
    packer.pack(linear, scale_t, zeros, g_idx)
    module = packer
    if kernel_class != TORCH_KERNEL_CLASS:
        module = _import_class(kernel_class)(**kwargs)
        for name, tensor in packer.state_dict().items():
            target = getattr(module, name, None)
            if target is None:
                raise ValueError(f"{kernel_class} has no buffer {name!r} to load")
            target.data = tensor.clone()
    module = module.to(device)
    module.post_init()
    module.eval()
    verify_linear(module, torch.as_tensor(weight), bias, dtype=dtype)
    return module


def build_int8_linear(linear: Any, threshold: float = LLM_INT8_THRESHOLD) -> Any:
    """A bitsandbytes ``Linear8bitLt`` (LLM.int8) with the weights of ``linear``.

    The int8 conversion happens when the module is moved to the GPU.
    """
    import bitsandbytes as bnb
    import torch

    device = linear.weight.device
    has_bias = linear.bias is not None
    module = bnb.nn.Linear8bitLt(
        linear.in_features,
        linear.out_features,
        bias=has_bias,
        has_fp16_weights=False,
        threshold=threshold,
    )
    module.weight = bnb.nn.Int8Params(
        linear.weight.data.detach().cpu(), requires_grad=False, has_fp16_weights=False
    )
    if has_bias:
        module.bias = torch.nn.Parameter(linear.bias.data.detach().cpu(), requires_grad=False)
    return module.to(device)


def verify_linear(
    module: Any,
    reference_weight: torch.Tensor,
    bias: Any | None,
    *,
    dtype: Any,
    rows: int = 8,
    tolerance: float = 1e-2,
    seed: int = 0,
) -> float:
    """Compare a quantized module with ``x @ W^T + b`` on random inputs; return the error.

    The error is ``max|out - ref| / max|ref|``; above ``tolerance`` it raises, because the
    module then does not compute with the weights we quantized (wrong format or kernel).
    """
    import torch

    device = next((t.device for t in module.buffers()), torch.device("cpu"))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, reference_weight.shape[1], generator=generator).to(device, dtype)
    with torch.no_grad():
        ref = x.float() @ reference_weight.to(device).float().T
        if bias is not None:
            ref = ref + bias.to(device).float()
        out = module(x).float()
    error = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    if not error <= tolerance:
        raise ValueError(
            f"{type(module).__qualname__} does not reproduce the quantized weights "
            f"(relative error {error:.3g} > {tolerance}); check the kernel class and version"
        )
    return error
