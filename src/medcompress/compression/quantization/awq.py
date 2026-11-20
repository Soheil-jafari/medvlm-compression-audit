"""AWQ through GPTQModel (E3 ``awq_w4``; pre-registration section 10, amendment of 2026-11-20).

AWQ (Lin et al., MLSys 2024) scales the input channels of each decoder linear layer by an
activation-aware factor searched on calibration activations, folds the inverse into the
preceding operation and rounds the weights to 4 bit in groups of 128 (asymmetric, with zero
points). The quantization and the kernel come from GPTQModel 7.6.0, already pinned for E3
(``requirements/train.txt``; AutoAWQ is archived):

* quantization: ``QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM)`` on a saved copy
  of the merged E2, with **our** calibration inputs: the same 128 cleaned calibration images,
  each with its question and ground-truth answer, built by the GPTQ code
  (:func:`~medcompress.compression.quantization.ptq.calibration_batches`). GPTQModel's own
  Qwen2.5-VL calibration would build prompts without the answer, so its ``prepare_dataset`` is
  replaced by our inputs;
* kernel: ``AwqTorchLinear`` (``BACKEND.AWQ_TORCH``), pure PyTorch, runs on a T4 (compute
  capability 7.5); AWQ Marlin needs 8.0 or more. It is the AWQ counterpart of the
  ``TorchLinear`` kernel E3's GPTQ arms use, so the latency comparison stays kernel for kernel.

The targets are GPTQModel's Qwen2.5-VL decoder modules (q/k/v/o, gate/up/down of every decoder
layer), the same as E3 ``w4``; :func:`check_targets` refuses any other set.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AWQ_SUBDIR = "awq"
"""The GPTQModel checkpoint inside the run's ``quantized`` folder (next to the manifest)."""
SOURCE_SUBDIR = "awq_source"
"""The temporary saved copy of the merged parent that GPTQModel loads (removed afterwards)."""
CALIBRATION_KEYS: tuple[str, ...] = (
    "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids",
)  # fmt: skip
"""The model inputs GPTQModel's Qwen2.5-VL input capture reads."""
KERNEL = "AwqTorchLinear"
QUANT_LOG = "quant_log.csv"
_DECODER = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")  # fmt: skip


def check_targets(targets: Sequence[str]) -> None:
    """GPTQModel's Qwen2.5-VL definition quantizes the seven decoder projections; the recipe
    must name exactly those (E3 ``w4``'s targets)."""
    wanted = {name.split(".")[-1] for name in _DECODER}
    if set(targets) != wanted:
        raise ValueError(f"AWQ through GPTQModel quantizes {sorted(wanted)}; the recipe names "
                         f"{sorted(targets)}")  # fmt: skip


def calibration_inputs(batches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Our calibration batches, restricted to the inputs GPTQModel reads (on the CPU)."""
    out = []
    for batch in batches:
        missing = {"input_ids", "pixel_values", "image_grid_thw"} - set(batch)
        if missing:
            raise ValueError(f"a calibration batch lacks {sorted(missing)}")
        out.append({k: batch[k] for k in CALIBRATION_KEYS if k in batch})
    return out


def quantize_folder(
    source: Path | str,
    target: Path | str,
    batches: Sequence[Mapping[str, Any]],
    *,
    bits: int,
    group_size: int,
    device: str | None = None,
) -> dict[str, Any]:
    """AWQ-quantize the saved model at ``source`` with ``batches`` and save it to ``target``.
    Returns what the manifest records (modules quantized, the mean AWQ loss, the kernel)."""
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.quantization import FORMAT, METHOD

    inputs = calibration_inputs(batches)
    where = {"device": device} if device else {}
    config = QuantizeConfig(bits=bits, group_size=group_size, quant_method=METHOD.AWQ,
                            format=FORMAT.GEMM, **where)  # fmt: skip
    import contextlib

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    # GPTQModel writes its AWQ logs to ./logs: run next to the checkpoint, not in the clone
    with contextlib.chdir(Path(target).parent):
        model = GPTQModel.load(str(Path(source).resolve()), config)
        # our inputs exactly (image, question, ground-truth answer), as the GPTQ arms are
        # calibrated; GPTQModel's own Qwen2.5-VL preparation would drop the answer
        model.prepare_dataset = lambda *args, **kwargs: inputs
        model.quantize(inputs, batch_size=1, backend=BACKEND.AWQ_TORCH)
        model.save(str(Path(target).resolve()))
        del model
    return {"kernel": KERNEL, "calibration_examples": len(inputs), **read_quant_log(target)}


def read_quant_log(folder: Path | str) -> dict[str, Any]:
    """The modules GPTQModel quantized (its ``quant_log.csv``) and their mean AWQ loss."""
    path = Path(folder) / QUANT_LOG
    if not path.is_file():
        return {"quantized_modules": 0}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    losses = []
    for row in rows:
        try:
            losses.append(float(row.get("loss", "")))
        except ValueError:
            continue
    return {
        "quantized_modules": len(rows),
        "layers": len({row.get("layer") for row in rows}),
        "mean_loss": sum(losses) / len(losses) if losses else None,
    }


def quantized_weight_bytes(folder: Path | str) -> int:
    """Bytes of the AWQ tensors (packed weights, zero points, scales) from the safetensors
    headers, the kernel's storage of the quantized matrices."""
    from safetensors import safe_open

    total = 0
    for path in sorted(Path(folder).glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith((".qweight", ".qzeros", ".scales")):
                    sl = handle.get_slice(key)
                    shape = sl.get_shape()
                    count = 1
                    for n in shape:
                        count *= int(n)
                    total += count * _dtype_bytes(sl.get_dtype())
    return total


def _dtype_bytes(dtype: str) -> int:
    return {"I32": 4, "F16": 2, "BF16": 2, "F32": 4, "I8": 1, "U8": 1, "I64": 8}.get(dtype, 4)


def load_folder(folder: Path | str, *, device: str, dtype: Any) -> Any:
    """The AWQ model saved in ``folder`` with the ``AwqTorchLinear`` kernel, as a transformers
    module; refused unless every decoder projection uses that kernel."""
    from gptqmodel import BACKEND, GPTQModel

    loaded = GPTQModel.load(str(folder), backend=BACKEND.AWQ_TORCH, device=device, dtype=dtype)
    module = loaded.model
    decoder = [(name, sub) for name, sub in module.named_modules()
               if "language_model.layers." in name and name.endswith(_DECODER)]  # fmt: skip
    wrong = [name for name, sub in decoder if type(sub).__name__ != KERNEL]
    found = len(decoder)
    if wrong or found == 0:
        raise ValueError(f"{folder}: {len(wrong)} of {found} decoder projections are not "
                         f"{KERNEL}, e.g. {wrong[:3]}")  # fmt: skip
    return module
