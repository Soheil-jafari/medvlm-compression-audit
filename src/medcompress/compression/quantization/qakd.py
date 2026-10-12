"""Quantization-aware distillation for E4 (D08 option B, a reduced GRACE).

GRACE (Chen, Habibian, Benini, Li, ICML 2026, arXiv:2601.22709) trains a vision-language
model with 4-bit weights while distilling from a teacher. E4 keeps the parts that fit a T4 and
an offline teacher cache:

* **Quantizer:** 4-bit symmetric group-wise quantization (groups of 128 input columns, the grid
  of ``grid.py``) with a learned step size per group, as in learned step-size quantization
  (LSQ; Esser et al., arXiv:1902.08153): rounding passes gradients straight through, and the
  step size gets the LSQ gradient with the scale ``1 / sqrt(group_size * Qp)``. Step sizes start
  at the scales of plain rounding (``grid.symmetric_scales``).
* **Adapter:** the decoder's weights stay frozen; a LoRA update ``B A`` is learned *through*
  the quantizer: every forward pass uses ``quantize(W + (alpha / r) B A)``, so the exported
  model (the merged weights rounded with the learned step sizes) is exactly the trained one.
  LoRA dropout is not applied (the update is merged into the weight before rounding).
* **Loss:** ``(1 - kd_weight) * CE + kd_weight * GDKD`` with GRACE's confidence-gated decoupled
  distillation (``grace_modules.py``: ``alpha = 1``, ``beta = 4``, ``T = 2``), computed on the
  teacher's cached top-20 tokens (:func:`gated_dkd_numpy` is the reference).

Left out (deviations, D08): GRACE's relational alignment of visual tokens (needs the teacher
live), its adaptive loss weight, and full fine-tuning of all weights.

The export writes the same checkpoint format as the post-training arms (``ptq.save_quantized``),
so E4 is evaluated and timed with the same kernel modules as E3 ``w4``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from medcompress.compression.quantization import grid
from medcompress.config import ConfigError, find_todos

if TYPE_CHECKING:
    import torch

    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

DKD_ALPHA = 1.0
DKD_BETA = 4.0
DKD_TEMPERATURE = 2.0
"""GRACE's defaults (``grace_modules.py``, read 2026-10-12)."""
_EPS = 1e-8


# --------------------------------------------------------------- the loss (reference)


def confidence_gate_numpy(topk_logprobs: np.ndarray, vocab_size: int) -> np.ndarray:
    """``exp(-H / log V)`` per position from the teacher's cached top-k log-probabilities.

    ``H`` is the entropy of the teacher at temperature 1. Only the top-k probabilities are
    known, so the mass outside them counts as one extra outcome (an approximation of GRACE,
    which has the full distribution).
    """
    p = np.exp(np.asarray(topk_logprobs, dtype=np.float64))
    tail = np.clip(1.0 - p.sum(axis=-1), 0.0, 1.0)
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)
    entropy -= tail * np.log(np.clip(tail, 1e-12, None))
    return np.exp(-entropy / math.log(vocab_size))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def dkd_per_token_numpy(
    student_topk_logits: np.ndarray,
    teacher_topk_logprobs: np.ndarray,
    target_index: np.ndarray,
    *,
    alpha: float = DKD_ALPHA,
    beta: float = DKD_BETA,
    temperature: float = DKD_TEMPERATURE,
) -> np.ndarray:
    """``(alpha * TCKD + beta * NCKD) * T^2`` per position, on the cached top-k tokens.

    ``student_topk_logits`` and ``teacher_topk_logprobs`` are ``(..., k)``; both are divided by
    ``T`` and renormalised over the k tokens. ``target_index`` (``(...)``) is the position of
    the target token among the k. TCKD is the KL of the two-way split {target, rest}; NCKD the
    KL of the distributions over the other ``k - 1`` tokens (GRACE's ``grace_modules.py``).
    """
    s = np.asarray(student_topk_logits, dtype=np.float64) / temperature
    t = np.asarray(teacher_topk_logprobs, dtype=np.float64) / temperature
    idx = np.asarray(target_index)[..., None]
    log_ps, log_pt = _log_softmax(s), _log_softmax(t)
    ps_t = np.exp(np.take_along_axis(log_ps, idx, -1))[..., 0]
    pt_t = np.exp(np.take_along_axis(log_pt, idx, -1))[..., 0]
    bin_s = np.clip(np.stack([ps_t, 1.0 - ps_t], -1), _EPS, None)
    bin_t = np.clip(np.stack([pt_t, 1.0 - pt_t], -1), _EPS, None)
    tckd = (bin_t * (np.log(bin_t) - np.log(bin_s))).sum(-1)
    masked = np.zeros_like(s, dtype=bool)
    np.put_along_axis(masked, idx, True, -1)
    floor = np.finfo(np.float64).min
    log_ps_nt = _log_softmax(np.where(masked, floor, s))
    log_pt_nt = _log_softmax(np.where(masked, floor, t))
    term = np.where(masked, 0.0, np.exp(log_pt_nt) * (log_pt_nt - log_ps_nt))
    nckd = term.sum(-1)
    return (alpha * tckd + beta * nckd) * temperature**2


def gated_dkd_numpy(
    student_logits: np.ndarray,
    topk_ids: np.ndarray,
    topk_logprobs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = DKD_ALPHA,
    beta: float = DKD_BETA,
    temperature: float = DKD_TEMPERATURE,
) -> float:
    """Gate-weighted mean of the decoupled distillation over the masked answer positions.

    ``student_logits`` ``(B, A, V)``; ``topk_ids`` / ``topk_logprobs`` ``(B, A, k)``;
    ``targets`` ``(B, A)`` (the teacher's cached token); ``mask`` ``(B, A)``. Positions whose
    target is not among the cached top-k are left out. Reference for :func:`gated_dkd_loss`.
    """
    logits = np.asarray(student_logits, dtype=np.float64)
    ids = np.asarray(topk_ids)
    found = ids == np.asarray(targets)[..., None]
    valid = np.asarray(mask, dtype=bool) & found.any(-1)
    if not valid.any():
        return 0.0
    index = found.argmax(-1)
    gathered = np.take_along_axis(logits, ids, -1)
    per_token = dkd_per_token_numpy(
        gathered, topk_logprobs, index, alpha=alpha, beta=beta, temperature=temperature
    )
    gate = confidence_gate_numpy(topk_logprobs, logits.shape[-1])
    weight = gate * valid
    return float((weight * per_token).sum() / max(weight.sum(), _EPS))


# ------------------------------------------------------------------ the loss (torch)


def gated_dkd_loss(
    student_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = DKD_ALPHA,
    beta: float = DKD_BETA,
    temperature: float = DKD_TEMPERATURE,
) -> torch.Tensor:
    """The torch version of :func:`gated_dkd_numpy` (same shapes; float32 inside).

    Used by: E4 (training mode ``qakd``).
    """
    import torch
    import torch.nn.functional as F

    found = topk_ids == targets.unsqueeze(-1)
    valid = mask.bool() & found.any(-1)
    if not valid.any():
        return student_logits.sum() * 0.0
    index = found.float().argmax(-1, keepdim=True)
    s = student_logits.float().gather(-1, topk_ids.long()) / temperature
    t = topk_logprobs.float() / temperature
    log_ps, log_pt = F.log_softmax(s, -1), F.log_softmax(t, -1)
    ps_t = log_ps.gather(-1, index).exp().squeeze(-1)
    pt_t = log_pt.gather(-1, index).exp().squeeze(-1)
    bin_s = torch.stack([ps_t, 1.0 - ps_t], -1).clamp_min(_EPS)
    bin_t = torch.stack([pt_t, 1.0 - pt_t], -1).clamp_min(_EPS)
    tckd = (bin_t * (bin_t.log() - bin_s.log())).sum(-1)
    masked = torch.zeros_like(s, dtype=torch.bool).scatter(-1, index, True)
    floor = torch.finfo(s.dtype).min
    log_ps_nt = F.log_softmax(s.masked_fill(masked, floor), -1)
    log_pt_nt = F.log_softmax(t.masked_fill(masked, floor), -1)
    term = (log_pt_nt.exp() * (log_pt_nt - log_ps_nt)).masked_fill(masked, 0.0)
    per_token = (alpha * tckd + beta * term.sum(-1)) * temperature**2
    with torch.no_grad():
        p = topk_logprobs.float().exp()
        tail = (1.0 - p.sum(-1)).clamp(0.0, 1.0)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) - tail * tail.clamp_min(1e-12).log()
        gate = torch.exp(-entropy / math.log(student_logits.shape[-1]))
    weight = gate * valid
    return (weight * per_token).sum() / weight.sum().clamp_min(_EPS)


# ------------------------------------------------------------------- the quantizer


def lsq_reference(
    weight: np.ndarray, scales: np.ndarray, bits: int, group_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(fake-quantized weight, d out / d scale)`` of learned step-size quantization (NumPy).

    The forward is ``grid.dequantize(grid.quantize_codes(...))``; the derivative of each output
    with respect to its group's step size is ``round(w / s) - w / s`` inside the grid and the
    clipped level (``-z`` or ``2**bits - 1 - z``) outside (LSQ). Reference for :func:`lsq_quantize`.
    """
    w = np.asarray(weight, dtype=np.float64)
    s = grid.expand_groups(scales, group_size)
    low, high = -grid.zero_point(bits), grid.max_code(bits) - grid.zero_point(bits)
    ratio = w / s
    rounded = np.clip(np.rint(ratio), low, high)
    grad = np.where(ratio <= low, low, np.where(ratio >= high, high, np.rint(ratio) - ratio))
    return rounded * s, grad


def lsq_quantize(weight: Any, scales: Any, bits: int, group_size: int) -> Any:
    """Fake-quantize ``weight`` (``out x in``) with learnable ``scales`` (``out x groups``).

    Straight-through for the weight (gradient 1 inside the grid, 0 where it clips) and the LSQ
    gradient, scaled by ``1 / sqrt(group_size * Qp)``, for the step sizes. Both run in float32.
    Only the ratio ``w / s`` is kept for the backward pass.
    """
    return _lsq_function(bits, group_size).apply(weight.float(), scales.float())


@functools.cache
def _lsq_function(bits: int, group_size: int) -> Any:
    """The autograd function of :func:`lsq_quantize` for one grid (built once, then reused)."""
    import torch

    low, high = -grid.zero_point(bits), grid.max_code(bits) - grid.zero_point(bits)
    g = 1.0 / math.sqrt(group_size * high)

    class _LSQ(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, w: Any, s: Any) -> Any:
            s_full = s.repeat_interleave(group_size, dim=1)
            ratio = w / s_full
            ctx.save_for_backward(ratio)
            return torch.clamp(torch.round(ratio), low, high) * s_full

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> tuple[Any, Any]:
            (ratio,) = ctx.saved_tensors
            inside = (ratio > low) & (ratio < high)
            grad_w = grad_out * inside
            d_s = torch.where(
                ratio <= low,
                torch.full_like(ratio, float(low)),
                torch.where(
                    ratio >= high, torch.full_like(ratio, float(high)), torch.round(ratio) - ratio
                ),
            )
            grad_s = (grad_out * d_s * g).reshape(ratio.shape[0], -1, group_size).sum(-1)
            return grad_w, grad_s

    return _LSQ


def _qat_linear_class() -> type:
    import torch

    class QATLinear(torch.nn.Module):
        """A decoder linear layer trained with LoRA through the 4-bit LSQ quantizer (E4)."""

        def __init__(
            self, linear: Any, *, bits: int, group_size: int, rank: int, alpha: float
        ) -> None:
            super().__init__()
            out_features, in_features = linear.weight.shape
            grid.check_shape(in_features, bits, group_size)
            self.bits, self.group_size, self.scaling = bits, group_size, alpha / rank
            self.in_features, self.out_features = in_features, out_features
            self.register_buffer("base", linear.weight.detach())
            self.register_buffer("bias", None if linear.bias is None else linear.bias.detach())
            device = linear.weight.device
            self.lora_A = torch.nn.Parameter(torch.empty(rank, in_features, device=device))
            self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # the LoRA default
            start = grid.symmetric_scales(self.base.float().cpu().numpy(), bits, group_size)
            self.step = torch.nn.Parameter(
                torch.as_tensor(start, dtype=torch.float32, device=device)
            )

        def merged(self) -> Any:
            """``W + scaling * B A`` in float32 (the weight the quantizer sees)."""
            return self.base.float() + self.scaling * (self.lora_B @ self.lora_A)

        def forward(self, x: Any) -> Any:
            weight = lsq_quantize(
                self.merged(), self.step.abs().clamp_min(1e-8), self.bits, self.group_size
            )
            return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)

    return QATLinear


def install_qat(
    model: VLMWrapper | Any,
    targets: Sequence[str],
    *,
    bits: int,
    group_size: int,
    rank: int,
    alpha: float,
) -> list[Any]:
    """Replace the decoder's target layers by :class:`QATLinear`; freeze everything else.

    Returns the trainable parameters (LoRA matrices and step sizes).
    """
    from medcompress.compression.quantization.kernels import decoder_targets

    module = getattr(model, "module", model)
    for p in module.parameters():
        p.requires_grad_(False)
    cls = _qat_linear_class()
    params = []
    for _key, parent, attr in decoder_targets(model, targets):
        qat = cls(getattr(parent, attr), bits=bits, group_size=group_size, rank=rank, alpha=alpha)
        setattr(parent, attr, qat)
        params += [qat.lora_A, qat.lora_B, qat.step]
    return params


def export_qat(model: VLMWrapper | Any, targets: Sequence[str]) -> dict[str, Any]:
    """Codes and learned scales of every :class:`QATLinear` (the trained model, exactly)."""
    import torch

    from medcompress.compression.quantization.kernels import decoder_targets
    from medcompress.compression.quantization.ptq import QuantizedMatrix

    records = {}
    with torch.no_grad():
        for key, parent, attr in decoder_targets(model, targets):
            qat = getattr(parent, attr)
            scales = qat.step.abs().clamp_min(1e-8).double().cpu().numpy()
            weight = qat.merged().double().cpu().numpy()
            codes = grid.quantize_codes(weight, scales, qat.bits, qat.group_size)
            records[key] = QuantizedMatrix(key, codes, scales, qat.bits, qat.group_size)
    return records


# ----------------------------------------------------------------------- the stage


@dataclass(frozen=True)
class QAKDSettings:
    """E4 settings: the w4 grid, the quantized targets and the distillation constants."""

    bits: int
    group_size: int
    targets: tuple[str, ...]
    alpha: float
    beta: float
    temperature: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> QAKDSettings:
        """Read ``quantization`` (recipe ``w4``, targets) and ``quantization.distillation``."""
        from medcompress.compression.quantization.ptq import PTQRecipe

        recipe = PTQRecipe.from_config(config)
        if recipe.backend != "gptq" or recipe.group_size is None:
            raise ConfigError("E4 exports with a GPTQ recipe (quantization.recipe: w4)")
        dist = (config.get("quantization") or {}).get("distillation") or {}
        if dist.get("loss", "gated_dkd") != "gated_dkd":
            raise ConfigError("E4 uses quantization.distillation.loss: gated_dkd (D08 option B)")
        values = {k: dist.get(k) for k in ("alpha", "beta", "temperature")}
        open_keys = [
            f"quantization.distillation.{k}"
            for k, v in values.items()
            if v is None or find_todos({"v": v})
        ]
        if open_keys:
            raise ConfigError(f"settings still open: {', '.join(open_keys)}")
        return cls(
            recipe.weight_bits,
            recipe.group_size,
            recipe.targets,
            float(values["alpha"]),
            float(values["beta"]),
            float(values["temperature"]),
        )


def run_qakd_training(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    student: VLMWrapper | None = None,
    train: Sequence[VQASample] | None = None,
    records: Mapping[str, Any] | None = None,
) -> None:
    """Stage entry point of E4 (``scripts/02_train_student.py`` with ``training.mode: qakd``).

    Loads the pretrained student, installs the quantizer with LoRA on the decoder's target
    layers, trains with the E2 recipe (same training pool, steps, optimiser) and the
    confidence-gated decoupled distillation, then saves the codes and learned scales with
    ``ptq.save_quantized`` (parent: the pretrained student) and the training numbers.
    ``student``, ``train`` and ``records`` replace what the stage loads (tests).

    Used by: E4 (3 seeds).
    """
    from medcompress.compression.distillation.offline_kd import (
        TrainConfig,
        data_seed,
        examples_from_samples,
        require_teacher_records,
        train_lora,
        training_pool,
    )
    from medcompress.compression.quantization.ptq import PTQRecipe, quantized_dir, save_quantized
    from medcompress.config import load_config
    from medcompress.utils.io import upsert_results

    if run is None:
        raise ConfigError("--exp is required")
    cfg = TrainConfig.from_config(config["training"])
    settings = QAKDSettings.from_config(config)
    seed_for_pool = data_seed(config)
    paths = config["paths"]
    if student is None:
        side = config.get("inputs", {}).get("image_longest_side")
        student_cfg = config.get("model", {}).get("config")
        open_keys = [
            n
            for n, v in (("model.config", student_cfg), ("inputs.image_longest_side", side))
            if v is None or find_todos({n: v})
        ]
        if open_keys:
            raise ConfigError(f"settings still open: {', '.join(open_keys)}")
        from medcompress.models.loading import ModelSpec, load_model

        student = load_model(
            ModelSpec.from_config(load_config(student_cfg)),
            device_map={"": 0},
            allow_candidate=True,
            image_longest_side=int(side),
        )
    if train is None:
        from medcompress.compression.pruning.stage import load_train_samples

        train = load_train_samples(config)
    if records is None:
        from medcompress.models.teacher_cache import load_teacher_cache

        records = {
            r.sample_id: r for r in load_teacher_cache(Path(paths["cache_root"]) / "teacher")
        }
    pool = training_pool(train, cfg.data_subset, seed_for_pool)
    examples = examples_from_samples(pool, Path(paths["data_root"]), records=records)
    require_teacher_records(pool, examples)
    install_qat(
        student,
        settings.targets,
        bits=settings.bits,
        group_size=settings.group_size,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
    )
    extra = {
        **dict(cfg.extra),
        "kd_loss": "gated_dkd",
        "dkd_alpha": settings.alpha,
        "dkd_beta": settings.beta,
        "dkd_temperature": settings.temperature,
    }
    from dataclasses import replace

    result = train_lora(
        student, examples, replace(cfg, extra=extra), seed=run.seed, attach_adapter=False
    )
    quantized = export_qat(student, settings.targets)
    recipe = PTQRecipe.from_config(config)
    directory = save_quantized(
        quantized,
        recipe,
        quantized_dir(config, run),
        start_from={"experiment": "pretrained", "budget": None, "seed": run.seed},
        extra={
            "qat": {
                "loss": "gated_dkd",
                "alpha": settings.alpha,
                "beta": settings.beta,
                "temperature": settings.temperature,
            }
        },
    )
    values = {
        "train_steps": result.steps,
        "train_examples_seen": result.examples_seen,
        "train_loss_final": result.final_loss(),
        "train_seconds": result.seconds,
        "quantized_matrices": len(quantized),
        "quantization_weight_bits": settings.bits,
    }
    if result.peak_gpu_bytes is not None:
        values["train_peak_gpu_bytes"] = result.peak_gpu_bytes
    upsert_results(
        [
            {
                "experiment": run.experiment_tag,
                "budget": "",
                "seed": run.seed,
                "metric": k,
                "value": float(v),
            }
            for k, v in values.items()
        ],
        run.run_dir(paths["results_root"]) / "metrics.csv",
    )
    print(f"E4: trained {result.steps} steps with 4-bit weights; saved {directory}")
