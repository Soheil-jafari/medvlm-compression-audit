"""Distillation losses on cached teacher outputs (shared tokenizer required).

All functions take logits for the answer positions only, shaped ``(batch, answer
length, vocabulary)``. Padding positions are excluded by ``-100`` targets (cross-entropy)
or a boolean mask (distillation). Logits are converted to float32 first, so fp16
training does not lose precision in the softmax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

IGNORE_INDEX = -100


def answer_ce_loss(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over answer tokens (positions with target ``-100`` are ignored).

    Used by: E1 (ground-truth answers), E2 (ground-truth answers next to the KD term).
    """
    import torch.nn.functional as F

    if student_logits.shape[:2] != target_ids.shape:
        raise ValueError(
            f"logits {tuple(student_logits.shape)} do not match targets {tuple(target_ids.shape)}"
        )
    target_ids = target_ids.to(student_logits.device)  # a model split over two GPUs (teacher)
    valid = target_ids != IGNORE_INDEX
    if not valid.any():
        return student_logits.sum() * 0.0
    flat = student_logits.float().reshape(-1, student_logits.shape[-1])
    return (
        F.cross_entropy(flat, target_ids.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")
        / valid.sum()
    )


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Mean KL(teacher || student) over the masked answer positions, on the teacher's top-k tokens.

    Both distributions are restricted to the teacher's ``k`` cached tokens and
    renormalised over them after dividing the logits (student) or log-probabilities
    (teacher) by ``temperature``. Probability mass outside the top-k is not modelled:
    the cached top-k is all that is stored, and a temperature cannot be applied
    to a residual bucket of unknown shape. The cross-entropy term on the answer
    keeps the student's mass on the correct tokens. The result is scaled by
    ``temperature ** 2`` (the usual correction for soft targets).

    Used by: E2, E4, recovery (E5-E7, A1-A6).
    """
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (student_logits.shape[:2] == teacher_topk_ids.shape[:2] == mask.shape):
        raise ValueError("logits, top-k tensors and mask must share batch and length")
    if not mask.any():
        return student_logits.sum() * 0.0
    device = student_logits.device  # a model split over two GPUs
    teacher_topk_ids, teacher_topk_logprobs = (
        teacher_topk_ids.to(device),
        teacher_topk_logprobs.to(device),
    )
    mask = mask.to(device)
    gathered = student_logits.float().gather(-1, teacher_topk_ids.long()) / temperature
    log_student = F.log_softmax(gathered, dim=-1)
    log_teacher = F.log_softmax(teacher_topk_logprobs.float() / temperature, dim=-1)
    kl = (log_teacher.exp() * (log_teacher - log_student)).sum(-1)
    return (kl * mask).sum() / mask.sum() * temperature**2


def combined_kd_loss(ce: torch.Tensor, kl: torch.Tensor, *, kd_weight: float) -> torch.Tensor:
    """Return ``(1 - kd_weight) * ce + kd_weight * kl``. Used by: E2, E4, recovery."""
    if not 0.0 <= kd_weight <= 1.0:
        raise ValueError("kd_weight must be between 0 and 1")
    return (1.0 - kd_weight) * ce + kd_weight * kl
