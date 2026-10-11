"""Distillation losses on cached teacher outputs (shared tokenizer required)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def answer_ce_loss(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on answer tokens only (prompt and image tokens masked).

    Used by: E1 (ground-truth answers), E2 (teacher answers as hard targets).
    """
    raise NotImplementedError("TODO: masked answer-token cross-entropy (E1, E2).")


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """KL between the teacher's cached top-k distribution and the student's.

    How the probability mass outside the top-k is handled (renormalise vs a
    residual bucket) must be decided and documented.

    Used by: E2, E4, recovery (E5-E7, A1-A6).
    """
    raise NotImplementedError("TODO: top-k KL with temperature (E2, E4, recovery).")


def combined_kd_loss(ce: torch.Tensor, kl: torch.Tensor, *, kd_weight: float) -> torch.Tensor:
    """Return (1 - kd_weight) * ce + kd_weight * kl. Used by: E2, E4, recovery."""
    raise NotImplementedError("TODO: weighted CE + KL.")
