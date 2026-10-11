"""Wrapper interface around a generative medical VLM (image + question -> answer).

Teacher and student candidates (configs/models/) must share a tokenizer so that
offline distillation on cached top-k token probabilities is well defined.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit


@dataclass(frozen=True)
class GenerationOutput:
    """Free-text answer plus optional per-token top-k log-probabilities."""

    text: str
    token_ids: Sequence[int] = ()
    topk_token_ids: Sequence[Sequence[int]] | None = None
    topk_logprobs: Sequence[Sequence[float]] | None = None


@dataclass(frozen=True)
class YesNoScore:
    """Single-forward-pass score of a yes/no probe.

    ``logp_yes`` / ``logp_no`` are log-probabilities of the answer tokens after
    the prompt (how multi-token answers are aggregated is decided per model).
    """

    logp_yes: float
    logp_no: float

    @property
    def predicted(self) -> str:
        return "yes" if self.logp_yes >= self.logp_no else "no"


class VLMWrapper(ABC):
    """Uniform interface used by distillation, compression and evaluation.

    Used by: every experiment (E0-E9, A1-A6).
    """

    @abstractmethod
    def generate(
        self,
        images: Sequence[Path | Any],
        questions: Sequence[str],
        *,
        max_new_tokens: int,
        return_topk: int | None = None,
    ) -> list[GenerationOutput]:
        """Generate answers (greedy unless configured otherwise).

        With ``return_topk`` set, also return per-token top-k log-probabilities
        (needed for the E0 teacher cache).
        """
        raise NotImplementedError("TODO: batched generation for the chosen VLM family.")

    @abstractmethod
    def score_yes_no(
        self, images: Sequence[Path | Any], questions: Sequence[str]
    ) -> list[YesNoScore]:
        """Score yes/no probes in one forward pass (no generation).

        Used by: clinical criterion (E7, A1-A6), mask audit, calibration metrics.
        """
        raise NotImplementedError("TODO: one forward pass, read yes/no answer log-probs.")

    @abstractmethod
    def list_prunable_units(self, *, include_vision: bool = False) -> list[PrunableUnit]:
        """Enumerate prunable units (layers, heads respecting GQA, MLP channel groups).

        The vision-language projector is never prunable. Delegates to
        :func:`medcompress.compression.pruning.units.enumerate_units`.
        """
        raise NotImplementedError("TODO: enumerate prunable units of this architecture.")

    def num_parameters(self) -> int:
        """Return the total parameter count (after any physical pruning)."""
        raise NotImplementedError("TODO: count parameters.")

    def tokenizer_fingerprint(self) -> str:
        """Return a hash of the tokenizer vocabulary (teacher and student must match)."""
        raise NotImplementedError("TODO: hash tokenizer vocab + special tokens.")
