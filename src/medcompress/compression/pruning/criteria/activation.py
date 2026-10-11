"""Activation criterion (Wanda-style): |weight| x input activation norm on calibration data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("activation")
class ActivationCriterion(PruningCriterion):
    """Structured, Wanda-style score aggregated over each unit. Used by: E5:activation."""

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return activation-weighted magnitude per unit using generic calibration prompts."""
        raise NotImplementedError("TODO: Wanda-style structured importance (E5:activation).")
